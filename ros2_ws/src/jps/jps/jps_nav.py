#!/usr/bin/env python3
"""JPS global path planner for ROS2 with hybrid obstacle checking.

Supports standalone mode (plans and drives via cmd_vel) and stacked mode
(publishes nav_msgs/Path for a downstream local planner). Requires a
pre-built static map loaded via the MAP_YAML env var.
"""

import heapq
import math
import os
import sys

import numpy as np
import yaml
from PIL import Image
from scipy.interpolate import CubicSpline
from scipy.ndimage import binary_dilation

import rclpy
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_interfaces.msg import NavStatus
from nav_interfaces.srv import CancelNav
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from tf2_ros import Buffer, TransformListener


class NavigatorConstants:
    """Default configuration constants for the JPS navigator."""

    ROBOT_RADIUS = 0.22
    SAFETY_CLEARANCE = 0.05

    LINEAR_SPEED = 0.2
    ANGULAR_SPEED = 0.5
    POSITION_TOLERANCE = 0.1
    ANGLE_TOLERANCE = 0.1
    CONTROL_TIMER_PERIOD = 0.1

    MAX_PATH_WAYPOINTS = 20
    TIGHT_SPACE_RADIUS = 3
    SIMPLIFICATION_EPSILON = 0.1
    MAX_JUMP_DEPTH = 500

    DEFAULT_NAMESPACE = '/don'


class JPSNavigator(Node):
    """ROS2 node for Jump Point Search path planning and waypoint following."""

    def __init__(self):
        _ns = NavigatorConstants.DEFAULT_NAMESPACE
        for i, arg in enumerate(sys.argv):
            if arg.startswith('namespace:='):
                _ns = arg.split(':=', 1)[1]
            elif arg == '-p' and i + 1 < len(sys.argv) and sys.argv[i+1].startswith('namespace:='):
                _ns = sys.argv[i+1].split(':=', 1)[1]

        _user_args = sys.argv[1:]
        _tf_remaps = ['-r', f'/tf:={_ns}/tf', '-r', f'/tf_static:={_ns}/tf_static']
        if '--ros-args' in _user_args:
            _combined = _user_args + _tf_remaps
        else:
            _combined = _user_args + ['--ros-args'] + _tf_remaps

        super().__init__('jps_navigator', cli_args=_combined, use_global_arguments=False)

        self.declare_parameter('namespace', NavigatorConstants.DEFAULT_NAMESPACE)
        self.ns = self.get_parameter('namespace').value

        default_radius = float(os.getenv('ROBOT_RADIUS', str(NavigatorConstants.ROBOT_RADIUS)))
        default_clearance = float(
            os.getenv('SAFETY_CLEARANCE', str(NavigatorConstants.SAFETY_CLEARANCE))
        )

        self.declare_parameter('robot_radius', default_radius)
        self.declare_parameter('safety_clearance', default_clearance)
        self.robot_radius = self.get_parameter('robot_radius').value
        self.safety_clearance = self.get_parameter('safety_clearance').value

        self.declare_parameter('linear_speed', NavigatorConstants.LINEAR_SPEED)
        self.declare_parameter('angular_speed', NavigatorConstants.ANGULAR_SPEED)
        self.declare_parameter('position_tolerance', NavigatorConstants.POSITION_TOLERANCE)
        self.declare_parameter('angle_tolerance', NavigatorConstants.ANGLE_TOLERANCE)
        self.declare_parameter('max_path_waypoints', NavigatorConstants.MAX_PATH_WAYPOINTS)
        self.declare_parameter('tight_space_radius', NavigatorConstants.TIGHT_SPACE_RADIUS)
        self.declare_parameter('simplification_epsilon', NavigatorConstants.SIMPLIFICATION_EPSILON)

        self.linear_speed = self.get_parameter('linear_speed').value
        self.angular_speed = self.get_parameter('angular_speed').value
        self.position_tolerance = self.get_parameter('position_tolerance').value
        self.angle_tolerance = self.get_parameter('angle_tolerance').value
        self.tight_space_radius = self.get_parameter('tight_space_radius').value
        self.simplification_epsilon = self.get_parameter('simplification_epsilon').value

        self.declare_parameter('stacked', False)
        self.standalone = not self.get_parameter('stacked').value

        self.plan_pub = self.create_publisher(Path, f'{self.ns}/jps/plan', 10)
        self.cmd_vel_pub = self.create_publisher(TwistStamped, f'{self.ns}/cmd_vel', 10)

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.odom_sub = self.create_subscription(
            Odometry,
            f'{self.ns}/odom',
            self.odom_callback,
            qos_profile,
        )
        self.goal_sub = self.create_subscription(
            PoseStamped,
            f'{self.ns}/goal_pose',
            self.goal_callback,
            10,
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self._planner_state = 'idle'
        self._goal_just_reached = False
        self._active_goal = None

        self.nav_status_pub = self.create_publisher(NavStatus, f'{self.ns}/jps/status', 10)
        self.cancel_srv = self.create_service(
            CancelNav,
            f'{self.ns}/jps/cancel',
            self._handle_cancel,
        )
        self._status_timer = self.create_timer(0.5, self._publish_nav_status)

        self.grid = None
        self.grid_original = None
        self.resolution = None
        self.origin = None

        self.current_pose = None
        self.path = []
        self.current_waypoint_idx = 0

        self.start_time = None
        self.total_distance = 0.0
        self.last_position = None

        self.control_timer = self.create_timer(
            NavigatorConstants.CONTROL_TIMER_PERIOD,
            self.control_loop,
        )
        self._path_republish_timer = self.create_timer(1.0, self._republish_path)

        map_yaml = os.getenv('MAP_YAML')
        map_pgm = os.getenv('MAP_PGM')

        if map_yaml is None:
            self.get_logger().error('No map provided. Set MAP_YAML env var.')
            raise SystemExit('Map YAML path required. Set MAP_YAML env var.')

        map_yaml = os.path.expanduser(map_yaml)
        if map_pgm is None:
            map_pgm = map_yaml.rsplit('.', 1)[0] + '.pgm'
        else:
            map_pgm = os.path.expanduser(map_pgm)

        self.load_map(map_yaml, map_pgm)

        mode_str = 'standalone' if self.standalone else 'stacked'
        self.get_logger().info(
            f'jps_navigator initialized | ns={self.ns} mode={mode_str} '
            f'radius={self.robot_radius}m clearance={self.safety_clearance}m')

    def load_map(self, yaml_file, pgm_file):
        """Load static map from YAML and PGM, then inflate obstacles.

        Changes:
            self.grid, self.grid_original, self.resolution, self.origin
        """
        try:
            with open(yaml_file, 'r') as f:
                map_data = yaml.safe_load(f)

            self.resolution = map_data['resolution']
            self.origin = map_data['origin']

            img = Image.open(pgm_file)
            occupancy_grid = np.array(img)

            self.grid = np.zeros_like(occupancy_grid)
            self.grid[occupancy_grid < 250] = 1
            self.grid[occupancy_grid >= 250] = 0
            self.grid = np.flipud(self.grid)
            self.grid_original = self.grid.copy()

            total_inflation = self.robot_radius + self.safety_clearance
            self.inflate_obstacles(total_inflation)

            self.get_logger().info(
                f'Map loaded: {self.grid.shape} res={self.resolution}m '
                f'inflation={total_inflation:.2f}m')
        except Exception as e:
            self.get_logger().error(f'Failed to load map: {e}')

    def inflate_obstacles(self, inflation_radius):
        """Inflate obstacles by inflation_radius using morphological dilation."""
        radius_pixels = int(inflation_radius / self.resolution)
        kernel_size = 2 * radius_pixels + 1
        kernel = np.ones((kernel_size, kernel_size))
        self.grid = binary_dilation(self.grid_original, kernel).astype(int)

    def odom_callback(self, msg):
        """Update current pose from odometry."""
        self.current_pose = {
            'x': msg.pose.pose.position.x,
            'y': msg.pose.pose.position.y,
            'theta': self.quaternion_to_yaw(msg.pose.pose.orientation),
        }

    def goal_callback(self, msg):
        """Handle goal pose from RViz 2D Goal Pose tool and start navigation."""
        goal_x = msg.pose.position.x
        goal_y = msg.pose.position.y
        self.get_logger().info(f'Goal received: ({goal_x:.2f}, {goal_y:.2f})')

        if self.current_pose is None:
            self.get_logger().warn('No odometry data yet, cannot navigate')
            return

        self._goal_just_reached = False
        self._active_goal = (goal_x, goal_y)
        self._planner_state = 'planning'

        start_x = self.current_pose['x']
        start_y = self.current_pose['y']

        if self.navigate_to_goal(start_x, start_y, goal_x, goal_y):
            self._planner_state = 'navigating'
        else:
            self._planner_state = 'idle'
            self._active_goal = None

    def navigate_to_goal(self, start_x, start_y, goal_x, goal_y):
        """Plan path from start to goal and begin navigation.

        Returns:
            True if a path was found and navigation started, False otherwise.

        Changes:
            self.path, self.start_time, self.total_distance, self.last_position,
            self.current_waypoint_idx, self._goal_just_reached
        """
        self.get_logger().info(
            f'Planning from ({start_x:.2f}, {start_y:.2f}) to ({goal_x:.2f}, {goal_y:.2f})')

        self._goal_just_reached = False
        self.path = self.plan_path(start_x, start_y, goal_x, goal_y)

        if self.path:
            self.current_waypoint_idx = 0
            self.start_time = self.get_clock().now()
            self.total_distance = 0.0
            self.last_position = (start_x, start_y)
            self.get_logger().info(f'Path found with {len(self.path)} waypoints')
            self._publish_path()
            return True
        else:
            self.get_logger().error('No path found! Check if start/goal are valid and reachable')
            return False

    def plan_path(self, start_x, start_y, goal_x, goal_y):
        """Validate positions, run JPS, and return a simplified world-coordinate path.

        Returns:
            List of (x, y) tuples in world coordinates, or [] if no path found.
        """
        start_grid = self.world_to_grid(start_x, start_y)
        goal_grid = self.world_to_grid(goal_x, goal_y)

        if not self.is_valid_in_original_grid(start_grid):
            self.get_logger().error(
                f'Start position ({start_x:.2f}, {start_y:.2f}) is in an obstacle.')
            return []

        if not self.is_valid(goal_grid):
            self.get_logger().error(
                f'Goal position ({goal_x:.2f}, {goal_y:.2f}) is invalid or too close to obstacles.')
            return []

        path_grid = self.jps(start_grid, goal_grid)
        if not path_grid:
            return []

        path_world = [self.grid_to_world(gx, gy) for gx, gy in path_grid]
        simplified = self._rdp_simplify(path_world, self.simplification_epsilon)
        return self._smooth_path(simplified)

    def jps(self, start, goal):
        """Jump Point Search with hybrid obstacle checking near the start cell."""
        def heuristic(a, b):
            return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}
        nodes_explored = 0

        while open_set:
            _, current = heapq.heappop(open_set)
            nodes_explored += 1

            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                self.get_logger().info(f'JPS explored {nodes_explored} nodes')
                return path[::-1]

            for successor in self.get_successors(current, start, goal):
                dx = successor[0] - current[0]
                dy = successor[1] - current[1]
                distance = math.sqrt(dx*dx + dy*dy)
                tentative_g = g_score[current] + distance

                if successor not in g_score or tentative_g < g_score[successor]:
                    came_from[successor] = current
                    g_score[successor] = tentative_g
                    f_score[successor] = tentative_g + heuristic(successor, goal)
                    heapq.heappush(open_set, (f_score[successor], successor))

        self.get_logger().error(f'JPS failed after exploring {nodes_explored} nodes')
        return []

    def get_successors(self, node, start, goal):
        """Return jump point successors in the 8-connected neighborhood."""
        successors = []
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0),
                       (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            jump_point = self.jump(node, (dx, dy), start, goal, depth=0)
            if jump_point is not None:
                successors.append(jump_point)
        return successors

    def jump(self, node, direction, start, goal, depth=0):
        """Jump recursively in direction until finding a jump point or dead end.

        Depth-limited to prevent RecursionError on large maps.
        """
        if depth > NavigatorConstants.MAX_JUMP_DEPTH:
            return None

        dx, dy = direction
        next_node = (node[0] + dx, node[1] + dy)

        if not self.is_valid_with_hybrid(next_node, start):
            return None

        if next_node == goal:
            return next_node

        if dx != 0 and dy != 0:
            if (self.is_valid_with_hybrid((next_node[0] - dx, next_node[1]), start) and
                    not self.is_valid_with_hybrid((next_node[0] - dx, next_node[1] + dy), start)):
                return next_node

            if (self.is_valid_with_hybrid((next_node[0], next_node[1] - dy), start) and
                    not self.is_valid_with_hybrid((next_node[0] + dx, next_node[1] - dy), start)):
                return next_node

            if self.jump(next_node, (dx, 0), start, goal, depth + 1) is not None:
                return next_node
            if self.jump(next_node, (0, dy), start, goal, depth + 1) is not None:
                return next_node

        else:
            if dx != 0:
                if (self.is_valid_with_hybrid((next_node[0], next_node[1] + 1), start) and
                        not self.is_valid_with_hybrid(
                            (next_node[0] + dx, next_node[1] + 1), start)):
                    return next_node
                if (self.is_valid_with_hybrid((next_node[0], next_node[1] - 1), start) and
                        not self.is_valid_with_hybrid(
                            (next_node[0] + dx, next_node[1] - 1), start)):
                    return next_node
            else:
                if (self.is_valid_with_hybrid((next_node[0] + 1, next_node[1]), start) and
                        not self.is_valid_with_hybrid(
                            (next_node[0] + 1, next_node[1] + dy), start)):
                    return next_node
                if (self.is_valid_with_hybrid((next_node[0] - 1, next_node[1]), start) and
                        not self.is_valid_with_hybrid(
                            (next_node[0] - 1, next_node[1] + dy), start)):
                    return next_node

        return self.jump(next_node, direction, start, goal, depth + 1)

    def is_valid_with_hybrid(self, grid_pos, start):
        """Return True if grid_pos is free, using original grid near the start cell."""
        gx, gy = grid_pos
        rows, cols = self.grid.shape

        if gx < 0 or gx >= cols or gy < 0 or gy >= rows:
            return False

        distance_from_start = abs(gx - start[0]) + abs(gy - start[1])
        if distance_from_start <= self.tight_space_radius:
            return self.grid_original[gy, gx] == 0
        return self.grid[gy, gx] == 0

    def is_valid(self, grid_pos):
        """Return True if grid_pos is in-bounds and free in the inflated grid."""
        gx, gy = grid_pos
        rows, cols = self.grid.shape
        if gx < 0 or gx >= cols or gy < 0 or gy >= rows:
            return False
        return self.grid[gy, gx] == 0

    def is_valid_in_original_grid(self, grid_pos):
        """Return True if grid_pos is in-bounds and free in the original (uninflated) grid."""
        gx, gy = grid_pos
        rows, cols = self.grid_original.shape
        if gx < 0 or gx >= cols or gy < 0 or gy >= rows:
            return False
        return self.grid_original[gy, gx] == 0

    def world_to_grid(self, x, y):
        """Convert world coordinates in meters to grid cell indices."""
        grid_x = int((x - self.origin[0]) / self.resolution)
        grid_y = int((y - self.origin[1]) / self.resolution)
        return (grid_x, grid_y)

    def grid_to_world(self, grid_x, grid_y):
        """Convert grid cell indices to world coordinates in meters."""
        x = grid_x * self.resolution + self.origin[0]
        y = grid_y * self.resolution + self.origin[1]
        return (x, y)

    def _rdp_simplify(self, path, epsilon):
        """Ramer-Douglas-Peucker line simplification."""
        if len(path) <= 2:
            return list(path)

        start = np.array(path[0])
        end = np.array(path[-1])
        line_vec = end - start
        line_len = np.linalg.norm(line_vec)

        if line_len < 1e-10:
            return [path[0], path[-1]]

        line_unit = line_vec / line_len
        max_dist = 0.0
        max_idx = 0
        for i in range(1, len(path) - 1):
            pt = np.array(path[i])
            proj = np.clip(np.dot(pt - start, line_unit), 0, line_len)
            closest = start + proj * line_unit
            dist = np.linalg.norm(pt - closest)
            if dist > max_dist:
                max_dist = dist
                max_idx = i

        if max_dist > epsilon:
            left = self._rdp_simplify(path[:max_idx + 1], epsilon)
            right = self._rdp_simplify(path[max_idx:], epsilon)
            return left[:-1] + right
        return [path[0], path[-1]]

    def _smooth_path(self, path):
        """Smooth waypoints with cubic spline interpolation, resampled at ~0.1 m."""
        if len(path) < 3:
            return path

        xs = [p[0] for p in path]
        ys = [p[1] for p in path]

        dists = [0.0]
        for i in range(1, len(xs)):
            d = math.sqrt((xs[i] - xs[i-1])**2 + (ys[i] - ys[i-1])**2)
            dists.append(dists[-1] + d)

        total_len = dists[-1]
        if total_len < 1e-6:
            return path

        t = np.array(dists)
        cs_x = CubicSpline(t, xs)
        cs_y = CubicSpline(t, ys)

        n_samples = max(int(total_len / 0.1), len(path))
        t_new = np.linspace(0, total_len, n_samples)
        return list(zip(cs_x(t_new).tolist(), cs_y(t_new).tolist()))

    def _publish_path(self):
        """Publish the planned path as a nav_msgs/Path."""
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'map'

        for x, y in self.path:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)

        self.plan_pub.publish(path_msg)

    def _republish_path(self):
        """Republish current path at 1 Hz for late-joining subscribers."""
        if self.path:
            self._publish_path()

    def _handle_cancel(self, request, response):
        """CancelNav service handler -- stop planning and driving."""
        self.get_logger().info('Cancel requested via service')
        self._active_goal = None
        self.path = []
        self.current_waypoint_idx = 0
        self._planner_state = 'idle'

        empty_path = Path()
        empty_path.header.stamp = self.get_clock().now().to_msg()
        empty_path.header.frame_id = 'map'
        self.plan_pub.publish(empty_path)

        if self.standalone:
            self.stop_robot()

        response.confirmed = True
        return response

    def _publish_nav_status(self):
        """Publish NavStatus at 2 Hz for the navigation server to consume."""
        msg = NavStatus()
        msg.nav_state = self._planner_state
        msg.has_active_goal = self._active_goal is not None
        msg.distance_traveled = self.total_distance

        if self.start_time is not None:
            msg.elapsed_time = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
        else:
            msg.elapsed_time = 0.0

        if self.current_pose is not None:
            msg.current_x = self.current_pose['x']
            msg.current_y = self.current_pose['y']
            if self._active_goal is not None:
                dx = self._active_goal[0] - self.current_pose['x']
                dy = self._active_goal[1] - self.current_pose['y']
                msg.distance_to_goal = math.sqrt(dx**2 + dy**2)
            else:
                msg.distance_to_goal = -1.0
        else:
            msg.current_x = 0.0
            msg.current_y = 0.0
            msg.distance_to_goal = -1.0

        msg.goal_reached = self._goal_just_reached
        self.nav_status_pub.publish(msg)

    def control_loop(self):
        """Waypoint-following control loop, called at 10 Hz."""
        if not self.path or self.current_pose is None:
            return

        if self.last_position is not None:
            current_x = self.current_pose['x']
            current_y = self.current_pose['y']
            dx = current_x - self.last_position[0]
            dy = current_y - self.last_position[1]
            self.total_distance += math.sqrt(dx**2 + dy**2)
            self.last_position = (current_x, current_y)

        if self.current_waypoint_idx >= len(self.path):
            if self.standalone:
                self.stop_robot()
            if self.start_time is not None:
                elapsed = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
                self.get_logger().info(
                    f'Goal reached | dist={self.total_distance:.2f}m time={elapsed:.2f}s')
            else:
                self.get_logger().info('Goal reached')
            self._planner_state = 'idle'
            self._goal_just_reached = True
            self.path = []
            return

        if self.standalone:
            self._follow_waypoint()

    def _follow_waypoint(self):
        """Drive toward the current waypoint with proportional control."""
        target = self.path[self.current_waypoint_idx]
        tx, ty = target

        dx = tx - self.current_pose['x']
        dy = ty - self.current_pose['y']
        distance = math.sqrt(dx**2 + dy**2)
        target_angle = math.atan2(dy, dx)
        angle_diff = normalize_angle(target_angle - self.current_pose['theta'])

        cmd = TwistStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = 'base_link'

        if distance < self.position_tolerance:
            self.current_waypoint_idx += 1
            return

        if abs(angle_diff) > self.angle_tolerance:
            cmd.twist.angular.z = self.angular_speed if angle_diff > 0 else -self.angular_speed
        else:
            cmd.twist.linear.x = min(self.linear_speed, distance)
            cmd.twist.angular.z = 0.3 * angle_diff

        self.cmd_vel_pub.publish(cmd)

    def stop_robot(self):
        """Stop the robot by publishing zero velocities."""
        cmd = TwistStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = 'base_link'
        self.cmd_vel_pub.publish(cmd)

    @staticmethod
    def quaternion_to_yaw(q):
        """Extract yaw angle from a quaternion."""
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)


def normalize_angle(angle):
    """Normalize angle to [-pi, pi] range."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def main(args=None):
    rclpy.init(args=args)
    navigator = JPSNavigator()
    try:
        rclpy.spin(navigator)
    except KeyboardInterrupt:
        pass
    finally:
        navigator.stop_robot()
        navigator.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
