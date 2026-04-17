#!/usr/bin/env python3
# Originally authored by the 2025 Carleton Senior Capstone Project
# (see AUTHORS.md). Substantially rewritten by Daniel Scheider, 2026.
"""A* global path planner for ROS2 with hybrid obstacle checking.

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

import rclpy
import rclpy.duration
import rclpy.time
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_interfaces.msg import NavStatus
from nav_interfaces.srv import CancelNav
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from tf2_ros import Buffer, TransformListener


class NavigatorConstants:
    """Default configuration constants for the A* navigator."""

    ROBOT_RADIUS = 0.25
    SAFETY_CLEARANCE = 0.05

    LINEAR_SPEED = 0.2
    ANGULAR_SPEED = 0.5
    POSITION_TOLERANCE = 0.1
    ANGLE_TOLERANCE = 0.1
    CONTROL_TIMER_PERIOD = 0.1

    MAX_PATH_WAYPOINTS = 20
    TIGHT_SPACE_RADIUS = 5      # grid cells to use original grid near start
    MAX_SEARCH_NODES = 100000
    SIMPLIFICATION_EPSILON = 0.1
    REPLAN_INTERVAL = 5.0

    DEFAULT_NAMESPACE = '/don'


class AStarNavigator(Node):
    """ROS2 node for A* global path planning and waypoint following."""

    def __init__(self, robot_radius=None, safety_clearance=None, map_yaml=None, map_pgm=None):
        """Initialize the A* navigator node.

        Args:
            robot_radius:     Robot footprint radius (m); falls back to ROBOT_RADIUS env var.
            safety_clearance: Additional inflation beyond radius (m); falls back to SAFETY_CLEARANCE.
            map_yaml:         Path to map YAML file; falls back to MAP_YAML env var.
            map_pgm:          Path to map PGM file; derived from map_yaml if not provided.
        """
        # Parse namespace from sys.argv before super().__init__() so TF remapping
        # can be passed via cli_args at construction time.
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

        super().__init__('astar_navigator', cli_args=_combined, use_global_arguments=False)

        self.declare_parameter('namespace', NavigatorConstants.DEFAULT_NAMESPACE)
        self.ns = self.get_parameter('namespace').value

        if robot_radius is not None:
            default_radius = robot_radius
        else:
            default_radius = float(os.getenv('ROBOT_RADIUS', str(NavigatorConstants.ROBOT_RADIUS)))

        if safety_clearance is not None:
            default_clearance = safety_clearance
        else:
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
        self.declare_parameter('max_search_nodes', NavigatorConstants.MAX_SEARCH_NODES)
        self.declare_parameter('simplification_epsilon', NavigatorConstants.SIMPLIFICATION_EPSILON)
        self.declare_parameter('replan_interval', NavigatorConstants.REPLAN_INTERVAL)

        # standalone: plan + drive via cmd_vel. stacked: publish path only for a local planner.
        self.declare_parameter('stacked', False)
        self.standalone = not self.get_parameter('stacked').value

        self.plan_pub = self.create_publisher(Path, self.ns + '/a_star/plan', 10)
        self.cmd_vel_pub = self.create_publisher(TwistStamped, self.ns + '/cmd_vel', 10)

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.odom_sub = self.create_subscription(
            Odometry,
            self.ns + '/odom',
            self.odom_callback,
            qos_profile,
        )
        self.goal_sub = self.create_subscription(
            PoseStamped,
            self.ns + '/goal_pose',
            self.goal_callback,
            10,
        )

        # TF remapped to namespaced topics via cli_args above
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.active_frame = 'map'

        self.grid = None
        self.grid_original = None
        self.resolution = None
        self.origin = None

        self.current_pose = None
        self.path = []
        self.current_waypoint_idx = 0
        self.latest_plan_msg = None

        self.linear_speed = self.get_parameter('linear_speed').value
        self.angular_speed = self.get_parameter('angular_speed').value
        self.position_tolerance = self.get_parameter('position_tolerance').value
        self.angle_tolerance = self.get_parameter('angle_tolerance').value
        self.max_path_waypoints = self.get_parameter('max_path_waypoints').value
        self.tight_space_radius = self.get_parameter('tight_space_radius').value
        self.max_search_nodes = self.get_parameter('max_search_nodes').value
        self.simplification_epsilon = self.get_parameter('simplification_epsilon').value
        self.replan_interval = self.get_parameter('replan_interval').value

        self.start_time = None
        self.total_distance = 0.0
        self.last_position = None

        self._active_goal = None

        if self.standalone:
            self.control_timer = self.create_timer(
                NavigatorConstants.CONTROL_TIMER_PERIOD,
                self.control_loop,
            )

        # Republish at 1 Hz so late-joining subscribers (e.g. DWA) receive the current plan.
        self.plan_republish_timer = self.create_timer(1.0, self.republish_plan)

        if self.replan_interval > 0:
            self.replan_timer = self.create_timer(self.replan_interval, self._replan_callback)

        self._planner_state = 'idle'
        self._goal_just_reached = False
        self.nav_status_pub = self.create_publisher(NavStatus, self.ns + '/a_star/status', 10)
        self.cancel_srv = self.create_service(
            CancelNav,
            self.ns + '/a_star/cancel',
            self._handle_cancel,
        )
        self._status_timer = self.create_timer(0.5, self._publish_nav_status)

        if map_yaml is None:
            map_yaml = os.getenv('MAP_YAML')
        if map_pgm is None:
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
            f'a_star_navigator initialized | ns={self.ns} mode={mode_str} '
            f'radius={self.robot_radius}m clearance={self.safety_clearance}m')

    def load_map(self, yaml_file, pgm_file):
        """Load static map from YAML and PGM, then inflate obstacles.

        Args:
            yaml_file: Path to map YAML metadata file.
            pgm_file:  Path to map PGM image file.

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
        from scipy.ndimage import binary_dilation

        radius_pixels = int(inflation_radius / self.resolution)
        kernel_size = 2 * radius_pixels + 1
        kernel = np.ones((kernel_size, kernel_size))
        self.grid = binary_dilation(self.grid_original, kernel).astype(int)

    def odom_callback(self, msg):
        """Update current pose from odometry."""
        self.current_pose = {
            'x': msg.pose.pose.position.x,
            'y': msg.pose.pose.position.y,
            'theta': self.quaternion_to_yaw(msg.pose.pose.orientation)
        }

    def goal_callback(self, msg):
        """Handle goal pose from RViz 2D Goal Pose tool and start navigation."""
        goal_x = msg.pose.position.x
        goal_y = msg.pose.position.y
        self.get_logger().info(f'Received goal from RViz: ({goal_x:.2f}, {goal_y:.2f})')

        start_pose = self.get_robot_pose_map_frame()
        if start_pose is None:
            self.get_logger().error('Cannot get robot position from TF2. Is localization running?')
            return

        start_x, start_y, _ = start_pose
        self.navigate_to_goal(start_x, start_y, goal_x, goal_y)

    def get_robot_pose_map_frame(self):
        """Return (x, y, theta) in map frame from TF2 map->base_link, or None."""
        try:
            transform = self.tf_buffer.lookup_transform(
                'map',
                'base_link',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
            x = transform.transform.translation.x
            y = transform.transform.translation.y
            q = transform.transform.rotation
            theta = self.quaternion_to_yaw(q)
            return (x, y, theta)
        except Exception:
            return None

    def navigate_to_goal(self, start_x, start_y, goal_x, goal_y):
        """Plan path from start to goal and begin navigation.

        In standalone mode, drives via cmd_vel. In planner-only mode, publishes
        nav_msgs/Path for a downstream local planner to follow.

        Args:
            start_x: Start X in world frame (m).
            start_y: Start Y in world frame (m).
            goal_x:  Goal X in world frame (m).
            goal_y:  Goal Y in world frame (m).

        Returns:
            True if a path was found and navigation started, False otherwise.

        Changes:
            self.path, self._active_goal, self._planner_state, self.start_time,
            self.total_distance, self.last_position, self.current_waypoint_idx
        """
        self.get_logger().info(
            f'Planning path from ({start_x:.2f}, {start_y:.2f}) to ({goal_x:.2f}, {goal_y:.2f})'
        )

        self._active_goal = (goal_x, goal_y)
        self._planner_state = 'planning'
        self._goal_just_reached = False

        self.path = self.plan_path(start_x, start_y, goal_x, goal_y)

        if self.path:
            self.current_waypoint_idx = 0
            self.start_time = self.get_clock().now()
            self.total_distance = 0.0
            self.last_position = (start_x, start_y)
            self.get_logger().info(f'Path found with {len(self.path)} waypoints')
            self._planner_state = 'ready'
            self.print_path()
            self.publish_global_plan()
            return True
        else:
            self.get_logger().error('No path found! Check if start/goal are valid and reachable')
            self.latest_plan_msg = None
            self._planner_state = 'idle'
            return False

    def publish_global_plan(self):
        """Build and publish the current path as nav_msgs/Path."""
        if not self.path:
            return

        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'map'

        yaw = 0.0
        for i, (wx, wy) in enumerate(self.path):
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = float(wx)
            pose.pose.position.y = float(wy)
            pose.pose.position.z = 0.0
            if i + 1 < len(self.path):
                nx, ny = self.path[i + 1]
                yaw = math.atan2(ny - wy, nx - wx)
            pose.pose.orientation.z = math.sin(yaw / 2.0)
            pose.pose.orientation.w = math.cos(yaw / 2.0)
            path_msg.poses.append(pose)

        self.latest_plan_msg = path_msg
        self.plan_pub.publish(path_msg)
        self.get_logger().info(
            f'Published global plan ({len(path_msg.poses)} poses) on {self.ns}/a_star/plan'
        )

    def republish_plan(self):
        """Periodically republish the latest plan so late-joining subscribers receive it."""
        if self.latest_plan_msg is not None:
            self.latest_plan_msg.header.stamp = self.get_clock().now().to_msg()
            self.plan_pub.publish(self.latest_plan_msg)

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
        self.latest_plan_msg = None
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

        pose = self.get_robot_pose_map_frame()
        if pose is not None:
            msg.current_x = pose[0]
            msg.current_y = pose[1]
            if self._active_goal is not None:
                dx = self._active_goal[0] - pose[0]
                dy = self._active_goal[1] - pose[1]
                msg.distance_to_goal = math.sqrt(dx**2 + dy**2)
            else:
                msg.distance_to_goal = -1.0
        else:
            msg.current_x = 0.0
            msg.current_y = 0.0
            msg.distance_to_goal = -1.0

        msg.goal_reached = self._goal_just_reached
        self.nav_status_pub.publish(msg)

    def _replan_callback(self):
        """Periodically replan from current TF2 position to the active goal."""
        if self._active_goal is None or not self.path:
            return

        start_pose = self.get_robot_pose_map_frame()
        if start_pose is None:
            return

        start_x, start_y, _ = start_pose
        goal_x, goal_y = self._active_goal

        dist = math.sqrt((goal_x - start_x)**2 + (goal_y - start_y)**2)
        if dist < self.position_tolerance * 2:
            return

        self.get_logger().info(
            f'Replanning from ({start_x:.2f}, {start_y:.2f}) to ({goal_x:.2f}, {goal_y:.2f})')
        new_path = self.plan_path(start_x, start_y, goal_x, goal_y)
        if new_path:
            self.path = new_path
            self.current_waypoint_idx = 0
            self.publish_global_plan()

    def plan_path(self, start_x, start_y, goal_x, goal_y):
        """Validate positions, run A*, and return a simplified world-coordinate path.

        Args:
            start_x: Start X in world frame (m).
            start_y: Start Y in world frame (m).
            goal_x:  Goal X in world frame (m).
            goal_y:  Goal Y in world frame (m).

        Returns:
            List of (x, y) tuples in world coordinates, or [] if no path found.
        """
        start_grid = self.world_to_grid(start_x, start_y)
        goal_grid = self.world_to_grid(goal_x, goal_y)

        self.get_logger().info(f'Grid coordinates: Start {start_grid}, Goal {goal_grid}')

        if not self.is_valid_in_original_grid(start_grid):
            self.get_logger().error(
                f'Start position ({start_x:.2f}, {start_y:.2f}) is in an obstacle.')
            return []

        if not self.is_valid(goal_grid):
            self.get_logger().error(
                f'Goal position ({goal_x:.2f}, {goal_y:.2f}) is invalid or too close to obstacles.')
            return []

        neighbors_valid = sum(
            1 for dx, dy in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]
            if self.is_valid((start_grid[0] + dx, start_grid[1] + dy))
        )

        if neighbors_valid == 0:
            self.get_logger().warn(
                f'Start surrounded by inflated obstacles. '
                f'Consider reducing robot_radius ({self.robot_radius}m) '
                f'or safety_clearance ({self.safety_clearance}m).')

        path_grid = self.astar(start_grid, goal_grid)

        if not path_grid:
            return []

        path_world = [self.grid_to_world(gx, gy) for gx, gy in path_grid]
        if not self.standalone:
            simplified_path = self.simplify_path(path_world, max_points=50)
        else:
            simplified_path = self.simplify_path(path_world)
        return simplified_path

    def astar(self, start, goal):
        """A* search with hybrid obstacle checking (original grid near start, inflated elsewhere).

        Args:
            start: (grid_x, grid_y) start cell.
            goal:  (grid_x, grid_y) goal cell.

        Returns:
            List of (grid_x, grid_y) tuples from start to goal, or [] if unreachable.
        """
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

            if nodes_explored >= self.max_search_nodes:
                self.get_logger().error(
                    f'A* exceeded {self.max_search_nodes} nodes -- goal may be unreachable.')
                return []

            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                self.get_logger().info(f'A* found path, explored {nodes_explored} nodes')
                return path[::-1]

            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                neighbor = (current[0] + dx, current[1] + dy)

                distance_from_start = abs(neighbor[0] - start[0]) + abs(neighbor[1] - start[1])
                if distance_from_start <= self.tight_space_radius:
                    # close to start -- use original grid to escape tight spaces
                    if not self.is_valid_in_original_grid(neighbor):
                        continue
                else:
                    # farther out -- use inflated grid for safety
                    if not self.is_valid(neighbor):
                        continue

                move_cost = 1.414 if dx != 0 and dy != 0 else 1.0  # diagonal = sqrt(2)
                tentative_g = g_score[current] + move_cost

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        self.get_logger().error(f'A* failed after exploring {nodes_explored} nodes')
        return []

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

    def simplify_path(self, path, max_points=None):
        """Simplify path with Ramer-Douglas-Peucker, then smooth with cubic spline.

        Args:
            path:       List of (x, y) tuples in world coordinates.
            max_points: Max waypoints in output (default: self.max_path_waypoints).

        Returns:
            Simplified and smoothed list of (x, y) tuples.
        """
        if max_points is None:
            max_points = self.max_path_waypoints

        if len(path) <= 2:
            return path

        rdp_path = self._rdp(path, self.simplification_epsilon)

        if len(rdp_path) >= 4:
            smoothed = self._smooth_path(rdp_path, max_points)
        else:
            smoothed = rdp_path

        smoothed[0] = path[0]
        smoothed[-1] = path[-1]
        return smoothed

    @staticmethod
    def _rdp(points, epsilon):
        """Ramer-Douglas-Peucker line simplification.

        Recursively removes points that deviate less than epsilon from the line
        between the current endpoints.
        """
        if len(points) <= 2:
            return list(points)

        start = np.array(points[0])
        end = np.array(points[-1])
        line_vec = end - start
        line_len = np.linalg.norm(line_vec)

        if line_len < 1e-10:
            return [points[0], points[-1]]

        line_unit = line_vec / line_len
        max_dist = 0.0
        max_idx = 0

        for i in range(1, len(points) - 1):
            pt = np.array(points[i])
            proj = np.dot(pt - start, line_unit)
            proj = np.clip(proj, 0, line_len)
            closest = start + proj * line_unit
            dist = np.linalg.norm(pt - closest)
            if dist > max_dist:
                max_dist = dist
                max_idx = i

        if max_dist > epsilon:
            left = AStarNavigator._rdp(points[:max_idx + 1], epsilon)
            right = AStarNavigator._rdp(points[max_idx:], epsilon)
            return left[:-1] + right
        else:
            return [points[0], points[-1]]

    @staticmethod
    def _smooth_path(waypoints, max_points):
        """Smooth waypoints with cubic spline interpolation.

        Fits a parametric spline through the waypoints and resamples at even
        intervals to remove grid-aligned staircase artifacts.
        """
        pts = np.array(waypoints)
        diffs = np.diff(pts, axis=0)
        chord_lengths = np.sqrt(np.sum(diffs**2, axis=1))
        t = np.concatenate([[0], np.cumsum(chord_lengths)])
        total_length = t[-1]

        if total_length < 1e-10:
            return list(waypoints)

        cs_x = CubicSpline(t, pts[:, 0])
        cs_y = CubicSpline(t, pts[:, 1])

        n_out = min(max_points, max(len(waypoints), 10))
        t_new = np.linspace(0, total_length, n_out)
        x_new = cs_x(t_new)
        y_new = cs_y(t_new)

        return list(zip(x_new.tolist(), y_new.tolist()))

    def print_path(self):
        """Log first and last waypoint of the planned path as a sanity check."""
        if not self.path:
            return
        start = self.path[0]
        end = self.path[-1]
        self.get_logger().info(
            f'Path: ({start[0]:.2f}, {start[1]:.2f}) -> ({end[0]:.2f}, {end[1]:.2f}) '
            f'| {len(self.path)} waypoints'
        )

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
            self.stop_robot()
            if self.start_time is not None:
                elapsed = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
                self.get_logger().info(
                    f'Goal reached | dist={self.total_distance:.2f}m time={elapsed:.2f}s')
            else:
                self.get_logger().info('Goal reached')
            self.path = []
            self._active_goal = None
            self._planner_state = 'idle'
            self._goal_just_reached = True
            return

        target = self.path[self.current_waypoint_idx]
        tx, ty = target

        # Path is planned in map frame -- odom fallback would cause drift-induced errors.
        map_pose = self.get_robot_pose_map_frame()
        if map_pose is not None:
            curr_x, curr_y, curr_theta = map_pose
            self.active_frame = 'map'
        else:
            self.get_logger().warn(
                'TF2 map->base_link unavailable -- stopping until localization recovers.')
            self.stop_robot()
            return

        dx = tx - curr_x
        dy = ty - curr_y
        distance = math.sqrt(dx**2 + dy**2)
        target_angle = math.atan2(dy, dx)
        angle_diff = self.normalize_angle(target_angle - curr_theta)

        cmd = TwistStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = 'base_link'

        if distance < self.position_tolerance:
            self.current_waypoint_idx += 1
            self.get_logger().info(
                f'Waypoint {self.current_waypoint_idx}/{len(self.path)} reached')
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

    def get_current_position(self):
        """Return current (x, y) from odometry, or None if unavailable."""
        if self.current_pose is None:
            self.get_logger().warn('No odometry data available yet')
            return None
        return (self.current_pose['x'], self.current_pose['y'])

    @staticmethod
    def quaternion_to_yaw(q):
        """Extract yaw angle from a quaternion."""
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def normalize_angle(angle):
        """Normalize angle to [-pi, pi] range."""
        return (angle + math.pi) % (2 * math.pi) - math.pi


def main(args=None):
    """Entry point: read env vars, create navigator node, spin.

    Requires MAP_YAML env var pointing to the map YAML file.
    MAP_PGM defaults to the same path with .pgm extension.
    ROBOT_RADIUS and SAFETY_CLEARANCE fall back to NavigatorConstants.
    """
    robot_radius = float(os.getenv('ROBOT_RADIUS', str(NavigatorConstants.ROBOT_RADIUS)))
    safety_clearance = float(
        os.getenv('SAFETY_CLEARANCE', str(NavigatorConstants.SAFETY_CLEARANCE))
    )
    map_yaml = os.getenv('MAP_YAML')
    map_pgm = os.getenv('MAP_PGM')

    rclpy.init(args=args)
    navigator = AStarNavigator(
        robot_radius=robot_radius,
        safety_clearance=safety_clearance,
        map_yaml=map_yaml,
        map_pgm=map_pgm,
    )

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
