#!/usr/bin/env python3
# Originally authored by Devin Dennis as part of the 2025 Carleton Senior
# Capstone Project (see AUTHORS.md). Updated by Daniel Scheider, 2026.
"""D* Lite incremental global path planner for ROS2 with dynamic replanning.

Supports standalone mode (plans and drives via cmd_vel) and stacked mode
(publishes nav_msgs/Path for a downstream local planner). Integrates with
SLAM for persistent obstacle tracking.
"""

import heapq
import math
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.ndimage import binary_dilation

import rclpy
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_interfaces.msg import NavStatus
from nav_interfaces.srv import CancelNav
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from tf2_ros import Buffer, TransformListener


class PlannerConstants:
    """Default configuration constants for the D* Lite navigator."""

    ROBOT_RADIUS = 0.22
    SAFETY_CLEARANCE = 0.05

    LINEAR_SPEED = 0.2
    ANGULAR_SPEED = 0.5
    POSITION_TOLERANCE = 0.1
    ANGLE_TOLERANCE = 0.1
    CONTROL_TIMER_PERIOD = 0.1

    MAX_PATH_ITERATIONS = 10000000
    MAX_WAYPOINTS = 10
    REPLAN_COOLDOWN = 1.0
    ROBOT_CLEARANCE_CELLS = 3
    PATH_CHECK_RADIUS = 2

    DEFAULT_NAMESPACE = '/don'


class DStarLite:
    """D* Lite incremental search algorithm for dynamic pathfinding."""

    def __init__(self, grid, start, goal, logger=None):
        self.grid = grid
        self.rows, self.cols = grid.shape
        self.start = start
        self.goal = goal
        self.s_last = start
        self.k_m = 0
        self.logger = logger

        self.g = defaultdict(lambda: float('inf'))
        self.rhs = defaultdict(lambda: float('inf'))

        self.open_list = []
        self._open_valid = {}

        self.rhs[self.goal] = 0
        start_key = self.calculate_key(self.goal)
        heapq.heappush(self.open_list, (start_key, self.goal))
        self._open_valid[self.goal] = start_key

    def heuristic(self, s1, s2):
        """Euclidean distance heuristic."""
        return math.sqrt((s1[0] - s2[0])**2 + (s1[1] - s2[1])**2)

    def calculate_key(self, s):
        """Calculate priority key for state s."""
        g_rhs = min(self.g[s], self.rhs[s])
        return (g_rhs + self.heuristic(self.start, s) + self.k_m, g_rhs)

    def is_valid(self, pos):
        """Check if position is valid (in bounds and not an obstacle)."""
        x, y = pos
        if x < 0 or x >= self.cols or y < 0 or y >= self.rows:
            return False
        return self.grid[y, x] == 0

    def get_neighbors(self, s):
        """Get valid 8-connected neighbors of state s."""
        neighbors = []
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
            neighbor = (s[0] + dx, s[1] + dy)
            if self.is_valid(neighbor):
                neighbors.append(neighbor)
        return neighbors

    def cost(self, s1, s2):
        """Calculate movement cost between adjacent states."""
        if not self.is_valid(s1) or not self.is_valid(s2):
            return float('inf')
        dx = abs(s1[0] - s2[0])
        dy = abs(s1[1] - s2[1])
        return 1.414 if (dx + dy) == 2 else 1.0

    def _open_push(self, key, u):
        """Push u onto the open list with lazy deletion."""
        heapq.heappush(self.open_list, (key, u))
        self._open_valid[u] = key

    def _open_remove(self, u):
        """Mark u as removed from the open list."""
        self._open_valid.pop(u, None)

    def _open_pop(self):
        """Pop the smallest valid entry."""
        while self.open_list:
            key, u = heapq.heappop(self.open_list)
            if self._open_valid.get(u) == key:
                del self._open_valid[u]
                return key, u
        return None, None

    def _open_top(self):
        """Peek at the smallest valid entry without removing it."""
        while self.open_list:
            key, u = self.open_list[0]
            if self._open_valid.get(u) == key:
                return key, u
            heapq.heappop(self.open_list)
        return None, None

    def update_vertex(self, u):
        """Update state u's cost values and priority queue status."""
        if u != self.goal:
            min_cost = float('inf')
            for s_prime in self.get_neighbors(u):
                cost = self.cost(u, s_prime) + self.g[s_prime]
                min_cost = min(min_cost, cost)
            self.rhs[u] = min_cost

        self._open_remove(u)

        if self.g[u] != self.rhs[u]:
            self._open_push(self.calculate_key(u), u)

    def compute_shortest_path(self):
        """Process states until start is locally consistent.

        Returns:
            True if path found, False if iteration limit reached.
        """
        iterations = 0

        while True:
            top_key, top_u = self._open_top()
            if top_key is None:
                break

            start_key = self.calculate_key(self.start)
            if not (self.compare_keys(top_key, start_key) or
                    self.rhs[self.start] != self.g[self.start]):
                break

            iterations += 1
            if iterations > PlannerConstants.MAX_PATH_ITERATIONS:
                if self.logger:
                    self.logger.warn('Max path iterations timeout')
                return False

            k_old, u = self._open_pop()
            if u is None:
                break

            k_new = self.calculate_key(u)

            if self.compare_keys(k_old, k_new):
                # Key has changed, re-insert with new key
                self._open_push(k_new, u)
            elif self.g[u] > self.rhs[u]:
                # Overconsistent: make consistent and propagate
                self.g[u] = self.rhs[u]
                for s in self.get_neighbors(u):
                    self.update_vertex(s)
            else:
                # Underconsistent: make overconsistent and propagate
                self.g[u] = float('inf')
                self.update_vertex(u)
                for s in self.get_neighbors(u):
                    self.update_vertex(s)

        return True

    def compare_keys(self, k1, k2):
        """Compare two priority keys. Returns True if k1 < k2."""
        return k1[0] < k2[0] or (k1[0] == k2[0] and k1[1] < k2[1])

    def extract_path(self, debug=False):
        """Extract path from start to goal by following gradient of g-values."""
        if self.g[self.start] == float('inf'):
            return []

        path = [self.start]
        current = self.start
        visited = set([self.start])
        max_iterations = self.rows * self.cols

        while current != self.goal:
            if len(path) > max_iterations:
                return []

            neighbors = self.get_neighbors(current)
            if not neighbors:
                return []

            valid_neighbors = [
                (n, self.g[n]) for n in neighbors
                if self.g[n] != float('inf') and n not in visited
            ]

            if not valid_neighbors:
                return []

            next_state, min_g = min(valid_neighbors, key=lambda x: x[1])

            if min_g > self.g[current] + 0.01 and next_state != self.goal:
                return []

            visited.add(next_state)
            path.append(next_state)
            current = next_state

        return path

    def update_start(self, new_start):
        """Update start position for replanning."""
        if new_start == self.start:
            return
        self.k_m += self.heuristic(self.s_last, self.start)
        self.s_last = self.start
        self.start = new_start

    def update_obstacles(self, changed_cells):
        """Update grid with changed obstacles and trigger replanning."""
        all_affected = set()

        for cell in changed_cells:
            x, y = cell
            if 0 <= x < self.cols and 0 <= y < self.rows:
                all_affected.add(cell)
                for dx, dy in [(-1,0), (1,0), (0,-1), (0,1),
                               (-1,-1), (-1,1), (1,-1), (1,1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.cols and 0 <= ny < self.rows:
                        all_affected.add((nx, ny))

        for vertex in all_affected:
            self.update_vertex(vertex)


class DStarNavigator(Node):
    """ROS2 node for D* Lite path planning with SLAM-driven replanning."""

    def __init__(self):
        _ns = PlannerConstants.DEFAULT_NAMESPACE
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

        super().__init__('dstar_navigator', cli_args=_combined, use_global_arguments=False)

        self.declare_parameter('namespace', PlannerConstants.DEFAULT_NAMESPACE)
        self.ns = self.get_parameter('namespace').value

        default_radius = float(os.getenv('ROBOT_RADIUS', str(PlannerConstants.ROBOT_RADIUS)))
        default_clearance = float(
            os.getenv('SAFETY_CLEARANCE', str(PlannerConstants.SAFETY_CLEARANCE))
        )

        self.declare_parameter('robot_radius', default_radius)
        self.declare_parameter('safety_clearance', default_clearance)
        self.robot_radius = self.get_parameter('robot_radius').value
        self.safety_clearance = self.get_parameter('safety_clearance').value

        self.declare_parameter('linear_speed', PlannerConstants.LINEAR_SPEED)
        self.declare_parameter('angular_speed', PlannerConstants.ANGULAR_SPEED)
        self.declare_parameter('position_tolerance', PlannerConstants.POSITION_TOLERANCE)
        self.declare_parameter('angle_tolerance', PlannerConstants.ANGLE_TOLERANCE)
        self.declare_parameter('max_waypoints', PlannerConstants.MAX_WAYPOINTS)
        self.declare_parameter('replan_cooldown', PlannerConstants.REPLAN_COOLDOWN)
        self.declare_parameter('optimistic_planning', True)

        self.linear_speed = self.get_parameter('linear_speed').value
        self.angular_speed = self.get_parameter('angular_speed').value
        self.position_tolerance = self.get_parameter('position_tolerance').value
        self.angle_tolerance = self.get_parameter('angle_tolerance').value
        self.optimistic_planning = self.get_parameter('optimistic_planning').value

        self.declare_parameter('stacked', False)
        self.standalone = not self.get_parameter('stacked').value

        self.path_pub = self.create_publisher(Path, f'{self.ns}/d_star/plan', 10)

        qos_cmd = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.cmd_vel_pub = self.create_publisher(TwistStamped, f'{self.ns}/cmd_vel', qos_cmd)
        self.dynamic_grid_pub = self.create_publisher(
            OccupancyGrid, f'{self.ns}/dynamic_grid', 10)

        qos_best_effort = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.odom_sub = self.create_subscription(
            Odometry,
            f'{self.ns}/odom',
            self.odom_callback,
            qos_best_effort,
        )

        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            f'{self.ns}/map',
            self.map_callback,
            map_qos,
        )
        self.goal_sub = self.create_subscription(
            PoseStamped,
            f'{self.ns}/goal_pose',
            self.goal_pose_callback,
            10,
        )
        self.obstacle_grid_sub = self.create_subscription(
            OccupancyGrid,
            f'{self.ns}/obstacle_grid',
            self.obstacle_grid_callback,
            10,
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self._planner_state = 'idle'
        self._goal_just_reached = False
        self._active_goal = None

        self.nav_status_pub = self.create_publisher(NavStatus, f'{self.ns}/d_star/status', 10)
        self.cancel_srv = self.create_service(
            CancelNav,
            f'{self.ns}/d_star/cancel',
            self._handle_cancel,
        )
        self._status_timer = self.create_timer(0.5, self._publish_nav_status)

        self.grid = None
        self.grid_original = None
        self.grid_base = None
        self.grid_dynamic = None
        self.grid_slam = None
        self.grid_unknown = None
        self.resolution = None
        self.origin = None
        self.map_received = False

        self.current_pose = None
        self.path = []
        self.current_waypoint_idx = 0

        self.dstar_planner = None
        self.goal_grid = None
        self.planner_grid_snapshot = None
        self.known_obstacles = set()

        self.replanning_needed = False
        self.last_replan_time = self.get_clock().now()
        self.blocked_waypoint_idx = None

        self.start_time = None
        self.total_distance = 0.0
        self.last_position = None

        self.control_timer = self.create_timer(
            PlannerConstants.CONTROL_TIMER_PERIOD,
            self.control_loop,
        )
        self.viz_timer = self.create_timer(0.1, self.publish_dynamic_grid)
        self._path_republish_timer = self.create_timer(1.0, self._republish_path)


        mode_str = 'standalone' if self.standalone else 'stacked'
        self.get_logger().info(
            f'd_star_navigator initialized | ns={self.ns} mode={mode_str} '
            f'radius={self.robot_radius}m clearance={self.safety_clearance}m')


    def _handle_cancel(self, request, response):
        """CancelNav service handler -- stop planning and driving."""
        self.get_logger().info('Cancel requested via service')
        self._active_goal = None
        self.path = []
        self.current_waypoint_idx = 0
        self._planner_state = 'idle'
        self.replanning_needed = False

        # Publish empty path so downstream knows to stop
        empty_path = Path()
        empty_path.header.stamp = self.get_clock().now().to_msg()
        empty_path.header.frame_id = 'map'
        self.path_pub.publish(empty_path)

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

    def _republish_path(self):
        """Republish current path at 1 Hz for late-joining RViz subscribers."""
        if self.path:
            self.publish_path()


    def goal_pose_callback(self, msg: PoseStamped):
        """Handle incoming goal pose from RViz 2D Goal Pose."""
        goal_x = msg.pose.position.x
        goal_y = msg.pose.position.y

        self.get_logger().info(f'Goal received: ({goal_x:.2f}, {goal_y:.2f})')

        if self.current_pose is None:
            self.get_logger().warn('No odometry data yet, cannot navigate to goal')
            return

        if not self.map_received:
            self.get_logger().warn('No SLAM map received yet, cannot plan')
            return

        # Reset goal-reached flag
        self._goal_just_reached = False
        self._active_goal = (goal_x, goal_y)
        self._planner_state = 'planning'

        start_x = self.current_pose['x']
        start_y = self.current_pose['y']

        if self.navigate_to_goal(start_x, start_y, goal_x, goal_y):
            self._planner_state = 'navigating'
            self.get_logger().info('Navigation started successfully')
        else:
            self._planner_state = 'idle'
            self._active_goal = None
            self.get_logger().error('Failed to plan path to goal')


    def _inflate_obstacles(self, grid, inflation_radius):
        """Inflate obstacles using morphological dilation."""
        radius_pixels = int(inflation_radius / self.resolution)
        kernel_size = 2 * radius_pixels + 1
        kernel = np.ones((kernel_size, kernel_size))
        return binary_dilation(grid, kernel).astype(int)


    def odom_callback(self, msg):
        """Update current robot pose from odometry."""
        self.current_pose = {
            'x': msg.pose.pose.position.x,
            'y': msg.pose.pose.position.y,
            'theta': self.quaternion_to_yaw(msg.pose.pose.orientation)
        }

    def obstacle_grid_callback(self, msg: OccupancyGrid):
        """Merge shared LIDAR obstacle grid into grid_dynamic and trigger replan if blocked."""
        if self.grid_dynamic is None or self.resolution is None:
            return

        if (msg.info.width != self.grid_dynamic.shape[1] or
                msg.info.height != self.grid_dynamic.shape[0]):
            return

        shared = np.array(msg.data, dtype=np.int8).reshape(
            msg.info.height, msg.info.width)

        previous_dynamic = self.grid_dynamic.copy()
        self.grid_dynamic = np.where(
            (self.grid_base == 1) | (shared >= 50),
            1, 0,
        ).astype(np.int8)

        diff = np.where(previous_dynamic != self.grid_dynamic)
        changed_cells = list(zip(diff[1].tolist(), diff[0].tolist()))

        if self.dstar_planner is not None and changed_cells:
            self.dstar_planner.grid = self.grid_dynamic.copy()
            self.dstar_planner.update_obstacles(changed_cells)

            if self.path and len(changed_cells) > 0:
                current_time = self.get_clock().now()
                replan_cooldown = self.get_parameter('replan_cooldown').value
                time_since = (current_time - self.last_replan_time).nanoseconds / 1e9

                if time_since > replan_cooldown:
                    # Check if any changed cell is an obstacle near the path
                    changed_set = set(changed_cells)
                    r = PlannerConstants.PATH_CHECK_RADIUS
                    for i in range(self.current_waypoint_idx,
                                   min(self.current_waypoint_idx + 5, len(self.path))):
                        wp = self.path[i]
                        wp_gx, wp_gy = self.world_to_grid(wp[0], wp[1])
                        for dy in range(-r, r + 1):
                            for dx in range(-r, r + 1):
                                cell = (wp_gx + dx, wp_gy + dy)
                                if cell in changed_set:
                                    cx, cy = cell
                                    if (0 <= cx < self.grid_dynamic.shape[1] and
                                            0 <= cy < self.grid_dynamic.shape[0] and
                                            self.grid_dynamic[cy, cx] == 1):
                                        self.replanning_needed = True
                                        self.blocked_waypoint_idx = i
                                        self.last_replan_time = current_time
                                        return

    def map_callback(self, msg: OccupancyGrid):
        """
        Process live SLAM map updates.

        Integrates SLAM-discovered obstacles into the planning grid and
        triggers replanning if they block the current path.
        """
        is_first_map = not self.map_received

        width, height = msg.info.width, msg.info.height
        slam_resolution = msg.info.resolution
        slam_origin = [msg.info.origin.position.x, msg.info.origin.position.y]

        occupancy_data = np.array(msg.data, dtype=np.int8).reshape((height, width))
        unknown_mask = (occupancy_data == -1)

        # Convert to binary (0=free, 1=occupied)
        slam_grid = np.zeros((height, width), dtype=np.int8)
        slam_grid[occupancy_data >= 50] = 1

        # Initialize grids from SLAM on first message
        if self.grid_original is None:
            self._initialize_grids_from_slam(height, width, slam_resolution, slam_origin)
            self.get_logger().info(
                f'Initialized grid from SLAM: {width}x{height}, '
                f'resolution={slam_resolution}m, '
                f'origin=({slam_origin[0]:.2f}, {slam_origin[1]:.2f})')

        if self.grid_original is None:
            return

        # Track unknowns for optimistic planning
        previous_unknown = self.grid_unknown.copy() if self.grid_unknown is not None else None
        self.grid_slam = slam_grid.copy()
        self.grid_unknown = np.zeros_like(self.grid_original, dtype=np.int8)

        # Map SLAM obstacles (vectorized)
        current_slam_obstacles, newly_discovered = self._map_slam_to_grid(
            slam_grid, unknown_mask, slam_resolution, slam_origin,
            height, width, previous_unknown)

        # Update dynamic grid
        self._update_dynamic_grid_with_slam(current_slam_obstacles)

        # Check if replanning needed
        min_new_obstacles = 5
        if newly_discovered >= min_new_obstacles and self.optimistic_planning and self.path:
            current_time = self.get_clock().now()
            time_since_last_replan = (current_time - self.last_replan_time).nanoseconds / 1e9

            if time_since_last_replan > self.get_parameter('replan_cooldown').value:
                if self._check_path_blocked_by_obstacles(current_slam_obstacles):
                    self.known_obstacles.clear()
                    self.get_logger().warn(f'Newly discovered SLAM obstacles block current path!')
                    self.replanning_needed = True
                    self.last_replan_time = current_time

        self.map_received = True

        if is_first_map:
            self.get_logger().info(
                f'SLAM map integrated: {current_slam_obstacles.sum()} obstacles detected')


    def _initialize_grids_from_slam(self, height, width, resolution, origin):
        """Initialize all grids from SLAM dimensions."""
        self.resolution = resolution
        self.origin = origin

        self.grid_original = np.zeros((height, width), dtype=np.int8)
        self.grid = np.zeros((height, width), dtype=np.int8)
        self.grid_base = np.zeros((height, width), dtype=np.int8)
        self.grid_dynamic = np.zeros((height, width), dtype=np.int8)
        self.grid_unknown = np.zeros((height, width), dtype=np.int8)

    def _map_slam_to_grid(self, slam_grid, unknown_mask, slam_resolution, slam_origin,
                          height, width, previous_unknown):
        """
        Map SLAM obstacles to our grid (vectorized).

        Returns:
            Tuple (obstacles_grid, newly_discovered_count)
        """
        obstacles_grid = np.zeros_like(self.grid_original, dtype=np.int8)
        newly_discovered_count = 0

        # Build coordinate arrays for vectorized mapping
        slam_ys, slam_xs = np.mgrid[0:height, 0:width]
        world_xs = slam_xs * slam_resolution + slam_origin[0]
        world_ys = slam_ys * slam_resolution + slam_origin[1]
        grid_xs = ((world_xs - self.origin[0]) / self.resolution).astype(int)
        grid_ys = ((world_ys - self.origin[1]) / self.resolution).astype(int)

        # Bounds mask
        in_bounds = ((grid_xs >= 0) & (grid_xs < self.grid_original.shape[1]) &
                     (grid_ys >= 0) & (grid_ys < self.grid_original.shape[0]))

        # Apply unknown mask
        unknown_in_bounds = in_bounds & unknown_mask
        gxs_unk = grid_xs[unknown_in_bounds]
        gys_unk = grid_ys[unknown_in_bounds]
        self.grid_unknown[gys_unk, gxs_unk] = 1

        # Apply obstacle mask
        obstacle_in_bounds = in_bounds & (slam_grid == 1)
        gxs_obs = grid_xs[obstacle_in_bounds]
        gys_obs = grid_ys[obstacle_in_bounds]
        obstacles_grid[gys_obs, gxs_obs] = 1

        # Count newly discovered
        if previous_unknown is not None:
            newly_discovered_count = int(np.sum(
                previous_unknown[gys_obs, gxs_obs] == 1))

        return obstacles_grid, newly_discovered_count

    def _update_dynamic_grid_with_slam(self, current_slam_obstacles):
        """Update grid_base to reflect current SLAM obstacle state."""
        previous_grid_base = self.grid_base.copy() if self.grid_base is not None else None

        self.grid_base = self.grid.copy()

        if current_slam_obstacles.sum() > 0:
            total_inflation = self.robot_radius + self.safety_clearance
            slam_obstacles_inflated = self._inflate_obstacles(
                current_slam_obstacles, total_inflation)
            self.grid_base = slam_obstacles_inflated

        self.grid_dynamic = self.grid_base.copy()

        # Find changed cells (vectorized)
        changed_cells = []
        if previous_grid_base is not None:
            diff = np.where(previous_grid_base != self.grid_base)
            changed_cells = list(zip(diff[1].tolist(), diff[0].tolist()))  # (x, y)

        if self.dstar_planner is not None and changed_cells:
            self.get_logger().info(f'Syncing {len(changed_cells)} SLAM changes to D* planner')
            self.dstar_planner.grid = self.grid_dynamic.copy()
            self.dstar_planner.update_obstacles(changed_cells)
            self.planner_grid_snapshot = self.grid_base.copy()

        self.publish_dynamic_grid()

    def _check_path_blocked_by_obstacles(self, obstacles_grid):
        """Check if obstacles block any waypoints in the current path."""
        if not self.path:
            return False

        for waypoint_idx, waypoint in enumerate(self.path):
            if waypoint_idx < self.current_waypoint_idx:
                continue

            gx, gy = self.world_to_grid(waypoint[0], waypoint[1])

            if not (0 <= gx < obstacles_grid.shape[1] and 0 <= gy < obstacles_grid.shape[0]):
                continue

            r = PlannerConstants.PATH_CHECK_RADIUS
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    nx, ny = gx + dx, gy + dy
                    if (0 <= nx < obstacles_grid.shape[1] and
                        0 <= ny < obstacles_grid.shape[0] and
                        obstacles_grid[ny, nx] != 0):
                        self.blocked_waypoint_idx = waypoint_idx
                        return True

        return False


    def navigate_to_goal(self, start_x, start_y, goal_x, goal_y):
        """Navigate from start to goal position."""
        self.get_logger().info(
            f'Planning path from ({start_x:.2f}, {start_y:.2f}) '
            f'to ({goal_x:.2f}, {goal_y:.2f})')

        self.known_obstacles.clear()
        self._goal_just_reached = False

        self.path = self.plan_path(start_x, start_y, goal_x, goal_y)

        if self.path:
            self.current_waypoint_idx = 0
            self.start_time = self.get_clock().now()
            self.total_distance = 0.0
            self.last_position = (start_x, start_y)
            self.get_logger().info(f'Path found with {len(self.path)} waypoints')
            self.publish_path()
            return True
        else:
            self.get_logger().error('No path found!')
            return False

    def plan_path(self, start_x, start_y, goal_x, goal_y):
        """Plan path using D* Lite algorithm."""
        if self.grid_dynamic is None or self.grid_original is None:
            self.get_logger().error('No map available!')
            return []

        start_grid = self.world_to_grid(start_x, start_y)
        goal_grid = self.world_to_grid(goal_x, goal_y)

        # Validate start in original grid (robot is already there)
        if not self._is_valid_in_original_grid(start_grid):
            self.get_logger().error('Start position is invalid!')
            return []

        # Validate goal in dynamic grid
        if not self._is_valid_in_dynamic_grid(goal_grid):
            self.get_logger().error('Goal position is invalid!')
            return []

        path_grid = self.dstar_plan(start_grid, goal_grid)

        if not path_grid:
            self.get_logger().error('D* Lite algorithm found no path!')
            return []

        path_world = [self.grid_to_world(gx, gy) for gx, gy in path_grid]
        max_wp = self.get_parameter('max_waypoints').value
        return self._simplify_path(path_world, max_points=max_wp)

    def dstar_plan(self, start_grid, goal_grid):
        """Plan or replan using D* Lite."""
        if self.dstar_planner is None or self.goal_grid != goal_grid:
            return self._initial_dstar_plan(start_grid, goal_grid)
        else:
            return self._replan_dstar(start_grid, goal_grid)

    def _initial_dstar_plan(self, start_grid, goal_grid):
        """Create initial D* Lite plan."""
        self.get_logger().info('Initializing D* Lite planner...')

        self.dstar_planner = DStarLite(
            self.grid_dynamic.copy(), start_grid, goal_grid,
            logger=self.get_logger())
        self.goal_grid = goal_grid
        self.planner_grid_snapshot = (
            self.grid_base.copy() if self.grid_base is not None else self.grid.copy())

        if not self.dstar_planner.compute_shortest_path():
            self.get_logger().error('D* Lite failed to find initial path')
            return []

        path = self.dstar_planner.extract_path()
        if path:
            self.get_logger().info(f'D* Lite found initial path with {len(path)} waypoints')
        return path

    def _replan_dstar(self, start_grid, goal_grid):
        """Replan using D* Lite with updated obstacles."""
        self.get_logger().info('Replanning with D* Lite...')

        start_x, start_y = start_grid
        if (0 <= start_x < self.grid_dynamic.shape[1] and
            0 <= start_y < self.grid_dynamic.shape[0]):
            if self.grid_dynamic[start_y, start_x] != 0:
                self.get_logger().warn('Start position is occupied, clearing robot area...')
                self._clear_robot_area(start_x, start_y)

        self.dstar_planner.update_start(start_grid)

        changed_cells = self._find_changed_cells()

        if changed_cells:
            self.get_logger().info(
                f'Persistent obstacle changes detected: {len(changed_cells)} cells')
            self.dstar_planner.grid = self.grid_dynamic.copy()
            self.dstar_planner.update_obstacles(changed_cells)
            self.planner_grid_snapshot = (
                self.grid_base.copy() if self.grid_base is not None else self.grid.copy())

        if not self.dstar_planner.compute_shortest_path():
            self.get_logger().error('D* Lite replanning failed')
            return []

        if self.dstar_planner.g[start_grid] == float('inf'):
            self.get_logger().warn('Start unreachable after replan, reinitializing...')
            return self._reinitialize_dstar(start_grid, goal_grid)

        path = self.dstar_planner.extract_path(debug=True)
        if not path:
            self.get_logger().warn('D* Lite extract failed, reinitializing...')
            return self._reinitialize_dstar(start_grid, goal_grid)

        self.get_logger().info(f'D* Lite replanning successful: {len(path)} waypoints')
        return path

    def _find_changed_cells(self):
        """Find cells that changed since last planning (vectorized)."""
        current_base = self.grid_base if self.grid_base is not None else self.grid

        if self.planner_grid_snapshot is None:
            return []

        diff = np.where(self.planner_grid_snapshot != current_base)
        return list(zip(diff[1].tolist(), diff[0].tolist()))  # (x, y)

    def _reinitialize_dstar(self, start_grid, goal_grid):
        """Reinitialize D* Lite planner from scratch (fallback)."""
        self.get_logger().info('Reinitializing D* Lite planner from scratch...')

        start_x, start_y = start_grid
        self._clear_robot_area(start_x, start_y)

        goal_x, goal_y = goal_grid
        if (0 <= goal_x < self.grid_dynamic.shape[1] and
            0 <= goal_y < self.grid_dynamic.shape[0] and
            self.grid_dynamic[goal_y, goal_x] != 0):
            new_goal = self._find_nearest_free_cell(goal_x, goal_y)
            if new_goal:
                goal_grid = new_goal
                self.goal_grid = new_goal
            else:
                self.get_logger().error('Could not find free cell near goal')
                return []

        self.dstar_planner = DStarLite(
            self.grid_dynamic.copy(), start_grid, goal_grid,
            logger=self.get_logger())
        self.planner_grid_snapshot = (
            self.grid_base.copy() if self.grid_base is not None else self.grid.copy())

        if not self.dstar_planner.compute_shortest_path():
            self.get_logger().error('Fresh D* Lite also failed to find path')
            return []

        if self.dstar_planner.g[start_grid] == float('inf'):
            self.get_logger().error('Start still unreachable after reinitialization')
            return []

        path = self.dstar_planner.extract_path(debug=True)
        if path:
            self.get_logger().info(f'Fresh D* Lite found path with {len(path)} waypoints')
        return path

    def _find_nearest_free_cell(self, gx, gy, max_radius=10):
        """Find the nearest free cell to the given position."""
        for radius in range(1, max_radius + 1):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if abs(dx) != radius and abs(dy) != radius:
                        continue
                    nx, ny = gx + dx, gy + dy
                    if (0 <= nx < self.grid_dynamic.shape[1] and
                        0 <= ny < self.grid_dynamic.shape[0] and
                        self.grid_dynamic[ny, nx] == 0):
                        return (nx, ny)
        return None

    def _clear_robot_area(self, robot_gx, robot_gy):
        """Clear the area around the robot in all grids."""
        clearance = PlannerConstants.ROBOT_CLEARANCE_CELLS
        changed_cells = []

        for dy in range(-clearance, clearance + 1):
            for dx in range(-clearance, clearance + 1):
                cx, cy = robot_gx + dx, robot_gy + dy
                if (0 <= cx < self.grid_dynamic.shape[1] and
                    0 <= cy < self.grid_dynamic.shape[0]):
                    if self.grid_dynamic[cy, cx] != 0:
                        changed_cells.append((cx, cy))
                    self.grid_dynamic[cy, cx] = 0
                    if self.grid_base is not None:
                        self.grid_base[cy, cx] = 0

        if self.dstar_planner is not None and changed_cells:
            self.dstar_planner.grid = self.grid_dynamic.copy()
            self.dstar_planner.update_obstacles(changed_cells)
            self.get_logger().info(f'Cleared {len(changed_cells)} cells around robot')


    def control_loop(self):
        """Main control loop -- executes at 10 Hz."""
        if not self.path or self.current_pose is None:
            return

        # Update distance traveled
        if self.last_position is not None:
            current_x = self.current_pose['x']
            current_y = self.current_pose['y']
            dx = current_x - self.last_position[0]
            dy = current_y - self.last_position[1]
            self.total_distance += math.sqrt(dx**2 + dy**2)
            self.last_position = (current_x, current_y)

        if self.replanning_needed:
            self._handle_replanning()
            return

        if self.current_waypoint_idx >= len(self.path):
            if self.standalone:
                self.stop_robot()
            if self.start_time is not None:
                elapsed_time = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
                self.get_logger().info('Goal reached!')
                self.get_logger().info(f'Total distance traveled: {self.total_distance:.2f} meters')
                self.get_logger().info(f'Total time elapsed: {elapsed_time:.2f} seconds')
            else:
                self.get_logger().info('Goal reached!')
            self._planner_state = 'idle'
            self._goal_just_reached = True
            self.path = []
            return

        if self.standalone:
            self._follow_waypoint()

    def _handle_replanning(self):
        """Handle replanning request."""
        self.get_logger().info('Stopping robot for replanning...')
        if self.standalone:
            self.stop_robot()

        if self.goal_grid is not None:
            goal_x, goal_y = self.grid_to_world(self.goal_grid[0], self.goal_grid[1])

            preserved_waypoints = []
            replan_start_x = self.current_pose['x']
            replan_start_y = self.current_pose['y']

            if (self.blocked_waypoint_idx is not None and
                self.blocked_waypoint_idx > self.current_waypoint_idx and
                self.path):
                preserved_waypoints = self.path[self.current_waypoint_idx:self.blocked_waypoint_idx]
                if preserved_waypoints:
                    replan_start_x, replan_start_y = preserved_waypoints[-1]

            new_path = self.plan_path(replan_start_x, replan_start_y, goal_x, goal_y)

            if new_path:
                if preserved_waypoints:
                    if (new_path and len(preserved_waypoints) > 0 and
                        abs(new_path[0][0] - preserved_waypoints[-1][0]) < 0.05 and
                        abs(new_path[0][1] - preserved_waypoints[-1][1]) < 0.05):
                        new_path = new_path[1:]
                    self.path = preserved_waypoints + new_path
                else:
                    self.path = new_path
                    self.current_waypoint_idx = 0

                self.get_logger().info(
                    f'Replanning successful! Path: {len(self.path)} waypoints '
                    f'({len(preserved_waypoints)} preserved + {len(new_path)} new)')
                self.publish_path()
            else:
                self.get_logger().error('Replanning failed! Stopping navigation.')
                self.path = []
                self._planner_state = 'idle'

        self.blocked_waypoint_idx = None
        self.replanning_needed = False

    def _follow_waypoint(self):
        """Follow current waypoint using simple proportional control."""
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
            self.known_obstacles.clear()
            self.get_logger().info(f'Waypoint {self.current_waypoint_idx}/{len(self.path)} reached')
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


    def publish_path(self):
        """Publish the planned path as a nav_msgs/Path."""
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'map'

        for x, y in self.path:
            pose = PoseStamped()
            pose.header.stamp = path_msg.header.stamp
            pose.header.frame_id = 'map'
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)

        self.path_pub.publish(path_msg)

    def publish_dynamic_grid(self):
        """Publish the dynamic grid for visualization."""
        if self.grid_dynamic is None or self.resolution is None:
            return

        grid_msg = OccupancyGrid()
        grid_msg.header.stamp = self.get_clock().now().to_msg()
        grid_msg.header.frame_id = 'map'

        grid_msg.info.resolution = self.resolution
        grid_msg.info.width = self.grid_dynamic.shape[1]
        grid_msg.info.height = self.grid_dynamic.shape[0]
        grid_msg.info.origin.position.x = self.origin[0]
        grid_msg.info.origin.position.y = self.origin[1]
        grid_msg.info.origin.position.z = 0.0

        occupancy_data = (self.grid_dynamic * 100).astype(np.int8).flatten().tolist()
        grid_msg.data = occupancy_data

        self.dynamic_grid_pub.publish(grid_msg)

    def _is_valid_in_dynamic_grid(self, grid_pos):
        """Check if position is valid in the dynamic grid."""
        gx, gy = grid_pos
        if self.grid_dynamic is None:
            return False
        if gx < 0 or gx >= self.grid_dynamic.shape[1] or gy < 0 or gy >= self.grid_dynamic.shape[0]:
            return False
        return self.grid_dynamic[gy, gx] == 0

    def _is_valid_in_original_grid(self, grid_pos):
        """Check if position is valid in original (uninflated) grid."""
        gx, gy = grid_pos
        if self.grid_original is None:
            return False
        if (gx < 0 or gx >= self.grid_original.shape[1] or
                gy < 0 or gy >= self.grid_original.shape[0]):
            return False
        return self.grid_original[gy, gx] == 0

    def world_to_grid(self, x, y):
        """Convert world coordinates to grid indices."""
        grid_x = int((x - self.origin[0]) / self.resolution)
        grid_y = int((y - self.origin[1]) / self.resolution)
        return (grid_x, grid_y)

    def grid_to_world(self, grid_x, grid_y):
        """Convert grid indices to world coordinates."""
        x = grid_x * self.resolution + self.origin[0]
        y = grid_y * self.resolution + self.origin[1]
        return (x, y)

    def _simplify_path(self, path, max_points=10):
        """Simplify path by keeping only key waypoints."""
        if len(path) <= max_points:
            return path
        step = len(path) // max_points
        simplified = [path[i] for i in range(0, len(path), step)]
        simplified.append(path[-1])
        return simplified

    @staticmethod
    def quaternion_to_yaw(q):
        """Convert quaternion to yaw angle."""
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)


def normalize_angle(angle):
    """Normalize angle to [-pi, pi] using modular arithmetic."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def main(args=None):
    """Main entry point for D* Lite navigator."""
    rclpy.init(args=args)
    navigator = DStarNavigator()

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
