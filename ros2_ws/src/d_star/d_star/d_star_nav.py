#!/usr/bin/env python3
"""
D* Lite Navigator with Dynamic Replanning for ROS2 TurtleBot4

This module implements a complete navigation system using D* Lite pathfinding
algorithm with dynamic replanning capabilities. It integrates with Nav2's
global costmap for persistent obstacle tracking and local costmap for
real-time dynamic obstacle detection.

Key Features:
    - D* Lite incremental pathfinding (efficient replanning)
    - Nav2 global costmap integration for persistent obstacle tracking
    - Nav2 local costmap integration for dynamic obstacle detection and replanning

Architecture:
    - DStarLite: Core pathfinding algorithm
    - DStarNavigator: ROS2 node integrating planning, sensing, and control

Grid Hierarchy:
    - grid_base: From global costmap (persistent obstacles)
    - grid_local_costmap: From local costmap (dynamic obstacles near robot)
    - grid_dynamic: max(grid_base, grid_local_costmap) — the planning grid

Authors: Devin Dennis, Assisted with Claude Code
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from geometry_msgs.msg import Twist, TwistStamped, PoseStamped
from nav_msgs.msg import Odometry, OccupancyGrid
from map_msgs.msg import OccupancyGridUpdate
import numpy as np
import heapq
import math
import os
from collections import defaultdict


# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

class PlannerConstants:
    """
    Centralized configuration for all planning and obstacle detection parameters.

    Adjust these values to tune robot behavior:
    - Adjust COSTMAP_OBSTACLE_THRESHOLD for obstacle sensitivity
    """

    # -------- Robot Physical Parameters (meters) --------
    NAMESPACE = '/don'          # Name of the physical robot
    ROBOT_RADIUS = 0.22           # Physical radius of the robot body
    SAFETY_CLEARANCE = 0.1      # Extra safety buffer around obstacles

    # -------- Grid Cell Tolerances (cells) --------
    PATH_CHECK_RADIUS = 2             # Cells around waypoint to check for obstacles

    # -------- Path Planning Parameters --------
    MAX_PATH_ITERATIONS = 10000000    # Max D* Lite iterations before timeout
    MAX_WAYPOINTS = 10

    # -------- Control Parameters --------
    REPLAN_COOLDOWN = 3.0         # Minimum seconds between replans
    ROBOT_CLEARANCE_CELLS = 3     # Cells around robot to keep free

    # -------- Topic Names --------
    CMD_VEL = f'{NAMESPACE}/cmd_vel'
    ODOMETRY = f'{NAMESPACE}/odom'
    DYNAMIC_GRID = f'{NAMESPACE}/dynamic_grid'
    GOAL_POSE = f'{NAMESPACE}/d_star_goal_pose'

    # Global costmap topics (Nav2) - used as base/persistent map
    GLOBAL_COSTMAP = f'{NAMESPACE}/global_costmap/costmap'
    GLOBAL_COSTMAP_UPDATES = f'{NAMESPACE}/global_costmap/costmap_updates'

    # Local costmap topics (Nav2) - used for dynamic obstacle detection
    LOCAL_COSTMAP = f'{NAMESPACE}/local_costmap/costmap'
    LOCAL_COSTMAP_UPDATES = f'{NAMESPACE}/local_costmap/costmap_updates'

    # Costmap configuration
    COSTMAP_OBSTACLE_THRESHOLD = 50  # Cost value above which cell is considered obstacle (0-254)


# ============================================================================
# D* LITE PATHFINDING ALGORITHM
# ============================================================================

class DStarLite:
    """
    D* Lite incremental search algorithm for dynamic pathfinding.

    D* Lite is an optimized version of D* that efficiently handles changing
    environments by only recomputing affected portions of the path when
    obstacles appear or disappear.

    Key Concepts:
        - g(s): Cost-to-come from start to state s
        - rhs(s): One-step lookahead value (min cost through neighbors)
        - k_m: Heuristic offset for incremental search
        - Locally consistent: g(s) == rhs(s)

    Usage:
        planner = DStarLite(grid, start, goal)
        planner.compute_shortest_path()  # Initial plan
        path = planner.extract_path()

        # When obstacles change:
        planner.update_start(new_start)
        planner.update_obstacles(changed_cells)
        planner.compute_shortest_path()  # Incremental replan
        path = planner.extract_path()
    """

    def __init__(self, grid, start, goal):
        """
        Initialize D* Lite planner.

        Args:
            grid: 2D numpy array (0=free, 1=occupied)
            start: Tuple (x, y) start position in grid coordinates
            goal: Tuple (x, y) goal position in grid coordinates
        """
        self.grid = grid
        self.rows, self.cols = grid.shape
        self.start = start
        self.goal = goal
        self.s_last = start  # Last start position for k_m calculation
        self.k_m = 0         # Heuristic modifier for incremental search

        # Cost values (defaultdict for automatic infinity initialization)
        self.g = defaultdict(lambda: float('inf'))    # Cost-to-come
        self.rhs = defaultdict(lambda: float('inf'))  # One-step lookahead
        self.open_list = []  # Priority queue of states to process

        # Initialize goal
        self.rhs[self.goal] = 0
        heapq.heappush(self.open_list, (self.calculate_key(self.goal), self.goal))

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

    def update_vertex(self, u):
        """Update state u's cost values and priority queue status."""
        if u != self.goal:
            min_cost = float('inf')
            for s_prime in self.get_neighbors(u):
                cost = self.cost(u, s_prime) + self.g[s_prime]
                min_cost = min(min_cost, cost)
            self.rhs[u] = min_cost

        self.open_list = [(k, s) for k, s in self.open_list if s != u]
        heapq.heapify(self.open_list)

        if self.g[u] != self.rhs[u]:
            heapq.heappush(self.open_list, (self.calculate_key(u), u))

    def compute_shortest_path(self):
        """
        Main D* Lite computation - processes states until start is consistent.

        Returns:
            True if path found, False if timeout or no path
        """
        iterations = 0

        while (self.open_list and
               (self.compare_keys(self.open_list[0][0], self.calculate_key(self.start)) or
                self.rhs[self.start] != self.g[self.start])):

            iterations += 1
            if iterations > PlannerConstants.MAX_PATH_ITERATIONS:
                return False

            k_old, u = heapq.heappop(self.open_list)
            k_new = self.calculate_key(u)

            if self.compare_keys(k_old, k_new):
                heapq.heappush(self.open_list, (k_new, u))
            elif self.g[u] > self.rhs[u]:
                self.g[u] = self.rhs[u]
                for s in self.get_neighbors(u):
                    self.update_vertex(s)
            else:
                self.g[u] = float('inf')
                self.update_vertex(u)
                for s in self.get_neighbors(u):
                    self.update_vertex(s)

        return True

    def compare_keys(self, k1, k2):
        """Compare two priority keys. Returns True if k1 < k2."""
        return k1[0] < k2[0] or (k1[0] == k2[0] and k1[1] < k2[1])

    def extract_path(self, debug=False):
        """
        Extract path from start to goal by following gradient of g-values.

        Returns:
            List of states [(x, y), ...] from start to goal, or [] if no path
        """
        if self.g[self.start] == float('inf'):
            if debug:
                print(f'[DEBUG] extract_path failed: g[start] = inf')
            return []

        path = [self.start]
        current = self.start
        visited = set([self.start])
        max_iterations = self.rows * self.cols

        while current != self.goal:
            if len(path) > max_iterations:
                if debug:
                    print(f'[DEBUG] extract_path failed: loop detected at {current}')
                return []

            neighbors = self.get_neighbors(current)
            if not neighbors:
                if debug:
                    print(f'[DEBUG] extract_path failed: dead end at {current}, no valid neighbors')
                return []

            valid_neighbors = [
                (n, self.g[n]) for n in neighbors
                if self.g[n] != float('inf') and n not in visited
            ]

            if not valid_neighbors:
                if debug:
                    all_neighbor_g = [(n, self.g[n]) for n in neighbors]
                    print(f'[DEBUG] extract_path failed: no valid moves from {current}')
                    print(f'[DEBUG]   All neighbors and g-values: {all_neighbor_g}')
                    print(f'[DEBUG]   Already visited: {[n for n in neighbors if n in visited]}')
                return []

            next_state, min_g = min(valid_neighbors, key=lambda x: x[1])

            if min_g > self.g[current] + 0.01 and next_state != self.goal:
                if debug:
                    print(f'[DEBUG] extract_path failed: going uphill from {current} (g={self.g[current]}) to {next_state} (g={min_g})')
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


# ============================================================================
# ROS2 NAVIGATOR NODE
# ============================================================================

class DStarNavigator(Node):
    """
    Navigation system using D* Lite with Nav2 global and local costmaps.

    This node manages:
    - Path planning with D* Lite
    - Nav2 global costmap for persistent obstacle tracking (base map)
    - Nav2 local costmap for dynamic obstacle detection and replanning
    - Robot motion control
    - Automatic replanning when obstacles block the path

    Grid Hierarchy:
        1. grid_base: From global costmap (persistent obstacles)
        2. grid_local_costmap: From local costmap (dynamic obstacles near robot)
        3. grid_dynamic: max(grid_base, grid_local_costmap) — the planning grid

    Replanning Triggers:
        - Local costmap detects obstacle changes near path
        - Cooldown prevents rapid replanning
    """

    # ========================================================================
    # INITIALIZATION
    # ========================================================================

    def __init__(self, robot_radius=0.22, safety_clearance=0.15):
        """
        Initialize the D* Lite navigator.

        Args:
            robot_radius: Physical radius of robot in meters
            safety_clearance: Extra safety buffer around obstacles in meters
        """
        super().__init__('dstar_navigator')

        # Robot configuration
        self.robot_radius = robot_radius
        self.safety_clearance = safety_clearance

        # ROS2 publishers
        self.cmd_vel_pub = self.create_publisher(TwistStamped, PlannerConstants.CMD_VEL, 10)
        self.dynamic_grid_pub = self.create_publisher(OccupancyGrid, PlannerConstants.DYNAMIC_GRID, 10)

        # ROS2 subscribers
        self._setup_subscribers()

        # Map data - Grid hierarchy
        self.grid_base = None             # From global costmap (persistent)
        self.grid_local_costmap = None    # From local costmap (dynamic)
        self.grid_dynamic = None          # max(grid_base, grid_local_costmap) — planning grid
        self.resolution = None            # Meters per grid cell
        self.origin = None                # Map origin [x, y] in world coordinates
        self.map_received = False

        # Robot state
        self.current_pose = None
        self.path = []
        self.current_waypoint_idx = 0

        # D* Lite planner state
        self.dstar_planner = None
        self.goal_grid = None
        self.planner_grid_snapshot = None
        self.known_obstacles = set()

        # Global costmap state
        self.global_costmap = None
        self.global_costmap_info = None

        # Local costmap state
        self.local_costmap = None
        self.local_costmap_info = None

        # Control parameters
        self.linear_speed = 0.2
        self.angular_speed = 0.5
        self.position_tolerance = 0.1
        self.angle_tolerance = 0.1
        self.replanning_needed = False
        self.last_replan_time = self.get_clock().now()
        self.blocked_waypoint_idx = None

        # Control loop timer (10 Hz)
        self.control_timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info('Grid will be initialized from first global costmap message')
        self._log_initialization()

    def _setup_subscribers(self):
        """Setup ROS2 subscribers with appropriate QoS profiles."""
        # Best-effort QoS for real-time data (odometry)
        qos_best_effort = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.odom_sub = self.create_subscription(
            Odometry, PlannerConstants.ODOMETRY, self.odom_callback, qos_best_effort
        )

        # Goal pose subscriber
        self.goal_pose_sub = self.create_subscription(
            PoseStamped, PlannerConstants.GOAL_POSE, self.goal_pose_callback, 10
        )

        # Global costmap subscribers (base/persistent map)
        self.global_costmap_sub = self.create_subscription(
            OccupancyGrid, PlannerConstants.GLOBAL_COSTMAP, self.global_costmap_callback, 10
        )
        self.global_costmap_update_sub = self.create_subscription(
            OccupancyGridUpdate, PlannerConstants.GLOBAL_COSTMAP_UPDATES,
            self.global_costmap_update_callback, 10
        )

        # Local costmap subscribers (dynamic obstacle detection)
        self.local_costmap_sub = self.create_subscription(
            OccupancyGrid, PlannerConstants.LOCAL_COSTMAP, self.local_costmap_callback, 10
        )
        self.local_costmap_update_sub = self.create_subscription(
            OccupancyGridUpdate, PlannerConstants.LOCAL_COSTMAP_UPDATES,
            self.local_costmap_update_callback, 10
        )

    def goal_pose_callback(self, msg: PoseStamped):
        """Handle incoming goal pose messages."""
        goal_x = msg.pose.position.x
        goal_y = msg.pose.position.y

        self.get_logger().info(f'Received goal pose: ({goal_x:.2f}, {goal_y:.2f})')

        if self.current_pose is None:
            self.get_logger().warn('No odometry data yet, cannot navigate to goal')
            return

        start_x = self.current_pose['x']
        start_y = self.current_pose['y']

        self.get_logger().info(f'Starting navigation from ({start_x:.2f}, {start_y:.2f}) to ({goal_x:.2f}, {goal_y:.2f})')

        if self.navigate_to_goal(start_x, start_y, goal_x, goal_y):
            self.get_logger().info('Navigation started successfully')
        else:
            self.get_logger().error('Failed to plan path to goal')

    def _log_initialization(self):
        """Log initialization information."""
        total_inflation = self.robot_radius + self.safety_clearance
        self.get_logger().info('D* Lite Navigator initialized')
        self.get_logger().info(f'Robot radius: {self.robot_radius}m, Safety: {self.safety_clearance}m, Total: {total_inflation}m')
        self.get_logger().info(f'Listening for goal poses on: {PlannerConstants.GOAL_POSE}')
        self.get_logger().info(f'Global costmap: {PlannerConstants.GLOBAL_COSTMAP}')
        self.get_logger().info(f'Local costmap: {PlannerConstants.LOCAL_COSTMAP}')

    # ========================================================================
    # ROS2 CALLBACKS
    # ========================================================================

    def odom_callback(self, msg):
        """Update current robot pose from odometry."""
        self.current_pose = {
            'x': msg.pose.pose.position.x,
            'y': msg.pose.pose.position.y,
            'theta': self.quaternion_to_yaw(msg.pose.pose.orientation)
        }

    # ========================================================================
    # GLOBAL COSTMAP CALLBACKS (Base/Persistent Map)
    # ========================================================================

    def global_costmap_callback(self, msg: OccupancyGrid):
        """
        Receive full global costmap from Nav2.

        On first message, initializes all grids. On subsequent messages, updates
        grid_base with the full costmap state.

        Args:
            msg: OccupancyGrid message from Nav2 global costmap
        """
        self.global_costmap = np.array(msg.data, dtype=np.int8).reshape(
            (msg.info.height, msg.info.width)
        )
        self.global_costmap_info = msg.info

        is_first = not self.map_received

        # Initialize grids on first message
        if not self.map_received:
            self.resolution = msg.info.resolution
            self.origin = [msg.info.origin.position.x, msg.info.origin.position.y]
            height, width = msg.info.height, msg.info.width

            self.grid_base = np.zeros((height, width), dtype=np.int8)
            self.grid_local_costmap = np.zeros((height, width), dtype=np.int8)
            self.grid_dynamic = np.zeros((height, width), dtype=np.int8)

            self.map_received = True
            self.get_logger().info(
                f'Initialized grid from global costmap: {width}x{height}, '
                f'resolution={self.resolution}m, origin=({self.origin[0]:.2f}, {self.origin[1]:.2f})'
            )

        if self.grid_base is None or self.resolution is None:
            return

        # Build new grid_base from global costmap
        changed_cells = []
        costmap_resolution = msg.info.resolution
        costmap_origin_x = msg.info.origin.position.x
        costmap_origin_y = msg.info.origin.position.y

        previous_base = self.grid_base.copy()

        for cy in range(msg.info.height):
            for cx in range(msg.info.width):
                cost = self.global_costmap[cy, cx]

                world_x = costmap_origin_x + cx * costmap_resolution
                world_y = costmap_origin_y + cy * costmap_resolution
                grid_x, grid_y = self.world_to_grid(world_x, world_y)

                if not self._in_bounds(grid_x, grid_y):
                    continue

                is_obstacle = 1 if cost > PlannerConstants.COSTMAP_OBSTACLE_THRESHOLD else 0
                self.grid_base[grid_y, grid_x] = is_obstacle

        # Rebuild grid_dynamic = max(grid_base, grid_local_costmap)
        new_dynamic = np.maximum(self.grid_base, self.grid_local_costmap)

        # Find cells that changed in grid_dynamic
        diff = (new_dynamic != self.grid_dynamic)
        changed_ys, changed_xs = np.where(diff)
        changed_cells = list(zip(changed_xs.tolist(), changed_ys.tolist()))

        self.grid_dynamic = new_dynamic

        if changed_cells and self.dstar_planner:
            self.dstar_planner.grid = self.grid_dynamic.copy()
            self.dstar_planner.update_obstacles(changed_cells)
            self.planner_grid_snapshot = self.grid_dynamic.copy()
            self.get_logger().debug(
                f'Global costmap processed: {len(changed_cells)} cells changed in dynamic grid')

        self.publish_dynamic_grid()

        if is_first:
            obstacle_count = int(self.grid_base.sum())
            self.get_logger().info(f'Global costmap integrated: {obstacle_count} obstacle cells')

    def global_costmap_update_callback(self, msg: OccupancyGridUpdate):
        """
        Process incremental global costmap updates from Nav2.

        Updates grid_base with changed cells, then rebuilds grid_dynamic.

        Args:
            msg: OccupancyGridUpdate message containing changed cells
        """
        if self.global_costmap_info is None:
            return

        if self.grid_base is None or self.resolution is None:
            return

        changed_cells = []

        for i, cost in enumerate(msg.data):
            local_x = msg.x + (i % msg.width)
            local_y = msg.y + (i // msg.width)

            world_x = (self.global_costmap_info.origin.position.x +
                      local_x * self.global_costmap_info.resolution)
            world_y = (self.global_costmap_info.origin.position.y +
                      local_y * self.global_costmap_info.resolution)

            grid_x, grid_y = self.world_to_grid(world_x, world_y)

            if not self._in_bounds(grid_x, grid_y):
                continue

            is_obstacle = 1 if cost > PlannerConstants.COSTMAP_OBSTACLE_THRESHOLD else 0

            # Update grid_base
            self.grid_base[grid_y, grid_x] = is_obstacle

            # Rebuild this cell in grid_dynamic
            new_val = max(is_obstacle, self.grid_local_costmap[grid_y, grid_x])
            if self.grid_dynamic[grid_y, grid_x] != new_val:
                self.grid_dynamic[grid_y, grid_x] = new_val
                changed_cells.append((grid_x, grid_y))

        if changed_cells and self.dstar_planner:
            self.dstar_planner.grid = self.grid_dynamic.copy()
            self.dstar_planner.update_obstacles(changed_cells)
            self.planner_grid_snapshot = self.grid_dynamic.copy()
            self.get_logger().debug(f'Global costmap update: {len(changed_cells)} cells changed')
            self.publish_dynamic_grid()

    # ========================================================================
    # LOCAL COSTMAP CALLBACKS (Dynamic Obstacle Detection + Replanning)
    # ========================================================================

    def local_costmap_callback(self, msg: OccupancyGrid):
        """
        Receive full local costmap from Nav2.

        Updates grid_local_costmap, rebuilds grid_dynamic, and triggers
        replanning if obstacles block the current path.

        Args:
            msg: OccupancyGrid message from Nav2 local costmap
        """
        self.local_costmap = np.array(msg.data, dtype=np.int8).reshape(
            (msg.info.height, msg.info.width)
        )
        self.local_costmap_info = msg.info

        if self.grid_base is None or self.resolution is None:
            return

        # Clear local costmap layer, then repopulate from this message
        self.grid_local_costmap[:] = 0

        costmap_resolution = msg.info.resolution
        costmap_origin_x = msg.info.origin.position.x
        costmap_origin_y = msg.info.origin.position.y

        for cy in range(msg.info.height):
            for cx in range(msg.info.width):
                cost = self.local_costmap[cy, cx]

                world_x = costmap_origin_x + cx * costmap_resolution
                world_y = costmap_origin_y + cy * costmap_resolution
                grid_x, grid_y = self.world_to_grid(world_x, world_y)

                if not self._in_bounds(grid_x, grid_y):
                    continue

                is_obstacle = 1 if cost > PlannerConstants.COSTMAP_OBSTACLE_THRESHOLD else 0
                self.grid_local_costmap[grid_y, grid_x] = is_obstacle

        # Rebuild grid_dynamic = max(grid_base, grid_local_costmap)
        new_dynamic = np.maximum(self.grid_base, self.grid_local_costmap)

        diff = (new_dynamic != self.grid_dynamic)
        changed_ys, changed_xs = np.where(diff)
        changed_cells = list(zip(changed_xs.tolist(), changed_ys.tolist()))

        self.grid_dynamic = new_dynamic

        if changed_cells and self.dstar_planner:
            self.dstar_planner.grid = self.grid_dynamic.copy()
            self.dstar_planner.update_obstacles(changed_cells)
            self.get_logger().debug(
                f'Local costmap processed: {len(changed_cells)} cells changed in dynamic grid')

            self.publish_dynamic_grid()

            # Check if replanning is needed
            if self.path:
                self._maybe_trigger_replan(changed_cells)

    def local_costmap_update_callback(self, msg: OccupancyGridUpdate):
        """
        Process incremental local costmap updates from Nav2.

        Updates grid_local_costmap with changed cells, rebuilds grid_dynamic,
        and triggers replanning if obstacles block the current path.

        Args:
            msg: OccupancyGridUpdate message containing changed cells
        """
        if self.local_costmap_info is None:
            return

        if self.grid_base is None or self.resolution is None:
            return

        changed_cells = []

        for i, cost in enumerate(msg.data):
            local_x = msg.x + (i % msg.width)
            local_y = msg.y + (i // msg.width)

            world_x = (self.local_costmap_info.origin.position.x +
                      local_x * self.local_costmap_info.resolution)
            world_y = (self.local_costmap_info.origin.position.y +
                      local_y * self.local_costmap_info.resolution)

            grid_x, grid_y = self.world_to_grid(world_x, world_y)

            if not self._in_bounds(grid_x, grid_y):
                continue

            is_obstacle = 1 if cost > PlannerConstants.COSTMAP_OBSTACLE_THRESHOLD else 0
            self.grid_local_costmap[grid_y, grid_x] = is_obstacle

            # Rebuild this cell in grid_dynamic
            new_val = max(self.grid_base[grid_y, grid_x], is_obstacle)
            if self.grid_dynamic[grid_y, grid_x] != new_val:
                self.grid_dynamic[grid_y, grid_x] = new_val
                changed_cells.append((grid_x, grid_y))

        if changed_cells and self.dstar_planner:
            self.dstar_planner.grid = self.grid_dynamic.copy()
            self.dstar_planner.update_obstacles(changed_cells)
            self.get_logger().debug(f'Local costmap update: {len(changed_cells)} cells changed')

            self.publish_dynamic_grid()

            if self.path:
                self._maybe_trigger_replan(changed_cells)

    def _maybe_trigger_replan(self, changed_cells):
        """
        Check if changed cells warrant replanning and trigger if so.

        Applies cooldown to prevent rapid replanning.

        Args:
            changed_cells: List of (grid_x, grid_y) tuples that changed
        """
        current_time = self.get_clock().now()
        time_since_last_replan = (current_time - self.last_replan_time).nanoseconds / 1e9

        if time_since_last_replan > PlannerConstants.REPLAN_COOLDOWN:
            path_blocked = self._check_path_blocked_by_costmap_changes(changed_cells)
            if path_blocked:
                self.get_logger().warn(
                    f'Local costmap detected obstacle in path, triggering replan '
                    f'({len(changed_cells)} cells changed)')
                self.replanning_needed = True
                self.last_replan_time = current_time

    def _check_path_blocked_by_costmap_changes(self, changed_cells):
        """
        Check if any changed cells block the current path.

        Args:
            changed_cells: List of (grid_x, grid_y) tuples that changed

        Returns:
            True if any changed cell is an obstacle near the path
        """
        if not self.path or self.current_waypoint_idx >= len(self.path):
            return False

        changed_set = set(changed_cells)

        for i in range(self.current_waypoint_idx, min(self.current_waypoint_idx + 5, len(self.path))):
            waypoint = self.path[i]
            wp_grid = self.world_to_grid(waypoint[0], waypoint[1])

            for dx in range(-PlannerConstants.PATH_CHECK_RADIUS, PlannerConstants.PATH_CHECK_RADIUS + 1):
                for dy in range(-PlannerConstants.PATH_CHECK_RADIUS, PlannerConstants.PATH_CHECK_RADIUS + 1):
                    check_cell = (wp_grid[0] + dx, wp_grid[1] + dy)
                    if check_cell in changed_set:
                        if self._in_bounds(check_cell[0], check_cell[1]):
                            if self.grid_dynamic[check_cell[1], check_cell[0]] == 1:
                                self.blocked_waypoint_idx = i
                                return True
        return False

    def _in_bounds(self, grid_x, grid_y):
        """Check if grid coordinates are within bounds."""
        if self.grid_dynamic is None:
            if self.grid_base is None:
                return False
            return (0 <= grid_x < self.grid_base.shape[1] and
                    0 <= grid_y < self.grid_base.shape[0])
        return (0 <= grid_x < self.grid_dynamic.shape[1] and
                0 <= grid_y < self.grid_dynamic.shape[0])

    # ========================================================================
    # PATH PLANNING
    # ========================================================================

    def navigate_to_goal(self, start_x, start_y, goal_x, goal_y):
        """
        Navigate from start to goal position.

        Args:
            start_x, start_y: Start position in meters (world coordinates)
            goal_x, goal_y: Goal position in meters (world coordinates)

        Returns:
            True if path found and navigation started, False otherwise
        """
        self.get_logger().info(f'Planning path from ({start_x:.2f}, {start_y:.2f}) to ({goal_x:.2f}, {goal_y:.2f})')

        self.known_obstacles.clear()

        self.path = self.plan_path(start_x, start_y, goal_x, goal_y)

        if self.path:
            self.current_waypoint_idx = 0
            self.get_logger().info(f'Path found with {len(self.path)} waypoints')
            return True
        else:
            self.get_logger().error('No path found!')
            return False

    def plan_path(self, start_x, start_y, goal_x, goal_y):
        """
        Plan path using D* Lite algorithm.

        Args:
            start_x, start_y: Start in world coordinates
            goal_x, goal_y: Goal in world coordinates

        Returns:
            List of waypoints [(x, y), ...] in world coordinates, or [] if no path
        """
        if self.grid_dynamic is None:
            self.get_logger().error('No map available!')
            return []

        start_grid = self.world_to_grid(start_x, start_y)
        goal_grid = self.world_to_grid(goal_x, goal_y)

        # Validate start position (must be free in base grid)
        if not self._is_valid_position(start_grid, self.grid_base):
            self.get_logger().error(f'Start position is invalid!')
            return []

        # Validate goal position (must be free in dynamic grid)
        if not self._is_valid_position(goal_grid, self.grid_dynamic):
            self.get_logger().error(f'Goal position is invalid!')
            return []

        path_grid = self.dstar_plan(start_grid, goal_grid)

        if not path_grid:
            self.get_logger().error(f'Dstar algorithm path not found!')
            return []

        path_world = [self.grid_to_world(gx, gy) for gx, gy in path_grid]

        return self._simplify_path(path_world, max_points=PlannerConstants.MAX_WAYPOINTS)

    def dstar_plan(self, start_grid, goal_grid):
        """Plan or replan using D* Lite."""
        if self.dstar_planner is None or self.goal_grid != goal_grid:
            return self._initial_dstar_plan(start_grid, goal_grid)
        else:
            return self._replan_dstar(start_grid, goal_grid)

    def _initial_dstar_plan(self, start_grid, goal_grid):
        """Create initial D* Lite plan."""
        self.get_logger().info('Initializing D* Lite planner...')

        self.dstar_planner = DStarLite(self.grid_dynamic.copy(), start_grid, goal_grid)
        self.goal_grid = goal_grid
        self.planner_grid_snapshot = self.grid_dynamic.copy()

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
            is_occupied = self.grid_dynamic[start_y, start_x] != 0
            self.get_logger().info(f'Start grid position: ({start_x}, {start_y}), occupied: {is_occupied}')

            if is_occupied:
                self.get_logger().warn('Start position is marked as occupied! Clearing robot area...')
                self._clear_robot_area(start_x, start_y)

        self.dstar_planner.update_start(start_grid)

        changed_cells = self._find_changed_cells()

        if changed_cells:
            self.get_logger().info(f'Grid changes detected: {len(changed_cells)} cells')
            self.dstar_planner.grid = self.grid_dynamic.copy()
            self.dstar_planner.update_obstacles(changed_cells)
            self.planner_grid_snapshot = self.grid_dynamic.copy()

        if not self.dstar_planner.compute_shortest_path():
            self.get_logger().error('D* Lite replanning failed - path is blocked')
            return []

        if self.dstar_planner.g[start_grid] == float('inf'):
            self.get_logger().warn('Start position unreachable after incremental replan - trying fresh reinitialization...')
            self._debug_unreachable_start(start_grid, goal_grid)
            return self._reinitialize_dstar(start_grid, goal_grid)

        path = self.dstar_planner.extract_path(debug=True)
        if not path:
            self.get_logger().warn('D* Lite incremental extract failed - trying fresh reinitialization...')
            return self._reinitialize_dstar(start_grid, goal_grid)

        self.get_logger().info(f'D* Lite replanning successful: {len(path)} waypoints')
        return path

    def _find_changed_cells(self):
        """Find cells that changed since last planning."""
        changed_cells = []

        if self.planner_grid_snapshot is None:
            return changed_cells

        diff = (self.grid_dynamic != self.planner_grid_snapshot)
        changed_ys, changed_xs = np.where(diff)
        changed_cells = list(zip(changed_xs.tolist(), changed_ys.tolist()))

        return changed_cells

    def _reinitialize_dstar(self, start_grid, goal_grid):
        """Reinitialize D* Lite planner from scratch."""
        self.get_logger().info('Reinitializing D* Lite planner from scratch...')

        start_x, start_y = start_grid
        self._clear_robot_area(start_x, start_y)

        goal_x, goal_y = goal_grid
        if (0 <= goal_x < self.grid_dynamic.shape[1] and
            0 <= goal_y < self.grid_dynamic.shape[0] and
            self.grid_dynamic[goal_y, goal_x] != 0):
            self.get_logger().warn(f'Goal ({goal_x}, {goal_y}) is occupied, finding nearest free cell...')
            new_goal = self._find_nearest_free_cell(goal_x, goal_y)
            if new_goal:
                goal_grid = new_goal
                self.goal_grid = new_goal
                self.get_logger().info(f'Using alternative goal: {new_goal}')
            else:
                self.get_logger().error('Could not find free cell near goal')
                return []

        self.dstar_planner = DStarLite(self.grid_dynamic.copy(), start_grid, goal_grid)
        self.planner_grid_snapshot = self.grid_dynamic.copy()

        if not self.dstar_planner.compute_shortest_path():
            self.get_logger().error('Fresh D* Lite also failed to find path')
            return []

        if self.dstar_planner.g[start_grid] == float('inf'):
            self.get_logger().error('Start still unreachable after reinitialization')
            self._debug_unreachable_start(start_grid, goal_grid)
            return []

        path = self.dstar_planner.extract_path(debug=True)
        if path:
            self.get_logger().info(f'Fresh D* Lite found path with {len(path)} waypoints')
        else:
            self.get_logger().error('Fresh D* Lite also failed to extract path')

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

    def _debug_unreachable_start(self, start_grid, goal_grid):
        """Debug helper to understand why start is unreachable."""
        start_x, start_y = start_grid
        goal_x, goal_y = goal_grid

        self.get_logger().error(f'  Start: ({start_x}, {start_y}), Goal: ({goal_x}, {goal_y})')

        obstacles_around_start = []
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                nx, ny = start_x + dx, start_y + dy
                if (0 <= nx < self.grid_dynamic.shape[1] and
                    0 <= ny < self.grid_dynamic.shape[0]):
                    if self.grid_dynamic[ny, nx] != 0:
                        obstacles_around_start.append((nx, ny))

        self.get_logger().error(f'  Obstacles within 3 cells of start: {len(obstacles_around_start)}')
        if obstacles_around_start and len(obstacles_around_start) <= 10:
            self.get_logger().error(f'  Obstacle positions: {obstacles_around_start}')

        if (0 <= goal_x < self.grid_dynamic.shape[1] and
            0 <= goal_y < self.grid_dynamic.shape[0]):
            goal_occupied = self.grid_dynamic[goal_y, goal_x] != 0
            self.get_logger().error(f'  Goal occupied: {goal_occupied}')

        goal_g = self.dstar_planner.g[goal_grid]
        self.get_logger().error(f'  Goal g-value: {goal_g}')

    def _clear_robot_area(self, robot_gx, robot_gy):
        """Clear the area around the robot in all grids."""
        clearance = PlannerConstants.ROBOT_CLEARANCE_CELLS
        changed_cells = []

        for dy in range(-clearance, clearance + 1):
            for dx in range(-clearance, clearance + 1):
                clear_x, clear_y = robot_gx + dx, robot_gy + dy

                if (0 <= clear_x < self.grid_dynamic.shape[1] and
                    0 <= clear_y < self.grid_dynamic.shape[0]):

                    if self.grid_dynamic[clear_y, clear_x] != 0:
                        changed_cells.append((clear_x, clear_y))

                    self.grid_dynamic[clear_y, clear_x] = 0
                    if self.grid_base is not None:
                        self.grid_base[clear_y, clear_x] = 0

        if self.dstar_planner is not None and changed_cells:
            self.dstar_planner.grid = self.grid_dynamic.copy()
            self.dstar_planner.update_obstacles(changed_cells)
            self.get_logger().info(f'Cleared {len(changed_cells)} cells around robot')

    # ========================================================================
    # ROBOT CONTROL
    # ========================================================================

    def control_loop(self):
        """Main control loop - executes at 10 Hz."""
        if not self.path or self.current_pose is None:
            return

        if self.replanning_needed:
            self._handle_replanning()
            return

        if self.current_waypoint_idx >= len(self.path):
            self.stop_robot()
            self.get_logger().info('Goal reached!')
            self.path = []
            return

        self._follow_waypoint()

    def _handle_replanning(self):
        """Handle replanning request."""
        self.get_logger().info('Stopping robot for replanning...')
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
                    self.get_logger().info(
                        f'Preserving {len(preserved_waypoints)} waypoints before obstacle at index {self.blocked_waypoint_idx}'
                    )

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

                self.get_logger().info(f'Replanning successful! Path: {len(self.path)} waypoints '
                                      f'({len(preserved_waypoints)} preserved + {len(new_path)} new)')
            else:
                self.get_logger().error('Replanning failed! Stopping navigation.')
                self.path = []

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
        angle_diff = self.normalize_angle(target_angle - self.current_pose['theta'])

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

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

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

    def _is_valid_position(self, grid_pos, grid):
        """
        Check if position is valid (in bounds and free) in the given grid.

        Args:
            grid_pos: Position (gx, gy)
            grid: Grid to check against

        Returns:
            True if in bounds and free, False otherwise
        """
        if grid is None:
            return False
        gx, gy = grid_pos
        if gx < 0 or gx >= grid.shape[1] or gy < 0 or gy >= grid.shape[0]:
            return False
        return grid[gy, gx] == 0

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

    def _simplify_path(self, path, max_points=20):
        """Simplify path by keeping only key waypoints."""
        if len(path) <= max_points:
            return path
        step = len(path) // max_points
        simplified = [path[i] for i in range(0, len(path), step)]
        simplified.append(path[-1])
        return simplified

    def get_current_position(self):
        """Get current robot position."""
        if self.current_pose is None:
            self.get_logger().warn('No odometry data available yet')
            return None
        return (self.current_pose['x'], self.current_pose['y'])

    @staticmethod
    def quaternion_to_yaw(q):
        """Convert quaternion to yaw angle."""
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def normalize_angle(angle):
        """Normalize angle to [-pi, pi]."""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main(args=None):
    """
    Main entry point for D* Lite navigator.

    Navigation:
        - Send goal poses to the goal_pose topic (PoseStamped messages)
        - Example: ros2 topic pub /don/d_star_goal_pose geometry_msgs/PoseStamped '{pose: {position: {x: 1.0, y: 2.0}}}' --once
    """
    robot_radius = float(os.getenv('ROBOT_RADIUS', str(PlannerConstants.ROBOT_RADIUS)))
    safety_clearance = float(os.getenv('SAFETY_CLEARANCE', str(PlannerConstants.SAFETY_CLEARANCE)))

    rclpy.init(args=args)
    navigator = DStarNavigator(
        robot_radius=robot_radius,
        safety_clearance=safety_clearance,
    )

    print("\n" + "="*60)
    print("TurtleBot4 D* Lite Navigator with Dynamic Replanning")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Robot radius: {robot_radius}m")
    print(f"  Safety clearance: {safety_clearance}m")
    print(f"\nTopics:")
    print(f"  Goal pose input: {PlannerConstants.GOAL_POSE}")
    print(f"  Odometry: {PlannerConstants.ODOMETRY}")
    print(f"  Global costmap: {PlannerConstants.GLOBAL_COSTMAP}")
    print(f"  Local costmap: {PlannerConstants.LOCAL_COSTMAP}")
    print(f"\nUsage:")
    print(f"  ros2 topic pub {PlannerConstants.GOAL_POSE} geometry_msgs/PoseStamped '{{pose: {{position: {{x: 1.0, y: 2.0}}}}}}' --once")
    print("="*60 + "\n")

    print("Waiting for odometry data...")
    while navigator.current_pose is None and rclpy.ok():
        rclpy.spin_once(navigator, timeout_sec=0.1)

    if navigator.current_pose:
        print(f"Odometry received. Current position: ({navigator.current_pose['x']:.2f}, {navigator.current_pose['y']:.2f})")

    print("Waiting for global costmap...")
    while not navigator.map_received and rclpy.ok():
        rclpy.spin_once(navigator, timeout_sec=0.1)

    print(f"\nReady! Waiting for goal poses on {PlannerConstants.GOAL_POSE}...")

    try:
        rclpy.spin(navigator)
    except KeyboardInterrupt:
        print("\nShutting down...")

    navigator.stop_robot()
    navigator.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
