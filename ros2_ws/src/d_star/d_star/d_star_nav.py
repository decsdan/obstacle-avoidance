#!/usr/bin/env python3
"""
D* Lite Navigator with Dynamic Replanning for ROS2 TurtleBot4

This module implements a complete navigation system using D* Lite pathfinding
algorithm with dynamic replanning capabilities. It integrates with SLAM for
real-time obstacle discovery and uses lidar for immediate obstacle detection.

Key Features:
    - D* Lite incremental pathfinding (efficient replanning)
    - SLAM integration for persistent obstacle tracking
    - Lidar-based dynamic obstacle avoidance
    - Optimistic planning mode for exploration
    - Obstacle inflation for safety margins

Architecture:
    - DStarLite: Core pathfinding algorithm
    - DStarNavigator: ROS2 node integrating planning, sensing, and control

Grid Hierarchy:
    - grid_original: Raw SLAM map (no inflation)
    - grid: Inflated SLAM map (obstacles + safety buffer)
    - grid_base: Accumulated SLAM obstacles (persistent)
    - grid_dynamic: Current planning grid = base + lidar obstacles (temporary)

Authors: Devin Dennis, Assisted with Claude Code
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from geometry_msgs.msg import Twist, TwistStamped, PoseStamped, Point
from nav_msgs.msg import Odometry, OccupancyGrid, Path
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
import numpy as np
import heapq
import math
import os
from collections import defaultdict
from scipy.ndimage import binary_dilation


# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

class PlannerConstants:
    """
    Centralized configuration for all planning and obstacle detection parameters.

    Adjust these values to tune robot behavior:
    - Increase SAFETY_CLEARANCE for more cautious navigation
    - Decrease LIDAR_MAX_RANGE to ignore distant obstacles
    - Increase MAX_OBSTACLE_CHECK_DISTANCE to trigger replanning earlier
    """

    # -------- Robot Physical Parameters (meters) --------
    NAMESPACE = '/don'          # Name of the physical robot
    ROBOT_RADIUS = 0.22           # Physical radius of the robot body
    SAFETY_CLEARANCE = 0.0001      # Extra safety buffer around obstacles
                                  # Total inflation = ROBOT_RADIUS + SAFETY_CLEARANCE

    # -------- Lidar Obstacle Detection (meters) --------
    LIDAR_MIN_RANGE = 0.1         # Ignore readings closer than this (robot body/noise)
    LIDAR_MAX_RANGE = 3.0         # Only detect obstacles up to this distance

    # -------- Grid Cell Tolerances (cells) --------
    KNOWN_OBSTACLE_TOLERANCE = 2     # Cells to check for known SLAM obstacle filtering
    NEARBY_OBSTACLE_TOLERANCE = 2     # Tolerance for coordinate transform errors
    PATH_CHECK_RADIUS = 3             # Cells around waypoint to check
    ROBOT_BODY_RADIUS_CELLS = 3       # Min distance from robot (self-detection filter)
    CLOSE_OBSTACLE_CELLS = 12         # Threshold for "very close" obstacles
    EMERGENCY_OBSTACLE_CELLS = 15      # Max distance to immediately mark in grid

    # -------- Path Planning Parameters --------
    MAX_PATH_ITERATIONS = 10000000      # Max D* Lite iterations before timeout
    PATH_BLOCKING_BUFFER = 5          # Extra buffer for blocking detection
    LOOKAHEAD_SEGMENTS = 20           # Path segments to check ahead for obstacles
    MAX_OBSTACLE_CHECK_DISTANCE = 40  # Max cells to check (prevents false triggers)

    # -------- Control Parameters --------
    REPLAN_COOLDOWN = 1.0         # Minimum seconds between replans
    ROBOT_CLEARANCE_CELLS = 3     # Cells around robot to keep free
    INFLATION_DIVISOR = 5         # Reduce inflation for dynamic obstacles
    MAX_WAYPOINTS = 10

    # Publishing/Subscribing Paths
    CMD_VEL = f'{NAMESPACE}/cmd_vel'
    ODOMETRY = f'{NAMESPACE}/odom'
    SCAN = f'{NAMESPACE}/scan'
    OCCUPANCY_GRID = f'{NAMESPACE}/map'
    PATH = f'{NAMESPACE}/path'
    DYNAMIC_GRID = f'{NAMESPACE}/dynamic_grid'
    GOAL_POSE = f'{NAMESPACE}/goal_pose'
    GRID_MARKERS = f'{NAMESPACE}/grid_markers'




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
        """
        Euclidean distance heuristic.

        Args:
            s1: State (x, y)
            s2: State (x, y)

        Returns:
            Euclidean distance between s1 and s2
        """
        return math.sqrt((s1[0] - s2[0])**2 + (s1[1] - s2[1])**2)

    def calculate_key(self, s):
        """
        Calculate priority key for state s.

        The key is used to determine processing order in the priority queue.
        States with lower keys are processed first.

        Args:
            s: State (x, y)

        Returns:
            Tuple (k1, k2) where k1 is primary key, k2 is tiebreaker
        """
        g_rhs = min(self.g[s], self.rhs[s])
        return (g_rhs + self.heuristic(self.start, s) + self.k_m, g_rhs)

    def is_valid(self, pos):
        """
        Check if position is valid (in bounds and not an obstacle).

        Args:
            pos: Position (x, y)

        Returns:
            True if valid and free, False otherwise
        """
        x, y = pos
        if x < 0 or x >= self.cols or y < 0 or y >= self.rows:
            return False
        return self.grid[y, x] == 0

    def get_neighbors(self, s):
        """
        Get valid 8-connected neighbors of state s.

        Args:
            s: State (x, y)

        Returns:
            List of valid neighbor states
        """
        neighbors = []
        # 8-connected grid: 4 cardinal + 4 diagonal directions
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
            neighbor = (s[0] + dx, s[1] + dy)
            if self.is_valid(neighbor):
                neighbors.append(neighbor)
        return neighbors

    def cost(self, s1, s2):
        """
        Calculate movement cost between adjacent states.

        Args:
            s1: State (x, y)
            s2: Adjacent state (x, y)

        Returns:
            Movement cost (1.0 for cardinal, 1.414 for diagonal)
        """
        if not self.is_valid(s1) or not self.is_valid(s2):
            return float('inf')

        dx = abs(s1[0] - s2[0])
        dy = abs(s1[1] - s2[1])
        return 1.414 if (dx + dy) == 2 else 1.0  # sqrt(2) for diagonal

    def update_vertex(self, u):
        """
        Update state u's cost values and priority queue status.

        This is the core of D* Lite. Makes u locally consistent if possible,
        otherwise adds it to the priority queue for further processing.

        Args:
            u: State (x, y) to update
        """
        # Update rhs value (one-step lookahead)
        if u != self.goal:
            min_cost = float('inf')
            for s_prime in self.get_neighbors(u):
                cost = self.cost(u, s_prime) + self.g[s_prime]
                min_cost = min(min_cost, cost)
            self.rhs[u] = min_cost

        # Remove u from open list if present
        self.open_list = [(k, s) for k, s in self.open_list if s != u]
        heapq.heapify(self.open_list)

        # If u is locally inconsistent, add to open list
        if self.g[u] != self.rhs[u]:
            heapq.heappush(self.open_list, (self.calculate_key(u), u))

    def compute_shortest_path(self):
        """
        Main D* Lite computation - processes states until start is consistent.

        This implements the core D* Lite algorithm, processing states from the
        priority queue until the start state has a consistent cost estimate.

        Returns:
            True if path found, False if timeout or no path
        """
        iterations = 0

        # Process until start is locally consistent and optimal
        while (self.open_list and
               (self.compare_keys(self.open_list[0][0], self.calculate_key(self.start)) or
                self.rhs[self.start] != self.g[self.start])):

            iterations += 1
            if iterations > PlannerConstants.MAX_PATH_ITERATIONS:
                self.get_logger().info('Max path iterations timeout')
                return False  # Timeout
            

            k_old, u = heapq.heappop(self.open_list)
            k_new = self.calculate_key(u)

            if self.compare_keys(k_old, k_new):
                # Key has changed, re-insert with new key
                heapq.heappush(self.open_list, (k_new, u))
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
        """
        Compare two priority keys.

        Args:
            k1: Key tuple (k1_1, k1_2)
            k2: Key tuple (k2_1, k2_2)

        Returns:
            True if k1 < k2
        """
        return k1[0] < k2[0] or (k1[0] == k2[0] and k1[1] < k2[1])

    def extract_path(self, debug=False):
        """
        Extract path from start to goal by following gradient of g-values.

        Starting from start, greedily moves to neighbor with lowest g-value
        until reaching the goal.

        Returns:
            List of states [(x, y), ...] from start to goal, or [] if no path
        """
        if self.g[self.start] == float('inf'):
            if debug:
                print(f'[DEBUG] extract_path failed: g[start] = inf')
            return []  # No path exists

        path = [self.start]
        current = self.start
        visited = set([self.start])
        max_iterations = self.rows * self.cols

        while current != self.goal:
            if len(path) > max_iterations:
                if debug:
                    print(f'[DEBUG] extract_path failed: loop detected at {current}')
                return []  # Loop detected

            neighbors = self.get_neighbors(current)
            if not neighbors:
                if debug:
                    print(f'[DEBUG] extract_path failed: dead end at {current}, no valid neighbors')
                return []  # Dead end

            # Find valid neighbors not yet visited
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
                return []  # No valid moves

            # Move to neighbor with lowest g-value
            next_state, min_g = min(valid_neighbors, key=lambda x: x[1])

            # Safety: don't go uphill (except for floating point errors)
            if min_g > self.g[current] + 0.01 and next_state != self.goal:
                if debug:
                    print(f'[DEBUG] extract_path failed: going uphill from {current} (g={self.g[current]}) to {next_state} (g={min_g})')
                return []  # Path is not optimal

            visited.add(next_state)
            path.append(next_state)
            current = next_state

        return path

    def update_start(self, new_start):
        """
        Update start position for replanning.

        When the robot moves, update k_m to account for heuristic changes.
        This is crucial for D* Lite's incremental search efficiency.

        Args:
            new_start: New start position (x, y)
        """
        if new_start == self.start:
            return

        # Update k_m to maintain heuristic consistency
        self.k_m += self.heuristic(self.s_last, self.start)
        self.s_last = self.start
        self.start = new_start

    def update_obstacles(self, changed_cells):
        """
        Update grid with changed obstacles and trigger replanning.

        This is the key to D* Lite's efficiency - only cells near changes
        are updated, not the entire grid.

        Args:
            changed_cells: List of (x, y) cells that changed occupancy
        """
        all_affected = set()

        # Collect all affected cells (changed cells + their neighbors)
        for cell in changed_cells:
            x, y = cell
            if 0 <= x < self.cols and 0 <= y < self.rows:
                all_affected.add(cell)

                # Add all 8-connected neighbors
                for dx, dy in [(-1,0), (1,0), (0,-1), (0,1),
                               (-1,-1), (-1,1), (1,-1), (1,1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.cols and 0 <= ny < self.rows:
                        all_affected.add((nx, ny))

        # Update all affected vertices
        for vertex in all_affected:
            self.update_vertex(vertex)


# ============================================================================
# ROS2 NAVIGATOR NODE
# ============================================================================

class DStarNavigator(Node):
    """
    Complete navigation system using D* Lite with SLAM and lidar integration.

    This node manages:
    - Path planning with D* Lite
    - SLAM map integration for persistent obstacles
    - Lidar-based dynamic obstacle detection
    - Robot motion control
    - Automatic replanning when obstacles block the path

    Operating Modes:
        OPTIMISTIC (default): Treats unexplored areas as free - good for exploration
        CONSERVATIVE: Only plans through known free space - good for navigation

    Grid Hierarchy:
        1. grid_original: Raw SLAM map (0=free, 1=occupied, no inflation)
        2. grid: SLAM map with safety inflation (planning baseline)
        3. grid_base: Accumulated SLAM obstacles (persistent)
        4. grid_dynamic: grid_base + lidar obstacles (current planning grid)

    Replanning Triggers:
        - SLAM discovers obstacles blocking current path
        - Lidar detects new obstacles near path (not already known)
        - Cooldown prevents rapid replanning (1 second default)
    """

    # ========================================================================
    # INITIALIZATION
    # ========================================================================

    def __init__(self, robot_radius=0.22, safety_clearance=0.15, optimistic_planning=True):
        """
        Initialize the D* Lite navigator.

        Args:
            robot_radius: Physical radius of robot in meters
            safety_clearance: Extra safety buffer around obstacles in meters
            optimistic_planning: If True, treat unknown areas as free
        """'/scan'
        super().__init__('dstar_navigator')

        # Robot configuration
        self.robot_radius = robot_radius
        self.safety_clearance = safety_clearance
        self.optimistic_planning = optimistic_planning

        # ROS2 publishers
        self.cmd_vel_pub = self.create_publisher(TwistStamped, PlannerConstants.CMD_VEL, 10)
        self.path_pub = self.create_publisher(Path, PlannerConstants.PATH, 10)
        self.dynamic_grid_pub = self.create_publisher(OccupancyGrid, PlannerConstants.DYNAMIC_GRID, 10)
        self.grid_markers_pub = self.create_publisher(MarkerArray, PlannerConstants.GRID_MARKERS, 10)

        # ROS2 subscribers
        self._setup_subscribers()

        # Map data - Grid hierarchy (see class docstring)
        self.grid = None              # Inflated SLAM map
        self.grid_original = None     # Original uninflated SLAM map
        self.grid_base = None         # SLAM obstacles (persistent)
        self.grid_dynamic = None      # grid_base + lidar obstacles (current)
        self.grid_slam = None         # SLAM map for comparison
        self.grid_unknown = None      # Track unexplored cells
        self.resolution = None        # Meters per grid cell
        self.origin = None            # Map origin [x, y] in world coordinates
        self.map_received = False
        self.use_live_slam = True

        # Robot state
        self.current_pose = None      # {'x': float, 'y': float, 'theta': float}
        self.path = []                # List of waypoints [(x, y), ...]
        self.current_waypoint_idx = 0
        self.latest_scan = None

        # D* Lite planner state
        self.dstar_planner = None
        self.goal_grid = None
        self.planner_grid_snapshot = None  # Snapshot for change detection
        self.known_obstacles = set()       # Track obstacles already replanned for
        self.lidar_obstacles_along_path = []  # Lidar detections that are along the path (for visualization)

        # Control parameters
        self.linear_speed = 0.2
        self.angular_speed = 0.4       # Reduced for smoother turns
        self.position_tolerance = 0.25  # Increased - don't need to hit waypoints exactly
        self.angle_tolerance = 0.3      # Increased - allow more angle error before slowing
        self.replanning_needed = False
        self.last_replan_time = self.get_clock().now()
        self.blocked_waypoint_idx = None  # Track which waypoint is blocked by obstacle

        # Control loop timer (10 Hz)
        self.control_timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info('Using SLAM-only mode')
        self.get_logger().info('Grid will be initialized from first SLAM message')

        self._log_initialization()

    def _setup_subscribers(self):
        """Setup ROS2 subscribers with appropriate QoS profiles."""
        # Best-effort QoS for real-time data (odometry, lidar)
        qos_best_effort = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.odom_sub = self.create_subscription(
            Odometry, PlannerConstants.ODOMETRY, self.odom_callback, qos_best_effort
        )

        self.scan_sub = self.create_subscription(
            LaserScan, PlannerConstants.SCAN, self.scan_callback, 10
        )

        # Reliable + transient-local QoS for maps (latched topic)
        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.map_sub = self.create_subscription(
            OccupancyGrid, PlannerConstants.OCCUPANCY_GRID, self.map_callback, map_qos
        )

        # Goal pose subscriber for receiving navigation goals
        self.goal_pose_sub = self.create_subscription(
            PoseStamped, PlannerConstants.GOAL_POSE, self.goal_pose_callback, 10
        )

    def goal_pose_callback(self, msg: PoseStamped):
        """
        Handle incoming goal pose messages.

        Extracts x, y coordinates from the PoseStamped message and
        initiates navigation from current position to the goal.

        Args:
            msg: PoseStamped message containing the goal position
        """
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

        self.get_logger().info('MAP MODE: SLAM-only')
        self.get_logger().info('  -> Unknown areas treated as FREE for exploration')

        if self.use_live_slam:
            self.get_logger().info('Using LIVE SLAM map from /map topic')
            mode = 'OPTIMISTIC' if self.optimistic_planning else 'CONSERVATIVE'
            self.get_logger().info(f'{mode} PLANNING enabled')

    # ========================================================================
    # MAP PROCESSING
    # ========================================================================

    def _inflate_obstacles(self, grid, inflation_radius):
        """
        Inflate obstacles using morphological dilation.

        Creates a safety buffer around all obstacles by expanding them.

        Args:
            grid: 2D numpy array (0=free, 1=occupied)
            inflation_radius: Inflation radius in meters

        Returns:
            Inflated grid with same shape as input
        """
        radius_pixels = int(inflation_radius / self.resolution)
        kernel_size = 2 * radius_pixels + 1
        kernel = np.ones((kernel_size, kernel_size))
        return binary_dilation(grid, kernel).astype(int)

    # ========================================================================
    # ROS2 CALLBACKS
    # ========================================================================

    def odom_callback(self, msg):
        """
        Update current robot pose from odometry.

        Args:
            msg: Odometry message
        """
        self.current_pose = {
            'x': msg.pose.pose.position.x,
            'y': msg.pose.pose.position.y,
            'theta': self.quaternion_to_yaw(msg.pose.pose.orientation)
        }

    def scan_callback(self, msg):
        """
        Process lidar scan and check for obstacles in path.

        Detects obstacles using lidar and triggers replanning if they block
        the current path (with cooldown to prevent rapid replanning).

        Args:
            msg: LaserScan message
        """
        self.latest_scan = msg

        if not self.path or self.current_pose is None:
            return

        # Check if obstacles block the path
        obstacles_detected = self.detect_obstacles_in_path(msg)

        if obstacles_detected:
            current_time = self.get_clock().now()
            time_since_last_replan = (current_time - self.last_replan_time).nanoseconds / 1e9

            # Respect cooldown period
            if time_since_last_replan > PlannerConstants.REPLAN_COOLDOWN:
                self.get_logger().warn(f'Obstacle in path ahead! Triggering replanning...')
                self.replanning_needed = True
                self.last_replan_time = current_time

    def map_callback(self, msg: OccupancyGrid):
        """
        Process live SLAM map updates.

        Integrates SLAM-discovered obstacles into the planning grid and
        triggers replanning if they block the current path.

        Initializes and maintains the entire grid from SLAM data,
        treating unknown areas as free.

        Args:
            msg: OccupancyGrid message from SLAM
        """
        if not self.use_live_slam:
            return

        is_first_map = not self.map_received

        # Extract SLAM map data
        width, height = msg.info.width, msg.info.height
        slam_resolution = msg.info.resolution
        slam_origin = [msg.info.origin.position.x, msg.info.origin.position.y]

        occupancy_data = np.array(msg.data, dtype=np.int8).reshape((height, width))
        unknown_mask = (occupancy_data == -1)

        # Convert to binary (0=free, 1=occupied)
        # Unknown (-1) treated as FREE for optimistic exploration
        slam_grid = np.zeros((height, width), dtype=np.int8)
        slam_grid[occupancy_data >= 50] = 1  # Occupied if >50% probability

        # Initialize grids from SLAM on first message
        if self.grid_original is None:
            self._initialize_grids_from_slam(height, width, slam_resolution, slam_origin)
            self.get_logger().info(
                f'Initialized grid from SLAM: {width}x{height}, '
                f'resolution={slam_resolution}m, origin=({slam_origin[0]:.2f}, {slam_origin[1]:.2f})'
            )

        # Skip if grids not initialized yet
        if self.grid_original is None:
            return

        # Track unknowns for optimistic planning
        previous_unknown = self.grid_unknown.copy() if self.grid_unknown is not None else None
        self.grid_slam = slam_grid.copy()
        self.grid_unknown = np.zeros_like(self.grid_original, dtype=np.int8)

        # Use SLAM obstacles directly (mapped to our grid)
        current_slam_obstacles, newly_discovered = self._map_slam_to_grid(
            slam_grid, unknown_mask, slam_resolution, slam_origin,
            height, width, previous_unknown
        )

        # Update grid_base to reflect current SLAM state (add AND remove)
        self._update_dynamic_grid_with_slam(current_slam_obstacles)

        # Check if replanning needed
        if newly_discovered > 0 and self.optimistic_planning and self.path:
            if self._check_path_blocked_by_obstacles(current_slam_obstacles):
                # Clear known obstacles - new SLAM discoveries
                self.known_obstacles.clear()
                self._log_replanning_trigger('Newly discovered SLAM obstacles', newly_discovered)
                self.replanning_needed = True

        self.map_received = True

        if is_first_map:
            self.get_logger().info(f'SLAM map integrated: {current_slam_obstacles.sum()} obstacles detected')

    # ========================================================================
    # SLAM OBSTACLE DETECTION
    # ========================================================================

    def _initialize_grids_from_slam(self, height, width, resolution, origin):
        """
        Initialize all grids from SLAM dimensions (SLAM-only mode).

        Creates empty grids matching SLAM map size. All cells start as free,
        allowing pathfinding to unknown areas for exploration.

        Args:
            height, width: SLAM map dimensions in cells
            resolution: SLAM map resolution (meters per cell)
            origin: SLAM map origin [x, y] in world coordinates
        """
        self.resolution = resolution
        self.origin = origin

        # Initialize all grids as free (zeros)
        self.grid_original = np.zeros((height, width), dtype=np.int8)
        self.grid = np.zeros((height, width), dtype=np.int8)
        self.grid_base = np.zeros((height, width), dtype=np.int8)
        self.grid_dynamic = np.zeros((height, width), dtype=np.int8)
        self.grid_unknown = np.zeros((height, width), dtype=np.int8)

    def _map_slam_to_grid(self, slam_grid, unknown_mask, slam_resolution, slam_origin,
                          height, width, previous_unknown):
        """
        Map SLAM obstacles directly to our grid.

        Maps all SLAM obstacles to the planning grid. Unknown areas remain free
        to allow exploration pathfinding.

        Args:
            slam_grid: SLAM occupancy grid (0=free, 1=occupied)
            unknown_mask: Boolean mask of unknown cells
            slam_resolution: SLAM map resolution
            slam_origin: SLAM map origin [x, y]
            height, width: SLAM map dimensions
            previous_unknown: Previous unknown mask (for tracking discoveries)

        Returns:
            Tuple (obstacles_grid, newly_discovered_count)
        """
        obstacles_grid = np.zeros_like(self.grid_original, dtype=np.int8)
        newly_discovered_count = 0

        for slam_y in range(height):
            for slam_x in range(width):
                # Convert SLAM coordinates to our grid coordinates
                world_x = slam_x * slam_resolution + slam_origin[0]
                world_y = slam_y * slam_resolution + slam_origin[1]
                grid_x = int((world_x - self.origin[0]) / self.resolution)
                grid_y = int((world_y - self.origin[1]) / self.resolution)

                # Check bounds
                if not (0 <= grid_x < self.grid_original.shape[1] and
                        0 <= grid_y < self.grid_original.shape[0]):
                    continue

                # Track unknown cells (for visualization, but treat as free for planning)
                if unknown_mask[slam_y, slam_x]:
                    self.grid_unknown[grid_y, grid_x] = 1

                # Map SLAM obstacles directly
                if slam_grid[slam_y, slam_x] == 1:
                    obstacles_grid[grid_y, grid_x] = 1

                    # Check if newly discovered (was unknown before)
                    if previous_unknown is not None and previous_unknown[grid_y, grid_x] == 1:
                        newly_discovered_count += 1

        return obstacles_grid, newly_discovered_count

    def _update_dynamic_grid_with_slam(self, current_slam_obstacles):
        """
        Update grid_base to reflect current SLAM obstacle state.

        IMPORTANT: This SYNCS with SLAM state - both adding new obstacles
        AND removing obstacles that SLAM no longer reports. This ensures the
        grid reflects reality when obstacles move or disappear.

        - Starts with empty grid (all free)
        - Only SLAM obstacles are marked
        - Unknown areas remain FREE for exploration pathfinding

        Args:
            current_slam_obstacles: Binary grid of ALL current SLAM obstacles
        """
        # Store previous state to detect changes
        previous_grid_base = self.grid_base.copy() if self.grid_base is not None else None

        # Start with base grid (empty initially, allowing exploration)
        self.grid_base = self.grid.copy()

        # Add current SLAM obstacles (inflated)
        if current_slam_obstacles.sum() > 0:
            total_inflation = self.robot_radius + self.safety_clearance
            slam_obstacles_inflated = self._inflate_obstacles(current_slam_obstacles, total_inflation)

            # Merge SLAM obstacles with base
            self.grid_base = np.maximum(self.grid_base, slam_obstacles_inflated)

        # Update dynamic grid to match base (lidar will add on top)
        self.grid_dynamic = self.grid_base.copy()

        # Find changed cells (both additions AND removals)
        changed_cells = []
        if previous_grid_base is not None:
            rows, cols = self.grid_base.shape
            for y in range(rows):
                for x in range(cols):
                    if previous_grid_base[y, x] != self.grid_base[y, x]:
                        changed_cells.append((x, y))

        # Update D* Lite planner with all changes
        if self.dstar_planner is not None and changed_cells:
            self.get_logger().info(f'Syncing {len(changed_cells)} SLAM changes to D* planner (adds & removes)')
            self.dstar_planner.grid = self.grid_dynamic.copy()
            self.dstar_planner.update_obstacles(changed_cells)
            self.planner_grid_snapshot = self.grid_base.copy()

        # Publish updated dynamic grid for visualization
        self.publish_dynamic_grid()

    def _check_path_blocked_by_obstacles(self, obstacles_grid):
        """
        Check if obstacles block any waypoints in the current path.

        Args:
            obstacles_grid: Binary grid of obstacle positions

        Returns:
            True if path is blocked, False otherwise
        """
        if not self.path:
            return False

        for waypoint_idx, waypoint in enumerate(self.path):
            # Skip waypoints we've already passed
            if waypoint_idx < self.current_waypoint_idx:
                continue

            gx, gy = self.world_to_grid(waypoint[0], waypoint[1])

            if not (0 <= gx < obstacles_grid.shape[1] and 0 <= gy < obstacles_grid.shape[0]):
                continue

            # Check waypoint and surrounding cells
            for dy in range(-PlannerConstants.PATH_CHECK_RADIUS,
                          PlannerConstants.PATH_CHECK_RADIUS + 1):
                for dx in range(-PlannerConstants.PATH_CHECK_RADIUS,
                              PlannerConstants.PATH_CHECK_RADIUS + 1):
                    nx, ny = gx + dx, gy + dy
                    if (0 <= nx < obstacles_grid.shape[1] and
                        0 <= ny < obstacles_grid.shape[0] and
                        obstacles_grid[ny, nx] != 0):
                        # Store blocked waypoint index for partial path preservation
                        self.blocked_waypoint_idx = waypoint_idx
                        return True

        return False

    # ========================================================================
    # LIDAR OBSTACLE DETECTION
    # ========================================================================

    def detect_obstacles_in_path(self, scan_msg):
        """
        Detect obstacles using lidar and check if they block the path.

        Processing pipeline:
        1. Process lidar scan to get obstacles along the path
        2. Store for visualization
        3. Update grid_dynamic with lidar obstacles
        4. Check if obstacles block upcoming path segments

        Args:
            scan_msg: LaserScan message

        Returns:
            True if obstacles block the path, False otherwise
        """
        if self.current_waypoint_idx >= len(self.path):
            self.lidar_obstacles_along_path = []  # Clear when no path
            return False

        robot_grid = self.world_to_grid(self.current_pose['x'], self.current_pose['y'])
        robot_gx, robot_gy = robot_grid
        robot_theta = self.current_pose['theta']

        # Process lidar scan to detect obstacles along the path
        detected_obstacles = self._process_lidar_scan(scan_msg, robot_gx, robot_gy, robot_theta)

        # Store for visualization (these are specifically obstacles along the path)
        self.lidar_obstacles_along_path = detected_obstacles.copy()

        if not detected_obstacles:
            return False

        # Update grid_dynamic with detected obstacles at their actual positions
        self._update_dynamic_grid_with_lidar(detected_obstacles, robot_gx, robot_gy)

        # Check if obstacles block the planned path
        return self._check_obstacles_block_path(detected_obstacles, robot_gx, robot_gy, detected_obstacles)

    def _process_lidar_scan(self, scan_msg, robot_gx, robot_gy, robot_theta):
        """
        Process lidar scan using ray-marching to detect obstacles along the path.

        For each lidar ray, marches along the ray checking all grid cells to see if:
        1. The cell contains an obstacle in grid_dynamic (from SLAM or previous detection)
        2. The cell is along the planned path

        This detects SLAM obstacles that the lidar ray passes through, even if the
        lidar didn't physically hit them at that exact point.

        Args:
            scan_msg: LaserScan message
            robot_gx, robot_gy: Robot position in grid coordinates
            robot_theta: Robot orientation in radians

        Returns:
            List of obstacle positions [(gx, gy), ...] that are along the path
        """
        obstacles_along_path = []
        detected_set = set()  # Avoid duplicates

        # Get path segments to check against
        path_segments = self._get_upcoming_path_segments(robot_gx, robot_gy)
        if not path_segments:
            return []

        # Process each lidar reading with ray-marching
        angle = scan_msg.angle_min
        for range_val in scan_msg.ranges:
            # Get the ray length (use max range if invalid reading)
            if self._is_valid_lidar_reading(range_val, scan_msg):
                ray_length = range_val
            else:
                # Skip invalid readings entirely
                angle += scan_msg.angle_increment
                continue

            # Ray-march along this lidar ray
            ray_obstacles = self._ray_march_for_obstacles(
                robot_gx, robot_gy, robot_theta, angle, ray_length,
                path_segments, detected_set
            )

            for obs in ray_obstacles:
                if obs not in detected_set:
                    detected_set.add(obs)
                    obstacles_along_path.append(obs)

            angle += scan_msg.angle_increment

        # Debug logging
        if hasattr(self, '_lidar_debug_counter'):
            self._lidar_debug_counter += 1
        else:
            self._lidar_debug_counter = 0

        if self._lidar_debug_counter % 50 == 0 and obstacles_along_path:
            self.get_logger().debug(
                f'Ray-march detected {len(obstacles_along_path)} obstacles along path'
            )

        return obstacles_along_path

    def _ray_march_for_obstacles(self, robot_gx, robot_gy, robot_theta, angle, ray_length, path_segments, detected_set):
        """
        March along a lidar ray checking for obstacles in grid_dynamic that are along the path.

        Uses Bresenham-style stepping to check each grid cell along the ray.

        Args:
            robot_gx, robot_gy: Robot position in grid
            robot_theta: Robot orientation
            angle: Lidar ray angle (relative to robot)
            ray_length: Length of ray in meters
            path_segments: Path segments to check against
            detected_set: Set of already detected obstacles (to avoid duplicates)

        Returns:
            List of obstacle positions [(gx, gy), ...] found along this ray
        """
        obstacles_found = []

        # Calculate ray direction in world frame
        world_angle = robot_theta + angle

        # Step size in meters (use resolution for grid-aligned stepping)
        step_size = self.resolution * 0.5  # Half-cell steps for better coverage
        num_steps = int(ray_length / step_size)

        # March along the ray
        for step in range(1, num_steps + 1):  # Start at 1 to skip robot position
            distance = step * step_size

            # Calculate world position along ray
            world_x = self.current_pose['x'] + distance * math.cos(world_angle)
            world_y = self.current_pose['y'] + distance * math.sin(world_angle)

            # Convert to grid
            gx, gy = self.world_to_grid(world_x, world_y)

            # Check bounds
            if not (0 <= gx < self.grid_dynamic.shape[1] and
                    0 <= gy < self.grid_dynamic.shape[0]):
                continue

            # Skip cells too close to robot (robot body)
            dist_from_robot = math.sqrt((gx - robot_gx)**2 + (gy - robot_gy)**2)
            if dist_from_robot < PlannerConstants.ROBOT_BODY_RADIUS_CELLS:
                continue

            # Skip known SLAM obstacles
            if self._is_near_known_obstacle(gx, gy, PlannerConstants.KNOWN_OBSTACLE_TOLERANCE):
                continue

            grid_pos = (gx, gy)

            # Check if this cell has an obstacle in dynamic grid AND is along the path
            if grid_pos not in detected_set:
                if self.grid_dynamic[gy, gx] != 0 and self._is_point_along_path(gx, gy, path_segments):
                    obstacles_found.append(grid_pos)
                    # Once we find an obstacle along this ray, we can stop
                    # (the ray would be blocked by it)
                    break

        # Also check the endpoint (where lidar actually hit)
        end_world = self._lidar_to_world(
            ray_length, angle, self.current_pose['x'],
            self.current_pose['y'], robot_theta
        )
        end_gx, end_gy = self.world_to_grid(end_world[0], end_world[1])

        if (0 <= end_gx < self.grid_dynamic.shape[1] and
            0 <= end_gy < self.grid_dynamic.shape[0]):
            dist_from_robot = math.sqrt((end_gx - robot_gx)**2 + (end_gy - robot_gy)**2)
            end_pos = (end_gx, end_gy)

            if (dist_from_robot >= PlannerConstants.ROBOT_BODY_RADIUS_CELLS and
                not self._is_near_known_obstacle(end_gx, end_gy, PlannerConstants.KNOWN_OBSTACLE_TOLERANCE) and
                end_pos not in detected_set and
                self._is_point_along_path(end_gx, end_gy, path_segments)):
                obstacles_found.append(end_pos)

        return obstacles_found

    def _get_upcoming_path_segments(self, robot_gx, robot_gy):
        """
        Get the upcoming path segments to check for obstacles.

        Returns:
            List of segment tuples: [((start_gx, start_gy), (end_gx, end_gy)), ...]
        """
        if not self.path or self.current_waypoint_idx >= len(self.path):
            return []

        segments = []
        num_segments = min(PlannerConstants.LOOKAHEAD_SEGMENTS,
                          len(self.path) - self.current_waypoint_idx)

        for i in range(self.current_waypoint_idx, self.current_waypoint_idx + num_segments):
            if i >= len(self.path):
                break

            # Get segment start
            if i == self.current_waypoint_idx:
                start_gx, start_gy = robot_gx, robot_gy
            else:
                prev_wp = self.path[i - 1]
                start_gx, start_gy = self.world_to_grid(prev_wp[0], prev_wp[1])

            # Get segment end
            end_wp = self.path[i]
            end_gx, end_gy = self.world_to_grid(end_wp[0], end_wp[1])

            segments.append(((start_gx, start_gy), (end_gx, end_gy)))

        return segments

    def _is_point_along_path(self, gx, gy, path_segments):
        """
        Check if a point is along any of the path segments.

        Uses the robot footprint as the threshold for "along the path".

        Args:
            gx, gy: Point to check
            path_segments: List of ((start_gx, start_gy), (end_gx, end_gy)) tuples

        Returns:
            True if point is within blocking threshold of any segment
        """
        # Use robot footprint as threshold
        blocking_threshold = int((self.robot_radius + self.safety_clearance) / self.resolution)

        for (start_gx, start_gy), (end_gx, end_gy) in path_segments:
            dist = self._point_to_segment_distance(gx, gy, start_gx, start_gy, end_gx, end_gy)
            if dist <= blocking_threshold:
                return True

        return False

    def _is_valid_lidar_reading(self, range_val, scan_msg):
        """
        Check if lidar reading is valid.

        Args:
            range_val: Range value in meters
            scan_msg: LaserScan message (for min/max range)

        Returns:
            True if valid reading, False otherwise
        """
        return (range_val >= scan_msg.range_min and
                range_val <= scan_msg.range_max and
                not math.isinf(range_val) and
                PlannerConstants.LIDAR_MIN_RANGE <= range_val <= PlannerConstants.LIDAR_MAX_RANGE)

    def _lidar_to_world(self, range_val, angle, robot_x, robot_y, robot_theta):
        """
        Convert lidar reading to world coordinates.

        Transforms from robot-local polar coordinates to world Cartesian.

        Args:
            range_val: Range in meters
            angle: Angle in radians (relative to robot)
            robot_x, robot_y: Robot position in world
            robot_theta: Robot orientation in world

        Returns:
            Tuple (world_x, world_y)
        """
        # Convert polar to Cartesian in robot frame
        obstacle_x_local = range_val * math.cos(angle)
        obstacle_y_local = range_val * math.sin(angle)

        # Rotate and translate to world frame
        obstacle_x_world = (robot_x +
                           obstacle_x_local * math.cos(robot_theta) -
                           obstacle_y_local * math.sin(robot_theta))
        obstacle_y_world = (robot_y +
                           obstacle_x_local * math.sin(robot_theta) +
                           obstacle_y_local * math.cos(robot_theta))

        return (obstacle_x_world, obstacle_y_world)

    def _is_valid_obstacle(self, grid_pos, robot_gx, robot_gy):
        """
        Check if detected obstacle is valid (not noise/known/robot body).

        Filtering logic:
        1. Check bounds
        2. Filter robot body (too close)
        3. Filter known SLAM obstacles (already in map)
        4. Special case: Free corridor walls need SLAM confirmation

        Args:
            grid_pos: Obstacle position (gx, gy)
            robot_gx, robot_gy: Robot position in grid

        Returns:
            True if valid obstacle, False if filtered out
        """
        gx, gy = grid_pos

        # Check bounds
        if not (0 <= gx < self.grid_original.shape[1] and
                0 <= gy < self.grid_original.shape[0]):
            return False

        # Filter robot body detections
        dist_from_robot = math.sqrt((gx - robot_gx)**2 + (gy - robot_gy)**2)
        if dist_from_robot < PlannerConstants.ROBOT_BODY_RADIUS_CELLS:
            return False

        # Filter known SLAM obstacles (already in map)
        if self._is_near_known_obstacle(gx, gy, PlannerConstants.KNOWN_OBSTACLE_TOLERANCE):
            return False

        # Check if completely free in all grids (likely corridor wall)
        is_completely_free = (self.grid_original[gy, gx] == 0 and
                             self.grid[gy, gx] == 0 and
                             self.grid_dynamic[gy, gx] == 0)

        if is_completely_free:
            # Only keep if confirmed by SLAM or very close to robot
            if (self.grid_slam is not None and
                0 <= gx < self.grid_slam.shape[1] and
                0 <= gy < self.grid_slam.shape[0] and
                self.grid_slam[gy, gx] != 0):
                return True
            return dist_from_robot <= PlannerConstants.CLOSE_OBSTACLE_CELLS

        return True

    def _debug_obstacle_filter(self, grid_pos, robot_gx, robot_gy, range_val, angle):
        """
        Debug helper to log why a front-facing obstacle was filtered.

        Only called for close front obstacles that were rejected.
        """
        gx, gy = grid_pos
        dist_from_robot = math.sqrt((gx - robot_gx)**2 + (gy - robot_gy)**2)

        reasons = []

        # Check bounds
        if not (0 <= gx < self.grid_original.shape[1] and
                0 <= gy < self.grid_original.shape[0]):
            reasons.append('OUT_OF_BOUNDS')
        else:
            # Check robot body filter
            if dist_from_robot < PlannerConstants.ROBOT_BODY_RADIUS_CELLS:
                reasons.append(f'ROBOT_BODY(dist={dist_from_robot:.1f}<{PlannerConstants.ROBOT_BODY_RADIUS_CELLS})')

            # Check known obstacle filter
            if self._is_near_known_obstacle(gx, gy, PlannerConstants.KNOWN_OBSTACLE_TOLERANCE):
                reasons.append(f'NEAR_KNOWN(tol={PlannerConstants.KNOWN_OBSTACLE_TOLERANCE})')

            # Check completely free filter
            if (0 <= gx < self.grid_original.shape[1] and 0 <= gy < self.grid_original.shape[0]):
                is_completely_free = (self.grid_original[gy, gx] == 0 and
                                     self.grid[gy, gx] == 0 and
                                     self.grid_dynamic[gy, gx] == 0)
                if is_completely_free:
                    slam_confirmed = (self.grid_slam is not None and
                                    0 <= gx < self.grid_slam.shape[1] and
                                    0 <= gy < self.grid_slam.shape[0] and
                                    self.grid_slam[gy, gx] != 0)
                    if not slam_confirmed and dist_from_robot > PlannerConstants.CLOSE_OBSTACLE_CELLS:
                        reasons.append(f'FREE_NO_SLAM(dist={dist_from_robot:.1f}>{PlannerConstants.CLOSE_OBSTACLE_CELLS})')

        if reasons:
            angle_deg = math.degrees(angle)
            # self.get_logger().warn(
            #     f'FILTERED front obstacle: pos=({gx},{gy}), range={range_val:.2f}m, '
            #     f'angle={angle_deg:.1f}deg, reasons={reasons}'
            # )

    def _is_near_known_obstacle(self, gx, gy, tolerance):
        """
        Check if position is near a known SLAM obstacle.

        Used to filter out lidar detections that are just seeing obstacles
        already known from SLAM.

        Args:
            gx, gy: Position to check
            tolerance: Search radius in cells

        Returns:
            True if near known obstacle, False otherwise
        """
        rows, cols = self.grid_original.shape

        if not (0 <= gx < cols and 0 <= gy < rows):
            return False

        # Check neighborhood
        for dx in range(-tolerance, tolerance + 1):
            for dy in range(-tolerance, tolerance + 1):
                check_x, check_y = gx + dx, gy + dy
                if (0 <= check_x < cols and 0 <= check_y < rows and
                    self.grid_original[check_y, check_x] != 0):
                    return True

        return False

    def _map_obstacles_to_neighbors(self, detected_obstacles, robot_gx, robot_gy):
        """
        Map detected obstacles to robot's 8-connected neighbor cells.

        For marking in the grid, obstacles are mapped to the robot's immediate
        neighbors for safety and control purposes.

        Args:
            detected_obstacles: List of obstacle positions [(gx, gy), ...]
            robot_gx, robot_gy: Robot position

        Returns:
            List of neighbor cells [(gx, gy), ...] with obstacles
        """
        # Get robot's 8-connected neighbors
        neighbors = []
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = robot_gx + dx, robot_gy + dy
                if 0 <= nx < self.grid.shape[1] and 0 <= ny < self.grid.shape[0]:
                    neighbors.append((nx, ny))

        # Map obstacles to nearest neighbors
        obst_neighbors = []
        for obs_gx, obs_gy in detected_obstacles:
            dist_to_robot = math.sqrt((obs_gx - robot_gx)**2 + (obs_gy - robot_gy)**2)

            # Find closest neighbor to this obstacle
            min_dist = float('inf')
            closest_neighbor = None
            for neighbor in neighbors:
                dist = (neighbor[0] - obs_gx)**2 + (neighbor[1] - obs_gy)**2
                if dist < min_dist:
                    min_dist = dist
                    closest_neighbor = neighbor

            # Only mark if close enough (emergency threshold)
            if closest_neighbor and dist_to_robot <= PlannerConstants.EMERGENCY_OBSTACLE_CELLS:
                if closest_neighbor not in obst_neighbors:
                    obst_neighbors.append(closest_neighbor)

        return obst_neighbors

    def _update_dynamic_grid_with_lidar(self, detected_obstacles, robot_gx, robot_gy):
        """
        Update grid_dynamic with lidar-detected obstacles at their actual positions.

        Starts from grid_base (persistent SLAM obstacles) and adds
        lidar obstacles on top. Also updates D* Lite planner's graph
        to keep it in sync with all detected obstacles.

        Args:
            detected_obstacles: List of obstacle positions [(gx, gy), ...] at actual locations
            robot_gx, robot_gy: Robot position
        """
        # Store previous grid state to detect changes
        previous_grid = self.grid_dynamic.copy() if self.grid_dynamic is not None else None

        # Start from base grid (preserves SLAM obstacles)
        self.grid_dynamic = self.grid_base.copy() if self.grid_base is not None else self.grid.copy()

        # Use reduced inflation for dynamic obstacles
        inflation_cells = max(3, int((self.robot_radius + self.safety_clearance) /
                                    self.resolution) // PlannerConstants.INFLATION_DIVISOR)

        # Track cells that changed for D* update
        changed_cells = []

        # Inflate and mark obstacles at their actual detected positions
        for gx, gy in detected_obstacles:
            for dy in range(-inflation_cells, inflation_cells + 1):
                for dx in range(-inflation_cells, inflation_cells + 1):
                    inflate_x, inflate_y = gx + dx, gy + dy
                    if (0 <= inflate_x < self.grid_dynamic.shape[1] and
                        0 <= inflate_y < self.grid_dynamic.shape[0] and
                        self.grid[inflate_y, inflate_x] == 0):  # Don't overwrite SLAM obstacles
                        # Track if this is a new obstacle
                        if self.grid_dynamic[inflate_y, inflate_x] == 0:
                            changed_cells.append((inflate_x, inflate_y))
                        self.grid_dynamic[inflate_y, inflate_x] = 1

        # Keep robot's immediate position free
        for dy in range(-PlannerConstants.ROBOT_CLEARANCE_CELLS,
                       PlannerConstants.ROBOT_CLEARANCE_CELLS + 1):
            for dx in range(-PlannerConstants.ROBOT_CLEARANCE_CELLS,
                          PlannerConstants.ROBOT_CLEARANCE_CELLS + 1):
                free_x, free_y = robot_gx + dx, robot_gy + dy
                if (0 <= free_x < self.grid_dynamic.shape[1] and
                    0 <= free_y < self.grid_dynamic.shape[0] and
                    self.grid[free_y, free_x] == 0):  # Don't clear SLAM obstacles
                    self.grid_dynamic[free_y, free_x] = 0

        # Update D* Lite planner with lidar changes to keep graph in sync
        if self.dstar_planner is not None and changed_cells:
            self.dstar_planner.grid = self.grid_dynamic.copy()
            self.dstar_planner.update_obstacles(changed_cells)

        # Publish updated dynamic grid for visualization
        self.publish_dynamic_grid()

    def _check_obstacles_block_path(self, detected_obstacles, robot_gx, robot_gy, _unused=None):
        """
        Check if detected obstacles block the planned path.

        Key features:
        - Only checks obstacles within MAX_OBSTACLE_CHECK_DISTANCE
        - Skips obstacles we've already replanned for (prevents loops)
        - Only checks next few path segments (not entire path)
        - Uses robot footprint as blocking threshold

        Args:
            detected_obstacles: Actual obstacle positions [(gx, gy), ...]
            robot_gx, robot_gy: Robot position
            _unused: Deprecated parameter, kept for compatibility

        Returns:
            True if obstacles block path, False otherwise
        """
        # Use robot's actual footprint as threshold
        robot_footprint_cells = int((self.robot_radius + self.safety_clearance) / self.resolution)
        blocking_threshold = robot_footprint_cells

        max_check_distance = PlannerConstants.MAX_OBSTACLE_CHECK_DISTANCE

        # Check actual detected obstacles, not neighbor cells
        for obs_gx, obs_gy in detected_obstacles:
            # Filter distant obstacles
            dist_from_robot = math.sqrt((obs_gx - robot_gx)**2 + (obs_gy - robot_gy)**2)
            if dist_from_robot > max_check_distance:
                continue

            # CRITICAL: Skip obstacles we've already replanned for
            # This prevents replanning loops
            if (obs_gx, obs_gy) in self.known_obstacles:
                continue

            # Check if in planner's grid snapshot (already considered)
            if (self.planner_grid_snapshot is not None and
                0 <= obs_gx < self.planner_grid_snapshot.shape[1] and
                0 <= obs_gy < self.planner_grid_snapshot.shape[0] and
                self.planner_grid_snapshot[obs_gy, obs_gx] != 0):
                # Already considered - add to known set
                self.known_obstacles.add((obs_gx, obs_gy))
                continue

            # Check if obstacle blocks next few path segments
            for i in range(self.current_waypoint_idx,
                          min(self.current_waypoint_idx + 3, len(self.path))):

                # Get segment endpoints
                if i == self.current_waypoint_idx:
                    start_gx, start_gy = robot_gx, robot_gy
                else:
                    prev_wp = self.path[i-1]
                    start_gx, start_gy = self.world_to_grid(prev_wp[0], prev_wp[1])

                end_wp = self.path[i]
                end_gx, end_gy = self.world_to_grid(end_wp[0], end_wp[1])

                # Calculate distance from obstacle to path segment
                dist_to_segment = self._point_to_segment_distance(
                    obs_gx, obs_gy, start_gx, start_gy, end_gx, end_gy
                )

                # Check if obstacle blocks path
                if dist_to_segment <= blocking_threshold:
                    # Add to known obstacles to prevent re-triggering
                    self.known_obstacles.add((obs_gx, obs_gy))
                    self._log_obstacle_detection(obs_gx, obs_gy, dist_to_segment,
                                                 blocking_threshold, len(detected_obstacles))
                    # Store which waypoint is blocked for partial path preservation
                    self.blocked_waypoint_idx = i
                    return True

        return False

    def _point_to_segment_distance(self, px, py, x1, y1, x2, y2):
        """
        Calculate minimum distance from point to line segment.

        Uses projection to find closest point on segment, then Euclidean distance.

        Args:
            px, py: Point coordinates
            x1, y1: Segment start
            x2, y2: Segment end

        Returns:
            Minimum distance in cells
        """
        dx = x2 - x1
        dy = y2 - y1

        # Degenerate case: segment is a point
        if dx == 0 and dy == 0:
            return math.sqrt((px - x1)**2 + (py - y1)**2)

        # Project point onto line, clamped to segment
        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))

        # Find closest point on segment
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy

        return math.sqrt((px - closest_x)**2 + (py - closest_y)**2)

    # ========================================================================
    # PATH PLANNING
    # ========================================================================

    def navigate_to_goal(self, start_x, start_y, goal_x, goal_y):
        """
        Navigate from start to goal position.

        This is the main entry point for navigation. Plans a path using D* Lite
        and begins following it.

        Args:
            start_x, start_y: Start position in meters (world coordinates)
            goal_x, goal_y: Goal position in meters (world coordinates)

        Returns:
            True if path found and navigation started, False otherwise
        """
        self.get_logger().info(f'Planning path from ({start_x:.2f}, {start_y:.2f}) to ({goal_x:.2f}, {goal_y:.2f})')

        # Clear known obstacles for fresh navigation
        self.known_obstacles.clear()

        self.path = self.plan_path(start_x, start_y, goal_x, goal_y)

        if self.path:
            self.current_waypoint_idx = 0
            self.get_logger().info(f'Path found with {len(self.path)} waypoints')
            self.publish_path()
            return True
        else:
            self.get_logger().error('No path found!')
            return False

    def plan_path(self, start_x, start_y, goal_x, goal_y):
        """
        Plan path using D* Lite algorithm.

        Converts world coordinates to grid, validates positions, runs D* Lite,
        and converts result back to world coordinates.

        Args:
            start_x, start_y: Start in world coordinates
            goal_x, goal_y: Goal in world coordinates

        Returns:
            List of waypoints [(x, y), ...] in world coordinates, or [] if no path
        """
        if self.grid is None or self.grid_original is None:
            self.get_logger().error('No map available!')
            return []

        # Convert to grid coordinates
        start_grid = self.world_to_grid(start_x, start_y)
        goal_grid = self.world_to_grid(goal_x, goal_y)

        # Validate start position (must be in free space in original grid)
        if not self._is_valid_in_original_grid(start_grid):
            self.get_logger().error(f'Start position is invalid!')
            return []

        # Validate goal position (must be in free space in inflated grid)
        # NOTE: This could be changed to allow to robot to get as close to the goal as possible
        if not self._is_valid_in_inflated_grid(goal_grid):
            self.get_logger().error(f'Goal position is invalid!')
            return []

        # Run D* Lite planning
        path_grid = self.dstar_plan(start_grid, goal_grid)

        if not path_grid:
            self.get_logger().error(f'Dstar algorithm path not found!')
            return []

        # Convert to world coordinates
        path_world = [self.grid_to_world(gx, gy) for gx, gy in path_grid]

        # Simplify for smoother motion
        return self._simplify_path(path_world, max_points=PlannerConstants.MAX_WAYPOINTS)

    def dstar_plan(self, start_grid, goal_grid):
        """
        Plan or replan using D* Lite.

        Decides whether to do initial planning or incremental replanning
        based on whether planner exists and goal changed.

        Args:
            start_grid: Start position (gx, gy)
            goal_grid: Goal position (gx, gy)

        Returns:
            Path as list of grid positions, or [] if no path
        """
        # Initial planning if planner doesn't exist or goal changed
        if self.dstar_planner is None or self.goal_grid != goal_grid:
            return self._initial_dstar_plan(start_grid, goal_grid)
        else:
            return self._replan_dstar(start_grid, goal_grid)

    def _initial_dstar_plan(self, start_grid, goal_grid):
        """
        Create initial D* Lite plan.

        Initializes new planner and computes first path.

        Args:
            start_grid: Start position
            goal_grid: Goal position

        Returns:
            Path or [] if failed
        """
        self.get_logger().info('Initializing D* Lite planner...')

        self.dstar_planner = DStarLite(self.grid_dynamic.copy(), start_grid, goal_grid)
        self.goal_grid = goal_grid

        # Snapshot base grid for change detection (persistent obstacles only)
        self.planner_grid_snapshot = self.grid_base.copy() if self.grid_base is not None else self.grid.copy()

        if not self.dstar_planner.compute_shortest_path():
            self.get_logger().error('D* Lite failed to find initial path')
            return []

        path = self.dstar_planner.extract_path()
        if path:
            self.get_logger().info(f'D* Lite found initial path with {len(path)} waypoints')
        return path

    def _replan_dstar(self, start_grid, goal_grid):
        """
        Replan using D* Lite with updated obstacles.

        This is the incremental replanning that makes D* Lite efficient.
        Only processes vertices affected by obstacle changes.

        Args:
            start_grid: New start position
            goal_grid: Goal position (unchanged)

        Returns:
            Updated path or [] if failed
        """
        self.get_logger().info('Replanning with D* Lite...')

        # Debug: Check if start position is occupied
        start_x, start_y = start_grid
        if (0 <= start_x < self.grid_dynamic.shape[1] and
            0 <= start_y < self.grid_dynamic.shape[0]):
            is_occupied = self.grid_dynamic[start_y, start_x] != 0
            self.get_logger().info(f'Start grid position: ({start_x}, {start_y}), occupied: {is_occupied}')

            if is_occupied:
                self.get_logger().warn('Start position is marked as occupied! Clearing robot area...')
                # Clear area around robot before replanning
                self._clear_robot_area(start_x, start_y)

        # Update start position (robot has moved)
        self.dstar_planner.update_start(start_grid)

        # Find changed cells (compare base grids for persistent obstacles only)
        changed_cells = self._find_changed_cells()

        if changed_cells:
            self.get_logger().info(f'Persistent obstacle changes detected: {len(changed_cells)} cells')

            # Update planner's grid
            self.dstar_planner.grid = self.grid_dynamic.copy()

            # Incremental update - only processes affected vertices!
            self.dstar_planner.update_obstacles(changed_cells)

            # Update snapshot
            self.planner_grid_snapshot = self.grid_base.copy() if self.grid_base is not None else self.grid.copy()

        # Recompute shortest path (incrementally)
        if not self.dstar_planner.compute_shortest_path():
            self.get_logger().error('D* Lite replanning failed - path is blocked')
            return []

        # Check if start is reachable
        if self.dstar_planner.g[start_grid] == float('inf'):
            self.get_logger().warn('Start position unreachable after incremental replan - trying fresh reinitialization...')
            self._debug_unreachable_start(start_grid, goal_grid)
            # Fallback: reinitialize D* Lite completely
            return self._reinitialize_dstar(start_grid, goal_grid)

        # Extract new path (with debug logging)
        path = self.dstar_planner.extract_path(debug=True)
        if not path:
            self.get_logger().warn('D* Lite incremental extract failed - trying fresh reinitialization...')
            # Fallback: reinitialize D* Lite completely
            return self._reinitialize_dstar(start_grid, goal_grid)

        self.get_logger().info(f'D* Lite replanning successful: {len(path)} waypoints')
        return path

    def _find_changed_cells(self):
        """
        Find cells that changed since last planning.

        Compares grid_base snapshots to track only persistent (SLAM) obstacles,
        not temporary lidar detections. This prevents the "massive grid changes"
        bug where 4000+ cells were detected as changed.

        Returns:
            List of changed cell positions [(x, y), ...]
        """
        changed_cells = []

        # Compare base grids (persistent SLAM obstacles only)
        current_base = self.grid_base if self.grid_base is not None else self.grid

        if self.planner_grid_snapshot is None:
            return changed_cells

        rows, cols = current_base.shape
        for y in range(rows):
            for x in range(cols):
                if self.planner_grid_snapshot[y, x] != current_base[y, x]:
                    changed_cells.append((x, y))

        return changed_cells

    def _reinitialize_dstar(self, start_grid, goal_grid):
        """
        Reinitialize D* Lite planner from scratch.

        This is a fallback when incremental replanning fails due to
        corrupted internal state. Creates a fresh planner with current grid.

        Args:
            start_grid: Start position
            goal_grid: Goal position

        Returns:
            Path or [] if still fails
        """
        self.get_logger().info('Reinitializing D* Lite planner from scratch...')

        # Clear robot area in grid first
        start_x, start_y = start_grid
        self._clear_robot_area(start_x, start_y)

        # Check if goal is occupied and find nearest free cell if so
        goal_x, goal_y = goal_grid
        if (0 <= goal_x < self.grid_dynamic.shape[1] and
            0 <= goal_y < self.grid_dynamic.shape[0] and
            self.grid_dynamic[goal_y, goal_x] != 0):
            self.get_logger().warn(f'Goal ({goal_x}, {goal_y}) is occupied, finding nearest free cell...')
            new_goal = self._find_nearest_free_cell(goal_x, goal_y)
            if new_goal:
                goal_grid = new_goal
                self.goal_grid = new_goal  # Update stored goal
                self.get_logger().info(f'Using alternative goal: {new_goal}')
            else:
                self.get_logger().error('Could not find free cell near goal')
                return []

        # Create fresh planner
        self.dstar_planner = DStarLite(self.grid_dynamic.copy(), start_grid, goal_grid)
        self.planner_grid_snapshot = self.grid_base.copy() if self.grid_base is not None else self.grid.copy()

        # Compute path
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
        """
        Find the nearest free cell to the given position.

        Searches in expanding circles until a free cell is found.

        Args:
            gx, gy: Center position in grid coordinates
            max_radius: Maximum search radius in cells

        Returns:
            Tuple (x, y) of nearest free cell, or None if not found
        """
        for radius in range(1, max_radius + 1):
            # Search in a square ring at this radius
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    # Only check cells on the ring perimeter
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

        # Check start neighborhood
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

        # Check goal
        if (0 <= goal_x < self.grid_dynamic.shape[1] and
            0 <= goal_y < self.grid_dynamic.shape[0]):
            goal_occupied = self.grid_dynamic[goal_y, goal_x] != 0
            self.get_logger().error(f'  Goal occupied: {goal_occupied}')

        # Check g-value at goal
        goal_g = self.dstar_planner.g[goal_grid]
        self.get_logger().error(f'  Goal g-value: {goal_g}')

    def _clear_robot_area(self, robot_gx, robot_gy):
        """
        Clear the area around the robot in all grids.

        This fixes the issue where SLAM inflation or noise marks the robot's
        current position as occupied, making replanning fail.

        Args:
            robot_gx, robot_gy: Robot position in grid coordinates
        """
        clearance = PlannerConstants.ROBOT_CLEARANCE_CELLS
        changed_cells = []

        for dy in range(-clearance, clearance + 1):
            for dx in range(-clearance, clearance + 1):
                clear_x, clear_y = robot_gx + dx, robot_gy + dy

                if (0 <= clear_x < self.grid_dynamic.shape[1] and
                    0 <= clear_y < self.grid_dynamic.shape[0]):

                    # Only clear if it was occupied (track for D* update)
                    if self.grid_dynamic[clear_y, clear_x] != 0:
                        changed_cells.append((clear_x, clear_y))

                    # Clear in all grids
                    self.grid_dynamic[clear_y, clear_x] = 0
                    if self.grid_base is not None:
                        self.grid_base[clear_y, clear_x] = 0

        # Update D* planner's grid and notify of changes
        if self.dstar_planner is not None and changed_cells:
            self.dstar_planner.grid = self.grid_dynamic.copy()
            self.dstar_planner.update_obstacles(changed_cells)
            self.get_logger().info(f'Cleared {len(changed_cells)} cells around robot')

    # ========================================================================
    # ROBOT CONTROL
    # ========================================================================

    def control_loop(self):
        """
        Main control loop - executes at 10 Hz.

        State machine:
        1. If replanning needed -> stop and replan
        2. If goal reached -> stop and clear path
        3. Otherwise -> follow current waypoint
        """
        if not self.path or self.current_pose is None:
            return

        if self.replanning_needed:
            self._handle_replanning()
            return

        if self.current_waypoint_idx >= len(self.path):
            self.stop_robot()
            self.get_logger().info('Goal reached! 😎')
            self.path = []
            return

        self._follow_waypoint()

    def _handle_replanning(self):
        """
        Handle replanning request.

        Stops robot, preserves valid waypoints up to the obstacle, and replans
        from the last valid waypoint to the original goal.
        """
        self.get_logger().info('Stopping robot for replanning...')
        self.stop_robot()

        if self.goal_grid is not None:
            goal_x, goal_y = self.grid_to_world(self.goal_grid[0], self.goal_grid[1])

            # Preserve valid waypoints up to the blocked segment
            preserved_waypoints = []
            replan_start_x = self.current_pose['x']
            replan_start_y = self.current_pose['y']

            if (self.blocked_waypoint_idx is not None and
                self.blocked_waypoint_idx > self.current_waypoint_idx and
                self.path):
                # Preserve waypoints from current position up to (not including) the blocked one
                preserved_waypoints = self.path[self.current_waypoint_idx:self.blocked_waypoint_idx]

                if preserved_waypoints:
                    # Replan from the last preserved waypoint
                    replan_start_x, replan_start_y = preserved_waypoints[-1]
                    self.get_logger().info(
                        f'Preserving {len(preserved_waypoints)} waypoints before obstacle at index {self.blocked_waypoint_idx}'
                    )

            # Plan new path from replan start point to goal
            new_path = self.plan_path(replan_start_x, replan_start_y, goal_x, goal_y)

            if new_path:
                if preserved_waypoints:
                    # Merge preserved waypoints with new path
                    # Skip first point of new_path if it's the same as last preserved waypoint
                    if (new_path and len(preserved_waypoints) > 0 and
                        abs(new_path[0][0] - preserved_waypoints[-1][0]) < 0.05 and
                        abs(new_path[0][1] - preserved_waypoints[-1][1]) < 0.05):
                        new_path = new_path[1:]
                    self.path = preserved_waypoints + new_path
                    # Keep current waypoint index since we preserved from there
                else:
                    self.path = new_path
                    self.current_waypoint_idx = 0

                self.get_logger().info(f'Replanning successful! Path: {len(self.path)} waypoints '
                                      f'({len(preserved_waypoints)} preserved + {len(new_path)} new)')
                self.publish_path()
            else:
                self.get_logger().error('Replanning failed! Stopping navigation.')
                self.path = []

        # Reset blocked waypoint tracker
        self.blocked_waypoint_idx = None
        self.replanning_needed = False

    def _follow_waypoint(self):
        """
        Follow current waypoint using smooth proportional control.

        Control strategy:
        - Always try to move forward while turning (no stop-and-rotate)
        - Reduce forward speed when angle error is large
        - Use proportional control for smooth motion
        - Look ahead to next waypoint for smoother curves
        """
        target = self.path[self.current_waypoint_idx]
        tx, ty = target

        # Calculate errors
        dx = tx - self.current_pose['x']
        dy = ty - self.current_pose['y']
        distance = math.sqrt(dx**2 + dy**2)
        target_angle = math.atan2(dy, dx)
        angle_diff = self.normalize_angle(target_angle - self.current_pose['theta'])

        cmd = TwistStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = 'base_link'

        # Check if waypoint reached
        if distance < self.position_tolerance:
            self.current_waypoint_idx += 1
            # Clear known obstacles when making progress
            self.known_obstacles.clear()
            self.get_logger().info(f'Waypoint {self.current_waypoint_idx}/{len(self.path)} reached')
            return

        # Smooth control - move and turn simultaneously
        # Only rotate in place if angle error is very large (> 90 degrees)
        large_angle_threshold = 1.57  # ~90 degrees

        if abs(angle_diff) > large_angle_threshold:
            # Large angle error - rotate in place but with proportional speed
            cmd.twist.linear.x = 0.0
            cmd.twist.angular.z = min(self.angular_speed, max(-self.angular_speed,
                                      0.8 * angle_diff))
        else:
            # Normal operation - move forward while turning
            # Reduce forward speed based on angle error (smoother curves)
            angle_factor = 1.0 - (abs(angle_diff) / large_angle_threshold) * 0.5
            cmd.twist.linear.x = min(self.linear_speed, distance) * angle_factor

            # Proportional angular control with higher gain for responsiveness
            cmd.twist.angular.z = min(self.angular_speed, max(-self.angular_speed,
                                      0.8 * angle_diff))

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

    def publish_path(self):
        """
        Publish planned path for visualization in RViz.

        Converts waypoint list to ROS Path message.
        """
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
            path_msg.poses.append(pose)

        self.path_pub.publish(path_msg)

    def publish_dynamic_grid(self):
        """
        Publish the dynamic grid for visualization.

        The dynamic grid shows the current planning grid including:
        - SLAM-discovered obstacles
        - Lidar-detected obstacles (temporary)
        """
        if self.grid_dynamic is None or self.resolution is None:
            return

        grid_msg = OccupancyGrid()
        grid_msg.header.stamp = self.get_clock().now().to_msg()
        grid_msg.header.frame_id = 'map'

        # Set map metadata
        grid_msg.info.resolution = self.resolution
        grid_msg.info.width = self.grid_dynamic.shape[1]
        grid_msg.info.height = self.grid_dynamic.shape[0]
        grid_msg.info.origin.position.x = self.origin[0]
        grid_msg.info.origin.position.y = self.origin[1]
        grid_msg.info.origin.position.z = 0.0

        # Convert grid to occupancy data (0=free, 100=occupied)
        occupancy_data = (self.grid_dynamic * 100).astype(np.int8).flatten().tolist()
        grid_msg.data = occupancy_data

        self.dynamic_grid_pub.publish(grid_msg)

        # Also publish markers for better visualization
        self.publish_grid_markers()

    def publish_grid_markers(self):
        """
        Publish grid obstacles as MarkerArray for RViz visualization.

        Color coding:
        - Red: Original SLAM obstacles (raw detections)
        - Orange: Inflated obstacle zones (safety buffer)
        - Purple: SLAM base obstacles
        - Cyan: Lidar-detected dynamic obstacles

        Subscribe to /don/grid_markers in RViz to see this.
        """
        if self.grid_dynamic is None or self.resolution is None:
            return

        marker_array = MarkerArray()
        stamp = self.get_clock().now().to_msg()

        # Create separate markers for different obstacle types
        # Using CUBE_LIST for efficiency (one marker with many cubes)

        # 1. Original SLAM obstacles (red) - raw detected obstacles
        original_marker = Marker()
        original_marker.header.stamp = stamp
        original_marker.header.frame_id = 'map'
        original_marker.ns = 'slam_original'
        original_marker.id = 0
        original_marker.type = Marker.CUBE_LIST
        original_marker.action = Marker.ADD
        original_marker.scale.x = self.resolution * 0.9
        original_marker.scale.y = self.resolution * 0.9
        original_marker.scale.z = 0.1
        original_marker.color = ColorRGBA(r=0.8, g=0.0, b=0.0, a=0.8)  # Red
        original_marker.pose.orientation.w = 1.0

        # 2. Inflated SLAM obstacles (orange) - safety buffer around obstacles
        inflated_marker = Marker()
        inflated_marker.header.stamp = stamp
        inflated_marker.header.frame_id = 'map'
        inflated_marker.ns = 'inflated_obstacles'
        inflated_marker.id = 1
        inflated_marker.type = Marker.CUBE_LIST
        inflated_marker.action = Marker.ADD
        inflated_marker.scale.x = self.resolution * 0.9
        inflated_marker.scale.y = self.resolution * 0.9
        inflated_marker.scale.z = 0.05
        inflated_marker.color = ColorRGBA(r=1.0, g=0.5, b=0.0, a=0.5)  # Orange
        inflated_marker.pose.orientation.w = 1.0

        # 3. SLAM base obstacles (purple) - in grid_base but not in grid
        slam_marker = Marker()
        slam_marker.header.stamp = stamp
        slam_marker.header.frame_id = 'map'
        slam_marker.ns = 'slam_obstacles'
        slam_marker.id = 2
        slam_marker.type = Marker.CUBE_LIST
        slam_marker.action = Marker.ADD
        slam_marker.scale.x = self.resolution * 0.9
        slam_marker.scale.y = self.resolution * 0.9
        slam_marker.scale.z = 0.15
        slam_marker.color = ColorRGBA(r=0.6, g=0.0, b=0.8, a=0.8)  # Purple
        slam_marker.pose.orientation.w = 1.0

        # 4. Lidar/dynamic obstacles (cyan) - in grid_dynamic but not in grid_base
        lidar_marker = Marker()
        lidar_marker.header.stamp = stamp
        lidar_marker.header.frame_id = 'map'
        lidar_marker.ns = 'lidar_obstacles'
        lidar_marker.id = 3
        lidar_marker.type = Marker.CUBE_LIST
        lidar_marker.action = Marker.ADD
        lidar_marker.scale.x = self.resolution * 0.9
        lidar_marker.scale.y = self.resolution * 0.9
        lidar_marker.scale.z = 0.2
        lidar_marker.color = ColorRGBA(r=0.0, g=0.8, b=0.8, a=0.9)  # Cyan
        lidar_marker.pose.orientation.w = 1.0

        # 5. Unknown/unexplored areas (gray)
        unknown_marker = Marker()
        unknown_marker.header.stamp = stamp
        unknown_marker.header.frame_id = 'map'
        unknown_marker.ns = 'unknown_areas'
        unknown_marker.id = 4
        unknown_marker.type = Marker.CUBE_LIST
        unknown_marker.action = Marker.ADD
        unknown_marker.scale.x = self.resolution * 0.9
        unknown_marker.scale.y = self.resolution * 0.9
        unknown_marker.scale.z = 0.02
        unknown_marker.color = ColorRGBA(r=0.5, g=0.5, b=0.5, a=0.3)  # Gray, semi-transparent
        unknown_marker.pose.orientation.w = 1.0

        # 6. Lidar obstacles along path (bright green) - what lidar sees on the path
        path_obstacle_marker = Marker()
        path_obstacle_marker.header.stamp = stamp
        path_obstacle_marker.header.frame_id = 'map'
        path_obstacle_marker.ns = 'lidar_along_path'
        path_obstacle_marker.id = 5
        path_obstacle_marker.type = Marker.CUBE_LIST
        path_obstacle_marker.action = Marker.ADD
        path_obstacle_marker.scale.x = self.resolution * 1.2  # Slightly larger to stand out
        path_obstacle_marker.scale.y = self.resolution * 1.2
        path_obstacle_marker.scale.z = 0.3  # Taller to stand out
        path_obstacle_marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)  # Bright green
        path_obstacle_marker.pose.orientation.w = 1.0

        # Iterate through grid and categorize obstacles
        rows, cols = self.grid_dynamic.shape
        for gy in range(rows):
            for gx in range(cols):
                # Convert to world coordinates
                world_x = gx * self.resolution + self.origin[0]
                world_y = gy * self.resolution + self.origin[1]
                point = Point(x=world_x, y=world_y, z=0.0)

                # Check what type of obstacle this is
                is_original = self.grid_original[gy, gx] != 0 if self.grid_original is not None else False
                is_inflated = self.grid[gy, gx] != 0 if self.grid is not None else False
                is_base = self.grid_base[gy, gx] != 0 if self.grid_base is not None else False
                is_dynamic = self.grid_dynamic[gy, gx] != 0
                is_unknown = self.grid_unknown[gy, gx] != 0 if self.grid_unknown is not None else False

                if is_original:
                    # Original SLAM obstacle
                    original_marker.points.append(point)
                elif is_inflated:
                    # Inflated safety zone
                    inflated_marker.points.append(point)
                elif is_base and not is_inflated:
                    # SLAM-discovered obstacle
                    slam_marker.points.append(point)
                elif is_dynamic and not is_base:
                    # Lidar-detected dynamic obstacle
                    lidar_marker.points.append(point)
                elif is_unknown and not is_dynamic:
                    # Unknown/unexplored area (shown as passable but unexplored)
                    unknown_marker.points.append(point)

        # Add lidar obstacles along path (stored separately from grid scan)
        for gx, gy in self.lidar_obstacles_along_path:
            world_x = gx * self.resolution + self.origin[0]
            world_y = gy * self.resolution + self.origin[1]
            point = Point(x=world_x, y=world_y, z=0.0)
            path_obstacle_marker.points.append(point)

        # Add markers to array (only if they have points)
        if original_marker.points:
            marker_array.markers.append(original_marker)
        if inflated_marker.points:
            marker_array.markers.append(inflated_marker)
        if slam_marker.points:
            marker_array.markers.append(slam_marker)
        if lidar_marker.points:
            marker_array.markers.append(lidar_marker)
        if unknown_marker.points:
            marker_array.markers.append(unknown_marker)
        if path_obstacle_marker.points:
            marker_array.markers.append(path_obstacle_marker)

        # Publish marker array
        self.grid_markers_pub.publish(marker_array)

    def _is_valid_in_inflated_grid(self, grid_pos):
        """
        Check if position is valid in inflated grid.

        Args:
            grid_pos: Position (gx, gy)

        Returns:
            True if in bounds and free, False otherwise
        """
        gx, gy = grid_pos
        if gx < 0 or gx >= self.grid.shape[1] or gy < 0 or gy >= self.grid.shape[0]:
            return False
        return self.grid[gy, gx] == 0

    def _is_valid_in_original_grid(self, grid_pos):
        """
        Check if position is valid in original (uninflated) grid.

        Args:
            grid_pos: Position (gx, gy)

        Returns:
            True if in bounds and free, False otherwise
        """
        gx, gy = grid_pos
        if gx < 0 or gx >= self.grid_original.shape[1] or gy < 0 or gy >= self.grid_original.shape[0]:
            return False
        return self.grid_original[gy, gx] == 0

    def world_to_grid(self, x, y):
        """
        Convert world coordinates to grid indices.

        Args:
            x, y: World coordinates in meters

        Returns:
            Tuple (grid_x, grid_y)
        """
        grid_x = int((x - self.origin[0]) / self.resolution)
        grid_y = int((y - self.origin[1]) / self.resolution)
        return (grid_x, grid_y)

    def grid_to_world(self, grid_x, grid_y):
        """
        Convert grid indices to world coordinates.

        Args:
            grid_x, grid_y: Grid coordinates

        Returns:
            Tuple (x, y) in meters
        """
        x = grid_x * self.resolution + self.origin[0]
        y = grid_y * self.resolution + self.origin[1]
        return (x, y)

    def _simplify_path(self, path, max_points=20):
        """
        Simplify path by keeping only key waypoints.

        Reduces waypoint count for smoother motion.

        Args:
            path: List of waypoints
            max_points: Maximum number of waypoints to keep

        Returns:
            Simplified path
        """
        if len(path) <= max_points:
            return path

        step = len(path) // max_points
        simplified = [path[i] for i in range(0, len(path), step)]
        simplified.append(path[-1])  # Always include goal
        return simplified

    def get_current_position(self):
        """
        Get current robot position.

        Returns:
            Tuple (x, y) or None if no odometry yet
        """
        if self.current_pose is None:
            self.get_logger().warn('No odometry data available yet')
            return None
        return (self.current_pose['x'], self.current_pose['y'])

    def _log_obstacle_detection(self, obst_gx, obst_gy, dist, threshold, total_obstacles):
        """Log obstacle detection event with details."""
        self.get_logger().warn('Obstacle blocking path detected!')
        self.get_logger().warn(f'  Position: ({obst_gx},{obst_gy}), Distance: {dist:.1f} cells')
        self.get_logger().warn(f'  Threshold: {threshold} cells, Total obstacles: {total_obstacles}')

    def _log_replanning_trigger(self, reason, count):
        """Log replanning trigger event."""
        self.get_logger().warn(f'{reason} block current path!')
        self.get_logger().warn(f'  Changed cells: {count}')
        self.get_logger().warn('  Triggering D* Lite replanning...')

    @staticmethod
    def quaternion_to_yaw(q):
        """
        Convert quaternion to yaw angle.

        Args:
            q: Quaternion with w, x, y, z components

        Returns:
            Yaw angle in radians
        """
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def normalize_angle(angle):
        """
        Normalize angle to [-pi, pi].

        Args:
            angle: Angle in radians

        Returns:
            Normalized angle in [-pi, pi]
        """
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

    Configuration:
        - Set environment variables to override defaults:
          ROBOT_RADIUS, SAFETY_CLEARANCE, OPTIMISTIC_PLANNING
        - Or modify PlannerConstants class directly

    Navigation:
        - Send goal poses to the goal_pose topic (PoseStamped messages)
        - Example: ros2 topic pub /don/goal_pose geometry_msgs/PoseStamped '{pose: {position: {x: 1.0, y: 2.0}}}'
    """

    # Load configuration from environment (with PlannerConstants as defaults)
    robot_radius = float(os.getenv('ROBOT_RADIUS', str(PlannerConstants.ROBOT_RADIUS)))
    safety_clearance = float(os.getenv('SAFETY_CLEARANCE', str(PlannerConstants.SAFETY_CLEARANCE)))
    optimistic_planning = os.getenv('OPTIMISTIC_PLANNING', '1') == '1'

    rclpy.init(args=args)
    navigator = DStarNavigator(
        robot_radius=robot_radius,
        safety_clearance=safety_clearance,
        optimistic_planning=optimistic_planning
    )

    # Print usage information
    print("\n" + "="*60)
    print("TurtleBot4 D* Lite Navigator with Dynamic Replanning")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Robot radius: {robot_radius}m")
    print(f"  Safety clearance: {safety_clearance}m")
    print(f"  Optimistic planning: {'ENABLED' if optimistic_planning else 'DISABLED'}")
    print(f"\nTopics:")
    print(f"  Goal pose input: {PlannerConstants.GOAL_POSE}")
    print(f"  Odometry: {PlannerConstants.ODOMETRY}")
    print(f"  Scan: {PlannerConstants.SCAN}")
    print(f"  Map: {PlannerConstants.OCCUPANCY_GRID}")
    print(f"\nUsage:")
    print(f"  ros2 topic pub {PlannerConstants.GOAL_POSE} geometry_msgs/PoseStamped '{{pose: {{position: {{x: 1.0, y: 2.0}}}}}}' --once")
    print("="*60 + "\n")

    # Wait for odometry
    print("Waiting for odometry data...")
    while navigator.current_pose is None and rclpy.ok():
        rclpy.spin_once(navigator, timeout_sec=0.1)

    if navigator.current_pose:
        print(f"Odometry received. Current position: ({navigator.current_pose['x']:.2f}, {navigator.current_pose['y']:.2f})")

    # Wait for SLAM map
    if navigator.use_live_slam:
        print("Listening for SLAM map updates...")
        for _ in range(20):
            rclpy.spin_once(navigator, timeout_sec=0.1)

    print(f"\nReady! Waiting for goal poses on {PlannerConstants.GOAL_POSE}...")

    try:
        rclpy.spin(navigator)
    except KeyboardInterrupt:
        print("\nShutting down...")

    # Cleanup
    navigator.stop_robot()
    navigator.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
