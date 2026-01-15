#!/usr/bin/env python3

"""
A* Navigator for ROS2 TurtleBot4

This ROS2 node provides autonomous navigation using the A* pathfinding algorithm.
Features hybrid obstacle checking to handle tight spaces while maintaining safety.

Features:
    - A* pathfinding with 8-connected grid movement
    - Hybrid obstacle validation (tight spaces + safety margins)
    - Pure pursuit-style waypoint following
    - Interactive goal selection via terminal input
    - Environment variable configuration for robot parameters

Publishes to:
    - /cmd_vel (TwistStamped): Velocity commands for robot control

Subscribes to:
    - /sim_ground_truth_pose (Odometry): Robot position in map frame

Usage:
    ros2 run a_star a_star_nav

    With custom parameters:
    ROBOT_RADIUS=0.22 SAFETY_CLEARANCE=0.20 ros2 run a_star a_star_nav

Architecture:
    - Hybrid obstacle checking near start position for escaping tight spaces
    - Standard inflated grid checking for safety in open areas
    - Simple proportional controller for robot movement
    - Path simplification to reduce waypoint count
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Twist, TwistStamped, PoseStamped
from nav_msgs.msg import Odometry
import numpy as np
import yaml
from PIL import Image
import heapq
import math
import os


# ============================================================================
# CONSTANTS
# ============================================================================

class NavigatorConstants:
    """Configuration constants for the A* navigator"""

    # Robot Physical Parameters (meters)
    ROBOT_RADIUS = 0.22          # TurtleBot4 radius
    SAFETY_CLEARANCE = 0.15      # Additional safety margin

    # Control Parameters
    LINEAR_SPEED = 0.2           # Forward speed (m/s)
    ANGULAR_SPEED = 0.5          # Rotation speed (rad/s)
    POSITION_TOLERANCE = 0.1     # Waypoint reached threshold (meters)
    ANGLE_TOLERANCE = 0.1        # Angular alignment threshold (radians)

    # Control Loop
    CONTROL_TIMER_PERIOD = 0.1   # Control loop frequency (seconds)

    # Path Planning
    MAX_PATH_WAYPOINTS = 20      # Maximum waypoints after simplification
    TIGHT_SPACE_RADIUS = 3       # Grid cells to use original grid near start

    # Publishing/Subscribing Paths
    CMD_VEL = '/don/cmd_vel'
    ODOMETRY = '/don/sim_ground_truth_pose'

    # Pgm and yaml paths
    SLAM_MAP_YAML = '~/obstacle-avoidance-comps/ros2_ws/maze_slamed.yaml'
    SLAM_MAP_PGM = '~/obstacle-avoidance-comps/ros2_ws/maze_slamed.pgm'



# ============================================================================
# MAIN NAVIGATOR CLASS
# ============================================================================

class AStarNavigator(Node):
    """
    ROS2 node for A* path planning and navigation.

    Implements autonomous navigation using A* pathfinding with hybrid
    obstacle checking to handle tight spaces while maintaining safety.
    """

    def __init__(self, robot_radius=None, safety_clearance=None):
        """
        Initialize the A* navigator.

        Args:
            robot_radius: Robot radius in meters (default from constants or env)
            safety_clearance: Safety margin in meters (default from constants or env)
        """
        super().__init__('astar_navigator')

        # ====================================================================
        # INITIALIZATION - Robot Parameters
        # ====================================================================

        # Use provided values, environment variables, or defaults
        self.robot_radius = robot_radius if robot_radius is not None else \
                           float(os.getenv('ROBOT_RADIUS', str(NavigatorConstants.ROBOT_RADIUS)))
        self.safety_clearance = safety_clearance if safety_clearance is not None else \
                               float(os.getenv('SAFETY_CLEARANCE', str(NavigatorConstants.SAFETY_CLEARANCE)))

        # ====================================================================
        # INITIALIZATION - ROS2 Publishers and Subscribers
        # ====================================================================

        # Command velocity publisher (TwistStamped for Jazzy, Twist for Humble)
        self.cmd_vel_pub = self.create_publisher(TwistStamped, NavigatorConstants.CMD_VEL, 10)

        # Odometry subscriber (best-effort QoS to match publisher)
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.odom_sub = self.create_subscription(
            Odometry,
            NavigatorConstants.ODOMETRY,
            self.odom_callback,
            qos_profile
        )

        # ====================================================================
        # INITIALIZATION - Map Data
        # ====================================================================

        self.grid = None                # Inflated grid for safety
        self.grid_original = None       # Original grid before inflation
        self.resolution = None          # Meters per grid cell
        self.origin = None              # Map origin [x, y, theta]

        # ====================================================================
        # INITIALIZATION - Navigation State
        # ====================================================================

        self.current_pose = None        # Current robot pose {x, y, theta}
        self.path = []                  # Planned path as list of (x, y) in world coords
        self.current_waypoint_idx = 0   # Index of current target waypoint

        # ====================================================================
        # INITIALIZATION - Control Parameters
        # ====================================================================

        self.linear_speed = NavigatorConstants.LINEAR_SPEED
        self.angular_speed = NavigatorConstants.ANGULAR_SPEED
        self.position_tolerance = NavigatorConstants.POSITION_TOLERANCE
        self.angle_tolerance = NavigatorConstants.ANGLE_TOLERANCE

        # ====================================================================
        # INITIALIZATION - Control Loop Timer
        # ====================================================================

        self.control_timer = self.create_timer(
            NavigatorConstants.CONTROL_TIMER_PERIOD,
            self.control_loop
        )

        # ====================================================================
        # INITIALIZATION - Load Map
        # ====================================================================

        # Default map path (same as a_star_nav.py)
        # yaml_file = '/opt/ros/jazzy/share/turtlebot4_navigation/maps/maze.yaml'
        # pgm_file = '/opt/ros/jazzy/share/turtlebot4_navigation/maps/maze.pgm'

        # Slammed maze map
        yaml_file = os.path.expanduser(NavigatorConstants.SLAM_MAP_YAML)
        pgm_file = os.path.expanduser(NavigatorConstants.SLAM_MAP_PGM)

        self.load_map(yaml_file, pgm_file)

        # ====================================================================
        # INITIALIZATION - Logging
        # ====================================================================

        self.get_logger().info('A* Navigator initialized')
        self.get_logger().info(f'Robot radius: {self.robot_radius}m, Safety clearance: {self.safety_clearance}m')
        self.get_logger().info(f'Total obstacle inflation: {self.robot_radius + self.safety_clearance}m')
        self.get_logger().info('Usage: navigator.navigate_to_goal(start_x, start_y, goal_x, goal_y)')

    # ========================================================================
    # MAP LOADING AND PROCESSING
    # ========================================================================

    def load_map(self, yaml_file, pgm_file):
        """
        Load map from YAML and PGM files.

        Loads the static map, applies coordinate transforms to match Gazebo,
        and inflates obstacles by robot radius + safety clearance.

        Args:
            yaml_file: Path to map metadata YAML file
            pgm_file: Path to map image PGM file
        """
        try:
            # Load map metadata
            with open(yaml_file, 'r') as f:
                map_data = yaml.safe_load(f)

            self.resolution = map_data['resolution']
            self.origin = map_data['origin']

            # Load map image
            img = Image.open(pgm_file)
            occupancy_grid = np.array(img)

            # Convert to binary: 0=free, 1=occupied
            # PGM: 255=free, <250=occupied/unknown
            self.grid = np.zeros_like(occupancy_grid)
            self.grid[occupancy_grid < 250] = 1  # Occupied/unknown
            self.grid[occupancy_grid >= 250] = 0  # Free space

            # Flip map vertically to match Gazebo coordinate system
            self.grid = np.flipud(self.grid)

            # Store original grid before inflation
            self.grid_original = self.grid.copy()

            # Inflate obstacles for robot safety
            total_inflation = self.robot_radius + self.safety_clearance
            self.inflate_obstacles(total_inflation)

            # Log map information
            self.get_logger().info(f'Map loaded: {self.grid.shape}, resolution: {self.resolution}')
            self.get_logger().info(f'Map origin: {self.origin}')
            self.get_logger().info(
                f'Obstacles inflated by {total_inflation}m ({int(total_inflation/self.resolution)} pixels)'
            )

            # Log map bounds
            map_width = self.grid.shape[1] * self.resolution
            map_height = self.grid.shape[0] * self.resolution
            self.get_logger().info(
                f'Map bounds: X=[{self.origin[0]:.2f}, {self.origin[0]+map_width:.2f}], '
                f'Y=[{self.origin[1]:.2f}, {self.origin[1]+map_height:.2f}]'
            )
        except Exception as e:
            self.get_logger().error(f'Failed to load map: {e}')

    def inflate_obstacles(self, inflation_radius):
        """
        Inflate obstacles by specified radius using morphological dilation.

        Creates a safety buffer around all obstacles equal to robot_radius +
        safety_clearance. This ensures the robot's center can safely follow
        paths without collision.

        Args:
            inflation_radius: Inflation distance in meters
        """
        from scipy.ndimage import binary_dilation

        # Convert radius to pixels
        radius_pixels = int(inflation_radius / self.resolution)
        kernel_size = 2 * radius_pixels + 1
        kernel = np.ones((kernel_size, kernel_size))

        # Apply dilation to original grid
        self.grid = binary_dilation(self.grid_original, kernel).astype(int)

    # ========================================================================
    # ODOMETRY CALLBACK
    # ========================================================================

    def odom_callback(self, msg):
        """
        Update current robot pose from odometry.

        Extracts position and orientation from odometry message and
        updates internal state for navigation control.

        Args:
            msg: Odometry message from /sim_ground_truth_pose
        """
        self.current_pose = {
            'x': msg.pose.pose.position.x,
            'y': msg.pose.pose.position.y,
            'theta': self.quaternion_to_yaw(msg.pose.pose.orientation)
        }

    # ========================================================================
    # NAVIGATION INTERFACE
    # ========================================================================

    def navigate_to_goal(self, start_x, start_y, goal_x, goal_y):
        """
        Plan path and start navigation from start to goal.

        Main entry point for navigation. Plans a path using A* and
        starts the waypoint following control loop.

        Args:
            start_x: Start X coordinate in world frame (meters)
            start_y: Start Y coordinate in world frame (meters)
            goal_x: Goal X coordinate in world frame (meters)
            goal_y: Goal Y coordinate in world frame (meters)

        Returns:
            bool: True if path found and navigation started, False otherwise
        """
        self.get_logger().info(
            f'Planning path from ({start_x:.2f}, {start_y:.2f}) to ({goal_x:.2f}, {goal_y:.2f})'
        )

        # Plan path using A*
        self.path = self.plan_path(start_x, start_y, goal_x, goal_y)

        if self.path:
            self.current_waypoint_idx = 0
            self.get_logger().info(f'✓ Path found with {len(self.path)} waypoints')
            self.print_path()
            return True
        else:
            self.get_logger().error('✗ No path found! Check if start/goal are valid and reachable')
            return False

    # ========================================================================
    # PATH PLANNING
    # ========================================================================

    def plan_path(self, start_x, start_y, goal_x, goal_y):
        """
        Plan path from start to goal using A*.

        Converts world coordinates to grid coordinates, validates positions,
        runs A* pathfinding, and converts result back to world coordinates.

        Args:
            start_x: Start X in world frame (meters)
            start_y: Start Y in world frame (meters)
            goal_x: Goal X in world frame (meters)
            goal_y: Goal Y in world frame (meters)

        Returns:
            list: Path as list of (x, y) tuples in world coordinates, or empty if no path
        """
        # Convert to grid coordinates
        start_grid = self.world_to_grid(start_x, start_y)
        goal_grid = self.world_to_grid(goal_x, goal_y)

        self.get_logger().info(f'Grid coordinates: Start {start_grid}, Goal {goal_grid}')

        # Verify conversion is reversible (debug)
        start_world_check = self.grid_to_world(start_grid[0], start_grid[1])
        goal_world_check = self.grid_to_world(goal_grid[0], goal_grid[1])
        self.get_logger().info(
            f'World coords check: Start ({start_world_check[0]:.2f}, {start_world_check[1]:.2f}), '
            f'Goal ({goal_world_check[0]:.2f}, {goal_world_check[1]:.2f})'
        )

        # Validate start position in original grid (robot is already there)
        if not self.is_valid_in_original_grid(start_grid):
            self.get_logger().error(
                f'Start position ({start_x:.2f}, {start_y:.2f}) is invalid or in actual obstacle!'
            )
            return []

        # Validate goal position in inflated grid (for safety)
        if not self.is_valid(goal_grid):
            self.get_logger().error(
                f'Goal position ({goal_x:.2f}, {goal_y:.2f}) is invalid or too close to obstacles!'
            )
            return []

        # Check connectivity from start (debug)
        neighbors_valid = sum(
            1 for dx, dy in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]
            if self.is_valid((start_grid[0] + dx, start_grid[1] + dy))
        )
        self.get_logger().info(f'Start has {neighbors_valid}/8 valid neighbors in inflated grid')

        if neighbors_valid == 0:
            self.get_logger().warn('Start position is surrounded by inflated obstacles!')
            self.get_logger().warn(
                f'Consider reducing safety_clearance (current: {self.safety_clearance}m) '
                f'or robot_radius (current: {self.robot_radius}m)'
            )
            self.get_logger().warn(f'Current total inflation: {self.robot_radius + self.safety_clearance}m')

        # Run A* pathfinding
        self.get_logger().info('Running A* pathfinding...')
        path_grid = self.astar(start_grid, goal_grid)

        if not path_grid:
            return []

        # Convert to world coordinates and simplify
        path_world = [self.grid_to_world(gx, gy) for gx, gy in path_grid]
        simplified_path = self.simplify_path(path_world)

        return simplified_path

    def astar(self, start, goal):
        """
        A* pathfinding algorithm with hybrid obstacle checking.

        Uses original grid near start position (tight spaces) and inflated
        grid far from start (safety). This allows the robot to escape from
        tight spaces while maintaining safety margins in open areas.

        Args:
            start: Start position as (grid_x, grid_y)
            goal: Goal position as (grid_x, grid_y)

        Returns:
            list: Path as list of (grid_x, grid_y) tuples, or empty list if no path
        """
        # Heuristic: Euclidean distance
        def heuristic(a, b):
            return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

        rows, cols = self.grid.shape

        # Initialize A* data structures
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}
        nodes_explored = 0

        # A* main loop
        while open_set:
            _, current = heapq.heappop(open_set)
            nodes_explored += 1

            # Check if reached goal
            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                self.get_logger().info(f'A* explored {nodes_explored} nodes')
                return path[::-1]  # Reverse to get start->goal order

            # Explore 8-connected neighbors
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                neighbor = (current[0] + dx, current[1] + dy)

                # HYBRID VALIDATION: Use original grid near start (tight spaces)
                # and inflated grid far from start (safety)
                distance_from_start = abs(neighbor[0] - start[0]) + abs(neighbor[1] - start[1])

                if distance_from_start <= NavigatorConstants.TIGHT_SPACE_RADIUS:
                    # Close to start - use original grid to escape tight spaces
                    if not self.is_valid_in_original_grid(neighbor):
                        continue
                else:
                    # Far from start - use inflated grid for safety
                    if not self.is_valid(neighbor):
                        continue

                # Diagonal moves cost sqrt(2) ≈ 1.414
                move_cost = 1.414 if dx != 0 and dy != 0 else 1.0
                tentative_g = g_score[current] + move_cost

                # Update if better path found
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        # No path found
        self.get_logger().error(f'A* failed after exploring {nodes_explored} nodes')
        return []

    # ========================================================================
    # GRID VALIDATION
    # ========================================================================

    def is_valid(self, grid_pos):
        """
        Check if grid position is valid and free in inflated grid.

        Args:
            grid_pos: Tuple (grid_x, grid_y) to check

        Returns:
            bool: True if position is valid and free, False otherwise
        """
        gx, gy = grid_pos
        rows, cols = self.grid.shape

        # Check bounds
        if gx < 0 or gx >= cols or gy < 0 or gy >= rows:
            return False

        # Check if free (0 = free, 1 = occupied)
        return self.grid[gy, gx] == 0  # Note: grid is [row, col] = [y, x]

    def is_valid_in_original_grid(self, grid_pos):
        """
        Check if grid position is valid and free in original grid (before inflation).

        Used for checking positions near the start to allow escape from tight spaces.

        Args:
            grid_pos: Tuple (grid_x, grid_y) to check

        Returns:
            bool: True if position is valid and free, False otherwise
        """
        gx, gy = grid_pos
        rows, cols = self.grid_original.shape

        # Check bounds
        if gx < 0 or gx >= cols or gy < 0 or gy >= rows:
            return False

        # Check if free (0 = free, 1 = occupied)
        return self.grid_original[gy, gx] == 0  # Note: grid is [row, col] = [y, x]

    # ========================================================================
    # COORDINATE CONVERSION
    # ========================================================================

    def world_to_grid(self, x, y):
        """
        Convert world coordinates (meters) to grid indices (cells).

        Args:
            x: X coordinate in world frame (meters)
            y: Y coordinate in world frame (meters)

        Returns:
            tuple: (grid_x, grid_y) in grid cell coordinates
        """
        grid_x = int((x - self.origin[0]) / self.resolution)
        grid_y = int((y - self.origin[1]) / self.resolution)
        return (grid_x, grid_y)

    def grid_to_world(self, grid_x, grid_y):
        """
        Convert grid indices (cells) to world coordinates (meters).

        Args:
            grid_x: X index in grid coordinates
            grid_y: Y index in grid coordinates

        Returns:
            tuple: (x, y) in world frame (meters)
        """
        x = grid_x * self.resolution + self.origin[0]
        y = grid_y * self.resolution + self.origin[1]
        return (x, y)

    # ========================================================================
    # PATH MANAGEMENT
    # ========================================================================

    def simplify_path(self, path, max_points=None):
        """
        Simplify path by keeping only key waypoints.

        Reduces the number of waypoints to improve navigation smoothness
        while maintaining the overall path shape.

        Args:
            path: List of (x, y) tuples in world coordinates
            max_points: Maximum waypoints to keep (default from constants)

        Returns:
            list: Simplified path as list of (x, y) tuples
        """
        if max_points is None:
            max_points = NavigatorConstants.MAX_PATH_WAYPOINTS

        if len(path) <= max_points:
            return path

        # Downsample to max_points waypoints
        step = len(path) // max_points
        simplified = [path[i] for i in range(0, len(path), step)]
        simplified.append(path[-1])  # Always include goal
        return simplified

    def print_path(self):
        """Print the planned path waypoints to logger."""
        self.get_logger().info('Planned waypoints:')
        for i, (x, y) in enumerate(self.path):
            self.get_logger().info(f'  {i+1}. ({x:.2f}, {y:.2f})')

    # ========================================================================
    # ROBOT CONTROL
    # ========================================================================

    def control_loop(self):
        """
        Main control loop for waypoint following.

        Called periodically by ROS2 timer. Implements simple proportional
        control to follow waypoints: rotate towards target, then drive forward.
        """
        # Check if navigation active
        if not self.path or self.current_pose is None:
            return

        # Check if goal reached
        if self.current_waypoint_idx >= len(self.path):
            self.stop_robot()
            self.get_logger().info('🎯 Goal reached!')
            self.path = []
            return

        # Get current target waypoint
        target = self.path[self.current_waypoint_idx]
        tx, ty = target

        # Calculate distance and angle to target
        dx = tx - self.current_pose['x']
        dy = ty - self.current_pose['y']
        distance = math.sqrt(dx**2 + dy**2)
        target_angle = math.atan2(dy, dx)
        angle_diff = self.normalize_angle(target_angle - self.current_pose['theta'])

        # Create velocity command (TwistStamped for Jazzy, Twist for Humble)
        cmd = TwistStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = 'base_link'

        # Check if waypoint reached
        if distance < self.position_tolerance:
            self.current_waypoint_idx += 1
            self.get_logger().info(
                f'Waypoint {self.current_waypoint_idx}/{len(self.path)} reached'
            )
            return

        # Simple proportional control:
        # 1. Rotate towards target if not aligned
        if abs(angle_diff) > self.angle_tolerance:
            cmd.twist.angular.z = self.angular_speed if angle_diff > 0 else -self.angular_speed
        else:
            # 2. Drive forward once aligned
            cmd.twist.linear.x = min(self.linear_speed, distance)
            # Small angular correction while moving
            cmd.twist.angular.z = 0.3 * angle_diff

        self.cmd_vel_pub.publish(cmd)

    def stop_robot(self):
        """Stop the robot by publishing zero velocities."""
        cmd = TwistStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = 'base_link'
        # cmd.twist is already zeros by default
        self.cmd_vel_pub.publish(cmd)

    def get_current_position(self):
        """
        Get the current robot position from odometry.

        Returns:
            tuple: (x, y) position in world frame, or None if no odometry data
        """
        if self.current_pose is None:
            self.get_logger().warn('No odometry data available yet')
            return None
        return (self.current_pose['x'], self.current_pose['y'])

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    @staticmethod
    def quaternion_to_yaw(q):
        """
        Convert quaternion to yaw angle.

        Args:
            q: Quaternion with w, x, y, z components

        Returns:
            float: Yaw angle in radians
        """
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def normalize_angle(angle):
        """
        Normalize angle to [-pi, pi] range.

        Args:
            angle: Angle in radians

        Returns:
            float: Normalized angle in [-pi, pi]
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
    Main entry point for the A* navigator.

    Initializes ROS2, creates the navigator node, waits for odometry data,
    prompts for goal input, and executes navigation.

    Configuration via environment variables:
        ROBOT_RADIUS: Robot radius in meters (default 0.22)
        SAFETY_CLEARANCE: Safety margin in meters (default 0.15)
    """
    # Get parameters from environment or use defaults
    robot_radius = float(os.getenv('ROBOT_RADIUS', str(NavigatorConstants.ROBOT_RADIUS)))
    safety_clearance = float(os.getenv('SAFETY_CLEARANCE', str(NavigatorConstants.SAFETY_CLEARANCE)))

    # Initialize ROS2
    rclpy.init(args=args)
    navigator = AStarNavigator(robot_radius=robot_radius, safety_clearance=safety_clearance)

    # Print usage information
    print("\n" + "="*60)
    print("TurtleBot4 A* Navigator")
    print("="*60)
    print(f"\nRobot Configuration:")
    print(f"  Robot radius: {robot_radius}m (set via ROBOT_RADIUS env var)")
    print(f"  Safety clearance: {safety_clearance}m (set via SAFETY_CLEARANCE env var)")
    print(f"  Total inflation: {robot_radius + safety_clearance}m")
    print("\nUsage Examples:")
    print("  1. Get current position:")
    print("     pos = navigator.get_current_position()")
    print("\n  2. Navigate to goal:")
    print("     navigator.navigate_to_goal(0.0, 0.0, 3.0, 2.0)")
    print("\n  3. Use current position:")
    print("     pos = navigator.get_current_position()")
    print("     if pos:")
    print("         navigator.navigate_to_goal(pos[0], pos[1], 3.0, 2.0)")
    print("\nTo customize clearance:")
    print("  ROBOT_RADIUS=0.22 SAFETY_CLEARANCE=0.20 ros2 run a_star a_star_nav")
    print("="*60 + "\n")

    # Wait for odometry data
    print("Waiting for odometry data...")
    while navigator.current_pose is None and rclpy.ok():
        rclpy.spin_once(navigator, timeout_sec=0.1)

    if navigator.current_pose:
        print(f"Current position: ({navigator.current_pose['x']:.2f}, {navigator.current_pose['y']:.2f})")

        # Get goal from user input
        try:
            goal_x = float(input("Enter goal X coordinate (meters): "))
            goal_y = float(input("Enter goal Y coordinate (meters): "))
        except (ValueError, EOFError, KeyboardInterrupt):
            print("\nInvalid input or interrupted. Exiting...")
            navigator.destroy_node()
            rclpy.shutdown()
            return

        # Use current position as start
        start_x = navigator.current_pose['x']
        start_y = navigator.current_pose['y']

        # Plan and execute navigation
        if navigator.navigate_to_goal(start_x, start_y, goal_x, goal_y):
            print("\nNavigation started! Press Ctrl+C to stop.\n")
            try:
                rclpy.spin(navigator)
            except KeyboardInterrupt:
                print("\nStopping navigation...")
        else:
            print("\nNavigation failed!")

    # Cleanup
    navigator.stop_robot()
    navigator.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
