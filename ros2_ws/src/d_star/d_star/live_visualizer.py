#!/usr/bin/env python3

"""
Live Path Visualizer for ROS2 TurtleBot4

This ROS2 node visualizes the robot's real-time path execution, showing:
- The planned path from the D* Lite planner
- The actual traveled path from odometry
- The current robot position
- The static map
- The dynamic grid (with lidar-detected obstacles)
- Raw lidar scan points

Subscribes to:
    - /don/odom (Odometry): Robot position
    - /don/path (Path): Planned path from d_star_nav node
    - /don/dynamic_grid (OccupancyGrid): Dynamic obstacle grid
    - /don/scan (LaserScan): Raw lidar data

Usage:
    ros2 run d_star live_visualizer

Architecture:
    - ROS2 node running in separate thread (rclpy.spin)
    - Matplotlib animation in main thread
    - Thread-safe access to shared data via threading.Lock
    - Real-time visualization updated at 10Hz (100ms interval)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from nav_msgs.msg import Odometry, Path, OccupancyGrid
from sensor_msgs.msg import LaserScan
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
import threading
import yaml
from PIL import Image
import os
import math


# ============================================================================
# CONSTANTS
# ============================================================================

class LiveVisualizerConstants:
    """Configuration constants for the live visualizer"""

    # Visualization Parameters
    FIGURE_SIZE = (14, 10)              # Figure dimensions (width, height)
    UPDATE_INTERVAL_MS = 100            # Visualization update interval (milliseconds)
    MAX_TRAVELED_POINTS = 1000          # Maximum traveled path points to keep

    # Map Display
    MAP_ALPHA = 0.6                     # Map transparency
    DYNAMIC_GRID_ALPHA = 0.4            # Dynamic grid transparency
    GRID_ALPHA = 0.3                    # Grid line transparency

    # Path Line Widths
    PLANNED_PATH_WIDTH = 2              # Width of planned path line
    TRAVELED_PATH_WIDTH = 3             # Width of traveled path line

    # Marker Sizes
    ROBOT_MARKER_SIZE = 15              # Size of robot position marker
    LIDAR_POINT_SIZE = 3                # Size of lidar point markers

    # Colors
    COLOR_PLANNED_PATH = 'b'            # Blue for planned path
    COLOR_TRAVELED_PATH = 'g'           # Green for traveled path
    COLOR_ROBOT = 'r'                   # Red for current robot position
    COLOR_OBSTACLE = [0.3, 0.3, 0.3]    # Dark gray for static obstacles
    COLOR_DYNAMIC_OBSTACLE = [1.0, 0.5, 0.0]  # Orange for dynamic obstacles
    COLOR_LIDAR_POINTS = 'cyan'         # Cyan for lidar points

    # Default Map Path
    DEFAULT_YAML = '~/obstacle-avoidance-comps/ros2_ws/olinmaze.yaml'
    DEFAULT_PGM = '~/obstacle-avoidance-comps/ros2_ws/olinmaze.pgm'

    # ROS2 Topics
    ODOMETRY = '/don/odom'
    PATH = '/don/path'
    DYNAMIC_GRID = '/don/dynamic_grid'
    SCAN = '/don/scan'


# ============================================================================
# LIVE PATH VISUALIZER NODE
# ============================================================================

class LivePathVisualizer(Node):
    """
    ROS2 node for real-time path visualization.

    Displays the planned path, actual traveled path, dynamic obstacles,
    and lidar scan on the static map, allowing real-time monitoring
    of the robot's navigation performance.
    """

    def __init__(self):
        """Initialize the live path visualizer node."""
        super().__init__('live_path_visualizer')

        # ====================================================================
        # INITIALIZATION - Robot State
        # ====================================================================

        self.current_position = None        # Current (x, y) position
        self.current_theta = 0.0            # Current orientation
        self.traveled_path = []             # List of (x, y) points traveled
        self.planned_path = []              # List of (x, y) points from planner

        # ====================================================================
        # INITIALIZATION - Map Data
        # ====================================================================

        self.grid = None                    # Occupancy grid (0=free, 1=occupied)
        self.grid_original = None           # Original grid before any processing
        self.resolution = None              # Meters per grid cell
        self.origin = None                  # Map origin [x, y, theta]

        # Dynamic grid data
        self.dynamic_grid = None            # Dynamic obstacle grid from planner
        self.dynamic_grid_resolution = None
        self.dynamic_grid_origin = None

        # Lidar data
        self.lidar_points = []              # List of (x, y) lidar hit points

        # ====================================================================
        # INITIALIZATION - ROS2 Subscriptions
        # ====================================================================

        # Odometry subscription (best-effort QoS for sensor data)
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.odom_sub = self.create_subscription(
            Odometry,
            LiveVisualizerConstants.ODOMETRY,
            self.odom_callback,
            qos_profile
        )

        # Planned path subscription (reliable QoS)
        self.path_sub = self.create_subscription(
            Path,
            LiveVisualizerConstants.PATH,
            self.path_callback,
            10
        )

        # Dynamic grid subscription
        self.dynamic_grid_sub = self.create_subscription(
            OccupancyGrid,
            LiveVisualizerConstants.DYNAMIC_GRID,
            self.dynamic_grid_callback,
            10
        )

        # Lidar scan subscription
        self.scan_sub = self.create_subscription(
            LaserScan,
            LiveVisualizerConstants.SCAN,
            self.scan_callback,
            qos_profile
        )

        # ====================================================================
        # INITIALIZATION - Map Loading
        # ====================================================================

        yaml_file = os.path.expanduser(LiveVisualizerConstants.DEFAULT_YAML)
        pgm_file = os.path.expanduser(LiveVisualizerConstants.DEFAULT_PGM)
        self.load_map(yaml_file, pgm_file)

        # ====================================================================
        # INITIALIZATION - Visualization
        # ====================================================================

        self.fig, self.ax = plt.subplots(figsize=LiveVisualizerConstants.FIGURE_SIZE)
        self.lock = threading.Lock()        # Thread-safe access to shared data

        self.get_logger().info('Live Path Visualizer initialized')
        self.get_logger().info(f'Subscribing to: {LiveVisualizerConstants.ODOMETRY}, '
                               f'{LiveVisualizerConstants.PATH}, '
                               f'{LiveVisualizerConstants.DYNAMIC_GRID}, '
                               f'{LiveVisualizerConstants.SCAN}')

    # ========================================================================
    # MAP LOADING
    # ========================================================================

    def load_map(self, yaml_file, pgm_file):
        """
        Load static map from YAML and PGM files.

        Loads the map that will be displayed as the background for the
        path visualization.

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
            self.grid_original = self.grid.copy()

            self.get_logger().info(
                f'Map loaded: {self.grid.shape}, resolution: {self.resolution}m'
            )
        except Exception as e:
            self.get_logger().error(f'Failed to load map: {e}')

    # ========================================================================
    # ROS2 CALLBACKS
    # ========================================================================

    def odom_callback(self, msg):
        """
        Callback for odometry updates.

        Tracks the robot's position and appends to the traveled path.
        Maintains a rolling window of recent positions to prevent
        memory growth.

        Args:
            msg: Odometry message with robot pose
        """
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        # Extract yaw from quaternion
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        theta = math.atan2(siny_cosp, cosy_cosp)

        with self.lock:
            self.current_position = (x, y)
            self.current_theta = theta
            self.traveled_path.append((x, y))

            # Keep only most recent points
            if len(self.traveled_path) > LiveVisualizerConstants.MAX_TRAVELED_POINTS:
                self.traveled_path = self.traveled_path[-LiveVisualizerConstants.MAX_TRAVELED_POINTS:]

    def path_callback(self, msg):
        """
        Callback for planned path updates.

        Updates the planned path visualization whenever the planner
        publishes a new path.

        Args:
            msg: Path message with planned waypoints
        """
        with self.lock:
            self.planned_path = [
                (pose.pose.position.x, pose.pose.position.y)
                for pose in msg.poses
            ]
            self.get_logger().info(
                f'Received planned path with {len(self.planned_path)} waypoints'
            )

    def dynamic_grid_callback(self, msg):
        """
        Callback for dynamic grid updates.

        Updates the dynamic obstacle grid visualization showing
        lidar-detected obstacles and SLAM updates.

        Args:
            msg: OccupancyGrid message with dynamic obstacles
        """
        with self.lock:
            width = msg.info.width
            height = msg.info.height
            self.dynamic_grid_resolution = msg.info.resolution
            self.dynamic_grid_origin = [
                msg.info.origin.position.x,
                msg.info.origin.position.y
            ]

            # Convert to numpy array
            data = np.array(msg.data, dtype=np.int8).reshape((height, width))
            self.dynamic_grid = data

    def scan_callback(self, msg):
        """
        Callback for lidar scan updates.

        Converts lidar scan to world coordinates for visualization.

        Args:
            msg: LaserScan message with lidar data
        """
        if self.current_position is None:
            return

        with self.lock:
            robot_x, robot_y = self.current_position
            robot_theta = self.current_theta

            points = []
            angle = msg.angle_min

            for range_val in msg.ranges:
                # Skip invalid readings
                if (range_val >= msg.range_min and
                    range_val <= msg.range_max and
                    not math.isinf(range_val) and
                    not math.isnan(range_val)):

                    # Convert to world coordinates
                    obstacle_x_local = range_val * math.cos(angle)
                    obstacle_y_local = range_val * math.sin(angle)

                    obstacle_x_world = (robot_x +
                                       obstacle_x_local * math.cos(robot_theta) -
                                       obstacle_y_local * math.sin(robot_theta))
                    obstacle_y_world = (robot_y +
                                       obstacle_x_local * math.sin(robot_theta) +
                                       obstacle_y_local * math.cos(robot_theta))

                    points.append((obstacle_x_world, obstacle_y_world))

                angle += msg.angle_increment

            self.lidar_points = points

    # ========================================================================
    # VISUALIZATION
    # ========================================================================

    def update_plot(self, frame):
        """
        Update the visualization plot.

        Called by FuncAnimation to update the display at regular intervals.
        Shows the static map, dynamic grid, lidar points, planned path,
        traveled path, and current robot position.

        Args:
            frame: Animation frame number (from FuncAnimation, unused)
        """
        with self.lock:
            self.ax.clear()

            # ----------------------------------------------------------------
            # STATIC MAP DISPLAY
            # ----------------------------------------------------------------
            if self.grid is not None:
                rows, cols = self.grid.shape
                extent = [
                    self.origin[0],
                    self.origin[0] + cols * self.resolution,
                    self.origin[1],
                    self.origin[1] + rows * self.resolution
                ]

                # Create colored map display
                map_display = np.ones((rows, cols, 3))
                map_display[self.grid == 1] = LiveVisualizerConstants.COLOR_OBSTACLE

                self.ax.imshow(
                    map_display,
                    extent=extent,
                    origin='lower',
                    alpha=LiveVisualizerConstants.MAP_ALPHA
                )

            # ----------------------------------------------------------------
            # DYNAMIC GRID OVERLAY
            # ----------------------------------------------------------------
            if self.dynamic_grid is not None and self.dynamic_grid_resolution is not None:
                rows, cols = self.dynamic_grid.shape
                extent = [
                    self.dynamic_grid_origin[0],
                    self.dynamic_grid_origin[0] + cols * self.dynamic_grid_resolution,
                    self.dynamic_grid_origin[1],
                    self.dynamic_grid_origin[1] + rows * self.dynamic_grid_resolution
                ]

                # Create overlay showing only dynamic obstacles (not static)
                # Compare with static grid to find new obstacles
                dynamic_overlay = np.zeros((rows, cols, 4))  # RGBA

                # Mark dynamic obstacles in orange with transparency
                dynamic_mask = self.dynamic_grid > 50  # Occupied cells
                if self.grid is not None and self.grid.shape == self.dynamic_grid.shape:
                    # Only show cells that are in dynamic but not in static
                    static_mask = self.grid == 1
                    new_obstacles = dynamic_mask & ~static_mask
                else:
                    new_obstacles = dynamic_mask

                dynamic_overlay[new_obstacles] = [1.0, 0.5, 0.0, 0.7]  # Orange with alpha

                self.ax.imshow(
                    dynamic_overlay,
                    extent=extent,
                    origin='lower',
                    alpha=LiveVisualizerConstants.DYNAMIC_GRID_ALPHA
                )

            # ----------------------------------------------------------------
            # LIDAR POINTS
            # ----------------------------------------------------------------
            if self.lidar_points:
                lx = [p[0] for p in self.lidar_points]
                ly = [p[1] for p in self.lidar_points]
                self.ax.scatter(
                    lx, ly,
                    c=LiveVisualizerConstants.COLOR_LIDAR_POINTS,
                    s=LiveVisualizerConstants.LIDAR_POINT_SIZE,
                    alpha=0.6,
                    label=f'Lidar ({len(self.lidar_points)} pts)'
                )

            # ----------------------------------------------------------------
            # PLANNED PATH
            # ----------------------------------------------------------------
            if self.planned_path:
                px = [p[0] for p in self.planned_path]
                py = [p[1] for p in self.planned_path]
                self.ax.plot(
                    px, py,
                    f'{LiveVisualizerConstants.COLOR_PLANNED_PATH}--',
                    linewidth=LiveVisualizerConstants.PLANNED_PATH_WIDTH,
                    alpha=0.7,
                    label='Planned Path'
                )

            # ----------------------------------------------------------------
            # TRAVELED PATH
            # ----------------------------------------------------------------
            if len(self.traveled_path) > 1:
                tx = [p[0] for p in self.traveled_path]
                ty = [p[1] for p in self.traveled_path]
                self.ax.plot(
                    tx, ty,
                    f'{LiveVisualizerConstants.COLOR_TRAVELED_PATH}-',
                    linewidth=LiveVisualizerConstants.TRAVELED_PATH_WIDTH,
                    alpha=0.8,
                    label='Traveled Path'
                )

            # ----------------------------------------------------------------
            # CURRENT ROBOT POSITION
            # ----------------------------------------------------------------
            if self.current_position:
                self.ax.plot(
                    self.current_position[0],
                    self.current_position[1],
                    f'{LiveVisualizerConstants.COLOR_ROBOT}o',
                    markersize=LiveVisualizerConstants.ROBOT_MARKER_SIZE,
                    label='Robot',
                    markeredgecolor='black',
                    markeredgewidth=2
                )

                # Draw heading indicator
                arrow_length = 0.3
                dx = arrow_length * math.cos(self.current_theta)
                dy = arrow_length * math.sin(self.current_theta)
                self.ax.arrow(
                    self.current_position[0],
                    self.current_position[1],
                    dx, dy,
                    head_width=0.1,
                    head_length=0.05,
                    fc='red',
                    ec='black'
                )

            # ----------------------------------------------------------------
            # AXIS CONFIGURATION
            # ----------------------------------------------------------------
            self.ax.set_xlabel('X (meters)')
            self.ax.set_ylabel('Y (meters)')

            # Build status string
            status_parts = [f'Traveled: {len(self.traveled_path)} pts']
            if self.lidar_points:
                status_parts.append(f'Lidar: {len(self.lidar_points)} pts')
            if self.dynamic_grid is not None:
                dynamic_count = np.sum(self.dynamic_grid > 50)
                status_parts.append(f'Dynamic obstacles: {dynamic_count} cells')

            self.ax.set_title(
                f'Live Robot Path Visualization\n({", ".join(status_parts)})'
            )
            self.ax.legend(loc='upper right')
            self.ax.grid(True, alpha=LiveVisualizerConstants.GRID_ALPHA)
            self.ax.set_aspect('equal')

    # ========================================================================
    # MAIN EXECUTION
    # ========================================================================

    def run(self):
        """
        Start the visualization.

        Creates a matplotlib animation that updates the plot at regular
        intervals, showing the real-time robot path execution.
        """
        ani = FuncAnimation(
            self.fig,
            self.update_plot,
            interval=LiveVisualizerConstants.UPDATE_INTERVAL_MS,
            cache_frame_data=False
        )
        plt.show()


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main(args=None):
    """
    Main entry point for the live visualizer.

    Initializes ROS2, creates the visualizer node, starts ROS2 spinning
    in a background thread, and runs the matplotlib visualization in the
    main thread.

    Args:
        args: Command line arguments (passed to rclpy.init)
    """
    rclpy.init(args=args)
    visualizer = LivePathVisualizer()

    # Run ROS2 spinning in separate thread
    spin_thread = threading.Thread(
        target=rclpy.spin,
        args=(visualizer,),
        daemon=True
    )
    spin_thread.start()

    # Run visualization in main thread
    visualizer.run()

    # Cleanup
    visualizer.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
