#!/usr/bin/env python3

"""
Live Path Visualizer for ROS2 TurtleBot4

This ROS2 node visualizes the robot's real-time path execution, showing:
- The planned path from the D* Lite planner
- The actual traveled path from odometry
- The current robot position
- The static map

Subscribes to:
    - /sim_ground_truth_pose (Odometry): Ground truth robot position
    - /planned_path (Path): Planned path from d_star_nav node

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
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from nav_msgs.msg import Odometry, Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
import yaml
from PIL import Image
import os


# ============================================================================
# CONSTANTS
# ============================================================================

class LiveVisualizerConstants:
    """Configuration constants for the live visualizer"""

    # Visualization Parameters
    FIGURE_SIZE = (12, 10)              # Figure dimensions (width, height)
    UPDATE_INTERVAL_MS = 100            # Visualization update interval (milliseconds)
    MAX_TRAVELED_POINTS = 1000          # Maximum traveled path points to keep

    # Map Display
    MAP_ALPHA = 0.7                     # Map transparency
    GRID_ALPHA = 0.3                    # Grid line transparency

    # Path Line Widths
    PLANNED_PATH_WIDTH = 2              # Width of planned path line
    TRAVELED_PATH_WIDTH = 3             # Width of traveled path line

    # Marker Sizes
    ROBOT_MARKER_SIZE = 15              # Size of robot position marker

    # Colors
    COLOR_PLANNED_PATH = 'b'            # Blue for planned path
    COLOR_TRAVELED_PATH = 'g'           # Green for traveled path
    COLOR_ROBOT = 'r'                   # Red for current robot position
    COLOR_OBSTACLE = [0.3, 0.3, 0.3]    # Dark gray for obstacles

    # Default Map Path
    DEFAULT_YAML = '~/code/obstacle-avoidance-comps/ros2_ws/maze_slamed.yaml'
    DEFAULT_PGM = '~/code/obstacle-avoidance-comps/ros2_ws/maze_slamed.pgm'


# ============================================================================
# LIVE PATH VISUALIZER NODE
# ============================================================================

class LivePathVisualizer(Node):
    """
    ROS2 node for real-time path visualization.

    Displays the planned path and actual traveled path on the static map,
    allowing real-time monitoring of the robot's navigation performance.
    """

    def __init__(self):
        """Initialize the live path visualizer node."""
        super().__init__('live_path_visualizer')

        # ====================================================================
        # INITIALIZATION - Robot State
        # ====================================================================

        self.current_position = None        # Current (x, y) position
        self.traveled_path = []             # List of (x, y) points traveled
        self.planned_path = []              # List of (x, y) points from planner

        # ====================================================================
        # INITIALIZATION - Map Data
        # ====================================================================

        self.grid = None                    # Occupancy grid (0=free, 1=occupied)
        self.grid_original = None           # Original grid before any processing
        self.resolution = None              # Meters per grid cell
        self.origin = None                  # Map origin [x, y, theta]

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
            '/sim_ground_truth_pose',
            self.odom_callback,
            qos_profile
        )

        # Planned path subscription (reliable QoS)
        self.path_sub = self.create_subscription(
            Path,
            '/planned_path',
            self.path_callback,
            10
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
        self.get_logger().info('Subscribing to /sim_ground_truth_pose and /planned_path')

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

        with self.lock:
            self.current_position = (x, y)
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

    # ========================================================================
    # VISUALIZATION
    # ========================================================================

    def update_plot(self, frame):
        """
        Update the visualization plot.

        Called by FuncAnimation to update the display at regular intervals.
        Shows the static map, planned path, traveled path, and current
        robot position.

        Args:
            frame: Animation frame number (from FuncAnimation, unused)
        """
        with self.lock:
            self.ax.clear()

            # ----------------------------------------------------------------
            # MAP DISPLAY
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
            # PLANNED PATH
            # ----------------------------------------------------------------
            if self.planned_path:
                px = [p[0] for p in self.planned_path]
                py = [p[1] for p in self.planned_path]
                self.ax.plot(
                    px, py,
                    f'{LiveVisualizerConstants.COLOR_PLANNED_PATH}--',
                    linewidth=LiveVisualizerConstants.PLANNED_PATH_WIDTH,
                    alpha=0.5,
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

            # ----------------------------------------------------------------
            # AXIS CONFIGURATION
            # ----------------------------------------------------------------
            self.ax.set_xlabel('X (meters)')
            self.ax.set_ylabel('Y (meters)')
            self.ax.set_title(
                f'Live Robot Path Visualization '
                f'(Traveled: {len(self.traveled_path)} points)'
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
