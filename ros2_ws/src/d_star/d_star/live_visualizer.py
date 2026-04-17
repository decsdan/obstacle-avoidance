#!/usr/bin/env python3
# Originally authored by Devin Dennis as part of the 2025 Carleton Senior
# Capstone Project (see AUTHORS.md). Updated by Daniel Scheider, 2026.
"""Matplotlib-based live visualizer for D* Lite navigation on TurtleBot4.

Subscribes to odometry, planned path, dynamic grid, and lidar scan and
overlays them on the static map. ROS2 spins on a background thread while
matplotlib animation runs on the main thread.
"""

import math
import os
import threading

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image
import yaml

import rclpy
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan


class LiveVisualizerConstants:
    """Default configuration constants for the live visualizer."""

    FIGURE_SIZE = (14, 10)
    UPDATE_INTERVAL_MS = 100
    MAX_TRAVELED_POINTS = 1000

    MAP_ALPHA = 0.6
    DYNAMIC_GRID_ALPHA = 0.4
    GRID_ALPHA = 0.3

    PLANNED_PATH_WIDTH = 2
    TRAVELED_PATH_WIDTH = 3

    ROBOT_MARKER_SIZE = 15
    LIDAR_POINT_SIZE = 3

    COLOR_PLANNED_PATH = 'b'
    COLOR_TRAVELED_PATH = 'g'
    COLOR_ROBOT = 'r'
    COLOR_OBSTACLE = [0.3, 0.3, 0.3]
    COLOR_DYNAMIC_OBSTACLE = [1.0, 0.5, 0.0]
    COLOR_LIDAR_POINTS = 'cyan'

    DEFAULT_YAML = '~/obstacle-avoidance-comps/ros2_ws/olinmaze.yaml'
    DEFAULT_PGM = '~/obstacle-avoidance-comps/ros2_ws/olinmaze.pgm'

    ODOMETRY = '/don/odom'
    PATH = '/don/d_star/plan'
    DYNAMIC_GRID = '/don/dynamic_grid'
    SCAN = '/don/scan'


class LivePathVisualizer(Node):
    """ROS2 node for real-time D* Lite path visualization."""

    def __init__(self):
        super().__init__('live_path_visualizer')

        self.current_position = None
        self.current_theta = 0.0
        self.traveled_path = []
        self.planned_path = []

        self.grid = None
        self.grid_original = None
        self.resolution = None
        self.origin = None

        self.dynamic_grid = None
        self.dynamic_grid_resolution = None
        self.dynamic_grid_origin = None

        self.lidar_points = []

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.odom_sub = self.create_subscription(
            Odometry,
            LiveVisualizerConstants.ODOMETRY,
            self.odom_callback,
            qos_profile,
        )
        self.path_sub = self.create_subscription(
            Path,
            LiveVisualizerConstants.PATH,
            self.path_callback,
            10,
        )
        self.dynamic_grid_sub = self.create_subscription(
            OccupancyGrid,
            LiveVisualizerConstants.DYNAMIC_GRID,
            self.dynamic_grid_callback,
            10,
        )
        self.scan_sub = self.create_subscription(
            LaserScan,
            LiveVisualizerConstants.SCAN,
            self.scan_callback,
            qos_profile,
        )

        yaml_file = os.path.expanduser(LiveVisualizerConstants.DEFAULT_YAML)
        pgm_file = os.path.expanduser(LiveVisualizerConstants.DEFAULT_PGM)
        self.load_map(yaml_file, pgm_file)

        self.fig, self.ax = plt.subplots(figsize=LiveVisualizerConstants.FIGURE_SIZE)
        self.lock = threading.Lock()

        self.get_logger().info(
            f'live_path_visualizer initialized | subs={LiveVisualizerConstants.ODOMETRY}, '
            f'{LiveVisualizerConstants.PATH}, {LiveVisualizerConstants.DYNAMIC_GRID}, '
            f'{LiveVisualizerConstants.SCAN}')

    def load_map(self, yaml_file, pgm_file):
        """Load static map from YAML metadata and PGM image."""
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

            self.get_logger().info(
                f'Map loaded: {self.grid.shape}, resolution: {self.resolution}m')
        except Exception as e:
            self.get_logger().error(f'Failed to load map: {e}')

    def odom_callback(self, msg):
        """Track robot pose and append to the traveled path."""
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        theta = math.atan2(siny_cosp, cosy_cosp)

        with self.lock:
            self.current_position = (x, y)
            self.current_theta = theta
            self.traveled_path.append((x, y))

            max_points = LiveVisualizerConstants.MAX_TRAVELED_POINTS
            if len(self.traveled_path) > max_points:
                self.traveled_path = self.traveled_path[-max_points:]

    def path_callback(self, msg):
        """Update the planned path from the D* Lite planner."""
        with self.lock:
            self.planned_path = [
                (pose.pose.position.x, pose.pose.position.y)
                for pose in msg.poses
            ]
            self.get_logger().info(
                f'Received planned path with {len(self.planned_path)} waypoints')

    def dynamic_grid_callback(self, msg):
        """Update the dynamic obstacle grid."""
        with self.lock:
            width = msg.info.width
            height = msg.info.height
            self.dynamic_grid_resolution = msg.info.resolution
            self.dynamic_grid_origin = [
                msg.info.origin.position.x,
                msg.info.origin.position.y,
            ]

            data = np.array(msg.data, dtype=np.int8).reshape((height, width))
            self.dynamic_grid = data

    def scan_callback(self, msg):
        """Convert lidar scan to world-frame points for display."""
        if self.current_position is None:
            return

        with self.lock:
            robot_x, robot_y = self.current_position
            robot_theta = self.current_theta

            points = []
            angle = msg.angle_min

            for range_val in msg.ranges:
                if (range_val >= msg.range_min and
                        range_val <= msg.range_max and
                        not math.isinf(range_val) and
                        not math.isnan(range_val)):
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

    def update_plot(self, frame):
        """Redraw the map, dynamic grid, lidar points, paths, and robot pose."""
        with self.lock:
            self.ax.clear()

            if self.grid is not None:
                rows, cols = self.grid.shape
                extent = [
                    self.origin[0],
                    self.origin[0] + cols * self.resolution,
                    self.origin[1],
                    self.origin[1] + rows * self.resolution,
                ]

                map_display = np.ones((rows, cols, 3))
                map_display[self.grid == 1] = LiveVisualizerConstants.COLOR_OBSTACLE

                self.ax.imshow(
                    map_display,
                    extent=extent,
                    origin='lower',
                    alpha=LiveVisualizerConstants.MAP_ALPHA,
                )

            if self.dynamic_grid is not None and self.dynamic_grid_resolution is not None:
                rows, cols = self.dynamic_grid.shape
                extent = [
                    self.dynamic_grid_origin[0],
                    self.dynamic_grid_origin[0] + cols * self.dynamic_grid_resolution,
                    self.dynamic_grid_origin[1],
                    self.dynamic_grid_origin[1] + rows * self.dynamic_grid_resolution,
                ]

                dynamic_overlay = np.zeros((rows, cols, 4))

                dynamic_mask = self.dynamic_grid > 50
                if self.grid is not None and self.grid.shape == self.dynamic_grid.shape:
                    static_mask = self.grid == 1
                    new_obstacles = dynamic_mask & ~static_mask
                else:
                    new_obstacles = dynamic_mask

                dynamic_overlay[new_obstacles] = [1.0, 0.5, 0.0, 0.7]

                self.ax.imshow(
                    dynamic_overlay,
                    extent=extent,
                    origin='lower',
                    alpha=LiveVisualizerConstants.DYNAMIC_GRID_ALPHA,
                )

            if self.lidar_points:
                lx = [p[0] for p in self.lidar_points]
                ly = [p[1] for p in self.lidar_points]
                self.ax.scatter(
                    lx, ly,
                    c=LiveVisualizerConstants.COLOR_LIDAR_POINTS,
                    s=LiveVisualizerConstants.LIDAR_POINT_SIZE,
                    alpha=0.6,
                    label=f'Lidar ({len(self.lidar_points)} pts)',
                )

            if self.planned_path:
                px = [p[0] for p in self.planned_path]
                py = [p[1] for p in self.planned_path]
                self.ax.plot(
                    px, py,
                    f'{LiveVisualizerConstants.COLOR_PLANNED_PATH}--',
                    linewidth=LiveVisualizerConstants.PLANNED_PATH_WIDTH,
                    alpha=0.7,
                    label='Planned Path',
                )

            if len(self.traveled_path) > 1:
                tx = [p[0] for p in self.traveled_path]
                ty = [p[1] for p in self.traveled_path]
                self.ax.plot(
                    tx, ty,
                    f'{LiveVisualizerConstants.COLOR_TRAVELED_PATH}-',
                    linewidth=LiveVisualizerConstants.TRAVELED_PATH_WIDTH,
                    alpha=0.8,
                    label='Traveled Path',
                )

            if self.current_position:
                self.ax.plot(
                    self.current_position[0],
                    self.current_position[1],
                    f'{LiveVisualizerConstants.COLOR_ROBOT}o',
                    markersize=LiveVisualizerConstants.ROBOT_MARKER_SIZE,
                    label='Robot',
                    markeredgecolor='black',
                    markeredgewidth=2,
                )

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
                    ec='black',
                )

            self.ax.set_xlabel('X (meters)')
            self.ax.set_ylabel('Y (meters)')

            status_parts = [f'Traveled: {len(self.traveled_path)} pts']
            if self.lidar_points:
                status_parts.append(f'Lidar: {len(self.lidar_points)} pts')
            if self.dynamic_grid is not None:
                dynamic_count = np.sum(self.dynamic_grid > 50)
                status_parts.append(f'Dynamic obstacles: {dynamic_count} cells')

            self.ax.set_title(
                f'Live Robot Path Visualization\n({", ".join(status_parts)})')
            self.ax.legend(loc='upper right')
            self.ax.grid(True, alpha=LiveVisualizerConstants.GRID_ALPHA)
            self.ax.set_aspect('equal')

    def run(self):
        """Start the matplotlib animation loop."""
        ani = FuncAnimation(
            self.fig,
            self.update_plot,
            interval=LiveVisualizerConstants.UPDATE_INTERVAL_MS,
            cache_frame_data=False,
        )
        plt.show()


def main(args=None):
    """Main entry point for the live visualizer."""
    rclpy.init(args=args)
    visualizer = LivePathVisualizer()

    spin_thread = threading.Thread(
        target=rclpy.spin,
        args=(visualizer,),
        daemon=True,
    )
    spin_thread.start()

    visualizer.run()

    visualizer.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
