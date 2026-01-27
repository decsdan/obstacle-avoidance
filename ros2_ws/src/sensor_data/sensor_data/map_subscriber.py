#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from nav_msgs.msg import OccupancyGrid, Odometry
import numpy as np
import sys
import math


class MapSubscriber(Node):
    def __init__(self):
        super().__init__('map_subscriber')

        # Use TRANSIENT_LOCAL durability to get the latest map even if published before subscription
        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            map_qos
        )

        # Odometry subscriber for robot position
        odom_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.create_subscription(
            Odometry,
            '/sim_ground_truth_pose',
            self.odom_callback,
            odom_qos
        )

        self.map_count = 0
        self.robot_pose = None  # Store robot position and orientation
        self.map_resolution = None
        self.map_origin = None
        # Don't use logger to avoid interfering with terminal output

    def odom_callback(self, msg: Odometry):
        """Update robot pose from odometry"""
        self.robot_pose = {
            'x': msg.pose.pose.position.x,
            'y': msg.pose.pose.position.y,
            'theta': self.quaternion_to_yaw(msg.pose.pose.orientation)
        }

    @staticmethod
    def quaternion_to_yaw(q):
        """Convert quaternion to yaw angle"""
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def world_to_grid(self, x, y):
        """Convert world coordinates to grid indices"""
        if self.map_resolution is None or self.map_origin is None:
            return None
        grid_x = int((x - self.map_origin[0]) / self.map_resolution)
        grid_y = int((y - self.map_origin[1]) / self.map_resolution)
        return (grid_x, grid_y)

    def get_direction_symbol(self, theta):
        """Get arrow symbol based on heading angle"""
        # Normalize angle to 0-2π
        angle = theta % (2 * math.pi)
        # Convert to degrees
        degrees = math.degrees(angle)

        # 8 directions
        if degrees < 22.5 or degrees >= 337.5:
            return '→'  # East
        elif degrees < 67.5:
            return '↗'  # Northeast
        elif degrees < 112.5:
            return '↑'  # North
        elif degrees < 157.5:
            return '↖'  # Northwest
        elif degrees < 202.5:
            return '←'  # West
        elif degrees < 247.5:
            return '↙'  # Southwest
        elif degrees < 292.5:
            return '↓'  # South
        else:
            return '↘'  # Southeast

    def map_callback(self, msg: OccupancyGrid):
        """Process and display SLAM map data"""
        self.map_count += 1

        # Clear screen and move cursor to top (ANSI escape codes)
        sys.stdout.write('\033[2J\033[H')

        # Extract map metadata
        width = msg.info.width
        height = msg.info.height
        resolution = msg.info.resolution
        origin_x = msg.info.origin.position.x
        origin_y = msg.info.origin.position.y

        # Store for coordinate conversion
        self.map_resolution = resolution
        self.map_origin = [origin_x, origin_y]

        # Convert to numpy array
        map_data = np.array(msg.data, dtype=np.int8)

        # Calculate statistics
        total_cells = len(map_data)
        free_cells = np.sum(map_data == 0)
        occupied_cells = np.sum(map_data == 100)
        unknown_cells = np.sum(map_data == -1)
        partial_occupied = np.sum((map_data > 0) & (map_data < 100))

        # Calculate percentages
        free_percent = (free_cells / total_cells) * 100
        occupied_percent = (occupied_cells / total_cells) * 100
        unknown_percent = (unknown_cells / total_cells) * 100
        partial_percent = (partial_occupied / total_cells) * 100

        # Get min/max occupied values (excluding -1 unknown)
        known_values = map_data[map_data >= 0]
        min_occupied = np.min(known_values) if len(known_values) > 0 else 0
        max_occupied = np.max(known_values) if len(known_values) > 0 else 0

        # Display map information
        print('=' * 60)
        print('SLAM MAP DATA')
        print('=' * 60)
        print(f'Update #{self.map_count}')
        print()

        print('Map Dimensions:')
        print(f'  Width:  {width} cells')
        print(f'  Height: {height} cells')
        print(f'  Total:  {total_cells:,} cells')
        print(f'  Resolution: {resolution:.4f} m/cell')
        print()

        print('Map Origin (world coordinates):')
        print(f'  X: {origin_x:.4f} m')
        print(f'  Y: {origin_y:.4f} m')
        print()

        print('Map Bounds (world coordinates):')
        print(f'  X: [{origin_x:.2f}, {origin_x + width * resolution:.2f}] m')
        print(f'  Y: [{origin_y:.2f}, {origin_y + height * resolution:.2f}] m')
        print()

        # Robot position
        if self.robot_pose:
            robot_grid = self.world_to_grid(self.robot_pose['x'], self.robot_pose['y'])
            direction_symbol = self.get_direction_symbol(self.robot_pose['theta'])
            print('Robot Position:')
            print(f'  World:  ({self.robot_pose["x"]:7.2f}, {self.robot_pose["y"]:7.2f}) m')
            if robot_grid:
                print(f'  Grid:   ({robot_grid[0]:4}, {robot_grid[1]:4}) cells')
            print(f'  Heading: {math.degrees(self.robot_pose["theta"]):6.1f}° {direction_symbol}')
            print()
        else:
            print('Robot Position: [Waiting for odometry...]')
            print()

        print('Occupancy Statistics:')
        print(f'  Free (0):           {free_cells:8,} cells ({free_percent:5.2f}%)')
        print(f'  Occupied (100):     {occupied_cells:8,} cells ({occupied_percent:5.2f}%)')
        print(f'  Partial (1-99):     {partial_occupied:8,} cells ({partial_percent:5.2f}%)')
        print(f'  Unknown (-1):       {unknown_cells:8,} cells ({unknown_percent:5.2f}%)')
        print()

        print('Occupancy Value Range:')
        print(f'  Min: {min_occupied}')
        print(f'  Max: {max_occupied}')
        print()

        # Show ASCII visualization centered on robot (or map center if no robot pose)
        # Reshape to 2D grid
        grid = map_data.reshape((height, width))

        section_height = min(20, height)
        section_width = min(40, width)

        # Center on robot if available, otherwise center of map
        if self.robot_pose:
            robot_grid = self.world_to_grid(self.robot_pose['x'], self.robot_pose['y'])
            if robot_grid:
                center_x, center_y = robot_grid
                print(f'Map Visualization (40x20 section centered on ROBOT):')
            else:
                center_x = width // 2
                center_y = height // 2
                print(f'Map Visualization (40x20 section - map center):')
        else:
            center_x = width // 2
            center_y = height // 2
            print(f'Map Visualization (40x20 section - map center):')

        print('  Legend: . = free, # = occupied, ? = unknown, o = partial, R = robot')
        print()

        start_y = max(0, center_y - section_height // 2)
        end_y = min(height, start_y + section_height)
        start_x = max(0, center_x - section_width // 2)
        end_x = min(width, start_x + section_width)

        # Show coordinates of displayed section
        section_world_x_min = start_x * resolution + origin_x
        section_world_x_max = end_x * resolution + origin_x
        section_world_y_min = start_y * resolution + origin_y
        section_world_y_max = end_y * resolution + origin_y
        print(f'  Section bounds: X=[{section_world_x_min:.2f}, {section_world_x_max:.2f}]m, Y=[{section_world_y_min:.2f}, {section_world_y_max:.2f}]m')
        print(f'  Section grid:   X=[{start_x}, {end_x}], Y=[{start_y}, {end_y}]')
        print()

        section = grid[start_y:end_y, start_x:end_x]

        # Draw the map with robot position
        for row_idx, row in enumerate(section):
            line = ''
            for col_idx, cell in enumerate(row):
                # Calculate actual grid position
                actual_y = start_y + row_idx
                actual_x = start_x + col_idx

                # Check if this is robot position
                is_robot = False
                if self.robot_pose and robot_grid:
                    if actual_x == robot_grid[0] and actual_y == robot_grid[1]:
                        is_robot = True

                if is_robot:
                    # Show robot with direction indicator
                    direction_symbol = self.get_direction_symbol(self.robot_pose['theta'])
                    line += direction_symbol
                elif cell == -1:
                    line += '?'
                elif cell == 0:
                    line += '.'
                elif cell == 100:
                    line += '#'
                else:
                    line += 'o'
            print('  ' + line)

        print()
        print('=' * 60)

        sys.stdout.flush()


def main(args=None):
    rclpy.init(args=args)
    node = MapSubscriber()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
