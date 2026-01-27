#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan

import pygame
import sys

class LidarSubscriber(Node):
    def __init__(self):
        super().__init__('lidar_subscriber')
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.screen_initialized = False

    def scan_callback(self, msg: LaserScan):
        # Clear screen and move cursor to top (ANSI escape codes)
        sys.stdout.write('\033[2J\033[H')

        # Print lidar data information
        print('=== LiDAR Data ===')
        print(f'Number of ranges: {len(msg.ranges)}')
        print(f'Angle min: {msg.angle_min:.2f} rad, max: {msg.angle_max:.2f} rad')
        print(f'Angle increment: {msg.angle_increment:.4f} rad')
        print(f'Range min: {msg.range_min:.2f} m, max: {msg.range_max:.2f} m')

        # Print sample ranges (front, left, right, back)
        num_ranges = len(msg.ranges)
        if num_ranges > 0:
            front = msg.ranges[0] if msg.ranges[0] != float('inf') else 'inf'
            left = msg.ranges[num_ranges // 4] if msg.ranges[num_ranges // 4] != float('inf') else 'inf'
            back = msg.ranges[num_ranges // 2] if msg.ranges[num_ranges // 2] != float('inf') else 'inf'
            right = msg.ranges[3 * num_ranges // 4] if msg.ranges[3 * num_ranges // 4] != float('inf') else 'inf'
            print(f'\nSample ranges:')
            print(f'  Front: {front}')
            print(f'  Left:  {left}')
            print(f'  Back:  {back}')
            print(f'  Right: {right}')

        sys.stdout.flush()

        # Pygame visualization
        if not self.screen_initialized:
            self.initialize_screen(len(msg.ranges))
        for i in range(len(msg.ranges)):
            color = tuple([round((msg.ranges[i] - msg.range_min) / msg.range_max * 255) if msg.ranges[i] < msg.range_max else 255] * 3)
            self.get_logger().info(str(color))
            
            pygame.draw.rect(self.window, color, pygame.rect.Rect(i, 0, 1, 100))
        pygame.display.flip()
        


    def initialize_screen(self, width):
        self.window = pygame.display.set_mode((width, 100))
        pygame.display.set_caption('LiDAR Output')
        self.screen_initialized = True
        

def main():
    pygame.init()
    rclpy.init()
    node = LidarSubscriber()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()
    pygame.quit()


if __name__ == '__main__':
    main()
