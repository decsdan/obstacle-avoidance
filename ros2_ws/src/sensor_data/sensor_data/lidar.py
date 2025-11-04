#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan

import pygame

class LidarSubscriber(Node):
    def __init__(self):
        super().__init__('lidar_subscriber')
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.screen_initialized = False

    def scan_callback(self, msg: LaserScan):
        if not self.screen_initialized:
            self.initialize_screen(len(msg.ranges))
        for i in range(len(msg.ranges)):
            color = tuple([round((msg.ranges[i] - msg.range_min) / msg.range_max * 255) if msg.ranges[i] < msg.range_max else 255] * 3)
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
