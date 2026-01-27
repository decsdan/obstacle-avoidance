#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import math


class JPSScriber(Node):
    def __init__(self):
        super().__init__('jps_scriber')
        
        # Subscribe to odometry topic
        self.subscription = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )
        
        self.get_logger().info('JPS Position Subscriber started')
        self.get_logger().info('Listening to /odom topic...\n')
    
    def odom_callback(self, msg):
        """Callback function that receives odometry messages"""
        # Extract position
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.position.z
        
        # Extract orientation (quaternion)
        orientation = msg.pose.pose.orientation
        
        # Convert quaternion to yaw (rotation around z-axis)
        yaw = self.quaternion_to_yaw(orientation)
        yaw_degrees = math.degrees(yaw)
        
        # Print current position (overwrites same line)
        print(f'\rPosition: X={x:7.3f}m  Y={y:7.3f}m  Z={z:7.3f}m  Yaw={yaw_degrees:7.2f}°', end='', flush=True)
    
    @staticmethod
    def quaternion_to_yaw(q):
        """Convert quaternion to yaw angle in radians"""
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)


def main(args=None):
    rclpy.init(args=args)
    
    jps_scriber = JPSScriber()
    
    try:
        rclpy.spin(jps_scriber)
    except KeyboardInterrupt:
        print('\n\nShutting down...')
    finally:
        jps_scriber.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()