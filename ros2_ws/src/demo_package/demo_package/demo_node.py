#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class SpinBot(Node):
    def __init__(self):
        super().__init__('spin_bot')
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.spin_robot)
        self.get_logger().info('Spinning node started!')
        
        self.twist = Twist()
        self.twist.linear.x = 0.0
        self.twist.angular.z = 0.5  # Positive = spin left (counterclockwise)

    def spin_robot(self):
        self.publisher_.publish(self.twist)
        self.get_logger().info('Spinning...')

def main(args=None):
    rclpy.init(args=args)
    node = SpinBot()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
