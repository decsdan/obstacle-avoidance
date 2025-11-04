#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class MoveAndSpin(Node):
    def __init__(self):
        super().__init__('move_and_spin')
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.timer_period = 0.1  # seconds
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        self.get_logger().info('Move-and-spin node started!')

        # Create a Twist message for forward + rotational motion
        self.twist = Twist()
        self.twist.linear.x = 0.5   # forward speed (m/s)
        self.twist.angular.z = 0.9    # rotation speed (rad/s)

    def timer_callback(self):
        # Publish the twist command repeatedly
        self.publisher_.publish(self.twist)
        self.get_logger().info(f"Moving forward at {self.twist.linear.x} m/s and spinning at {self.twist.angular.z} rad/s")

def main(args=None):
    rclpy.init(args=args)
    node = MoveAndSpin()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
