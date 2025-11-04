import rclpy
from rclpy.node import Node
import rclpy.logging

from sensor_msgs.msg import Joy
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry

class ControlNode(Node):
    def __init__(self):
        super().__init__('control_node')
        self.create_subscription(Joy, '/joy', self.joy_callback, 5)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.control_publisher = self.create_publisher(TwistStamped, '/cmd_vel', 5)

    def joy_callback(self, data):
        controls = TwistStamped()
        time = self.get_clock().now().seconds_nanoseconds()
        controls.header.stamp.sec = time[0]
        controls.header.stamp.nanosec = time[1]
        controls.twist.linear.x = data.axes[1]
        controls.twist.angular.z = data.axes[3]
        
        # Print joystick data
        self.get_logger().info(f'Joystick Input - Axes: {data.axes}')
        self.get_logger().info(f'Linear X: {controls.twist.linear.x}, Angular Z: {controls.twist.angular.z}')
        
        self.control_publisher.publish(controls)
    
    def odom_callback(self, msg: Odometry):
        # Extract position
        pos = msg.pose.pose.position
        # Print position data
        self.get_logger().info(f'Robot Position - X: {pos.x:.2f}, Y: {pos.y:.2f}, Z: {pos.z:.2f}')


def main():
    rclpy.init()
    node = ControlNode()
    print("hello")

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()