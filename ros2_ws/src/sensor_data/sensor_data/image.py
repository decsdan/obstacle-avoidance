#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from cv_bridge import CvBridge
from sensor_msgs.msg import Image

import cv2

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        self.create_subscription(Image, '/oakd/rgb/preview/image_raw', self.image_callback, 10)

        self.bridge = CvBridge()

    def image_callback(self, msg):
        image = self.bridge.imgmsg_to_cv2(msg)
        cv2.imshow('Video Feed', image)
        if cv2.waitKey(25) & 0xFF == ord('q'): 
            self.destroy_node()


def main():
    rclpy.init()
    node = ImageSubscriber()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
    