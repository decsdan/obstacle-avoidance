#!/usr/bin/env python3
"""Republish Create3 ``HazardDetectionVector`` as plain ``Bool`` triggers.

Keeps ``safety_mux_node`` free of ``irobot_create_msgs`` so the mux runs
identically in sim and on hardware. This adapter is the only place in
the stack that depends on the Create3 message package; if ``irobot_create_msgs``
is unavailable (sim-only environments, hardware-less unit tests), the
node logs once and exits cleanly so the launch file does not block.
"""

import sys

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy,
)
from std_msgs.msg import Bool

try:
    from irobot_create_msgs.msg import HazardDetection, HazardDetectionVector
    _CREATE_MSGS_AVAILABLE = True
except ImportError:
    _CREATE_MSGS_AVAILABLE = False


class HazardAdapter(Node):
    """Map ``HazardDetectionVector`` types to bumper / cliff Bool triggers."""

    def __init__(self):
        super().__init__('hazard_adapter')

        self.declare_parameter('namespace', '/don')
        self.declare_parameter('hazard_topic', 'hazard_detection')
        self.ns = self.get_parameter('namespace').value
        hazard_topic = self.get_parameter('hazard_topic').value

        latched = QoSProfile(
            depth=1,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self._bumper_pub = self.create_publisher(
            Bool, f'{self.ns}/bumper_trigger', latched)
        self._cliff_pub = self.create_publisher(
            Bool, f'{self.ns}/cliff_trigger', latched)

        # Publish the initial false state so the mux's latched subscription
        # gets a baseline rather than waiting for the first hazard.
        self._publish(self._bumper_pub, False)
        self._publish(self._cliff_pub, False)

        self.create_subscription(
            HazardDetectionVector,
            f'{self.ns}/{hazard_topic}',
            self._on_hazard,
            10,
        )

        self.get_logger().info(
            f'hazard_adapter ready | ns={self.ns} topic={hazard_topic}')

    def _on_hazard(self, msg: HazardDetectionVector):
        bumper = False
        cliff = False
        for det in msg.detections:
            if det.type == HazardDetection.BUMP:
                bumper = True
            elif det.type == HazardDetection.CLIFF:
                cliff = True
        self._publish(self._bumper_pub, bumper)
        self._publish(self._cliff_pub, cliff)

    def _publish(self, pub, value: bool) -> None:
        msg = Bool()
        msg.data = value
        pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    if not _CREATE_MSGS_AVAILABLE:
        # Stay alive under launch so dependent nodes don't trip on a missing
        # publisher; warn once and idle.
        node = Node('hazard_adapter_disabled')
        node.get_logger().warn(
            'irobot_create_msgs unavailable; hazard_adapter idling. '
            'Bumper/cliff triggers will not be published.')
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        finally:
            node.destroy_node()
            rclpy.shutdown()
        return

    node = HazardAdapter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
