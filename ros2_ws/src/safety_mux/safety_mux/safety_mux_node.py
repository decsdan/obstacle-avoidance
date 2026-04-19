#!/usr/bin/env python3
"""cmd_vel multiplexer and safety override.

Sits between the local planner and the robot base: passes ``cmd_vel_raw``
through to ``cmd_vel`` until a safety trigger fires, at which point it
latches a zero-velocity command and holds until explicitly cleared via
the ``safety/clear`` service.

Safety triggers:

* ``bumper_trigger`` / ``cliff_trigger`` (``std_msgs/Bool``): a thin
  adapter republishes Create3's ``HazardDetectionVector`` as these two
  booleans. Keeps this node free of ``irobot_create_msgs`` so it runs
  identically in sim and on hardware.
* Sensor-staleness watchdog: if ``/scan`` or ``/odom`` stops arriving
  within ``stale_timeout`` seconds of being seen at least once, the mux
  latches and publishes ``safety/latched`` (without asserting
  ``conditions/collision``).

The orchestrator observes ``safety/latched`` and ``conditions/collision``
but never mediates the override path. The zero-velocity command is
published from this node directly; nav_server is not in the loop.
"""

import rclpy
from geometry_msgs.msg import Twist, TwistStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool
from std_srvs.srv import Trigger


class SafetyMux(Node):
    """cmd_vel multiplexer with bumper/cliff/staleness override."""

    def __init__(self):
        super().__init__('safety_mux')

        self.declare_parameter('namespace', '/don')
        self.declare_parameter('stale_timeout', 0.5)
        self.declare_parameter('publish_zero_hz', 20.0)
        self.declare_parameter('input_is_stamped', True)

        self.ns = self.get_parameter('namespace').value
        self.stale_timeout = float(self.get_parameter('stale_timeout').value)
        publish_zero_hz = float(self.get_parameter('publish_zero_hz').value)
        self.input_is_stamped = bool(self.get_parameter('input_is_stamped').value)

        self._latched = False
        self._collision = False
        self._reason = ''
        self._scan_last_stamp = None
        self._odom_last_stamp = None

        self._twist_out_stamped = self.create_publisher(
            TwistStamped, f'{self.ns}/cmd_vel',
            QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=10))

        latched_qos = QoSProfile(
            depth=1,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._latched_pub = self.create_publisher(
            Bool, f'{self.ns}/safety/latched', latched_qos)
        self._collision_pub = self.create_publisher(
            Bool, f'{self.ns}/conditions/collision', latched_qos)
        self._publish_latched()
        self._publish_collision()

        if self.input_is_stamped:
            self.create_subscription(
                TwistStamped, f'{self.ns}/cmd_vel_raw',
                self._on_twist_stamped,
                QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=10))
        else:
            self.create_subscription(
                Twist, f'{self.ns}/cmd_vel_raw',
                self._on_twist,
                QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=10))

        self.create_subscription(
            Bool, f'{self.ns}/bumper_trigger',
            self._on_bumper, latched_qos)
        self.create_subscription(
            Bool, f'{self.ns}/cliff_trigger',
            self._on_cliff, latched_qos)

        self.create_subscription(
            LaserScan, f'{self.ns}/scan',
            self._on_scan, qos_profile_sensor_data)
        self.create_subscription(
            Odometry, f'{self.ns}/odom',
            self._on_odom, qos_profile_sensor_data)

        self.create_service(
            Trigger, f'{self.ns}/safety/clear', self._handle_clear)

        self._watchdog_timer = self.create_timer(0.1, self._check_staleness)
        self._zero_timer = self.create_timer(
            1.0 / publish_zero_hz, self._publish_zero_if_latched)

        self.get_logger().info(
            f'safety_mux ready | ns={self.ns} '
            f'stale_timeout={self.stale_timeout:.2f}s '
            f'input_stamped={self.input_is_stamped}')

    def _on_twist_stamped(self, msg: TwistStamped):
        """Pass through unless latched; latched state publishes zeros itself."""
        if self._latched:
            return
        self._twist_out_stamped.publish(msg)

    def _on_twist(self, msg: Twist):
        if self._latched:
            return
        out = TwistStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = 'base_link'
        out.twist = msg
        self._twist_out_stamped.publish(out)

    def _on_bumper(self, msg: Bool):
        if msg.data:
            self._latch(reason='bumper', is_collision=True)

    def _on_cliff(self, msg: Bool):
        if msg.data:
            self._latch(reason='cliff', is_collision=True)

    def _on_scan(self, msg: LaserScan):
        self._scan_last_stamp = self.get_clock().now()

    def _on_odom(self, msg: Odometry):
        self._odom_last_stamp = self.get_clock().now()

    def _check_staleness(self):
        """Watchdog: latch when /scan or /odom stops arriving."""
        if self._latched:
            return
        now = self.get_clock().now()
        if self._scan_last_stamp is not None:
            dt = (now - self._scan_last_stamp).nanoseconds / 1e9
            if dt > self.stale_timeout:
                self._latch(reason='scan_stale', is_collision=False)
                return
        if self._odom_last_stamp is not None:
            dt = (now - self._odom_last_stamp).nanoseconds / 1e9
            if dt > self.stale_timeout:
                self._latch(reason='odom_stale', is_collision=False)

    def _latch(self, reason: str, is_collision: bool):
        if self._latched and self._collision == is_collision:
            return
        self._latched = True
        self._collision = is_collision
        self._reason = reason
        self.get_logger().warn(f'[safety] latched: {reason}')
        self._publish_latched()
        self._publish_collision()
        self._publish_zero()

    def _publish_zero_if_latched(self):
        if self._latched:
            self._publish_zero()

    def _publish_zero(self):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        self._twist_out_stamped.publish(msg)

    def _publish_latched(self):
        msg = Bool()
        msg.data = self._latched
        self._latched_pub.publish(msg)

    def _publish_collision(self):
        msg = Bool()
        msg.data = self._collision
        self._collision_pub.publish(msg)

    def _handle_clear(self, _request, response):
        """Unlatch and resume pass-through.

        The caller is responsible for confirming the hazard has physically
        cleared; the mux does not infer this on its own.
        """
        was_latched = self._latched
        self._latched = False
        self._collision = False
        self._reason = ''
        self._publish_latched()
        self._publish_collision()
        response.success = True
        response.message = ('cleared' if was_latched else 'no-op')
        self.get_logger().info(f'[safety] clear requested ({response.message})')
        return response


def main(args=None):
    rclpy.init(args=args)
    node = SafetyMux()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
