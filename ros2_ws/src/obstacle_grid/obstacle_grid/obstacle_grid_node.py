#!/usr/bin/env python3
"""Shared LIDAR obstacle grid node."""
import math
import sys

import numpy as np

import rclpy
from nav_interfaces.srv import GetGridSnapshot
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import LaserScan
from tf2_ros import (
    Buffer,
    ConnectivityException,
    ExtrapolationException,
    LookupException,
    TransformListener,
)

from obstacle_grid import log_odds
from obstacle_grid.distance_grid import inflate_binary


class ObstacleGridNode(Node):
    """Shared log-odds occupancy grid for global and local planners."""

    def __init__(self):
        super().__init__('obstacle_grid_node')

        self.declare_parameter('namespace', '/don')
        self.ns = self.get_parameter('namespace').value

        self.declare_parameter('grid_width', 200)
        self.declare_parameter('grid_height', 200)
        self.declare_parameter('grid_resolution', 0.05)
        self.declare_parameter('grid_origin_x', -5.0)
        self.declare_parameter('grid_origin_y', -5.0)
        self.declare_parameter('use_slam_dimensions', True)

        self.declare_parameter('max_lidar_range', 8.0)
        self.declare_parameter('min_lidar_range', 0.05)
        self.declare_parameter('lidar_downsample', 2)

        self.declare_parameter('robot_radius', 0.22)
        self.declare_parameter('safety_clearance', 0.05)
        self.declare_parameter('publish_rate', 10.0)

        # Log-odds inverse sensor model
        self.declare_parameter('l_free', -0.4)
        self.declare_parameter('l_occ', 0.85)
        self.declare_parameter('l_clamp', 5.0)
        self.declare_parameter('sensor_noise_sigma', 0.02)
        # NOTE: decay_tau < 5s causes flickering on the real robot; 30s is a safe default
        self.declare_parameter('decay_tau_seconds', 30.0)
        self.declare_parameter('occ_threshold_log_odds', 0.4)

        self.max_range = self.get_parameter('max_lidar_range').value
        self.min_range = self.get_parameter('min_lidar_range').value
        self.downsample = self.get_parameter('lidar_downsample').value
        self.publish_rate = self.get_parameter('publish_rate').value

        robot_radius = self.get_parameter('robot_radius').value
        safety_clearance = self.get_parameter('safety_clearance').value
        self.resolution = self.get_parameter('grid_resolution').value
        self.inflation_cells = int(
            (robot_radius + safety_clearance) / self.resolution)

        self.l_free = float(self.get_parameter('l_free').value)
        self.l_occ = float(self.get_parameter('l_occ').value)
        self.l_clamp = float(self.get_parameter('l_clamp').value)
        self.sensor_noise_sigma = float(
            self.get_parameter('sensor_noise_sigma').value)
        self.decay_tau = float(self.get_parameter('decay_tau_seconds').value)
        self.occ_threshold = float(
            self.get_parameter('occ_threshold_log_odds').value)

        self.log_odds = None
        self.grid_width = 0
        self.grid_height = 0
        self.origin_x = 0.0
        self.origin_y = 0.0
        self._grid_initialized = False
        self._last_update_time = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.scan_msg = None
        self.create_subscription(
            LaserScan,
            f'{self.ns}/scan',
            self._scan_callback,
            qos_profile_sensor_data,
        )

        self._slam_initialized = False
        if self.get_parameter('use_slam_dimensions').value:
            map_qos = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
            )
            self.create_subscription(
                OccupancyGrid,
                f'{self.ns}/map',
                self._map_callback,
                map_qos,
            )
        else:
            self._init_grid_from_params()

        self.grid_pub = self.create_publisher(
            OccupancyGrid, f'{self.ns}/obstacle_grid', 10)
        self.raw_grid_pub = self.create_publisher(
            OccupancyGrid, f'{self.ns}/obstacle_grid_raw', 10)

        self._latest_raw_msg = None
        self._latest_inflated_msg = None

        self.create_service(
            GetGridSnapshot,
            f'{self.ns}/get_grid_snapshot',
            self._handle_get_snapshot,
        )

        self.create_timer(1.0 / self.publish_rate, self._update_cycle)

        self.get_logger().info(
            f'obstacle_grid (log-odds) ready | ns={self.ns} '
            f'res={self.resolution}m inflation={self.inflation_cells}cells '
            f'l_free={self.l_free} l_occ={self.l_occ} '
            f'sigma={self.sensor_noise_sigma}m tau={self.decay_tau}s')

    def _init_grid_from_params(self):
        self.grid_width = self.get_parameter('grid_width').value
        self.grid_height = self.get_parameter('grid_height').value
        self.resolution = self.get_parameter('grid_resolution').value
        self.origin_x = self.get_parameter('grid_origin_x').value
        self.origin_y = self.get_parameter('grid_origin_y').value
        self._allocate_grid()

    def _map_callback(self, msg: OccupancyGrid):
        # one-shot: adopt SLAM map dims on first /map message
        if self._slam_initialized:
            return
        self._slam_initialized = True
        self.grid_width = msg.info.width
        self.grid_height = msg.info.height
        self.resolution = msg.info.resolution
        self.origin_x = msg.info.origin.position.x
        self.origin_y = msg.info.origin.position.y
        self.inflation_cells = int(
            (self.get_parameter('robot_radius').value +
             self.get_parameter('safety_clearance').value) / self.resolution)
        self._allocate_grid()
        self.get_logger().info(
            f'Adopted SLAM map dims: {self.grid_width}x{self.grid_height} '
            f'res={self.resolution}m inflation={self.inflation_cells}cells')

    def _allocate_grid(self):
        self.log_odds = np.zeros(
            (self.grid_height, self.grid_width), dtype=np.float32)
        self._grid_initialized = True

    def _scan_callback(self, msg: LaserScan):
        self.scan_msg = msg

    def _update_cycle(self):
        """Apply latest scan, decay, threshold + inflate, publish."""
        if not self._grid_initialized or self.scan_msg is None:
            return

        sensor_frame = self.scan_msg.header.frame_id
        try:
            tf = self.tf_buffer.lookup_transform(
                'map', sensor_frame, rclpy.time.Time())
        except (LookupException, ConnectivityException, ExtrapolationException):
            return

        sensor_x = tf.transform.translation.x
        sensor_y = tf.transform.translation.y
        q = tf.transform.rotation
        sensor_yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z))

        ranges = np.array(self.scan_msg.ranges)[::self.downsample]
        beam_angles = np.linspace(
            self.scan_msg.angle_min,
            self.scan_msg.angle_max,
            len(self.scan_msg.ranges),
        )[::self.downsample]
        world_angles = beam_angles + sensor_yaw

        now = self.get_clock().now().nanoseconds / 1e9
        if self._last_update_time is not None:
            log_odds.decay(
                self.log_odds, now - self._last_update_time, self.decay_tau)
        self._last_update_time = now

        log_odds.apply_scan(
            self.log_odds,
            ranges,
            world_angles,
            sensor_x,
            sensor_y,
            self.origin_x,
            self.origin_y,
            self.resolution,
            self.max_range,
            self.min_range,
            self.l_free,
            self.l_occ,
            self.l_clamp,
            self.sensor_noise_sigma,
        )

        raw = log_odds.to_occupancy_raw(self.log_odds)
        binary = log_odds.to_occupancy_binary(self.log_odds, self.occ_threshold)
        inflated = inflate_binary(binary, self.inflation_cells)

        stamp = self.get_clock().now().to_msg()
        self._latest_raw_msg = self._build_grid_msg(raw, stamp)
        self._latest_inflated_msg = self._build_grid_msg(inflated, stamp)
        self.raw_grid_pub.publish(self._latest_raw_msg)
        self.grid_pub.publish(self._latest_inflated_msg)

    def _build_grid_msg(self, grid_data: np.ndarray, stamp) -> OccupancyGrid:
        """Wrap a 2D int8 grid as an ``OccupancyGrid`` in the map frame."""
        msg = OccupancyGrid()
        msg.header.stamp = stamp
        msg.header.frame_id = 'map'
        msg.info.resolution = self.resolution
        msg.info.width = self.grid_width
        msg.info.height = self.grid_height
        msg.info.origin.position.x = self.origin_x
        msg.info.origin.position.y = self.origin_y
        msg.info.origin.position.z = 0.0
        msg.data = grid_data.flatten().tolist()
        return msg

    def _handle_get_snapshot(self, _request, response):
        """Return the latest raw + inflated pair, or empty grids if none yet."""
        if self._latest_raw_msg is None or self._latest_inflated_msg is None:
            stamp = self.get_clock().now().to_msg()
            response.raw = OccupancyGrid()
            response.raw.header.stamp = stamp
            response.raw.header.frame_id = 'map'
            response.inflated = OccupancyGrid()
            response.inflated.header.stamp = stamp
            response.inflated.header.frame_id = 'map'
            response.stamp = stamp
            self.get_logger().warn(
                'GetGridSnapshot called before first scan; returning empty grids')
            return response

        response.raw = self._latest_raw_msg
        response.inflated = self._latest_inflated_msg
        response.stamp = self._latest_inflated_msg.header.stamp
        return response


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleGridNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
