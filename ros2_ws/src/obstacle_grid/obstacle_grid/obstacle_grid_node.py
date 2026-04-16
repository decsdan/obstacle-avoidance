#!/usr/bin/env python3
"""Shared LIDAR obstacle grid node.

Converts LaserScan data to a map-frame OccupancyGrid using Bresenham
raycasting with temporal decay and EDT inflation.
"""

import math
import sys

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformListener

from obstacle_grid.raycaster import apply_temporal_decay, inflate_grid, process_scan


class ObstacleGridNode(Node):
    """Converts LIDAR scans to a shared obstacle occupancy grid."""

    def __init__(self):
        _ns = '/don'
        for i, arg in enumerate(sys.argv):
            if arg.startswith('namespace:='):
                _ns = arg.split(':=', 1)[1]
            elif arg == '-p' and i + 1 < len(sys.argv) and sys.argv[i+1].startswith('namespace:='):
                _ns = sys.argv[i+1].split(':=', 1)[1]

        _user_args = sys.argv[1:]
        _tf_remaps = ['-r', f'/tf:={_ns}/tf', '-r', f'/tf_static:={_ns}/tf_static']
        if '--ros-args' in _user_args:
            _combined = _user_args + _tf_remaps
        else:
            _combined = _user_args + ['--ros-args'] + _tf_remaps

        super().__init__('obstacle_grid_node', cli_args=_combined, use_global_arguments=False)

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
        self.declare_parameter('obstacle_decay_seconds', 20.0)
        self.declare_parameter('publish_rate', 10.0)

        self.max_range = self.get_parameter('max_lidar_range').value
        self.min_range = self.get_parameter('min_lidar_range').value
        self.downsample = self.get_parameter('lidar_downsample').value
        self.decay_seconds = self.get_parameter('obstacle_decay_seconds').value
        self.publish_rate = self.get_parameter('publish_rate').value

        robot_radius = self.get_parameter('robot_radius').value
        safety_clearance = self.get_parameter('safety_clearance').value
        self.resolution = self.get_parameter('grid_resolution').value
        self.inflation_cells = int((robot_radius + safety_clearance) / self.resolution)

        self.grid = None
        self.last_occupied = None
        self.grid_width = 0
        self.grid_height = 0
        self.origin_x = 0.0
        self.origin_y = 0.0
        self._grid_initialized = False

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.scan_msg = None
        self.scan_sub = self.create_subscription(
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
                depth=1
            )
            self.map_sub = self.create_subscription(
                OccupancyGrid,
                f'{self.ns}/map',
                self._map_callback,
                map_qos,
            )
        else:
            self._init_grid_from_params()

        self.grid_pub = self.create_publisher(
            OccupancyGrid,
            f'{self.ns}/obstacle_grid',
            10,
        )

        self.update_timer = self.create_timer(1.0 / self.publish_rate, self._update_cycle)

        self.get_logger().info(
            f'obstacle_grid initialized | ns={self.ns} res={self.resolution}m '
            f'inflation={self.inflation_cells}cells decay={self.decay_seconds}s')

    def _init_grid_from_params(self):
        """Initialize grid from ROS parameters when no SLAM map is available."""
        self.grid_width = self.get_parameter('grid_width').value
        self.grid_height = self.get_parameter('grid_height').value
        self.resolution = self.get_parameter('grid_resolution').value
        self.origin_x = self.get_parameter('grid_origin_x').value
        self.origin_y = self.get_parameter('grid_origin_y').value

        self.grid = np.zeros((self.grid_height, self.grid_width), dtype=np.int8)
        self.last_occupied = np.zeros((self.grid_height, self.grid_width), dtype=np.float64)
        self._grid_initialized = True
        self.get_logger().info(
            f'Grid from params: {self.grid_width}x{self.grid_height} '
            f'res={self.resolution}m origin=({self.origin_x:.2f}, {self.origin_y:.2f})')

    def _map_callback(self, msg: OccupancyGrid):
        """One-shot callback to grab SLAM map dimensions."""
        if self._slam_initialized:
            return

        self._slam_initialized = True
        self.grid_width = msg.info.width
        self.grid_height = msg.info.height
        self.resolution = msg.info.resolution
        self.origin_x = msg.info.origin.position.x
        self.origin_y = msg.info.origin.position.y

        self.grid = np.zeros((self.grid_height, self.grid_width), dtype=np.int8)
        self.last_occupied = np.zeros((self.grid_height, self.grid_width), dtype=np.float64)
        self._grid_initialized = True

        self.inflation_cells = int(
            (self.get_parameter('robot_radius').value +
             self.get_parameter('safety_clearance').value) / self.resolution)

        self.get_logger().info(
            f'Grid from SLAM map: {self.grid_width}x{self.grid_height} '
            f'res={self.resolution}m inflation={self.inflation_cells}cells')

    def _scan_callback(self, msg: LaserScan):
        """Store latest scan for processing."""
        self.scan_msg = msg

    def _update_cycle(self):
        """Process latest scan, update grid, publish."""
        if not self._grid_initialized or self.scan_msg is None:
            return

        sensor_frame = self.scan_msg.header.frame_id
        try:
            tf = self.tf_buffer.lookup_transform(
                'map', sensor_frame, rclpy.time.Time())
        except Exception:
            return

        sensor_x = tf.transform.translation.x
        sensor_y = tf.transform.translation.y
        q = tf.transform.rotation
        sensor_yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z))

        ranges = np.array(self.scan_msg.ranges)[::self.downsample]
        angles = np.linspace(
            self.scan_msg.angle_min,
            self.scan_msg.angle_max,
            len(self.scan_msg.ranges)
        )[::self.downsample]

        current_time = self.get_clock().now().nanoseconds / 1e9

        process_scan(
            self.grid,
            self.last_occupied,
            ranges,
            angles,
            sensor_x,
            sensor_y,
            sensor_yaw,
            self.origin_x,
            self.origin_y,
            self.resolution,
            self.max_range,
            self.min_range,
            current_time,
        )

        apply_temporal_decay(
            self.grid,
            self.last_occupied,
            current_time,
            self.decay_seconds,
        )

        inflated = inflate_grid(self.grid, self.inflation_cells)
        self._publish_grid(inflated)

    def _publish_grid(self, grid_data: np.ndarray):
        """Publish the inflated grid as OccupancyGrid."""
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'

        msg.info.resolution = self.resolution
        msg.info.width = self.grid_width
        msg.info.height = self.grid_height
        msg.info.origin.position.x = self.origin_x
        msg.info.origin.position.y = self.origin_y
        msg.info.origin.position.z = 0.0

        msg.data = grid_data.flatten().tolist()
        self.grid_pub.publish(msg)


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
