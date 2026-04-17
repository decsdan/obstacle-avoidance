#!/usr/bin/env python3
"""Dynamic Window Approach local planner for ROS2 with NavStatus and CancelNav.

Each control cycle samples the reachable velocity window, simulates a short
trajectory for each (v, w) pair, scores them, and publishes the best as a
TwistStamped command. Supports standalone mode (drives directly to a goal
pose) and stacked mode (follows an upstream nav_msgs/Path).
"""

import math
import sys
from enum import Enum

import numpy as np
from scipy.ndimage import distance_transform_edt, zoom
from scipy.spatial import cKDTree

import rclpy
import rclpy.duration
import rclpy.time
from geometry_msgs.msg import Point, PoseStamped, TwistStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from nav_interfaces.msg import NavStatus
from nav_interfaces.srv import CancelNav
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import ColorRGBA
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker


class NavState(Enum):
    """Top-level navigation state for the DWA control loop."""

    IDLE = 0
    NAVIGATING = 1
    EMERGENCY_STOPPED = 2
    RECOVERING = 3


class DWA(Node):
    """ROS2 node for Dynamic Window Approach local planning on TurtleBot4."""

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

        super().__init__('dynamic_window_approach',
                         cli_args=_combined,
                         use_global_arguments=False)

        self.declare_parameter('stacked', False)
        self.declare_parameter('lookahead', 0.85)
        self.stacked = self.get_parameter('stacked').value
        self.lookahead = self.get_parameter('lookahead').value

        self.declare_parameter('max_velocity', 0.4)
        self.declare_parameter('min_velocity', 0.0)
        self.declare_parameter('max_angular_velocity', 1.8)
        self.declare_parameter('min_angular_velocity', -1.8)
        self.declare_parameter('max_linear_acceleration', 0.5)
        self.declare_parameter('max_angular_acceleration', 2.0)
        self.declare_parameter('v_samples', 20)
        self.declare_parameter('w_samples', 20)
        self.declare_parameter('lidar_angle_offset', 1.5708)

        self.max_v = self.get_parameter('max_velocity').value
        self.min_v = self.get_parameter('min_velocity').value
        self.max_w = self.get_parameter('max_angular_velocity').value
        self.min_w = self.get_parameter('min_angular_velocity').value
        self.v_accel = self.get_parameter('max_linear_acceleration').value
        self.w_accel = self.get_parameter('max_angular_acceleration').value
        self.v_samples = self.get_parameter('v_samples').value
        self.w_samples = self.get_parameter('w_samples').value
        self.lidar_offset = self.get_parameter('lidar_angle_offset').value

        self.declare_parameter('dt', 0.1)
        self.declare_parameter('prediction_steps', 25)
        self.declare_parameter('window_steps', 5)
        self.declare_parameter('LIDAR_downsample', 2)
        self.declare_parameter('max_path_deviation', 1.0)

        self.dt = self.get_parameter('dt').value
        self.steps = self.get_parameter('prediction_steps').value
        self.window_steps = self.get_parameter('window_steps').value
        self.LIDAR_downsample = self.get_parameter('LIDAR_downsample').value
        self.max_path_deviation = self.get_parameter('max_path_deviation').value

        self.declare_parameter('grid_size', 161)
        self.declare_parameter('grid_resolution', 10.0)

        self.grid_size = self.get_parameter('grid_size').value
        self.grid_resolution = self.get_parameter('grid_resolution').value

        self.declare_parameter('critical_radius', 0.20)
        self.declare_parameter('emergency_stop_distance', 0.17)
        self.declare_parameter('max_lidar_range', 8.0)
        self.critical_radius = self.get_parameter('critical_radius').value
        self.emergency_stop_dist = self.get_parameter('emergency_stop_distance').value
        self.max_lidar_range = self.get_parameter('max_lidar_range').value

        self.declare_parameter('weights.goal', 0.35)
        self.declare_parameter('weights.heading', 0.05)
        self.declare_parameter('weights.velocity', 0.10)
        self.declare_parameter('weights.smoothness', 0.05)
        self.declare_parameter('weights.obstacle', 0.40)
        self.declare_parameter('weights.dist_path', 0.10)
        self.declare_parameter('weights.heading_path', 0.05)

        self.w_goal = self.get_parameter('weights.goal').value
        # In stacked mode, heading-to-goal is redundant with heading-to-path.
        # Skip h_score entirely; use hp_score for near-goal heading alignment.
        self.w_heading = 0.0 if self.stacked else self.get_parameter('weights.heading').value
        self.w_velocity = self.get_parameter('weights.velocity').value
        self.w_smoothness = self.get_parameter('weights.smoothness').value
        self.w_obstacle = self.get_parameter('weights.obstacle').value
        self.w_dist_path = self.get_parameter('weights.dist_path').value
        self.w_heading_path = self.get_parameter('weights.heading_path').value

        self.declare_parameter('recovery.linear_velocity', 0.0)
        self.declare_parameter('recovery.angular_velocity', 0.5)
        self.declare_parameter('recovery.backup_velocity', -0.05)
        self.declare_parameter('recovery.rotate_timeout', 3.0)
        self.declare_parameter('recovery.total_timeout', 10.0)

        self.recovery_v = self.get_parameter('recovery.linear_velocity').value
        self.recovery_w = self.get_parameter('recovery.angular_velocity').value
        self.recovery_backup_v = self.get_parameter('recovery.backup_velocity').value
        self.recovery_rotate_timeout = self.get_parameter('recovery.rotate_timeout').value
        self.recovery_total_timeout = self.get_parameter('recovery.total_timeout').value

        self.declare_parameter('visualize_trajectories', True)
        self.declare_parameter('trajectory_visualization_downsample', 10)
        self.visualize_trajectories = self.get_parameter('visualize_trajectories').value
        self.traj_downsample = self.get_parameter('trajectory_visualization_downsample').value

        self.declare_parameter('goal_tolerance', 0.2)
        self.goal_tolerance = self.get_parameter('goal_tolerance').value

        self.nav_state = NavState.IDLE
        self.goal = None
        self.global_path = None
        self.global_path_displacement = None
        self.scan_msg = None
        self.odom_msg = None
        self._warned_no_shared_grid = False
        self.start_time = None
        self.total_distance = 0.0
        self.last_position = None
        self._tracked_final_goal = None
        self._goal_just_reached = False
        self.active_frame = 'odom'

        self._estop_clear_count = 0
        self._estop_clear_needed = 5

        self._recovery_start_time = None

        self.declare_parameter('namespace', '/don')
        self.namespace = self.get_parameter('namespace').value
        if self.stacked:
            self.global_path_sub = self.create_subscription(
                Path,
                f'{self.namespace}/a_star/plan',
                self.global_path_callback,
                10,
            )
        else:
            self.goal_sub = self.create_subscription(
                PoseStamped,
                f'{self.namespace}/goal_pose',
                self.goal_callback,
                10,
            )

        self.odom_sub = Subscriber(
            self, Odometry, f'{self.namespace}/odom',
            qos_profile=qos_profile_sensor_data)
        self.scan_sub = Subscriber(
            self, LaserScan, f'{self.namespace}/scan',
            qos_profile=qos_profile_sensor_data)

        qos_policy = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=10)
        self.twist_publish = self.create_publisher(
            TwistStamped, f'{self.namespace}/cmd_vel', qos_policy)
        self.debug_pub = self.create_publisher(
            Marker, f'{self.namespace}/debug_obstacles', 10)
        self.traj_pub = self.create_publisher(
            Marker, f'{self.namespace}/dwa/trajectories', 10)
        self.best_traj_pub = self.create_publisher(
            Marker, f'{self.namespace}/dwa/best_trajectory', 10)

        self.nav_status_pub = self.create_publisher(
            NavStatus, f'{self.namespace}/dwa/nav_status', 10)
        self.cancel_srv = self.create_service(
            CancelNav, f'{self.namespace}/dwa/cancel', self._handle_cancel)
        self._status_timer = self.create_timer(0.2, self._publish_nav_status)

        self._shared_grid_msg = None
        self.shared_grid_sub = self.create_subscription(
            OccupancyGrid,
            f'{self.namespace}/obstacle_grid',
            self._shared_grid_callback,
            10,
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.sync = ApproximateTimeSynchronizer(
            [self.odom_sub, self.scan_sub], 10, 0.1)
        self.sync.registerCallback(self.synchronized_callback)

        self.timer = self.create_timer(self.dt, self.nav_loop)

    def global_path_callback(self, msg):
        """Handle a new global path from the upstream planner (stacked mode)."""
        if self.global_path is None or self.global_path.poses != msg.poses:
            self.global_path = msg
            self.get_logger().info('New global path has been grabbed!')
            self.get_path_displacement()

            if msg.poses:
                last = msg.poses[-1].pose.position
                new_final = np.array([last.x, last.y])
                if (self._tracked_final_goal is None or
                        np.linalg.norm(new_final - self._tracked_final_goal) > 0.5):
                    self._tracked_final_goal = new_final
                    self.start_time = self.get_clock().now()
                    self.total_distance = 0.0
                    self.last_position = None
                    self.get_logger().info(
                        f'Tracking reset for new destination: '
                        f'({new_final[0]:.2f}, {new_final[1]:.2f})')

            if self.nav_state == NavState.IDLE:
                self.nav_state = NavState.NAVIGATING

    def goal_callback(self, msg):
        """Handle a new goal pose (standalone mode)."""
        new_goal = np.array([msg.pose.position.x, msg.pose.position.y])
        self.goal = new_goal
        self.get_logger().info(f'New Goal Set: {self.goal}')
        if not self.stacked:
            self._tracked_final_goal = new_goal
            self.start_time = self.get_clock().now()
            self.total_distance = 0.0
            self.last_position = None

        if self.nav_state == NavState.IDLE:
            self.nav_state = NavState.NAVIGATING

    def synchronized_callback(self, odom_msg, scan_msg):
        """Cache the latest synchronized odom and scan messages for the loop."""
        self.scan_msg = scan_msg
        self.odom_msg = odom_msg

    def _shared_grid_callback(self, msg: OccupancyGrid):
        """Store the latest shared obstacle grid message for cost lookup."""
        self._shared_grid_msg = msg

    def _extract_dist_grid_from_shared(self, curr_x, curr_y):
        """Extract a robot-centric distance grid from the shared obstacle grid."""
        msg = self._shared_grid_msg
        if msg is None:
            return None

        shared_res = msg.info.resolution
        shared_w = msg.info.width
        shared_h = msg.info.height
        shared_ox = msg.info.origin.position.x
        shared_oy = msg.info.origin.position.y

        # Robot position in shared grid cells
        robot_gx = int((curr_x - shared_ox) / shared_res)
        robot_gy = int((curr_y - shared_oy) / shared_res)

        # DWA grid physical span
        half_span_m = (self.grid_size / self.grid_resolution) / 2.0
        half_span_cells = int(half_span_m / shared_res)

        # Window bounds in shared grid
        y_lo = robot_gy - half_span_cells
        y_hi = robot_gy + half_span_cells
        x_lo = robot_gx - half_span_cells
        x_hi = robot_gx + half_span_cells

        # Parse shared grid
        shared = np.array(msg.data, dtype=np.int8).reshape(shared_h, shared_w)

        # Extract with bounds clamping + padding
        # Pad shared grid so we can always extract the full window
        pad_y = max(0, -y_lo, y_hi - shared_h + 1)
        pad_x = max(0, -x_lo, x_hi - shared_w + 1)

        if pad_y > 0 or pad_x > 0:
            shared_padded = np.pad(shared, ((pad_y, pad_y), (pad_x, pad_x)),
                                   mode='constant', constant_values=0)
            y_lo += pad_y
            y_hi += pad_y
            x_lo += pad_x
            x_hi += pad_x
        else:
            shared_padded = shared

        window = shared_padded[y_lo:y_hi, x_lo:x_hi]

        # Resample to DWA resolution if needed
        # shared_res is typically 0.05m, DWA resolution is cells/meter (10.0 = 0.1m/cell)
        dwa_cell_m = 1.0 / self.grid_resolution  # 0.1m per cell
        scale = shared_res / dwa_cell_m  # e.g., 0.05/0.1 = 0.5

        if abs(scale - 1.0) > 0.01:
            resampled = zoom(window, scale, order=0)
        else:
            resampled = window

        # Ensure output matches DWA grid size
        target = self.grid_size
        if resampled.shape[0] != target or resampled.shape[1] != target:
            result = np.zeros((target, target), dtype=np.int8)
            h = min(resampled.shape[0], target)
            w = min(resampled.shape[1], target)
            offset_y = (target - h) // 2
            offset_x = (target - w) // 2
            result[offset_y:offset_y+h, offset_x:offset_x+w] = resampled[:h, :w]
            resampled = result

        obstacle_grid = np.where(resampled >= 50, 100, 0).astype(int)
        free_mask = (obstacle_grid != 100)
        dist_grid = distance_transform_edt(free_mask)

        return dist_grid

    def get_path_displacement(self):
        """Cache per-segment distances along the global path for lookahead."""
        if self.global_path is None or len(self.global_path.poses) == 0:
            return None
        self.global_path_displacement = []
        prev_x = None
        prev_y = None
        for pose in self.global_path.poses:
            x = pose.pose.position.x
            y = pose.pose.position.y
            if prev_x is None:
                prev_x = x
            if prev_y is None:
                prev_y = y
            dx = x - prev_x
            dy = y - prev_y
            prev_x = x
            prev_y = y
            self.global_path_displacement.append(math.sqrt(dx*dx + dy*dy))

    def get_moving_goal(self, curr_x, curr_y):
        """Pick a lookahead pose along the global path for the local planner."""
        if self.global_path is None or len(self.global_path.poses) == 0:
            return None
        distances = []
        for pose in self.global_path.poses:
            x = pose.pose.position.x
            y = pose.pose.position.y
            dx = x - curr_x
            dy = y - curr_y
            distances.append(math.sqrt(dx*dx + dy*dy))

        closestpose = int(np.argmin(distances))
        i = 1
        target_idx = closestpose + i
        traveled = 0
        while (
            target_idx < len(self.global_path.poses) - 1 and
            traveled < self.lookahead
        ):
            traveled += self.global_path_displacement[target_idx]
            i += 1
            target_idx = closestpose + i

        target_idx = min(target_idx, len(self.global_path.poses) - 1)
        goal_x = self.global_path.poses[target_idx].pose.position.x
        goal_y = self.global_path.poses[target_idx].pose.position.y
        goal_theta = self.quat_to_yaw(self.global_path.poses[target_idx].pose.orientation)
        return goal_x, goal_y, goal_theta

    def quat_to_yaw(self, q):
        """Convert quaternion to yaw angle."""
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def get_robot_pose_map_frame(self):
        """Look up robot pose in the map frame via TF, or None if unavailable."""
        try:
            transform = self.tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1))
            x = transform.transform.translation.x
            y = transform.transform.translation.y
            q = transform.transform.rotation
            theta = self.quat_to_yaw(q)
            return (x, y, theta)
        except Exception:
            return None

    def _get_current_pose(self):
        """Return (x, y, theta, v, w) using map-frame TF when available."""
        if self.odom_msg is None:
            return None
        map_pose = self.get_robot_pose_map_frame()
        if map_pose is not None:
            curr_x, curr_y, curr_theta = map_pose
            self.active_frame = 'map'
        else:
            curr_x = self.odom_msg.pose.pose.position.x
            curr_y = self.odom_msg.pose.pose.position.y
            curr_theta = self.quat_to_yaw(self.odom_msg.pose.pose.orientation)
            self.active_frame = 'odom'
        curr_v = self.odom_msg.twist.twist.linear.x
        curr_w = self.odom_msg.twist.twist.angular.z
        return curr_x, curr_y, curr_theta, curr_v, curr_w

    def _update_distance_tracking(self, curr_x, curr_y):
        """Accumulate total distance traveled."""
        if self.last_position is None:
            self.total_distance = 0.0
            self.last_position = (curr_x, curr_y)
        else:
            dx = curr_x - self.last_position[0]
            dy = curr_y - self.last_position[1]
            self.total_distance += math.sqrt(dx**2 + dy**2)
            self.last_position = (curr_x, curr_y)

    def _publish_stop(self):
        """Publish zero velocity command."""
        stop_cmd = TwistStamped()
        stop_cmd.header.stamp = self.get_clock().now().to_msg()
        stop_cmd.header.frame_id = 'base_link'
        stop_cmd.twist.linear.x = 0.0
        stop_cmd.twist.angular.z = 0.0
        self.twist_publish.publish(stop_cmd)

    def _publish_cmd(self, v, w):
        """Publish velocity command."""
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.twist.linear.x = v
        msg.twist.angular.z = w
        self.twist_publish.publish(msg)

    def _check_front_cone_clear(self):
        """Return True if the front LIDAR cone is clear of close obstacles."""
        if self.scan_msg is None:
            return True

        ranges = np.array(self.scan_msg.ranges)
        ranges = np.where(np.isfinite(ranges) & (ranges > 0.05), ranges, 10.0)
        n = len(ranges)
        if n == 0:
            return True
        angle_range = self.scan_msg.angle_max - self.scan_msg.angle_min
        front_idx = int((-self.lidar_offset - self.scan_msg.angle_min) / angle_range * n) % n
        cone_width = int(n / 8)
        indices = np.arange(front_idx - cone_width, front_idx + cone_width) % n
        front_ranges = ranges[indices]
        return np.min(front_ranges) >= self.emergency_stop_dist

    def publish_trajectory_markers(self, trajectories, scores, best_idx):
        """Publish candidate trajectory visualization to RViz."""
        if not self.visualize_trajectories:
            return

        now = self.get_clock().now().to_msg()
        marker = Marker()
        marker.header.frame_id = self.active_frame
        marker.header.stamp = now
        marker.ns = 'candidate_trajectories'
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 0.01
        marker.pose.orientation.w = 1.0

        valid_scores = scores[np.isfinite(scores)]
        if len(valid_scores) == 0:
            return

        smin = np.min(valid_scores)
        smax = np.max(valid_scores)
        srange = smax - smin if smax > smin else 1.0

        for i in range(0, len(trajectories), self.traj_downsample):
            if i == best_idx:
                continue
            traj = trajectories[i]
            score = scores[i]
            if np.isfinite(score):
                norm = (score - smin) / srange
                if norm < 0.5:
                    r, g, b = 1.0, norm * 2.0, 0.0
                else:
                    r, g, b = 1.0 - (norm - 0.5) * 2.0, 1.0, 0.0
                a = 0.3 + 0.4 * norm
            else:
                r, g, b, a = 0.5, 0.5, 0.5, 0.1

            for j in range(len(traj) - 1):
                p1 = Point()
                p1.x, p1.y, p1.z = float(traj[j, 0]), float(traj[j, 1]), 0.05
                p2 = Point()
                p2.x, p2.y, p2.z = float(traj[j+1, 0]), float(traj[j+1, 1]), 0.05
                marker.points.append(p1)
                marker.points.append(p2)
                c = ColorRGBA()
                c.r, c.g, c.b, c.a = r, g, b, a
                marker.colors.append(c)
                marker.colors.append(c)

        self.traj_pub.publish(marker)

        best_marker = Marker()
        best_marker.header.frame_id = self.active_frame
        best_marker.header.stamp = now
        best_marker.ns = 'best_trajectory'
        best_marker.id = 1
        best_marker.type = Marker.LINE_STRIP
        best_marker.action = Marker.ADD
        best_marker.scale.x = 0.05
        best_marker.color.r, best_marker.color.g = 0.0, 1.0
        best_marker.color.b, best_marker.color.a = 1.0, 1.0
        best_marker.pose.orientation.w = 1.0

        for pt in trajectories[best_idx]:
            p = Point()
            p.x, p.y, p.z = float(pt[0]), float(pt[1]), 0.1
            best_marker.points.append(p)

        self.best_traj_pub.publish(best_marker)

    def get_obstacles(self):
        """Convert the latest lidar scan into world-frame obstacle points."""
        if self.scan_msg is None:
            return np.array([])

        map_pose = self.get_robot_pose_map_frame()
        if map_pose is not None:
            x, y, theta = map_pose
        elif self.odom_msg is not None:
            theta = self.quat_to_yaw(self.odom_msg.pose.pose.orientation)
            x = self.odom_msg.pose.pose.position.x
            y = self.odom_msg.pose.pose.position.y
        else:
            return np.array([])

        ranges = np.array(self.scan_msg.ranges)[::self.LIDAR_downsample]
        angles = np.linspace(
            self.scan_msg.angle_min,
            self.scan_msg.angle_max,
            len(self.scan_msg.ranges)
        )[::self.LIDAR_downsample]
        mask = (
            np.isfinite(ranges) &
            (ranges > 0.05) &
            (ranges > self.scan_msg.range_min) &
            (ranges < self.scan_msg.range_max) &
            (ranges < self.max_lidar_range)
        )
        valid_ranges = ranges[mask]
        valid_angles = angles[mask]

        world_angles = theta + valid_angles + self.lidar_offset
        o_x = x + valid_ranges * np.cos(world_angles)
        o_y = y + valid_ranges * np.sin(world_angles)

        return np.column_stack((o_x, o_y))

    def cost_function(self, trajectories, final_vs, final_ws, obstacles, curr_x, curr_y,
                      dist_to_final_goal, final_goal_pos):
        """Score each candidate trajectory and hard-reject ones too close to obstacles."""

        final_pos = trajectories[:, -1, :2]
        curr_dist = dist_to_final_goal
        goal_dists = np.linalg.norm(final_goal_pos - final_pos, axis=1)

        # Near-goal weight blending: smoothly transitions within 0.5m of goal
        if curr_dist < 0.5:
            near_t = 1.0 - (curr_dist / 0.5)
            w_goal_eff = self.w_goal + near_t * (0.60 - self.w_goal)
            w_obstacle_eff = self.w_obstacle
            w_velocity_eff = self.w_velocity * (1.0 - near_t)

            if self.stacked:
                # In stacked mode, blend hp_score weight UP near goal for final heading alignment
                w_heading_eff = 0.0
                w_dist_path_eff = self.w_dist_path * (1.0 - near_t * 0.7)
                w_heading_path_eff = self.w_heading_path + near_t * (0.25 - self.w_heading_path)
            else:
                w_heading_eff = self.w_heading + near_t * (0.25 - self.w_heading)
                w_dist_path_eff = self.w_dist_path
                w_heading_path_eff = self.w_heading_path
        else:
            w_goal_eff = self.w_goal
            w_heading_eff = self.w_heading
            w_obstacle_eff = self.w_obstacle
            w_velocity_eff = self.w_velocity
            w_dist_path_eff = self.w_dist_path
            w_heading_path_eff = self.w_heading_path

        # Path-following scores (stacked only)
        if self.stacked and self.global_path is not None and len(self.global_path.poses) >= 2:
            path_pts = np.array([[p.pose.position.x, p.pose.position.y]
                                 for p in self.global_path.poses])
            A = path_pts[:-1]
            B = path_pts[1:]
            AB = B - A
            AB_len_sq = np.maximum(np.sum(AB ** 2, axis=1), 1e-10)

            PA = final_pos[:, None, :] - A[None, :, :]
            t = np.clip(np.sum(PA * AB[None, :, :], axis=2) / AB_len_sq[None, :], 0.0, 1.0)
            closest = A[None, :, :] + t[:, :, None] * AB[None, :, :]
            seg_dists = np.linalg.norm(final_pos[:, None, :] - closest, axis=2)
            min_path_dists = np.min(seg_dists, axis=1)

            # distance from path score
            dp_score = np.sqrt(np.clip(1.0 - (min_path_dists / self.max_path_deviation), 0.0, 1.0))

            # heading to path score
            nearest_seg_idx = np.argmin(seg_dists, axis=1)
            seg_tangents = np.arctan2(AB[:, 1], AB[:, 0])
            path_theta = seg_tangents[nearest_seg_idx]
            final_theta = trajectories[:, -1, 2]
            heading_err = np.abs(np.arctan2(
                np.sin(path_theta - final_theta),
                np.cos(path_theta - final_theta)))
            hp_score = 1.0 - (heading_err / np.pi)
        else:
            dp_score = np.ones(len(final_vs))
            hp_score = np.ones(len(final_vs))

        # goal score
        max_prog = max(dist_to_final_goal, self.max_v * self.dt * trajectories.shape[1])
        progress = curr_dist - goal_dists
        g_score = np.clip((progress + max_prog) / (2 * max_prog), 0.0, 1.0)

        # heading score (standalone only -- skipped in stacked mode)
        if not self.stacked:
            dx = self.goal[0] - final_pos[:, 0]
            dy = self.goal[1] - final_pos[:, 1]
            goal_theta = np.arctan2(dy, dx)
            final_theta = trajectories[:, -1, 2]
            heading_err = np.abs(np.arctan2(
                np.sin(goal_theta - final_theta),
                np.cos(goal_theta - final_theta)))
            h_score = 1.0 - (heading_err / np.pi)
        else:
            h_score = np.zeros(len(final_vs))

        # obstacle hard-rejection via cKDTree (replaces 4D broadcast)
        if len(obstacles) > 0:
            tree = cKDTree(obstacles)
            flat_pts = trajectories[:, :, :2].reshape(-1, 2)
            dists, _ = tree.query(flat_pts)
            min_dists = dists.reshape(trajectories.shape[0], trajectories.shape[1]).min(axis=1)
        else:
            min_dists = np.full(len(final_vs), self.max_lidar_range)

        dist_grid = self._extract_dist_grid_from_shared(curr_x, curr_y)
        if dist_grid is None:
            if not self._warned_no_shared_grid:
                self.get_logger().warn(
                    'obstacle_grid node not available -- obstacle distance scoring '
                    'disabled. Ensure obstacle_grid node is running alongside DWA.')
                self._warned_no_shared_grid = True
            o_score = np.full(len(trajectories), 0.5)
        else:
            self._warned_no_shared_grid = False
            o_score = np.array(_get_all_path_costs_with_grid(
                trajectories, dist_grid, curr_x, curr_y,
                self.grid_size, self.grid_resolution))

        v_score = np.clip(final_vs / self.max_v, 0.0, 1.0)
        w_score = np.clip(1.0 - (np.abs(final_ws) / self.max_w), 0.0, 1.0)

        # combine effective weights
        scores = (
            w_goal_eff         * g_score +
            w_heading_eff      * h_score +
            w_velocity_eff     * v_score +
            self.w_smoothness  * w_score +
            w_obstacle_eff     * o_score +
            w_dist_path_eff    * dp_score +
            w_heading_path_eff * hp_score
        )

        crit = np.where(
            np.abs(final_vs) > 0.1,
            self.critical_radius,
            self.emergency_stop_dist + 0.01)
        scores[min_dists < crit] = -np.inf

        return scores, min_dists

    def predict_trajectories(self, v_arr, w_arr, curr_x, curr_y, curr_theta,
                             steps=30, dt=0.1):
        """Forward-simulate all (v, w) pairs in parallel."""
        vs = np.tile(v_arr[:, None], (1, steps))
        ws = np.tile(w_arr[:, None], (1, steps))

        thetas = np.cumsum(ws * dt, axis=1) + curr_theta
        xs = np.cumsum(vs * np.cos(thetas) * dt, axis=1) + curr_x
        ys = np.cumsum(vs * np.sin(thetas) * dt, axis=1) + curr_y

        trajectories = np.stack((xs, ys, thetas), axis=2)
        return trajectories, vs[:, -1], ws[:, -1]

    def dynamic_window(self, curr_v, curr_w):
        """Compute reachable (v, w) bounds for the next window_steps * dt seconds."""
        window_time = self.dt * self.window_steps

        v_range = self.v_accel * window_time
        w_range = self.w_accel * window_time

        poss_v_max = min(curr_v + v_range, self.max_v)
        poss_v_min = max(curr_v - v_range, self.min_v)
        poss_w_max = min(curr_w + w_range, self.max_w)
        poss_w_min = max(curr_w - w_range, self.min_w)

        return poss_v_max, poss_v_min, poss_w_max, poss_w_min

    def _handle_cancel(self, request, response):
        """CancelNav service handler -- stop navigation immediately."""
        self.get_logger().info('Cancel requested via service')
        self._publish_stop()
        self.goal = None
        if self.stacked:
            self.global_path = None
        self._recovery_start_time = None
        self.nav_state = NavState.IDLE
        response.confirmed = True
        return response

    def _publish_nav_status(self):
        """Publish NavStatus at 5 Hz for the navigation server to consume."""
        msg = NavStatus()
        msg.nav_state = self.nav_state.name.lower()
        msg.has_active_goal = (self.goal is not None or self.global_path is not None)
        msg.distance_traveled = self.total_distance

        if self.start_time is not None:
            msg.elapsed_time = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
        else:
            msg.elapsed_time = 0.0

        pose = self._get_current_pose()
        if pose is not None:
            curr_x, curr_y, _, _, _ = pose
            msg.current_x = curr_x
            msg.current_y = curr_y

            final_goal = self._tracked_final_goal
            if final_goal is not None:
                msg.distance_to_goal = float(np.linalg.norm(
                    final_goal - np.array([curr_x, curr_y])))
            else:
                msg.distance_to_goal = -1.0
        else:
            msg.current_x = 0.0
            msg.current_y = 0.0
            msg.distance_to_goal = -1.0

        msg.goal_reached = self._goal_just_reached

        self.nav_status_pub.publish(msg)

    def nav_loop(self):
        """Main control loop dispatching to state handlers."""
        if self.odom_msg is None or self.scan_msg is None:
            return

        if self.nav_state == NavState.IDLE:
            if self.stacked and self.global_path is not None:
                self.nav_state = NavState.NAVIGATING
            elif not self.stacked and self.goal is not None:
                self.nav_state = NavState.NAVIGATING
            else:
                return

        if self.nav_state == NavState.NAVIGATING:
            self._handle_navigating()
        elif self.nav_state == NavState.EMERGENCY_STOPPED:
            self._handle_emergency_stopped()
        elif self.nav_state == NavState.RECOVERING:
            self._handle_recovering()

    def _handle_navigating(self):
        """Run the DWA selection loop: e-stop check, goal tracking, scoring."""
        if not self._check_front_cone_clear():
            self.get_logger().warn('EMERGENCY STOP: Obstacle too close!')
            self._publish_stop()
            self._estop_clear_count = 0
            self.nav_state = NavState.EMERGENCY_STOPPED
            return

        pose = self._get_current_pose()
        if pose is None:
            return
        curr_x, curr_y, curr_theta, curr_v, curr_w = pose
        self._update_distance_tracking(curr_x, curr_y)

        if self.stacked:
            if self.global_path is None:
                return
            result = self.get_moving_goal(curr_x, curr_y)
            if result is None:
                return
            goal_x, goal_y, goal_theta = result
            if not np.array_equal(self.goal, np.array([goal_x, goal_y])):
                self.get_logger().info(
                    f'New Goal Assigned: {goal_x, goal_y, goal_theta}')
                self.goal = np.array([goal_x, goal_y])
            final_goal_pos = np.array([
                self.global_path.poses[-1].pose.position.x,
                self.global_path.poses[-1].pose.position.y,
            ])
        else:
            if self.goal is None:
                return
            final_goal_pos = self.goal

        dist = np.linalg.norm(final_goal_pos - np.array([curr_x, curr_y]))
        if dist < self.goal_tolerance:
            self.get_logger().info('Goal reached!')
            self._publish_stop()
            if self.start_time is not None:
                elapsed = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
                self.get_logger().info(
                    f'Total distance traveled: {self.total_distance:.2f} meters')
                self.get_logger().info(f'Total time elapsed: {elapsed:.2f} seconds')
            self.goal = None
            if self.stacked:
                self.global_path = None
            self._goal_just_reached = True
            self.nav_state = NavState.IDLE
            return

        effective_max_v = float(
            np.clip(self.max_v * min(1.0, dist / 0.3), 0.15, self.max_v))
        _saved_max_v = self.max_v
        self.max_v = effective_max_v
        poss_v_max, poss_v_min, poss_w_max, poss_w_min = self.dynamic_window(
            curr_v, curr_w)
        self.max_v = _saved_max_v

        effective_steps = max(10, int(self.steps * min(1.0, dist / 0.5)))

        poss_v = np.linspace(poss_v_min, poss_v_max, self.v_samples)
        poss_w = np.linspace(poss_w_min, poss_w_max, self.w_samples)
        v_arr, w_arr = np.meshgrid(poss_v, poss_w)
        v_arr = v_arr.flatten()
        w_arr = w_arr.flatten()
        trajectories, final_vs, final_ws = self.predict_trajectories(
            v_arr, w_arr, curr_x, curr_y, curr_theta, effective_steps, self.dt)

        obstacles = self.get_obstacles()
        if len(obstacles) > 0:
            marker = Marker()
            marker.header.frame_id = self.active_frame
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.type = Marker.SPHERE_LIST
            marker.action = Marker.ADD
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            for obs in obstacles:
                p = Point()
                p.x = float(obs[0])
                p.y = float(obs[1])
                p.z = 0.0
                marker.points.append(p)
            self.debug_pub.publish(marker)

        scores, min_dists = self.cost_function(
            trajectories, final_vs, final_ws, obstacles, curr_x, curr_y,
            dist, final_goal_pos)
        best_idx = np.argmax(scores)
        self.publish_trajectory_markers(trajectories, scores, best_idx)

        if scores[best_idx] == -np.inf:
            self.get_logger().warn('All trajectories rejected, entering recovery')
            self._recovery_start_time = self.get_clock().now()
            self.nav_state = NavState.RECOVERING
            return

        self._publish_cmd(float(final_vs[best_idx]), float(final_ws[best_idx]))

    def _handle_emergency_stopped(self):
        """Hold the robot stopped until the front cone is clear for several ticks."""
        if self._check_front_cone_clear():
            self._estop_clear_count += 1
            if self._estop_clear_count >= self._estop_clear_needed:
                self.get_logger().info('Front cone clear, resuming navigation')
                self.nav_state = NavState.NAVIGATING
                return
        else:
            self._estop_clear_count = 0

        self._publish_stop()

    def _handle_recovering(self):
        """Rotate in place, then back up slowly if still stuck."""
        if self._recovery_start_time is None:
            self._recovery_start_time = self.get_clock().now()

        elapsed = (self.get_clock().now() - self._recovery_start_time).nanoseconds / 1e9

        if not self._check_front_cone_clear():
            self._publish_stop()
            self._estop_clear_count = 0
            self.nav_state = NavState.EMERGENCY_STOPPED
            return

        if elapsed > self.recovery_total_timeout:
            self.get_logger().error(
                'Recovery timeout -- stopping. Manual intervention may be needed.')
            self._publish_stop()
            self.goal = None
            if self.stacked:
                self.global_path = None
            self.nav_state = NavState.IDLE
            return

        if elapsed < self.recovery_rotate_timeout:
            self._publish_cmd(self.recovery_v, self.recovery_w)
        else:
            self._publish_cmd(self.recovery_backup_v, 0.0)

        pose = self._get_current_pose()
        if pose is None:
            return
        curr_x, curr_y, curr_theta, curr_v, curr_w = pose

        poss_v_max, poss_v_min, poss_w_max, poss_w_min = self.dynamic_window(
            curr_v, curr_w)
        poss_v = np.linspace(poss_v_min, poss_v_max, self.v_samples)
        poss_w = np.linspace(poss_w_min, poss_w_max, self.w_samples)
        v_arr, w_arr = np.meshgrid(poss_v, poss_w)
        v_arr = v_arr.flatten()
        w_arr = w_arr.flatten()

        obstacles = self.get_obstacles()
        if len(obstacles) > 0:
            tree = cKDTree(obstacles)
            trajectories, final_vs, final_ws = self.predict_trajectories(
                v_arr, w_arr, curr_x, curr_y, curr_theta, self.steps, self.dt)
            flat_pts = trajectories[:, :, :2].reshape(-1, 2)
            dists, _ = tree.query(flat_pts)
            min_dists = dists.reshape(
                trajectories.shape[0], trajectories.shape[1]).min(axis=1)
            crit = np.where(
                np.abs(final_vs) > 0.1,
                self.critical_radius,
                self.emergency_stop_dist + 0.01)
            viable = np.any(min_dists >= crit)
            if viable:
                self.get_logger().info(
                    'Recovery complete -- viable trajectories found')
                self.nav_state = NavState.NAVIGATING


def _normalize_path_costs(all_costs: list) -> list:
    """Min-max normalize costs in [0, 1], preserving math.inf entries."""
    finite = [c for c in all_costs if not math.isinf(c)]
    if not finite:
        return all_costs
    mn, mx = min(finite), max(finite)
    if mx == mn:
        return [0.0 if not math.isinf(c) else math.inf for c in all_costs]
    return [
        (c - mn) / (mx - mn) if not math.isinf(c) else math.inf
        for c in all_costs
    ]


def _get_all_path_costs_with_grid(
    trajectories: np.ndarray,
    dist_grid: np.ndarray,
    curr_x: float,
    curr_y: float,
    grid_size: int,
    grid_resolution: float,
) -> list:
    """Score trajectories by tail-segment proximity to obstacles using a distance grid."""
    center = grid_size // 2
    max_idx = grid_size - 1

    pts = trajectories[:, :, :2]
    grid_pts = ((pts - np.array([curr_x, curr_y])) * grid_resolution + center).astype(int)
    gx = grid_pts[:, :, 0]
    gy = grid_pts[:, :, 1]
    in_bounds = (gx >= 0) & (gx <= max_idx) & (gy >= 0) & (gy <= max_idx)
    gx_c = np.clip(gx, 0, max_idx)
    gy_c = np.clip(gy, 0, max_idx)
    dist_vals = dist_grid[gy_c, gx_c].astype(float)

    # OOB points -> 0 cost; obstacle points -> inf; valid -> exp decay
    point_costs = np.where(
        in_bounds,
        np.where(dist_vals > 0, np.exp(-0.5 * dist_vals), np.inf),
        0.0
    )

    all_oob = ~in_bounds.any(axis=1)
    has_inf = np.isinf(point_costs).any(axis=1)

    # Score by the mean cost of the last 20% of trajectory steps
    n_tail = max(1, point_costs.shape[1] // 5)
    tail_mean = point_costs[:, -n_tail:].mean(axis=1)
    total = np.where(all_oob | has_inf, np.inf, tail_mean)

    normalised = _normalize_path_costs(total.tolist())
    return [1 - cost for cost in normalised]


def main(args=None):
    """Main entry point for the DWA local planner."""
    rclpy.init(args=args)
    node = DWA()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
