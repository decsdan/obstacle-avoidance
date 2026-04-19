#!/usr/bin/env python3
# Originally authored by the 2025 Carleton Senior Capstone Project
# (see AUTHORS.md). Substantially rewritten by Daniel Scheider, 2026.
"""Dynamic Window Approach local planner.

Follower only, the navigation server sends a reference path via the
``FollowPath`` action, and this node drives the robot along it until the
endpoint is reached, the path is blocked, or the goal is cancelled. It
does not listen for goal poses or global paths on its own.
"""

import math
import time

import numpy as np
from scipy.ndimage import distance_transform_edt, zoom
from scipy.spatial import cKDTree

import rclpy
import rclpy.duration
import rclpy.time
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Point, Twist, TwistStamped
from nav_interfaces.action import FollowPath
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import ColorRGBA
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker


class DWAConstants:
    """Default configuration constants."""

    DEFAULT_NAMESPACE = '/don'
    BLOCKED_TICKS_TO_ABORT = 10   # rejected-trajectory ticks before path_blocked
    STUCK_TIMEOUT = 8.0           # seconds e-stopped before giving up
    DATA_WAIT_TIMEOUT = 5.0       # seconds to wait for first odom/scan


class DWAFollower(Node):
    """FollowPath action server that tracks a reference path using DWA."""

    def __init__(self):
        super().__init__('dwa_follower')

        self._declare_params()
        self._read_params()

        self._cbg = ReentrantCallbackGroup()

        # Cached sensor state.
        self._scan_msg = None
        self._odom_msg = None
        self._shared_grid_msg = None
        self._warned_no_shared_grid = False

        # Publishers / subscribers.
        qos_cmd = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=10)
        self._twist_pub = self.create_publisher(
            TwistStamped, f'{self.ns}/cmd_vel', qos_cmd)
        self._debug_pub = self.create_publisher(
            Marker, f'{self.ns}/debug_obstacles', 10)
        self._traj_pub = self.create_publisher(
            Marker, f'{self.ns}/dwa/trajectories', 10)
        self._best_traj_pub = self.create_publisher(
            Marker, f'{self.ns}/dwa/best_trajectory', 10)

        self.create_subscription(
            Odometry, f'{self.ns}/odom',
            self._odom_cb, qos_profile_sensor_data,
            callback_group=self._cbg)
        self.create_subscription(
            LaserScan, f'{self.ns}/scan',
            self._scan_cb, qos_profile_sensor_data,
            callback_group=self._cbg)
        self.create_subscription(
            OccupancyGrid, f'{self.ns}/obstacle_grid',
            self._grid_cb, 10,
            callback_group=self._cbg)

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._active_frame = 'odom'

        # Action server.
        self._action_server = ActionServer(
            self, FollowPath, f'{self.ns}/dwa/follow_path',
            execute_callback=self._execute,
            goal_callback=self._goal_cb,
            cancel_callback=self._cancel_cb,
            callback_group=self._cbg,
        )

        self.get_logger().info(
            f'dwa follower ready on {self.ns}/dwa/follow_path')

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------

    def _declare_params(self):
        self.declare_parameter('namespace', DWAConstants.DEFAULT_NAMESPACE)
        self.declare_parameter('lookahead', 0.85)

        self.declare_parameter('max_velocity', 0.31)
        self.declare_parameter('min_velocity', 0.0)
        self.declare_parameter('max_angular_velocity', 1.8)
        self.declare_parameter('min_angular_velocity', -1.8)
        self.declare_parameter('max_linear_acceleration', 0.5)
        self.declare_parameter('max_angular_acceleration', 2.0)
        self.declare_parameter('v_samples', 20)
        self.declare_parameter('w_samples', 20)
        self.declare_parameter('lidar_angle_offset', 1.5708)

        self.declare_parameter('dt', 0.1)
        self.declare_parameter('prediction_steps', 25)
        self.declare_parameter('window_steps', 5)
        self.declare_parameter('LIDAR_downsample', 2)
        self.declare_parameter('max_path_deviation', 1.0)

        self.declare_parameter('grid_size', 161)
        self.declare_parameter('grid_resolution', 10.0)

        self.declare_parameter('critical_radius', 0.20)
        self.declare_parameter('emergency_stop_distance', 0.17)
        self.declare_parameter('max_lidar_range', 8.0)
        self.declare_parameter('goal_tolerance', 0.2)

        self.declare_parameter('weights.goal', 0.35)
        self.declare_parameter('weights.velocity', 0.10)
        self.declare_parameter('weights.smoothness', 0.05)
        self.declare_parameter('weights.obstacle', 0.40)
        self.declare_parameter('weights.dist_path', 0.10)
        self.declare_parameter('weights.heading_path', 0.05)

        self.declare_parameter('visualize_trajectories', True)
        self.declare_parameter('trajectory_visualization_downsample', 10)

        self.declare_parameter('blocked_ticks_to_abort',
                               DWAConstants.BLOCKED_TICKS_TO_ABORT)
        self.declare_parameter('stuck_timeout', DWAConstants.STUCK_TIMEOUT)

    def _read_params(self):
        self.ns = self.get_parameter('namespace').value
        self.lookahead = self.get_parameter('lookahead').value

        self.max_v = self.get_parameter('max_velocity').value
        self.min_v = self.get_parameter('min_velocity').value
        self.max_w = self.get_parameter('max_angular_velocity').value
        self.min_w = self.get_parameter('min_angular_velocity').value
        self.v_accel = self.get_parameter('max_linear_acceleration').value
        self.w_accel = self.get_parameter('max_angular_acceleration').value
        self.v_samples = self.get_parameter('v_samples').value
        self.w_samples = self.get_parameter('w_samples').value
        self.lidar_offset = self.get_parameter('lidar_angle_offset').value

        self.dt = self.get_parameter('dt').value
        self.steps = self.get_parameter('prediction_steps').value
        self.window_steps = self.get_parameter('window_steps').value
        self.LIDAR_downsample = self.get_parameter('LIDAR_downsample').value
        self.max_path_deviation = self.get_parameter('max_path_deviation').value

        self.grid_size = self.get_parameter('grid_size').value
        self.grid_resolution = self.get_parameter('grid_resolution').value

        self.critical_radius = self.get_parameter('critical_radius').value
        self.emergency_stop_dist = self.get_parameter(
            'emergency_stop_distance').value
        self.max_lidar_range = self.get_parameter('max_lidar_range').value
        self.goal_tolerance = self.get_parameter('goal_tolerance').value

        self.w_goal = self.get_parameter('weights.goal').value
        self.w_velocity = self.get_parameter('weights.velocity').value
        self.w_smoothness = self.get_parameter('weights.smoothness').value
        self.w_obstacle = self.get_parameter('weights.obstacle').value
        self.w_dist_path = self.get_parameter('weights.dist_path').value
        self.w_heading_path = self.get_parameter('weights.heading_path').value

        self.visualize_trajectories = self.get_parameter(
            'visualize_trajectories').value
        self.traj_downsample = self.get_parameter(
            'trajectory_visualization_downsample').value

        self.blocked_ticks_to_abort = int(
            self.get_parameter('blocked_ticks_to_abort').value)
        self.stuck_timeout = float(self.get_parameter('stuck_timeout').value)

    # ------------------------------------------------------------------
    # Subscriber callbacks
    # ------------------------------------------------------------------

    def _odom_cb(self, msg):
        self._odom_msg = msg

    def _scan_cb(self, msg):
        self._scan_msg = msg

    def _grid_cb(self, msg):
        self._shared_grid_msg = msg

    # ------------------------------------------------------------------
    # Action lifecycle
    # ------------------------------------------------------------------

    def _goal_cb(self, goal_request):
        if not goal_request.reference_path.poses:
            self.get_logger().warn(
                'FollowPath goal rejected: empty reference_path '
                '(DWA requires a path to follow).')
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _cancel_cb(self, goal_handle):
        return CancelResponse.ACCEPT

    def _execute(self, goal_handle):
        """Drive the robot along the reference path until done."""
        path_pts = np.array([
            [p.pose.position.x, p.pose.position.y]
            for p in goal_handle.request.reference_path.poses
        ])
        cumlen = _cumulative_length(path_pts)
        final_goal = path_pts[-1]

        # Wait briefly for first odom/scan if we haven't seen any yet.
        if not self._wait_for_first_data(DWAConstants.DATA_WAIT_TIMEOUT):
            return self._finalize(goal_handle, 'failed',
                                  distance=0.0, elapsed=0.0,
                                  log='no odom/scan within startup timeout')

        start_time = time.monotonic()
        last_pos = None
        distance_traveled = 0.0
        blocked_tick_count = 0
        stuck_since = None

        while rclpy.ok():
            tick_start = time.monotonic()

            if goal_handle.is_cancel_requested:
                self._publish_stop()
                return self._finalize(goal_handle, 'cancelled',
                                      distance=distance_traveled,
                                      elapsed=time.monotonic() - start_time,
                                      log='cancel requested')

            if self._odom_msg is None or self._scan_msg is None:
                # Rare mid-run sensor dropout: publish a safe stop and try again.
                self._publish_stop()
                self._sleep_until(tick_start + self.dt)
                continue

            pose = self._current_pose()
            if pose is None:
                self._publish_stop()
                self._sleep_until(tick_start + self.dt)
                continue
            curr_x, curr_y, curr_theta, curr_v, curr_w = pose

            # Distance accumulation from pose deltas.
            if last_pos is None:
                last_pos = (curr_x, curr_y)
            else:
                distance_traveled += math.hypot(
                    curr_x - last_pos[0], curr_y - last_pos[1])
                last_pos = (curr_x, curr_y)

            # Reached-goal check.
            dist_to_goal = float(np.linalg.norm(
                final_goal - np.array([curr_x, curr_y])))
            if dist_to_goal < self.goal_tolerance:
                self._publish_stop()
                self._publish_feedback(
                    goal_handle, curr_x, curr_y, curr_theta, curr_v, curr_w,
                    dist_to_goal, path_pts, state='running',
                    waypoint_index=len(path_pts) - 1)
                return self._finalize(goal_handle, 'reached',
                                      distance=distance_traveled,
                                      elapsed=time.monotonic() - start_time,
                                      log=f'goal reached, '
                                          f'dist={distance_traveled:.2f} m')

            # Front-cone e-stop: hold until clear, or give up after stuck_timeout.
            # Terminal is 'stuck' -- the local planner cannot make forward
            # progress through the immediate geometry. The global path may
            # still be valid, which is the orchestrator's cue to replan.
            if not self._front_cone_clear():
                self._publish_stop()
                if stuck_since is None:
                    stuck_since = time.monotonic()
                stuck_for = time.monotonic() - stuck_since
                if stuck_for > self.stuck_timeout:
                    return self._finalize(goal_handle, 'stuck',
                                          distance=distance_traveled,
                                          elapsed=time.monotonic() - start_time,
                                          log=f'front-cone blocked '
                                              f'{stuck_for:.1f}s')
                self._publish_feedback(
                    goal_handle, curr_x, curr_y, curr_theta, 0.0, 0.0,
                    dist_to_goal, path_pts, state='stuck',
                    waypoint_index=-1)
                self._sleep_until(tick_start + self.dt)
                continue
            stuck_since = None

            # Pick a lookahead target along the path.
            look_x, look_y = self._lookahead_target(
                path_pts, cumlen, curr_x, curr_y)

            # Sample the dynamic window and score trajectories.
            trajectories, final_vs, final_ws = self._sample_and_predict(
                curr_x, curr_y, curr_theta, curr_v, curr_w, dist_to_goal)
            obstacles = self._world_obstacles(curr_x, curr_y, curr_theta)
            self._publish_debug_obstacles(obstacles)

            scores, _min_dists = self._cost(
                trajectories, final_vs, final_ws, obstacles,
                curr_x, curr_y, dist_to_goal, final_goal,
                path_pts, look_x, look_y)
            best_idx = int(np.argmax(scores))
            self._publish_trajectory_markers(trajectories, scores, best_idx)

            if scores[best_idx] == -np.inf:
                # Dynamic-window exhaustion: every sampled velocity would
                # hit an obstacle. Semantically the same as the front-cone
                # case -- local planner cannot proceed -- so terminal is
                # 'stuck'. Feedback state still uses path_blocked to give
                # observers a finer-grained view of *why* we're stuck.
                blocked_tick_count += 1
                self._publish_stop()
                self._publish_feedback(
                    goal_handle, curr_x, curr_y, curr_theta, 0.0, 0.0,
                    dist_to_goal, path_pts, state='path_blocked',
                    waypoint_index=-1)
                if blocked_tick_count >= self.blocked_ticks_to_abort:
                    return self._finalize(
                        goal_handle, 'stuck',
                        distance=distance_traveled,
                        elapsed=time.monotonic() - start_time,
                        log=f'no viable trajectories for '
                            f'{blocked_tick_count} ticks')
                self._sleep_until(tick_start + self.dt)
                continue

            blocked_tick_count = 0
            self._publish_cmd(float(final_vs[best_idx]),
                              float(final_ws[best_idx]))

            self._publish_feedback(
                goal_handle, curr_x, curr_y, curr_theta,
                float(final_vs[best_idx]), float(final_ws[best_idx]),
                dist_to_goal, path_pts, state='running',
                waypoint_index=_nearest_idx(path_pts, curr_x, curr_y))

            self._sleep_until(tick_start + self.dt)

        # rclpy shut down mid-run.
        self._publish_stop()
        return self._finalize(goal_handle, 'failed',
                              distance=distance_traveled,
                              elapsed=time.monotonic() - start_time,
                              log='rclpy shutdown during execute')

    def _finalize(self, goal_handle, terminal, distance, elapsed, log):
        self.get_logger().info(
            f'[follow] terminal={terminal} distance={distance:.2f} '
            f'elapsed={elapsed:.2f} | {log}')
        result = FollowPath.Result()
        result.terminal_outcome = terminal
        result.total_distance = distance
        result.total_time = elapsed
        if terminal == 'cancelled':
            goal_handle.canceled()
        elif terminal == 'reached':
            goal_handle.succeed()
        else:
            goal_handle.abort()
        return result

    # ------------------------------------------------------------------
    # Feedback
    # ------------------------------------------------------------------

    def _publish_feedback(self, goal_handle, x, y, theta, v, w,
                          dist_to_goal, path_pts, state, waypoint_index):
        fb = FollowPath.Feedback()
        fb.current_pose.position.x = x
        fb.current_pose.position.y = y
        fb.current_pose.orientation.z = math.sin(theta / 2.0)
        fb.current_pose.orientation.w = math.cos(theta / 2.0)
        vel = Twist()
        vel.linear.x = v
        vel.angular.z = w
        fb.current_velocity = vel
        fb.distance_to_goal = dist_to_goal
        fb.cross_track_error = float(_cross_track_error(path_pts, x, y))
        fb.local_state = state
        fb.waypoint_index = int(waypoint_index)
        if goal_handle.status == GoalStatus.STATUS_EXECUTING:
            goal_handle.publish_feedback(fb)

    # ------------------------------------------------------------------
    # Pose / sensor helpers
    # ------------------------------------------------------------------

    def _wait_for_first_data(self, timeout_s):
        deadline = time.monotonic() + timeout_s
        while (self._odom_msg is None or self._scan_msg is None) and \
                time.monotonic() < deadline and rclpy.ok():
            time.sleep(0.05)
        return self._odom_msg is not None and self._scan_msg is not None

    def _sleep_until(self, deadline):
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)

    def _current_pose(self):
        """Return (x, y, theta, v, w) in the best available frame."""
        if self._odom_msg is None:
            return None
        map_pose = self._map_pose()
        if map_pose is not None:
            x, y, theta = map_pose
            self._active_frame = 'map'
        else:
            x = self._odom_msg.pose.pose.position.x
            y = self._odom_msg.pose.pose.position.y
            theta = _yaw(self._odom_msg.pose.pose.orientation)
            self._active_frame = 'odom'
        v = self._odom_msg.twist.twist.linear.x
        w = self._odom_msg.twist.twist.angular.z
        return x, y, theta, v, w

    def _map_pose(self):
        try:
            tf = self._tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1))
            x = tf.transform.translation.x
            y = tf.transform.translation.y
            theta = _yaw(tf.transform.rotation)
            return (x, y, theta)
        except Exception:
            return None

    def _front_cone_clear(self):
        if self._scan_msg is None:
            return True
        ranges = np.array(self._scan_msg.ranges)
        ranges = np.where(np.isfinite(ranges) & (ranges > 0.05), ranges, 10.0)
        n = len(ranges)
        if n == 0:
            return True
        angle_range = self._scan_msg.angle_max - self._scan_msg.angle_min
        front_idx = int((-self.lidar_offset - self._scan_msg.angle_min) /
                        angle_range * n) % n
        cone_width = int(n / 8)
        indices = np.arange(front_idx - cone_width, front_idx + cone_width) % n
        return np.min(ranges[indices]) >= self.emergency_stop_dist

    def _world_obstacles(self, curr_x, curr_y, curr_theta):
        if self._scan_msg is None:
            return np.array([])
        ranges = np.array(self._scan_msg.ranges)[::self.LIDAR_downsample]
        angles = np.linspace(
            self._scan_msg.angle_min, self._scan_msg.angle_max,
            len(self._scan_msg.ranges))[::self.LIDAR_downsample]
        mask = (
            np.isfinite(ranges) &
            (ranges > 0.05) &
            (ranges > self._scan_msg.range_min) &
            (ranges < self._scan_msg.range_max) &
            (ranges < self.max_lidar_range)
        )
        valid_ranges = ranges[mask]
        valid_angles = angles[mask]
        world_angles = curr_theta + valid_angles + self.lidar_offset
        ox = curr_x + valid_ranges * np.cos(world_angles)
        oy = curr_y + valid_ranges * np.sin(world_angles)
        return np.column_stack((ox, oy))

    # ------------------------------------------------------------------
    # DWA core
    # ------------------------------------------------------------------

    def _sample_and_predict(self, curr_x, curr_y, curr_theta, curr_v, curr_w,
                            dist_to_goal):
        """Sample (v, w) pairs in the dynamic window and forward-simulate."""
        # Clamp max_v near the goal so we don't overshoot.
        effective_max_v = float(
            np.clip(self.max_v * min(1.0, dist_to_goal / 0.3),
                    0.15, self.max_v))

        window_time = self.dt * self.window_steps
        v_range = self.v_accel * window_time
        w_range = self.w_accel * window_time
        v_max = min(curr_v + v_range, effective_max_v)
        v_min = max(curr_v - v_range, self.min_v)
        w_max = min(curr_w + w_range, self.max_w)
        w_min = max(curr_w - w_range, self.min_w)

        effective_steps = max(10, int(self.steps * min(1.0, dist_to_goal / 0.5)))

        poss_v = np.linspace(v_min, v_max, self.v_samples)
        poss_w = np.linspace(w_min, w_max, self.w_samples)
        v_arr, w_arr = np.meshgrid(poss_v, poss_w)
        v_arr = v_arr.flatten()
        w_arr = w_arr.flatten()

        vs = np.tile(v_arr[:, None], (1, effective_steps))
        ws = np.tile(w_arr[:, None], (1, effective_steps))
        thetas = np.cumsum(ws * self.dt, axis=1) + curr_theta
        xs = np.cumsum(vs * np.cos(thetas) * self.dt, axis=1) + curr_x
        ys = np.cumsum(vs * np.sin(thetas) * self.dt, axis=1) + curr_y
        trajectories = np.stack((xs, ys, thetas), axis=2)
        return trajectories, vs[:, -1], ws[:, -1]

    def _cost(self, trajectories, final_vs, final_ws, obstacles,
              curr_x, curr_y, dist_to_goal, final_goal,
              path_pts, look_x, look_y):
        """Score each candidate trajectory; reject ones that hit obstacles."""
        final_pos = trajectories[:, -1, :2]
        goal_dists = np.linalg.norm(final_goal - final_pos, axis=1)

        # Near-goal blend: crank up goal / heading-to-path weight.
        if dist_to_goal < 0.5:
            t_blend = 1.0 - (dist_to_goal / 0.5)
            w_goal_eff = self.w_goal + t_blend * (0.60 - self.w_goal)
            w_velocity_eff = self.w_velocity * (1.0 - t_blend)
            w_dist_path_eff = self.w_dist_path * (1.0 - t_blend * 0.7)
            w_heading_path_eff = (
                self.w_heading_path + t_blend * (0.25 - self.w_heading_path))
        else:
            w_goal_eff = self.w_goal
            w_velocity_eff = self.w_velocity
            w_dist_path_eff = self.w_dist_path
            w_heading_path_eff = self.w_heading_path

        # dp_score / hp_score: distance-to-path and heading-to-path.
        if len(path_pts) >= 2:
            A = path_pts[:-1]
            B = path_pts[1:]
            AB = B - A
            AB_len_sq = np.maximum(np.sum(AB ** 2, axis=1), 1e-10)
            PA = final_pos[:, None, :] - A[None, :, :]
            t = np.clip(np.sum(PA * AB[None, :, :], axis=2) / AB_len_sq[None, :],
                        0.0, 1.0)
            closest = A[None, :, :] + t[:, :, None] * AB[None, :, :]
            seg_dists = np.linalg.norm(final_pos[:, None, :] - closest, axis=2)
            min_path_dists = np.min(seg_dists, axis=1)
            dp_score = np.sqrt(np.clip(
                1.0 - (min_path_dists / self.max_path_deviation), 0.0, 1.0))

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

        # g_score: progress toward the final goal.
        max_prog = max(dist_to_goal, self.max_v * self.dt * trajectories.shape[1])
        progress = dist_to_goal - goal_dists
        g_score = np.clip((progress + max_prog) / (2 * max_prog), 0.0, 1.0)

        # Obstacle rejection via lidar kd-tree + shared obstacle grid cost.
        if len(obstacles) > 0:
            tree = cKDTree(obstacles)
            flat_pts = trajectories[:, :, :2].reshape(-1, 2)
            dists, _ = tree.query(flat_pts)
            min_dists = dists.reshape(
                trajectories.shape[0], trajectories.shape[1]).min(axis=1)
        else:
            min_dists = np.full(len(final_vs), self.max_lidar_range)

        dist_grid = self._extract_dist_grid(curr_x, curr_y)
        if dist_grid is None:
            if not self._warned_no_shared_grid:
                self.get_logger().warn(
                    'obstacle_grid not available; obstacle-distance scoring '
                    'disabled until it appears.')
                self._warned_no_shared_grid = True
            o_score = np.full(len(trajectories), 0.5)
        else:
            self._warned_no_shared_grid = False
            o_score = np.array(_path_costs_from_dist_grid(
                trajectories, dist_grid, curr_x, curr_y,
                self.grid_size, self.grid_resolution))

        v_score = np.clip(final_vs / self.max_v, 0.0, 1.0)
        w_score = np.clip(1.0 - (np.abs(final_ws) / self.max_w), 0.0, 1.0)

        scores = (
            w_goal_eff         * g_score +
            w_velocity_eff     * v_score +
            self.w_smoothness  * w_score +
            self.w_obstacle    * o_score +
            w_dist_path_eff    * dp_score +
            w_heading_path_eff * hp_score
        )

        crit = np.where(
            np.abs(final_vs) > 0.1,
            self.critical_radius,
            self.emergency_stop_dist + 0.01)
        scores[min_dists < crit] = -np.inf

        return scores, min_dists

    def _extract_dist_grid(self, curr_x, curr_y):
        """Return a robot-centric distance grid from the shared obstacle grid."""
        msg = self._shared_grid_msg
        if msg is None:
            return None

        shared_res = msg.info.resolution
        shared_w = msg.info.width
        shared_h = msg.info.height
        shared_ox = msg.info.origin.position.x
        shared_oy = msg.info.origin.position.y

        robot_gx = int((curr_x - shared_ox) / shared_res)
        robot_gy = int((curr_y - shared_oy) / shared_res)

        half_span_m = (self.grid_size / self.grid_resolution) / 2.0
        half_span_cells = int(half_span_m / shared_res)

        y_lo = robot_gy - half_span_cells
        y_hi = robot_gy + half_span_cells
        x_lo = robot_gx - half_span_cells
        x_hi = robot_gx + half_span_cells

        shared = np.array(msg.data, dtype=np.int8).reshape(shared_h, shared_w)

        pad_y = max(0, -y_lo, y_hi - shared_h + 1)
        pad_x = max(0, -x_lo, x_hi - shared_w + 1)
        if pad_y > 0 or pad_x > 0:
            shared_padded = np.pad(
                shared, ((pad_y, pad_y), (pad_x, pad_x)),
                mode='constant', constant_values=0)
            y_lo += pad_y
            y_hi += pad_y
            x_lo += pad_x
            x_hi += pad_x
        else:
            shared_padded = shared

        window = shared_padded[y_lo:y_hi, x_lo:x_hi]

        dwa_cell_m = 1.0 / self.grid_resolution
        scale = shared_res / dwa_cell_m
        if abs(scale - 1.0) > 0.01:
            resampled = zoom(window, scale, order=0)
        else:
            resampled = window

        target = self.grid_size
        if resampled.shape[0] != target or resampled.shape[1] != target:
            result = np.zeros((target, target), dtype=np.int8)
            h = min(resampled.shape[0], target)
            w = min(resampled.shape[1], target)
            off_y = (target - h) // 2
            off_x = (target - w) // 2
            result[off_y:off_y + h, off_x:off_x + w] = resampled[:h, :w]
            resampled = result

        obstacle_grid = np.where(resampled >= 50, 100, 0).astype(int)
        free_mask = (obstacle_grid != 100)
        return distance_transform_edt(free_mask)

    def _lookahead_target(self, path_pts, cumlen, curr_x, curr_y):
        """Pick a point `lookahead` metres ahead of the nearest path point."""
        dists = np.linalg.norm(path_pts - np.array([curr_x, curr_y]), axis=1)
        closest = int(np.argmin(dists))
        target_total = cumlen[closest] + self.lookahead
        target_idx = int(np.searchsorted(cumlen, target_total))
        target_idx = min(target_idx, len(path_pts) - 1)
        return float(path_pts[target_idx, 0]), float(path_pts[target_idx, 1])

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    def _publish_cmd(self, v, w):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.twist.linear.x = v
        msg.twist.angular.z = w
        self._twist_pub.publish(msg)

    def _publish_stop(self):
        self._publish_cmd(0.0, 0.0)

    def _publish_debug_obstacles(self, obstacles):
        if len(obstacles) == 0:
            return
        marker = Marker()
        marker.header.frame_id = self._active_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.type = Marker.SPHERE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color.a = 1.0
        marker.color.r = 1.0
        for obs in obstacles:
            p = Point()
            p.x = float(obs[0])
            p.y = float(obs[1])
            marker.points.append(p)
        self._debug_pub.publish(marker)

    def _publish_trajectory_markers(self, trajectories, scores, best_idx):
        if not self.visualize_trajectories:
            return
        now = self.get_clock().now().to_msg()

        marker = Marker()
        marker.header.frame_id = self._active_frame
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
        smin = float(np.min(valid_scores))
        smax = float(np.max(valid_scores))
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
                p2.x, p2.y, p2.z = float(traj[j + 1, 0]), float(traj[j + 1, 1]), 0.05
                marker.points.append(p1)
                marker.points.append(p2)
                c = ColorRGBA()
                c.r, c.g, c.b, c.a = r, g, b, a
                marker.colors.append(c)
                marker.colors.append(c)
        self._traj_pub.publish(marker)

        best_marker = Marker()
        best_marker.header.frame_id = self._active_frame
        best_marker.header.stamp = now
        best_marker.ns = 'best_trajectory'
        best_marker.id = 1
        best_marker.type = Marker.LINE_STRIP
        best_marker.action = Marker.ADD
        best_marker.scale.x = 0.05
        best_marker.color.g = 1.0
        best_marker.color.b = 1.0
        best_marker.color.a = 1.0
        best_marker.pose.orientation.w = 1.0
        for pt in trajectories[best_idx]:
            p = Point()
            p.x, p.y, p.z = float(pt[0]), float(pt[1]), 0.1
            best_marker.points.append(p)
        self._best_traj_pub.publish(best_marker)


# ----------------------------------------------------------------------
# Module-level helpers
# ----------------------------------------------------------------------

def _yaw(q):
    siny_cosp = 2 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def _cumulative_length(pts):
    if len(pts) == 0:
        return np.array([0.0])
    d = np.zeros(len(pts))
    d[1:] = np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))
    return d


def _nearest_idx(pts, x, y):
    if len(pts) == 0:
        return -1
    return int(np.argmin(np.linalg.norm(pts - np.array([x, y]), axis=1)))


def _cross_track_error(pts, x, y):
    if len(pts) < 2:
        return 0.0
    A = pts[:-1]
    B = pts[1:]
    AB = B - A
    AB_len_sq = np.maximum(np.sum(AB ** 2, axis=1), 1e-10)
    P = np.array([x, y])
    t = np.clip(np.sum((P - A) * AB, axis=1) / AB_len_sq, 0.0, 1.0)
    closest = A + t[:, None] * AB
    return float(np.min(np.linalg.norm(P - closest, axis=1)))


def _path_costs_from_dist_grid(trajectories, dist_grid,
                               curr_x, curr_y, grid_size, grid_resolution):
    """Score each trajectory's tail by the distance grid; higher = safer."""
    center = grid_size // 2
    max_idx = grid_size - 1

    pts = trajectories[:, :, :2]
    grid_pts = ((pts - np.array([curr_x, curr_y])) * grid_resolution +
                center).astype(int)
    gx = grid_pts[:, :, 0]
    gy = grid_pts[:, :, 1]
    in_bounds = (gx >= 0) & (gx <= max_idx) & (gy >= 0) & (gy <= max_idx)
    gx_c = np.clip(gx, 0, max_idx)
    gy_c = np.clip(gy, 0, max_idx)
    dist_vals = dist_grid[gy_c, gx_c].astype(float)

    point_costs = np.where(
        in_bounds,
        np.where(dist_vals > 0, np.exp(-0.5 * dist_vals), np.inf),
        0.0,
    )

    all_oob = ~in_bounds.any(axis=1)
    has_inf = np.isinf(point_costs).any(axis=1)

    n_tail = max(1, point_costs.shape[1] // 5)
    tail_mean = point_costs[:, -n_tail:].mean(axis=1)
    total = np.where(all_oob | has_inf, np.inf, tail_mean)

    # Min-max normalize finite costs to [0, 1]; infinite stays inf.
    finite_mask = np.isfinite(total)
    if not finite_mask.any():
        return [0.0] * len(total)
    finite_vals = total[finite_mask]
    mn = float(finite_vals.min())
    mx = float(finite_vals.max())
    if mx == mn:
        normed = np.where(finite_mask, 0.0, np.inf)
    else:
        normed = np.where(finite_mask, (total - mn) / (mx - mn), np.inf)
    return [1.0 - c if np.isfinite(c) else 0.0 for c in normed]


def main(args=None):
    rclpy.init(args=args)
    node = DWAFollower()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
