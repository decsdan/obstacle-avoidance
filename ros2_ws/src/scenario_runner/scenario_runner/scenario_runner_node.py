#!/usr/bin/env python3
"""Episode lifecycle driver; serves the ``RunBatch`` action.

Loads a manifest, runs each scenario's episode lifecycle (PRD \u00a74.2),
and produces matched rosbag2 + HDF5 sidecar artifacts per episode
(PRD \u00a74.4). Terminal classification lives in
``scenario_runner.terminal_detector`` so the same logic can be unit
tested in isolation.

This node is the one node in the stack that uses wall-clock timeouts
(for service calls to Gazebo). Every control-path decision still reads
``use_sim_time``-backed clocks so determinism is preserved.
"""

import math
import os
import sys
import time
from typing import Optional

import numpy as np
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped, Twist, TwistStamped
from nav_interfaces.action import Navigate, RunBatch
from nav_interfaces.srv import GetGridSnapshot
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool
from tf2_ros import (
    Buffer,
    ConnectivityException,
    ExtrapolationException,
    LookupException,
    TransformListener,
)

from scenario_runner.episode import Episode, EpisodePaths, pose_from_transform
from scenario_runner.dataset_writer import TickRecord
from scenario_runner.scenario_spec import (
    Manifest, Scenario, ScenarioSchemaError, load_manifest, load_scenario,
)
from scenario_runner.versioning import capture as capture_env


class ScenarioRunnerNode(Node):
    """Drives episodes end-to-end; implements the RunBatch action server."""

    def __init__(self):
        ns = '/don'
        for i, arg in enumerate(sys.argv):
            if arg.startswith('namespace:='):
                ns = arg.split(':=', 1)[1]
            elif arg == '-p' and i + 1 < len(sys.argv) and \
                    sys.argv[i + 1].startswith('namespace:='):
                ns = sys.argv[i + 1].split(':=', 1)[1]

        user_args = sys.argv[1:]
        tf_remaps = ['-r', f'/tf:={ns}/tf', '-r', f'/tf_static:={ns}/tf_static']
        if '--ros-args' in user_args:
            combined = user_args + tf_remaps
        else:
            combined = user_args + ['--ros-args'] + tf_remaps

        super().__init__(
            'scenario_runner',
            cli_args=combined,
            use_global_arguments=False,
        )

        self.declare_parameter('namespace', '/don')
        self.declare_parameter('tick_hz', 10.0)
        self.declare_parameter('repo_root', os.getcwd())
        self.declare_parameter('scenarios_dir', '')
        self.declare_parameter('reset_mode', 'service')  # service | relaunch
        self.declare_parameter('gazebo_world_name', 'default')
        self.declare_parameter('entity_name', 'turtlebot4')

        self.ns = self.get_parameter('namespace').value
        self.tick_hz = float(self.get_parameter('tick_hz').value)
        self.repo_root = self.get_parameter('repo_root').value
        self.scenarios_dir = self.get_parameter('scenarios_dir').value
        self.reset_mode = self.get_parameter('reset_mode').value
        self.gazebo_world_name = self.get_parameter('gazebo_world_name').value
        self.entity_name = self.get_parameter('entity_name').value

        self._cb_group = ReentrantCallbackGroup()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self._scan: Optional[LaserScan] = None
        self._odom: Optional[Odometry] = None
        self._cmd_vel = (0.0, 0.0)
        self._active_path: Optional[Path] = None
        self._waypoint_idx = -1
        self._collision = False
        self._episode: Optional[Episode] = None
        self._goal_handle = None
        self._nav_terminal: Optional[str] = None

        self.create_subscription(
            LaserScan, f'{self.ns}/scan',
            self._on_scan, qos_profile_sensor_data,
            callback_group=self._cb_group)
        self.create_subscription(
            Odometry, f'{self.ns}/odom',
            self._on_odom, qos_profile_sensor_data,
            callback_group=self._cb_group)
        self.create_subscription(
            TwistStamped, f'{self.ns}/cmd_vel',
            self._on_cmd_vel_stamped,
            QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=10),
            callback_group=self._cb_group)
        latched = QoSProfile(
            depth=1,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(
            Path, f'{self.ns}/nav_server/active_path',
            self._on_active_path, latched,
            callback_group=self._cb_group)
        self.create_subscription(
            Bool, f'{self.ns}/conditions/collision',
            self._on_collision, latched,
            callback_group=self._cb_group)

        self._grid_client = self.create_client(
            GetGridSnapshot, f'{self.ns}/get_grid_snapshot',
            callback_group=self._cb_group)
        self._nav_client = ActionClient(
            self, Navigate, f'{self.ns}/navigate',
            callback_group=self._cb_group)

        self._run_batch = ActionServer(
            self,
            RunBatch,
            f'{self.ns}/run_batch',
            execute_callback=self._execute_run_batch,
            goal_callback=lambda _goal: GoalResponse.ACCEPT,
            cancel_callback=lambda _goal: CancelResponse.ACCEPT,
            callback_group=self._cb_group,
        )

        self.get_logger().info(
            f'scenario_runner ready | ns={self.ns} tick_hz={self.tick_hz} '
            f'reset_mode={self.reset_mode}')

    # ------------------------------------------------------------------
    # Subscription callbacks

    def _on_scan(self, msg: LaserScan):
        self._scan = msg

    def _on_odom(self, msg: Odometry):
        self._odom = msg

    def _on_cmd_vel_stamped(self, msg: TwistStamped):
        self._cmd_vel = (msg.twist.linear.x, msg.twist.angular.z)

    def _on_active_path(self, msg: Path):
        self._active_path = msg

    def _on_collision(self, msg: Bool):
        if msg.data:
            self._collision = True
            if self._episode is not None:
                self._episode.detector.note_collision()

    # ------------------------------------------------------------------
    # RunBatch action server

    async def _execute_run_batch(self, goal_handle):
        goal = goal_handle.request
        feedback = RunBatch.Feedback()
        result = RunBatch.Result()

        try:
            manifest = load_manifest(goal.manifest_path)
        except (FileNotFoundError, ScenarioSchemaError) as exc:
            self.get_logger().error(f'manifest load failed: {exc}')
            goal_handle.abort()
            result.output_dir = goal.output_dir
            return result

        result.output_dir = goal.output_dir
        counts = {
            'reached': 0, 'collision': 0, 'timeout': 0,
            'stuck': 0, 'cancelled': 0, 'failed': 0,
        }

        env_versions = capture_env(self.repo_root)

        for i, entry in enumerate(manifest.entries):
            if i < goal.start_index:
                continue
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                return result

            scenario_path = self._resolve_scenario_path(entry.scenario_id)
            try:
                scenario = load_scenario(scenario_path)
            except (FileNotFoundError, ScenarioSchemaError) as exc:
                self.get_logger().error(
                    f'scenario load failed: {entry.scenario_id}: {exc}')
                counts['failed'] += 1
                if goal.stop_on_first_failure:
                    break
                continue

            seed = entry.seed_override if entry.seed_override is not None \
                else scenario.seed

            feedback.current_index = i
            feedback.current_scenario_id = scenario.id
            feedback.current_outcome = ''
            goal_handle.publish_feedback(feedback)

            with open(scenario_path, 'r') as handle:
                scenario_yaml_text = handle.read()
            paths = EpisodePaths.for_episode(
                goal.output_dir, scenario.id, seed)

            outcome = self._run_episode(
                scenario=scenario,
                seed=seed,
                paths=paths,
                env_versions=env_versions,
                scenario_yaml_text=scenario_yaml_text,
            )
            counts[outcome] = counts.get(outcome, 0) + 1

            feedback.current_outcome = outcome
            goal_handle.publish_feedback(feedback)

            if goal.stop_on_first_failure and outcome in ('failed', 'collision'):
                break

        result.episodes_run = sum(counts.values())
        result.episodes_reached = counts.get('reached', 0)
        result.episodes_collision = counts.get('collision', 0)
        result.episodes_timeout = counts.get('timeout', 0)
        result.episodes_stuck = counts.get('stuck', 0)
        goal_handle.succeed()
        return result

    # ------------------------------------------------------------------
    # Episode lifecycle

    def _run_episode(
        self,
        scenario: Scenario,
        seed: int,
        paths: EpisodePaths,
        env_versions,
        scenario_yaml_text: str,
    ) -> str:
        """Run one episode start to finish; return the terminal outcome string."""
        self._reset_per_episode_state()

        if not self._reset_world(scenario):
            self.get_logger().error(
                f'[{scenario.id}] world reset failed; marking episode failed')
            return 'failed'

        if not self._nav_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error(
                f'[{scenario.id}] navigate action server unavailable')
            return 'failed'

        episode = Episode(
            scenario=scenario,
            seed=seed,
            paths=paths,
            tick_hz=self.tick_hz,
            env_versions=env_versions,
            scenario_yaml_text=scenario_yaml_text,
        )
        self._episode = episode

        goal = Navigate.Goal()
        goal.goal = PoseStamped()
        goal.goal.header.frame_id = 'map'
        goal.goal.pose.position.x = scenario.goal_pose.x
        goal.goal.pose.position.y = scenario.goal_pose.y
        half = 0.5 * scenario.goal_pose.theta
        goal.goal.pose.orientation.z = math.sin(half)
        goal.goal.pose.orientation.w = math.cos(half)
        goal.global_planner = scenario.strategy.global_planner
        goal.local_planner = scenario.strategy.local_planner
        goal.global_budget = 0.0
        goal.scenario_id = scenario.id

        send_future = self._nav_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=5.0)
        if not send_future.done():
            self.get_logger().error(
                f'[{scenario.id}] send_goal timed out')
            episode.close(terminal_outcome='failed')
            self._episode = None
            return 'failed'

        self._goal_handle = send_future.result()
        if self._goal_handle is None or not self._goal_handle.accepted:
            self.get_logger().error(f'[{scenario.id}] navigate goal rejected')
            episode.close(terminal_outcome='failed')
            self._episode = None
            return 'failed'

        result_future = self._goal_handle.get_result_async()

        outcome = self._tick_until_terminal(episode, scenario, result_future)

        if not result_future.done() and self._goal_handle is not None:
            self._goal_handle.cancel_goal_async()

        episode.close(terminal_outcome=outcome)
        self._episode = None
        self._goal_handle = None
        return outcome

    def _tick_until_terminal(
        self,
        episode: Episode,
        scenario: Scenario,
        result_future,
    ) -> str:
        period = 1.0 / self.tick_hz
        next_tick = time.monotonic()
        goal_xy = (scenario.goal_pose.x, scenario.goal_pose.y)

        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=period)

            record = self._capture_tick(episode, scenario)
            if record is not None:
                episode.record_tick(record)

                terminal = episode.detector.update(
                    sim_time=record.timestamp_sim,
                    pose_xy=(record.pose[0], record.pose[1]),
                    goal_xy=goal_xy,
                )
                if terminal is not None:
                    return terminal

            if result_future.done():
                result_wrap = result_future.result()
                outcome = ''
                if result_wrap is not None and result_wrap.result is not None:
                    outcome = getattr(
                        result_wrap.result, 'terminal_outcome', '') or 'failed'
                if outcome:
                    return outcome
                return 'failed'

            next_tick += period
            sleep = next_tick - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_tick = time.monotonic()

        return 'cancelled'

    # ------------------------------------------------------------------
    # Per-tick data capture

    def _capture_tick(
        self,
        episode: Episode,
        scenario: Scenario,
    ) -> Optional[TickRecord]:
        pose = self._current_pose()
        if pose is None:
            return None

        grid, grid_origin, grid_res = self._fetch_grid_snapshot()
        if grid is None:
            return None

        scan_arr = None
        if scenario.logging.include_raw_scan and self._scan is not None:
            scan_arr = np.asarray(self._scan.ranges, dtype=np.float32)

        if not episode._writer_open:
            scan_len = scan_arr.shape[0] if scan_arr is not None else 0
            episode.ensure_writer_open(grid.shape, scan_len)

        path_arr = self._active_path_as_array()

        vel = (0.0, 0.0)
        if self._odom is not None:
            vel = (
                float(self._odom.twist.twist.linear.x),
                float(self._odom.twist.twist.angular.z),
            )

        sim_time = self.get_clock().now().nanoseconds / 1e9
        return TickRecord(
            timestamp_sim=sim_time,
            timestamp_wall=time.time(),
            tick_index=episode.tick_index,
            pose=pose,
            velocity=vel,
            goal=(scenario.goal_pose.x, scenario.goal_pose.y,
                  scenario.goal_pose.theta),
            cmd_vel=self._cmd_vel,
            path_waypoint_idx=self._waypoint_idx,
            obstacle_grid=grid,
            grid_origin=grid_origin,
            grid_resolution=grid_res,
            global_path=path_arr,
            scan_ranges=scan_arr,
        )

    def _current_pose(self) -> Optional[tuple[float, float, float]]:
        try:
            tf = self.tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time())
        except (LookupException, ConnectivityException, ExtrapolationException):
            return None
        return pose_from_transform(tf.transform.translation, tf.transform.rotation)

    def _fetch_grid_snapshot(self):
        if not self._grid_client.wait_for_service(timeout_sec=0.01):
            return (None, None, None)
        req = GetGridSnapshot.Request()
        future = self._grid_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=0.5)
        if not future.done():
            return (None, None, None)
        resp = future.result()
        raw = resp.raw
        if raw.info.width == 0 or raw.info.height == 0:
            return (None, None, None)
        arr = np.asarray(raw.data, dtype=np.int16).reshape(
            raw.info.height, raw.info.width)
        # Map 0..100 OccupancyGrid to 0..255 uint8 (PRD: "Log-odds grid, clipped to uint8")
        packed = np.clip(arr * 255 // 100, 0, 255).astype(np.uint8)
        origin = (
            float(raw.info.origin.position.x),
            float(raw.info.origin.position.y),
            0.0,
        )
        return (packed, origin, float(raw.info.resolution))

    def _active_path_as_array(self) -> np.ndarray:
        if self._active_path is None or not self._active_path.poses:
            return np.zeros((0, 3), dtype=np.float32)
        out = np.empty((len(self._active_path.poses), 3), dtype=np.float32)
        for i, ps in enumerate(self._active_path.poses):
            yaw = 2.0 * math.atan2(
                ps.pose.orientation.z, ps.pose.orientation.w)
            out[i] = (ps.pose.position.x, ps.pose.position.y, yaw)
        return out

    # ------------------------------------------------------------------
    # Helpers

    def _reset_per_episode_state(self) -> None:
        self._collision = False
        self._waypoint_idx = -1
        self._nav_terminal = None

    def _reset_world(self, scenario: Scenario) -> bool:
        """Best-effort Gazebo reset; tolerate unavailable services."""
        from scenario_runner.gazebo_reset import is_available
        if not is_available():
            self.get_logger().warn(
                'ros_gz_interfaces unavailable; skipping world reset')
            return True
        from scenario_runner.gazebo_reset import GazeboResetClient
        client = GazeboResetClient(self, self.gazebo_world_name)
        if not client.wait_for_services(timeout_sec=2.0):
            self.get_logger().warn(
                'gazebo reset services unavailable; proceeding without reset')
            return True
        if not client.reset_world():
            return False
        if not client.teleport(
                self.entity_name,
                scenario.start_pose.x,
                scenario.start_pose.y,
                scenario.start_pose.theta):
            return False
        return True

    def _resolve_scenario_path(self, scenario_id: str) -> str:
        if self.scenarios_dir:
            return os.path.join(self.scenarios_dir, f'{scenario_id}.yaml')
        share = get_package_share_directory('scenario_runner')
        return os.path.join(share, 'scenarios', f'{scenario_id}.yaml')


def main(args=None):
    rclpy.init(args=args)
    node = ScenarioRunnerNode()
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
