#!/usr/bin/env python3
# Originally authored by the 2025 Carleton Senior Capstone Project
# (see AUTHORS.md). Substantially rewritten by Daniel Scheider, 2026.
"""Call/response orchestrator for the obstacle-avoidance stack.

Serves the top-level ``Navigate`` action and drives global planners
(``PlanPath`` service) and local planners (``FollowPath`` action) 
This node manages all replanning for the other nodes

* ``_monitor_*`` -- read external state (TF, feedback topics).
* ``_analyze_*`` -- decide whether to adapt.
* ``_plan_*``    -- request a new strategy or path.
* ``_execute_*`` -- dispatch a strategy to the managed system.

Shared state lives on ``self.K`` (Knowledge). In v2.0 all four phases
run in the same node -- the labels are there to keep the structural
door open for a future managing-system split.
"""

import asyncio
import math
from dataclasses import dataclass, field
from typing import Optional

import rclpy
import tf2_ros
from builtin_interfaces.msg import Time as TimeMsg
from geometry_msgs.msg import Pose, PoseStamped, Twist
from nav_interfaces.action import FollowPath, Navigate
from nav_interfaces.srv import GetGridSnapshot, PlanPath
from nav_msgs.msg import Path
from rclpy.action import (
    ActionClient, ActionServer, CancelResponse, GoalResponse,
)
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy,
)
from rclpy.time import Time
from std_msgs.msg import Bool
from tf2_ros import Buffer, TransformListener


KNOWN_GLOBALS = {'a_star', 'd_star', 'jps', 'rrt_star', 'fm2'}
KNOWN_LOCALS = {'dwa', 'mppi', 'rl_policy'}


@dataclass
class NavKnowledge:
    """The K in MAPE-K. All state observed or decided by the orchestrator."""

    # Current goal context
    goal: Optional[PoseStamped] = None
    scenario_id: str = ''
    global_planner: str = ''
    local_planner: str = ''
    global_budget: float = 0.0

    # Plan state
    active_path: Optional[Path] = None
    last_path_stamp: Optional[TimeMsg] = None
    last_grid_stamp: Optional[TimeMsg] = None

    # Pose/velocity (from TF or FollowPath feedback, whichever is fresher)
    current_pose: Pose = field(default_factory=Pose)
    current_velocity: Twist = field(default_factory=Twist)
    pose_valid: bool = False

    # Derived metrics
    distance_to_goal: float = -1.0
    distance_traveled: float = 0.0
    cross_track_error: float = 0.0
    local_state: str = ''  # mirrors FollowPath feedback: running|path_blocked|stuck
    waypoint_index: int = -1

    # Orchestrator state machine; values restricted to the Navigate feedback
    # enum (planning | following | replanning | recovering).
    nav_state: str = 'planning'
    replan_count: int = 0
    replan_reason: str = ''

    # Terminal outcome from the FollowPath action, set by its result callback
    follow_terminal: Optional[str] = None

    # Safety mux latched flag; any transition to True forces an abort
    safety_latched: bool = False

    # Timing
    start_time: Optional[Time] = None


class NavigationServer(Node):
    """Orchestrates call/response navigation; implements the Navigate action."""

    def __init__(self):
        super().__init__('navigation_server')

        # ------------------------------------------------------------------
        # Parameters
        # ------------------------------------------------------------------
        self.declare_parameter('namespace', '/don')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('default_global_planner', 'a_star')
        self.declare_parameter('default_local_planner', 'dwa')
        self.declare_parameter('navigation_timeout', 300.0)
        self.declare_parameter('plan_timeout', 5.0)
        self.declare_parameter('plan_budget_default', 0.0)
        self.declare_parameter('feedback_hz', 10.0)
        self.declare_parameter('replan_on_path_blocked', True)
        self.declare_parameter('replan_deviation_threshold', 0.5)
        # Periodic replanning is OFF by default; replan decisions must be
        # deterministic functions of grid/path, not wall-clock timers.
        self.declare_parameter('replan_period_sec', 0.0)

        p = self.get_parameter
        self.ns = p('namespace').value
        self.map_frame = p('map_frame').value
        self.base_frame = p('base_frame').value
        self.default_global = p('default_global_planner').value
        self.default_local = p('default_local_planner').value
        self.nav_timeout = float(p('navigation_timeout').value)
        self.plan_timeout = float(p('plan_timeout').value)
        self.plan_budget_default = float(p('plan_budget_default').value)
        self.feedback_period = 1.0 / float(p('feedback_hz').value)
        self.replan_on_path_blocked = bool(p('replan_on_path_blocked').value)
        self.replan_deviation_m = float(p('replan_deviation_threshold').value)
        self.replan_period_sec = float(p('replan_period_sec').value)

        # ------------------------------------------------------------------
        # Callback groups & knowledge
        # ------------------------------------------------------------------
        self._cbg = ReentrantCallbackGroup()
        self.K = NavKnowledge()
        self._last_pose_for_distance: Optional[Pose] = None
        self._follow_goal_handle = None
        self._navigate_goal_handle = None

        # ------------------------------------------------------------------
        # TF listener -- pose source during planning phase (Monitor)
        # ------------------------------------------------------------------
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        # ------------------------------------------------------------------
        # Clients
        # ------------------------------------------------------------------
        self._plan_clients = {}
        self._follow_clients = {}
        self._grid_client = self.create_client(
            GetGridSnapshot, f'{self.ns}/get_grid_snapshot',
            callback_group=self._cbg)

        # ------------------------------------------------------------------
        # Safety mux latch subscription -- observe-only; the mux owns the
        # cmd_vel override path directly and does not depend on this node.
        # ------------------------------------------------------------------
        latched_qos = QoSProfile(
            depth=1,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._safety_sub = self.create_subscription(
            Bool, f'{self.ns}/safety/latched',
            self._monitor_safety_latched, latched_qos,
            callback_group=self._cbg)

        # ------------------------------------------------------------------
        # Navigate action server
        # ------------------------------------------------------------------
        self._action_server = ActionServer(
            self,
            Navigate,
            f'{self.ns}/navigate',
            execute_callback=self._execute_navigate,
            goal_callback=self._monitor_accept_goal,
            cancel_callback=self._monitor_accept_cancel,
            callback_group=self._cbg,
        )

        # ------------------------------------------------------------------
        # RViz convenience: single-click 2D Goal Pose dispatcher
        # ------------------------------------------------------------------
        self._rviz_sub = self.create_subscription(
            PoseStamped, f'{self.ns}/goal_pose',
            self._monitor_rviz_goal, 10, callback_group=self._cbg)
        self._rviz_self_client = ActionClient(
            self, Navigate, f'{self.ns}/navigate',
            callback_group=self._cbg)
        self._rviz_goal_handle = None

        self.get_logger().info(
            f'nav_server ready | ns={self.ns} | '
            f'default_global={self.default_global} '
            f'default_local={self.default_local} | '
            f'feedback_hz={1.0 / self.feedback_period:.1f}')

    # ----------------------------------------------------------------------
    # RViz convenience entry point
    # ----------------------------------------------------------------------

    def _monitor_rviz_goal(self, msg: PoseStamped):
        """Route an RViz goal_pose through our own Navigate action."""
        if not self._rviz_self_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().warn(
                'Own Navigate server not reachable from RViz hook')
            return

        goal = Navigate.Goal()
        goal.goal = msg
        goal.global_planner = self.default_global
        goal.local_planner = self.default_local
        goal.global_budget = self.plan_budget_default
        goal.scenario_id = 'rviz'

        self.get_logger().info(
            f'RViz goal → ({msg.pose.position.x:.2f}, '
            f'{msg.pose.position.y:.2f})')

        if self._rviz_goal_handle is not None:
            self._rviz_goal_handle.cancel_goal_async()
            self._rviz_goal_handle = None

        self._rviz_self_client.send_goal_async(goal).add_done_callback(
            self._rviz_goal_accepted)

    def _rviz_goal_accepted(self, future):
        handle = future.result()
        if handle.accepted:
            self._rviz_goal_handle = handle

    # ----------------------------------------------------------------------
    # Monitor: action-server accept gates
    # ----------------------------------------------------------------------

    def _monitor_accept_goal(self, goal_request):
        g = goal_request.global_planner.lower() or self.default_global
        loc = goal_request.local_planner.lower() or self.default_local
        if g not in KNOWN_GLOBALS:
            self.get_logger().error(
                f'Unknown global_planner "{g}"; valid: {sorted(KNOWN_GLOBALS)}')
            return GoalResponse.REJECT
        if loc not in KNOWN_LOCALS:
            self.get_logger().error(
                f'Unknown local_planner "{loc}"; valid: {sorted(KNOWN_LOCALS)}')
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _monitor_accept_cancel(self, goal_handle):
        return CancelResponse.ACCEPT

    # ----------------------------------------------------------------------
    # Execute: the main call/response loop
    # ----------------------------------------------------------------------

    async def _execute_navigate(self, goal_handle):
        req = goal_handle.request
        self.K = NavKnowledge(
            goal=req.goal,
            scenario_id=req.scenario_id,
            global_planner=(req.global_planner.lower() or self.default_global),
            local_planner=(req.local_planner.lower() or self.default_local),
            global_budget=(req.global_budget if req.global_budget > 0.0
                           else self.plan_budget_default),
            start_time=self.get_clock().now(),
        )
        self._last_pose_for_distance = None
        self._navigate_goal_handle = goal_handle

        self.get_logger().info(
            f'[execute] scenario={self.K.scenario_id or "-"} '
            f'global={self.K.global_planner} '
            f'local={self.K.local_planner} '
            f'goal=({req.goal.pose.position.x:.2f}, '
            f'{req.goal.pose.position.y:.2f})')

        # Plan once up front; replan is driven by analyze-phase triggers.
        if not await self._plan_request_new_path('initial'):
            goal_handle.abort()
            return self._finalize_result('failed', success=False)

        if not await self._execute_start_follow():
            goal_handle.abort()
            return self._finalize_result('failed', success=False)

        last_periodic_replan = self.get_clock().now()
        while rclpy.ok():
            if goal_handle.is_cancel_requested:
                await self._execute_cancel_follow()
                goal_handle.canceled()
                return self._finalize_result('cancelled', success=False)

            elapsed = self._elapsed(self.K.start_time)
            if self.nav_timeout > 0 and elapsed > self.nav_timeout:
                await self._execute_cancel_follow()
                goal_handle.abort()
                return self._finalize_result('timeout', success=False)

            if self.K.safety_latched:
                await self._execute_cancel_follow()
                goal_handle.abort()
                return self._finalize_result('failed', success=False)

            # Monitor: refresh pose from TF when FollowPath isn't feeding us
            if self.K.nav_state != 'following':
                self._monitor_pose_from_tf()

            self._publish_navigate_feedback(goal_handle, elapsed)

            # Did the FollowPath action terminate?
            if self.K.follow_terminal is not None:
                outcome = self.K.follow_terminal
                self.K.follow_terminal = None
                self._follow_goal_handle = None

                if outcome == 'reached':
                    goal_handle.succeed()
                    return self._finalize_result('reached', success=True)

                if outcome == 'path_blocked' and self.replan_on_path_blocked:
                    if await self._analyze_and_replan('path_blocked'):
                        continue
                    goal_handle.abort()
                    return self._finalize_result('failed', success=False)

                # cancelled or failed from the local planner itself
                goal_handle.abort()
                return self._finalize_result(outcome, success=False)

            # Analyze-phase replan triggers
            if self._analyze_deviation_exceeded():
                await self._analyze_and_replan('deviation')
            elif (self.replan_period_sec > 0.0 and
                  self._elapsed(last_periodic_replan) > self.replan_period_sec):
                await self._analyze_and_replan('periodic')
                last_periodic_replan = self.get_clock().now()

            await asyncio.sleep(self.feedback_period)

        # rclpy shutdown during execution
        await self._execute_cancel_follow()
        goal_handle.abort()
        return self._finalize_result('failed', success=False)

    # ----------------------------------------------------------------------
    # Monitor
    # ----------------------------------------------------------------------

    def _monitor_pose_from_tf(self):
        try:
            t = self._tf_buffer.lookup_transform(
                self.map_frame, self.base_frame, Time())
        except (tf2_ros.LookupException,
                tf2_ros.ExtrapolationException,
                tf2_ros.ConnectivityException):
            return
        self.K.current_pose.position.x = t.transform.translation.x
        self.K.current_pose.position.y = t.transform.translation.y
        self.K.current_pose.position.z = t.transform.translation.z
        self.K.current_pose.orientation = t.transform.rotation
        self.K.pose_valid = True
        self.K.distance_to_goal = self._distance_to_goal()

    def _monitor_safety_latched(self, msg: Bool):
        """Latched-QoS subscription for the safety mux override signal."""
        was_latched = self.K.safety_latched
        self.K.safety_latched = bool(msg.data)
        if self.K.safety_latched and not was_latched:
            self.get_logger().warn(
                '[monitor] safety mux latched -- aborting active Navigate goal')

    def _monitor_follow_feedback(self, feedback_msg):
        fb = feedback_msg.feedback
        new_pose = fb.current_pose
        # Accumulate distance_traveled locally; FollowPath doesn't publish it.
        if self._last_pose_for_distance is not None:
            dx = new_pose.position.x - self._last_pose_for_distance.position.x
            dy = new_pose.position.y - self._last_pose_for_distance.position.y
            self.K.distance_traveled += math.hypot(dx, dy)
        self._last_pose_for_distance = new_pose

        self.K.current_pose = new_pose
        self.K.current_velocity = fb.current_velocity
        self.K.distance_to_goal = fb.distance_to_goal
        self.K.cross_track_error = fb.cross_track_error
        self.K.local_state = fb.local_state
        self.K.waypoint_index = fb.waypoint_index
        self.K.pose_valid = True

    # ----------------------------------------------------------------------
    # Analyze
    # ----------------------------------------------------------------------

    def _analyze_deviation_exceeded(self) -> bool:
        """True when the robot has drifted beyond the threshold from the path."""
        if not self.K.pose_valid or self.K.active_path is None:
            return False
        if not self.K.active_path.poses:
            return False
        px = self.K.current_pose.position.x
        py = self.K.current_pose.position.y
        best_sq = math.inf
        for p in self.K.active_path.poses:
            dx = p.pose.position.x - px
            dy = p.pose.position.y - py
            d2 = dx * dx + dy * dy
            if d2 < best_sq:
                best_sq = d2
        return math.sqrt(best_sq) > self.replan_deviation_m

    async def _analyze_and_replan(self, reason: str) -> bool:
        self.K.replan_reason = reason
        self.K.replan_count += 1
        self.K.nav_state = 'replanning'
        self.get_logger().info(f'[analyze] replan triggered: {reason}')

        await self._execute_cancel_follow()
        if not await self._plan_request_new_path(reason):
            return False
        return await self._execute_start_follow()

    # ----------------------------------------------------------------------
    # Plan
    # ----------------------------------------------------------------------

    async def _plan_request_new_path(self, reason: str) -> bool:
        self.K.nav_state = 'planning'
        self._monitor_pose_from_tf()

        snap = await self._plan_get_grid_snapshot()
        if snap is None:
            self.get_logger().error('[plan] grid snapshot unavailable')
            return False
        self.K.last_grid_stamp = snap.stamp

        client = self._plan_client_for(self.K.global_planner)
        if not client.wait_for_service(timeout_sec=self.plan_timeout):
            self.get_logger().error(
                f'[plan] PlanPath service unavailable for '
                f'{self.K.global_planner}')
            return False

        req = PlanPath.Request()
        req.grid_snapshot = snap.inflated
        req.start = self._current_pose_stamped()
        req.goal = self.K.goal
        req.budget = self.K.global_budget
        req.timeout = self.plan_timeout

        self.get_logger().info(
            f'[plan] PlanPath({self.K.global_planner}, reason={reason}) '
            f'budget={req.budget:.0f} timeout={req.timeout:.1f}s')

        future = client.call_async(req)
        await future
        resp = future.result()
        if resp is None or not resp.success:
            fr = resp.failure_reason if resp is not None else 'NO_RESPONSE'
            self.get_logger().error(f'[plan] PlanPath failed: {fr}')
            return False

        self.K.active_path = resp.path
        self.K.last_path_stamp = self.get_clock().now().to_msg()
        self.get_logger().info(
            f'[plan] path received: {len(resp.path.poses)} poses in '
            f'{resp.compute_time:.3f}s '
            f'(nodes_expanded={resp.nodes_expanded}, cached={resp.cached})')
        return True

    async def _plan_get_grid_snapshot(self):
        if not self._grid_client.wait_for_service(timeout_sec=self.plan_timeout):
            return None
        future = self._grid_client.call_async(GetGridSnapshot.Request())
        await future
        return future.result()

    def _plan_client_for(self, name: str):
        if name not in self._plan_clients:
            self._plan_clients[name] = self.create_client(
                PlanPath, f'{self.ns}/{name}/plan_path',
                callback_group=self._cbg)
        return self._plan_clients[name]

    # ----------------------------------------------------------------------
    # Execute
    # ----------------------------------------------------------------------

    async def _execute_start_follow(self) -> bool:
        self.K.nav_state = 'following'
        self.K.follow_terminal = None

        client = self._follow_client_for(self.K.local_planner)
        if not client.wait_for_server(timeout_sec=self.plan_timeout):
            self.get_logger().error(
                f'[execute] FollowPath server unavailable for '
                f'{self.K.local_planner}')
            return False

        goal = FollowPath.Goal()
        goal.reference_path = self.K.active_path or Path()
        goal.mode = ('stacked' if goal.reference_path.poses
                     else 'standalone')

        send_future = client.send_goal_async(
            goal, feedback_callback=self._monitor_follow_feedback)
        await send_future
        handle = send_future.result()
        if handle is None or not handle.accepted:
            self.get_logger().error('[execute] FollowPath goal rejected')
            return False

        self._follow_goal_handle = handle
        handle.get_result_async().add_done_callback(
            self._execute_follow_result)
        return True

    def _execute_follow_result(self, future):
        """Callback invoked when the FollowPath action terminates."""
        try:
            result = future.result().result
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f'[execute] FollowPath result error: {e}')
            self.K.follow_terminal = 'failed'
            return
        self.K.follow_terminal = result.terminal_outcome
        # Prefer the planner's own total_distance if it's non-zero; otherwise
        # keep the locally-accumulated value from feedback-pose deltas.
        if result.total_distance > 0.0:
            self.K.distance_traveled = result.total_distance
        self.get_logger().info(
            f'[execute] FollowPath terminal={result.terminal_outcome} '
            f'distance={result.total_distance:.2f} '
            f'time={result.total_time:.2f}')

    async def _execute_cancel_follow(self):
        if self._follow_goal_handle is None:
            return
        try:
            cancel_future = self._follow_goal_handle.cancel_goal_async()
            await cancel_future
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f'[execute] follow cancel raised: {e}')
        self._follow_goal_handle = None

    def _follow_client_for(self, name: str):
        if name not in self._follow_clients:
            self._follow_clients[name] = ActionClient(
                self, FollowPath, f'{self.ns}/{name}/follow_path',
                callback_group=self._cbg)
        return self._follow_clients[name]

    # ----------------------------------------------------------------------
    # Feedback / result helpers
    # ----------------------------------------------------------------------

    def _publish_navigate_feedback(self, goal_handle, elapsed: float):
        fb = Navigate.Feedback()
        fb.distance_to_goal = self.K.distance_to_goal
        fb.distance_traveled = self.K.distance_traveled
        fb.elapsed_time = elapsed
        fb.nav_state = self.K.nav_state
        fb.current_pose = self.K.current_pose
        fb.current_velocity = self.K.current_velocity
        goal_handle.publish_feedback(fb)

    def _finalize_result(self, outcome: str, success: bool) -> Navigate.Result:
        r = Navigate.Result()
        r.success = success
        r.terminal_outcome = outcome
        r.total_distance = self.K.distance_traveled
        r.total_time = (self._elapsed(self.K.start_time)
                        if self.K.start_time else 0.0)
        r.replan_count = self.K.replan_count
        self.get_logger().info(
            f'[result] outcome={outcome} '
            f'distance={r.total_distance:.2f} '
            f'time={r.total_time:.2f} replans={r.replan_count}')
        return r

    # ----------------------------------------------------------------------
    # Utilities
    # ----------------------------------------------------------------------

    def _elapsed(self, start) -> float:
        if start is None:
            return 0.0
        return (self.get_clock().now() - start).nanoseconds / 1e9

    def _current_pose_stamped(self) -> PoseStamped:
        ps = PoseStamped()
        ps.header.frame_id = self.map_frame
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose = self.K.current_pose
        return ps

    def _distance_to_goal(self) -> float:
        if self.K.goal is None or not self.K.pose_valid:
            return -1.0
        dx = self.K.goal.pose.position.x - self.K.current_pose.position.x
        dy = self.K.goal.pose.position.y - self.K.current_pose.position.y
        return math.hypot(dx, dy)


def main(args=None):
    rclpy.init(args=args)
    node = NavigationServer()
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
