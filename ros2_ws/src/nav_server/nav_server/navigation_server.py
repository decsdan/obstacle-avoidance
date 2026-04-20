#!/usr/bin/env python3
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


KNOWN_GLOBALS = {'a_star', 'd_star', 'jps'}
KNOWN_LOCALS = {'dwa'}


@dataclass
class State:
    # goal context
    goal: Optional[PoseStamped] = None
    scenario_id: str = ''
    global_planner: str = ''
    local_planner: str = ''
    global_budget: float = 0.0

    # path/grid state
    active_path: Optional[Path] = None
    last_path_stamp: Optional[TimeMsg] = None
    last_grid_stamp: Optional[TimeMsg] = None

    # pose/vel
    current_pose: Pose = field(default_factory=Pose)
    current_velocity: Twist = field(default_factory=Twist)
    pose_valid: bool = False

    # metrics
    dist_to_goal: float = -1.0
    dist_traveled: float = 0.0
    cte: float = 0.0
    local_state: str = ''
    waypoint_index: int = -1

    # status
    nav_state: str = 'planning'
    replan_count: int = 0
    replan_reason: str = ''

    follow_terminal: Optional[str] = None
    last_plan_failure: str = ''

    stuck_detected: bool = False
    collision_detected: bool = False
    safety_latched: bool = False

    start_time: Optional[Time] = None


class NavigationServer(Node):

    def __init__(self):
        super().__init__('navigation_server')

        self.declare_parameter('namespace', '/don')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('default_global_planner', 'a_star')
        self.declare_parameter('default_local_planner', 'dwa')
        self.declare_parameter('navigation_timeout', 300.0)
        self.declare_parameter('plan_timeout', 5.0)
        self.declare_parameter('plan_budget_default', 0.0)
        self.declare_parameter('feedback_hz', 10.0)
        self.declare_parameter('replan_on_local_block', True)
        self.declare_parameter('replan_deviation_threshold', 0.5)
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
        self.replan_on_local_block = bool(p('replan_on_local_block').value)
        self.replan_deviation_m = float(p('replan_deviation_threshold').value)
        self.replan_period_sec = float(p('replan_period_sec').value)

        self._cbg = ReentrantCallbackGroup()
        self.state = State()
        self._last_pose_for_distance: Optional[Pose] = None
        self._follow_gh = None
        self._navigate_gh = None

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._plan_clients = {}
        self._follow_clients = {}
        self._grid_client = self.create_client(
            GetGridSnapshot, f'{self.ns}/get_grid_snapshot',
            callback_group=self._cbg)

        latched_qos = QoSProfile(
            depth=1,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._safety_sub = self.create_subscription(
            Bool, f'{self.ns}/safety/latched',
            self._on_safety_latched, latched_qos,
            callback_group=self._cbg)
        self._collision_sub = self.create_subscription(
            Bool, f'{self.ns}/conditions/collision',
            self._on_collision, latched_qos,
            callback_group=self._cbg)
        self._stuck_sub = self.create_subscription(
            Bool, f'{self.ns}/conditions/stuck',
            self._on_stuck, latched_qos,
            callback_group=self._cbg)

        path_qos = QoSProfile(
            depth=1,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._active_path_pub = self.create_publisher(
            Path, f'{self.ns}/nav_server/active_path', path_qos)

        self._action_server = ActionServer(
            self,
            Navigate,
            f'{self.ns}/navigate',
            execute_callback=self._execute_navigate,
            goal_callback=self._on_goal_request,
            cancel_callback=self._on_cancel_request,
            callback_group=self._cbg,
        )

        self._rviz_sub = self.create_subscription(
            PoseStamped, f'{self.ns}/goal_pose',
            self._monitor_rviz_goal, 10, callback_group=self._cbg)
        self._rviz_self_client = ActionClient(
            self, Navigate, f'{self.ns}/navigate',
            callback_group=self._cbg)
        self._rviz_gh = None

        self.get_logger().info(
            f'nav_server ready | ns={self.ns} | '
            f'global={self.default_global} local={self.default_local}')


    def _monitor_rviz_goal(self, msg: PoseStamped):
        if not self._rviz_self_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().warn('Navigate server unreachable')
            return

        goal = Navigate.Goal()
        goal.goal = msg
        goal.global_planner = self.default_global
        goal.local_planner = self.default_local
        goal.global_budget = self.plan_budget_default
        goal.scenario_id = 'rviz'

        self.get_logger().info(f'RViz goal set to ({msg.pose.position.x:.1f}, {msg.pose.position.y:.1f})')

        if self._rviz_gh is not None:
            self._rviz_gh.cancel_goal_async()
            self._rviz_gh = None

        self._rviz_self_client.send_goal_async(goal).add_done_callback(
            self._rviz_goal_accepted)

    def _rviz_goal_accepted(self, future):
        handle = future.result()
        if handle.accepted:
            self._rviz_gh = handle

    def _on_goal_request(self, goal_request):
        g = goal_request.global_planner.lower() or self.default_global
        loc = goal_request.local_planner.lower() or self.default_local
        if g not in KNOWN_GLOBALS or loc not in KNOWN_LOCALS:
            self.get_logger().error(f'Invalid planners: {g}, {loc}')
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _on_cancel_request(self, gh):
        return CancelResponse.ACCEPT

    async def _execute_navigate(self, gh):
        req = gh.request
        self.state = State(
            goal=req.goal,
            scenario_id=req.scenario_id,
            global_planner=(req.global_planner.lower() or self.default_global),
            local_planner=(req.local_planner.lower() or self.default_local),
            global_budget=(req.global_budget if req.global_budget > 0.0
                           else self.plan_budget_default),
            start_time=self.get_clock().now(),
        )
        self._last_pose_for_distance = None
        self._navigate_gh = gh

        self.get_logger().info(f'Heading to ({req.goal.pose.position.x:.1f}, {req.goal.pose.position.y:.1f})')

        if not await self._request_path('initial'):
            gh.abort()
            return self._finalize_result(self._plan_failure_terminal(), success=False)

        if not await self._start_follow():
            gh.abort()
            return self._finalize_result('failed', success=False)

        last_periodic_replan = self.get_clock().now()
        while rclpy.ok():
            if gh.is_cancel_requested:
                await self._cancel_follow()
                gh.canceled()
                return self._finalize_result('cancelled', success=False)

            elapsed = self._elapsed(self.state.start_time)
            if self.nav_timeout > 0 and elapsed > self.nav_timeout:
                await self._cancel_follow()
                gh.abort()
                return self._finalize_result('timeout', success=False)

            if self.state.collision_detected:
                self.state.nav_state = 'collision'
                await self._cancel_follow()
                gh.abort()
                return self._finalize_result('collision', success=False)

            if self.state.safety_latched:
                await self._cancel_follow()
                gh.abort()
                return self._finalize_result('failed', success=False)

            if self.state.nav_state != 'following':
                self._refresh_pose()

            self._publish_navigate_feedback(gh, elapsed)

            if self.state.follow_terminal is not None:
                outcome = self.state.follow_terminal
                self.state.follow_terminal = None
                self._follow_gh = None

                if outcome == 'reached':
                    gh.succeed()
                    return self._finalize_result('reached', success=True)

                if outcome == 'collision':
                    gh.abort()
                    return self._finalize_result('collision', success=False)

                if outcome in ('stuck', 'path_blocked'):
                    if self.replan_on_local_block:
                        if await self._try_replan(outcome):
                            continue
                        gh.abort()
                        return self._finalize_result(self._plan_failure_terminal(), success=False)
                    gh.abort()
                    return self._finalize_result(outcome, success=False)

                gh.abort()
                return self._finalize_result(outcome, success=False)

            if self._deviation_exceeded():
                await self._try_replan('deviation')
            elif (self.replan_period_sec > 0.0 and
                  self._elapsed(last_periodic_replan) > self.replan_period_sec):
                await self._try_replan('periodic')
                last_periodic_replan = self.get_clock().now()

            await asyncio.sleep(self.feedback_period)

        await self._cancel_follow()
        gh.abort()
        return self._finalize_result('failed', success=False)


    def _refresh_pose(self):
        try:
            t = self._tf_buffer.lookup_transform(
                self.map_frame, self.base_frame, Time())
        except Exception as e:
            self.get_logger().debug(f'TF lookup failed: {e}')
            return
        self.state.current_pose.position.x = t.transform.translation.x
        self.state.current_pose.position.y = t.transform.translation.y
        self.state.current_pose.position.z = t.transform.translation.z
        self.state.current_pose.orientation = t.transform.rotation
        self.state.pose_valid = True
        self.state.dist_to_goal = self._distance_to_goal()

    def _on_safety_latched(self, msg: Bool):
        was_latched = self.state.safety_latched
        self.state.safety_latched = bool(msg.data)
        if self.state.safety_latched and not was_latched:
            self.get_logger().warn('Safety mux latched — aborting')

    def _on_collision(self, msg: Bool):
        self.state.collision_detected = bool(msg.data)
        if self.state.collision_detected:
            self.get_logger().warn('Collision detected!')

    def _on_stuck(self, msg: Bool):
        self.state.stuck_detected = bool(msg.data)

    def _on_follow_feedback(self, feedback_msg):
        fb = feedback_msg.feedback
        new_pose = fb.current_pose
        if self._last_pose_for_distance is not None:
            dx = new_pose.position.x - self._last_pose_for_distance.position.x
            dy = new_pose.position.y - self._last_pose_for_distance.position.y
            self.state.dist_traveled += math.hypot(dx, dy)
        self._last_pose_for_distance = new_pose

        self.state.current_pose = new_pose
        self.state.current_velocity = fb.current_velocity
        self.state.dist_to_goal = fb.dist_to_goal
        self.state.cte = fb.cross_track_error
        self.state.local_state = fb.local_state
        self.state.waypoint_index = fb.waypoint_index
        self.state.pose_valid = True


    def _deviation_exceeded(self) -> bool:
        if not self.state.pose_valid or self.state.active_path is None:
            return False
        if not self.state.active_path.poses:
            return False
        px = self.state.current_pose.position.x
        py = self.state.current_pose.position.y
        best_sq = math.inf
        for p in self.state.active_path.poses:
            dx = p.pose.position.x - px
            dy = p.pose.position.y - py
            d2 = dx * dx + dy * dy
            if d2 < best_sq:
                best_sq = d2
        return math.sqrt(best_sq) > self.replan_deviation_m

    async def _try_replan(self, reason: str) -> bool:
        self.state.replan_reason = reason
        self.state.replan_count += 1
        self.state.nav_state = 'replanning'
        self.get_logger().info(f'Replanning because of {reason}')

        await self._cancel_follow()
        if not await self._request_path(reason):
            return False
        return await self._start_follow()



    async def _request_path(self, reason: str) -> bool:
        self.state.nav_state = 'planning'
        self._refresh_pose()

        snap = await self._get_grid_snapshot()
        if snap is None:
            self.state.last_plan_failure = 'INFRASTRUCTURE'
            self.get_logger().error('Grid snapshot unavailable')
            return False
        self.state.last_grid_stamp = snap.stamp

        client = self._get_plan_client(self.state.global_planner)
        if not client.wait_for_service(timeout_sec=self.plan_timeout):
            self.state.last_plan_failure = 'INFRASTRUCTURE'
            self.get_logger().error(f'PlanPath service unavailable for {self.state.global_planner}')
            return False

        req = PlanPath.Request()
        req.grid_snapshot = snap.inflated
        req.start = self._current_pose_stamped()
        req.goal = self.state.goal
        req.budget = self.state.global_budget
        req.timeout = self.plan_timeout

        self.get_logger().info(f'Planning with {self.state.global_planner} ({reason})')

        future = client.call_async(req)
        try:
            # HACK: server might hang, so we pad the timeout
            await asyncio.wait_for(future, timeout=self.plan_timeout + 1.0)
        except asyncio.TimeoutError:
            self.state.last_plan_failure = 'TIMEOUT'
            self.get_logger().error('Planner timed out')
            return False

        resp = future.result()
        if resp is None or not resp.success:
            self.state.last_plan_failure = getattr(resp, 'failure_reason', 'NO_PATH') or 'NO_PATH'
            self.get_logger().error(f'Planning failed: {self.state.last_plan_failure}')
            return False

        self.state.last_plan_failure = ''
        self.state.active_path = resp.path
        self.state.last_path_stamp = self.get_clock().now().to_msg()
        self._active_path_pub.publish(resp.path)
        self.get_logger().info(f'Path found with {len(resp.path.poses)} points')
        return True

    def _plan_failure_terminal(self) -> str:
        if self.state.last_plan_failure in ('NO_PATH', 'TIMEOUT'):
            return 'path_blocked'
        return 'failed'

    async def _get_grid_snapshot(self):
        if not self._grid_client.wait_for_service(timeout_sec=self.plan_timeout):
            return None
        future = self._grid_client.call_async(GetGridSnapshot.Request())
        await future
        return future.result()

    def _get_plan_client(self, name: str):
        if name not in self._plan_clients:
            self._plan_clients[name] = self.create_client(
                PlanPath, f'{self.ns}/{name}/plan_path',
                callback_group=self._cbg)
        return self._plan_clients[name]


    async def _start_follow(self) -> bool:
        self.state.nav_state = 'following'
        self.state.follow_terminal = None

        client = self._get_follow_client(self.state.local_planner)
        if not client.wait_for_server(timeout_sec=self.plan_timeout):
            self.get_logger().error(f'FollowPath server unavailable for {self.state.local_planner}')
            return False

        goal = FollowPath.Goal()
        goal.reference_path = self.state.active_path or Path()

        send_future = client.send_goal_async(
            goal, feedback_callback=self._on_follow_feedback)
            
        try:
            await asyncio.wait_for(send_future, timeout=self.plan_timeout)
        except asyncio.TimeoutError:
            self.get_logger().error('FollowPath send_goal timed out')
            return False

        handle = send_future.result()
        if handle is None or not handle.accepted:
            self.get_logger().error('FollowPath goal rejected')
            return False

        self._follow_gh = handle
        handle.get_result_async().add_done_callback(
            self._on_follow_done)
        return True

    def _on_follow_done(self, future):
        try:
            wrapper = future.result()
        except Exception as e:
            self.get_logger().error(f'FollowPath result error: {e}')
            self.state.follow_terminal = 'failed'
            return

        if wrapper is None or wrapper.result is None:
            self.get_logger().error('FollowPath returned no result')
            self.state.follow_terminal = 'failed'
            return

        result = wrapper.result
        self.state.follow_terminal = result.terminal_outcome
        if result.total_distance > 0.0:
            self.state.dist_traveled = result.total_distance
        self.get_logger().info(f'FollowPath finished: {result.terminal_outcome}')

    async def _cancel_follow(self):
        if self._follow_gh is None:
            return
        try:
            cancel_future = self._follow_gh.cancel_goal_async()
            await asyncio.wait_for(cancel_future, timeout=2.0)
        except Exception:
            pass
        self._follow_gh = None

    def _get_follow_client(self, name: str):
        if name not in self._follow_clients:
            self._follow_clients[name] = ActionClient(
                self, FollowPath, f'{self.ns}/{name}/follow_path',
                callback_group=self._cbg)
        return self._follow_clients[name]


    def _publish_navigate_feedback(self, gh, elapsed: float):
        fb = Navigate.Feedback()
        fb.dist_to_goal = self.state.dist_to_goal
        fb.dist_traveled = self.state.dist_traveled
        fb.elapsed_time = elapsed
        fb.nav_state = self.state.nav_state
        fb.current_pose = self.state.current_pose
        fb.current_velocity = self.state.current_velocity
        gh.publish_feedback(fb)

    def _finalize_result(self, outcome: str, success: bool) -> Navigate.Result:
        r = Navigate.Result()
        r.success = success
        r.terminal_outcome = outcome
        r.total_distance = self.state.dist_traveled
        r.total_time = self._elapsed(self.state.start_time) if self.state.start_time else 0.0
        r.replan_count = self.state.replan_count
        self.get_logger().info(f'Finished with {outcome} after {r.replan_count} replans')
        return r


    def _elapsed(self, start) -> float:
        if start is None:
            return 0.0
        return (self.get_clock().now() - start).nanoseconds / 1e9

    def _current_pose_stamped(self) -> PoseStamped:
        ps = PoseStamped()
        ps.header.frame_id = self.map_frame
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose = self.state.current_pose
        return ps

    def _distance_to_goal(self) -> float:
        if self.state.goal is None or not self.state.pose_valid:
            return -1.0
        dx = self.state.goal.pose.position.x - self.state.current_pose.position.x
        dy = self.state.goal.pose.position.y - self.state.current_pose.position.y
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
