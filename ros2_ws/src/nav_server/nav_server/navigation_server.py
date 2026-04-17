#!/usr/bin/env python3
"""Navigate action server and RViz goal monitor for the obstacle-avoidance fork.

Watches /{ns}/goal_pose for RViz clicks, dispatches them to the configured
global and local planners, and prints live NavStatus feedback to the launch
terminal. Also exposes a Navigate action for programmatic use. New goals
preempt any in-flight navigation via the planners' CancelNav services.
"""

import asyncio
import math

import rclpy
import rclpy.time
from geometry_msgs.msg import PoseStamped
from nav_interfaces.action import Navigate
from nav_interfaces.msg import NavStatus
from nav_interfaces.srv import CancelNav
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node


class NavigationServer(Node):
    """Orchestrate planners and stream feedback for RViz-driven navigation."""

    def __init__(self):
        super().__init__('navigation_server')

        self.declare_parameter('namespace', '/don')
        self.declare_parameter('navigation_timeout', 300.0)
        self.declare_parameter('global_planner', 'a_star')
        self.declare_parameter('local_planner', 'dwa')

        self.ns = self.get_parameter('namespace').value
        self.nav_timeout = self.get_parameter('navigation_timeout').value
        self.global_planner = self.get_parameter('global_planner').value
        self.local_planner = self.get_parameter('local_planner').value

        self._action_cb_group = ReentrantCallbackGroup()
        self._action_server = ActionServer(
            self,
            Navigate,
            f'{self.ns}/navigate',
            execute_callback=self._action_execute_callback,
            goal_callback=self._action_goal_callback,
            cancel_callback=self._action_cancel_callback,
            callback_group=self._action_cb_group,
        )

        self._local_status = None
        self._global_status = None

        self.local_status_sub = self.create_subscription(
            NavStatus, f'{self.ns}/dwa/nav_status', self._local_status_cb, 10)

        global_status_map = {
            'a_star': f'{self.ns}/a_star/status',
            'd_star': f'{self.ns}/d_star/status',
            'jps': f'{self.ns}/jps/status',
        }
        global_status_topic = global_status_map.get(
            self.global_planner, f'{self.ns}/a_star/status')
        self.global_status_sub = self.create_subscription(
            NavStatus, global_status_topic, self._global_status_cb, 10)

        self._cancel_clients = {}

        self._rviz_active = False
        self._rviz_goal = None
        self._rviz_start_time = None
        self._rviz_last_status_line = ''
        self._self_published_goal = False

        self._rviz_sub = self.create_subscription(
            PoseStamped, f'{self.ns}/goal_pose', self._goal_pose_cb, 10)
        self._goal_pub = self.create_publisher(
            PoseStamped, f'{self.ns}/goal_pose', 10)

        self._monitor_timer = self.create_timer(0.2, self._monitor_loop)

        self.get_logger().info(
            f'Navigation server ready | namespace={self.ns} '
            f'| global={self.global_planner} local={self.local_planner}')
        self.get_logger().info(
            f'Send goals via RViz "2D Goal Pose" tool on {self.ns}/goal_pose')

    def _goal_pose_cb(self, msg):
        """Handle goal_pose messages from RViz or the action server."""
        if self._self_published_goal:
            self._self_published_goal = False
            return

        goal_x = msg.pose.position.x
        goal_y = msg.pose.position.y
        q = msg.pose.orientation
        goal_theta = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z))

        if self._rviz_active:
            self.get_logger().info('New RViz goal -- preempting previous navigation')
            self._cancel_active_planners()

        self._rviz_goal = (goal_x, goal_y, goal_theta)
        self._rviz_start_time = self.get_clock().now()
        self._rviz_active = True
        self._rviz_last_status_line = ''

        self.get_logger().info(
            f'RViz goal received: ({goal_x:.2f}, {goal_y:.2f}, {goal_theta:.2f})')

    def _monitor_loop(self):
        """Tick at 5 Hz; print live feedback while a goal is active."""
        if not self._rviz_active:
            return

        use_local = self.local_planner not in ('none', '')
        status = self._local_status if use_local else self._global_status
        elapsed = self._elapsed(self._rviz_start_time)

        if self.nav_timeout > 0 and elapsed > self.nav_timeout:
            self.get_logger().error(
                f'Navigation timeout ({self.nav_timeout:.0f}s) -- cancelling')
            self._cancel_active_planners()
            self._finish_rviz('TIMEOUT', elapsed, status)
            return

        if status is None:
            self.get_logger().info(
                f'Waiting for planner status... ({elapsed:.1f}s)',
                throttle_duration_sec=2.0)
            return

        if status.goal_reached:
            self._finish_rviz('REACHED', elapsed, status)
            return

        if (elapsed > 3.0 and
                status.nav_state == 'idle' and
                not status.has_active_goal and
                not status.goal_reached):
            self._finish_rviz('FAILED', elapsed, status)
            return

        state = status.nav_state.upper()
        dist = (f'{status.distance_to_goal:.2f}m'
                if status.distance_to_goal >= 0 else '---')
        line = (
            f'  [{state:20s}] '
            f'to_goal: {dist}  '
            f'traveled: {status.distance_traveled:.2f}m  '
            f'time: {elapsed:.1f}s  '
            f'pos: ({status.current_x:.2f}, {status.current_y:.2f})')

        if line != self._rviz_last_status_line:
            self.get_logger().info(line)
            self._rviz_last_status_line = line

    def _finish_rviz(self, outcome, elapsed, status):
        """Print final result and reset RViz monitoring state."""
        dist = status.distance_traveled if status else 0.0
        self.get_logger().info('')
        self.get_logger().info(f'  == {outcome} ==')
        self.get_logger().info(f'  Distance traveled: {dist:.2f} m')
        self.get_logger().info(f'  Total time:        {elapsed:.2f} s')
        if elapsed > 0 and dist > 0:
            self.get_logger().info(
                f'  Average speed:     {dist / elapsed:.3f} m/s')
        self.get_logger().info('')

        self._rviz_active = False
        self._rviz_goal = None
        self._rviz_start_time = None
        self._rviz_last_status_line = ''

    def _local_status_cb(self, msg):
        self._local_status = msg

    def _global_status_cb(self, msg):
        self._global_status = msg

    def _action_goal_callback(self, goal_request):
        """Accept or reject action goals."""
        planner = goal_request.global_planner.lower()
        local = goal_request.local_planner.lower()

        known_global = {'a_star', 'd_star', 'jps'}
        known_local = {'dwa', 'none', ''}

        if planner not in known_global:
            self.get_logger().error(
                f'Unknown global_planner: "{planner}". Valid: {known_global}')
            return GoalResponse.REJECT

        if local not in known_local:
            self.get_logger().error(
                f'Unknown local_planner: "{local}". Valid: {known_local}')
            return GoalResponse.REJECT

        # Preempt any active RViz navigation
        if self._rviz_active:
            self.get_logger().info('Action goal preempts active RViz navigation')
            self._cancel_active_planners()
            self._rviz_active = False

        self.get_logger().info(
            f'Action goal accepted: ({goal_request.goal_x:.2f}, '
            f'{goal_request.goal_y:.2f}) global={planner} local={local}')
        return GoalResponse.ACCEPT

    def _action_cancel_callback(self, goal_handle):
        """Accept all cancel requests."""
        self.get_logger().info('Action cancel request received')
        return CancelResponse.ACCEPT

    async def _action_execute_callback(self, goal_handle):
        """Execute a Navigate action goal (programmatic interface)."""
        request = goal_handle.request
        global_planner = request.global_planner.lower()
        local_planner = request.local_planner.lower()

        self._local_status = None
        self._global_status = None

        use_local = local_planner not in ('none', '')

        # Publish goal_pose (set flag so our subscription ignores it)
        goal_msg = PoseStamped()
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.header.frame_id = 'map'
        goal_msg.pose.position.x = request.goal_x
        goal_msg.pose.position.y = request.goal_y
        goal_msg.pose.position.z = 0.0
        goal_msg.pose.orientation.z = math.sin(request.goal_theta / 2.0)
        goal_msg.pose.orientation.w = math.cos(request.goal_theta / 2.0)

        self._self_published_goal = True
        self._goal_pub.publish(goal_msg)

        result = Navigate.Result()
        feedback = Navigate.Feedback()
        start_time = self.get_clock().now()
        rate_period = 1.0 / 5.0

        while rclpy.ok():
            if goal_handle.is_cancel_requested:
                self._cancel_planners(global_planner, local_planner)
                goal_handle.canceled()
                result.success = False
                result.final_state = 'cancelled'
                result.total_time = self._elapsed(start_time)
                result.total_distance = self._get_distance_traveled(use_local)
                return result

            elapsed = self._elapsed(start_time)
            if self.nav_timeout > 0 and elapsed > self.nav_timeout:
                self._cancel_planners(global_planner, local_planner)
                goal_handle.abort()
                result.success = False
                result.final_state = 'timeout'
                result.total_time = elapsed
                result.total_distance = self._get_distance_traveled(use_local)
                return result

            status = self._local_status if use_local else self._global_status

            if status is not None:
                feedback.nav_state = status.nav_state
                feedback.distance_to_goal = status.distance_to_goal
                feedback.distance_traveled = status.distance_traveled
                feedback.elapsed_time = elapsed
                feedback.current_x = status.current_x
                feedback.current_y = status.current_y
                goal_handle.publish_feedback(feedback)

                if status.goal_reached:
                    goal_handle.succeed()
                    result.success = True
                    result.final_state = 'reached'
                    result.total_time = elapsed
                    result.total_distance = status.distance_traveled
                    return result

                if (elapsed > 2.0 and
                        status.nav_state == 'idle' and
                        not status.has_active_goal and
                        not status.goal_reached):
                    goal_handle.abort()
                    result.success = False
                    result.final_state = 'failed'
                    result.total_time = elapsed
                    result.total_distance = status.distance_traveled
                    return result

            await asyncio.sleep(rate_period)

    def _elapsed(self, start_time):
        return (self.get_clock().now() - start_time).nanoseconds / 1e9

    def _get_distance_traveled(self, use_local):
        status = self._local_status if use_local else self._global_status
        return status.distance_traveled if status is not None else 0.0

    def _cancel_active_planners(self):
        """Cancel whichever planners are configured for this server."""
        self._cancel_planners(self.global_planner, self.local_planner)

    def _cancel_planners(self, global_planner, local_planner):
        """Call CancelNav services on specified planners."""
        cancel_map = {
            'a_star': f'{self.ns}/a_star/cancel',
            'd_star': f'{self.ns}/d_star/cancel',
            'jps': f'{self.ns}/jps/cancel',
            'dwa': f'{self.ns}/dwa/cancel',
        }

        planners_to_cancel = [global_planner]
        if local_planner not in ('none', ''):
            planners_to_cancel.append(local_planner)

        for planner in planners_to_cancel:
            service_name = cancel_map.get(planner)
            if service_name is None:
                continue

            if service_name not in self._cancel_clients:
                self._cancel_clients[service_name] = self.create_client(
                    CancelNav, service_name)

            client = self._cancel_clients[service_name]
            if client.service_is_ready():
                req = CancelNav.Request()
                client.call_async(req)
                self.get_logger().info(f'Cancel sent to {service_name}')
            else:
                self.get_logger().warn(
                    f'Cancel service {service_name} not available')


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
