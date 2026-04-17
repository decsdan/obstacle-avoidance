#!/usr/bin/env python3
"""Command-line Navigate action client for the obstacle-avoidance fork.

Sends a Navigate goal to the navigation server, prints live feedback, and
treats Ctrl+C as a cancel request. Run with::

    ros2 run nav_server navigate -- --goal X Y [THETA] \\
        [--global-planner a_star|d_star|jps] [--local-planner dwa|none] \\
        [--namespace /don]
"""

import argparse
import sys

import rclpy
from nav_interfaces.action import Navigate
from rclpy.action import ActionClient
from rclpy.node import Node


class NavigateCLI(Node):
    """Action client wrapper that mirrors server feedback to the terminal."""

    def __init__(self, namespace):
        super().__init__('navigate_cli')
        self.ns = namespace
        self._action_client = ActionClient(
            self, Navigate, f'{self.ns}/navigate')
        self._goal_handle = None
        self._done = False
        self._result = None

    def send_goal(self, goal_x, goal_y, goal_theta,
                  global_planner, local_planner):
        """Send a Navigate goal; return True if the server is reachable."""
        if not self._action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error(
                f'Navigation server not available on {self.ns}/navigate. '
                f'Is nav_server running?')
            return False

        goal_msg = Navigate.Goal()
        goal_msg.global_planner = global_planner
        goal_msg.local_planner = local_planner
        goal_msg.goal_x = goal_x
        goal_msg.goal_y = goal_y
        goal_msg.goal_theta = goal_theta

        print(f'\n  Navigating to '
              f'({goal_x:.2f}, {goal_y:.2f}, {goal_theta:.2f})')
        print(f'  Global planner: {global_planner}')
        print(f'  Local planner:  {local_planner}')
        print(f'  Namespace:      {self.ns}')
        print('  Press Ctrl+C to cancel\n')

        self._send_future = self._action_client.send_goal_async(
            goal_msg, feedback_callback=self._feedback_cb)
        self._send_future.add_done_callback(self._goal_response_cb)
        return True

    def _goal_response_cb(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            print('  Goal REJECTED by server')
            self._done = True
            return

        print('  Goal accepted\n')
        self._goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._result_cb)

    def _feedback_cb(self, feedback_msg):
        fb = feedback_msg.feedback
        state_str = fb.nav_state.upper()
        dist_str = (f'{fb.distance_to_goal:.2f}m'
                    if fb.distance_to_goal >= 0 else '---')
        print(
            f'\r  [{state_str:20s}] '
            f'dist_to_goal: {dist_str}  '
            f'traveled: {fb.distance_traveled:.2f}m  '
            f'time: {fb.elapsed_time:.1f}s  '
            f'pos: ({fb.current_x:.2f}, {fb.current_y:.2f})   ',
            end='', flush=True)

    def _result_cb(self, future):
        result = future.result().result
        self._result = result
        self._done = True

        labels = {
            'reached': 'SUCCESS',
            'cancelled': 'CANCELLED',
            'timeout': 'TIMEOUT',
            'failed': 'FAILED',
        }
        label = labels.get(result.final_state, result.final_state.upper())

        print(f'\n\n  === {label} ===')
        print(f'  Distance traveled: {result.total_distance:.2f} m')
        print(f'  Total time:        {result.total_time:.2f} s')
        if result.total_time > 0:
            avg_speed = result.total_distance / result.total_time
            print(f'  Average speed:     {avg_speed:.3f} m/s')
        print()

    def cancel_goal(self):
        """Send a cancel request for the active goal, if any."""
        if self._goal_handle is not None:
            print('\n\n  Cancelling navigation...')
            self._goal_handle.cancel_goal_async()


def main(args=None):
    parser = argparse.ArgumentParser(
        description='Send navigation goals to the navigation server',
        prog='ros2 run nav_server navigate --')
    parser.add_argument(
        '--goal', nargs='+', type=float, required=True,
        metavar=('X', 'Y'),
        help='Goal coordinates: X Y [THETA]')
    parser.add_argument(
        '--global-planner', default='a_star',
        choices=['a_star', 'd_star', 'jps'],
        help='Global planner to use (default: a_star)')
    parser.add_argument(
        '--local-planner', default='dwa',
        choices=['dwa', 'none'],
        help='Local planner to use (default: dwa)')
    parser.add_argument(
        '--namespace', default='/don',
        help='Robot namespace (default: /don)')

    filtered_args = []
    for arg in sys.argv[1:]:
        if arg == '--ros-args':
            break
        filtered_args.append(arg)

    parsed = parser.parse_args(filtered_args)

    if len(parsed.goal) < 2:
        parser.error('--goal requires at least X and Y coordinates')
    goal_x = parsed.goal[0]
    goal_y = parsed.goal[1]
    goal_theta = parsed.goal[2] if len(parsed.goal) >= 3 else 0.0

    rclpy.init(args=args)
    node = NavigateCLI(parsed.namespace)

    success = node.send_goal(
        goal_x, goal_y, goal_theta,
        parsed.global_planner, parsed.local_planner)

    if not success:
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)

    try:
        while rclpy.ok() and not node._done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        node.cancel_goal()
        for _ in range(20):
            rclpy.spin_once(node, timeout_sec=0.1)
            if node._done:
                break

    node.destroy_node()
    rclpy.shutdown()

    if node._result is not None and node._result.success:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()
