#!/usr/bin/env python3
# Originally authored by Daniel Scheider, 2026.
"""Reinforcement Learning based local planner node."""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from nav_interfaces.action import FollowPath
from geometry_msgs.msg import Twist
import numpy as np

class RLLocalPlanner(Node):
    """
    FollowPath action server that uses a trained RL policy to drive the robot.
    Adheres to PRD v2.0 §5.3 Gymnasium-style contract.
    """

    def __init__(self):
        super().__init__('rl_local_planner')

        self.declare_parameter('namespace', '/don')
        self.ns = self.get_parameter('namespace').value

        self._cbg = ReentrantCallbackGroup()

        # FollowPath Action Server
        self._action_server = ActionServer(
            self,
            FollowPath,
            f'{self.ns}/rl_local_planner/follow_path',
            execute_callback=self._execute_callback,
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback,
            callback_group=self._cbg
        )

        self.get_logger().info(f'RL Local Planner ready on {self.ns}/rl_local_planner/follow_path')

    def _goal_callback(self, goal_request):
        """Accept or reject a goal request."""
        self.get_logger().info('Received FollowPath goal request')
        if not goal_request.reference_path.poses:
            self.get_logger().warn('Goal rejected: reference_path is empty')
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _cancel_callback(self, goal_handle):
        """Accept or reject a cancellation request."""
        self.get_logger().info('Received cancellation request')
        return CancelResponse.ACCEPT

    async def _execute_callback(self, goal_handle):
        """Execute the FollowPath goal."""
        self.get_logger().info('Executing FollowPath goal')
        
        feedback_msg = FollowPath.Feedback()
        result = FollowPath.Result()

        # TODO: Implement RL inference loop
        
        # For the skeleton, we'll just succeed immediately or wait for cancel
        while rclpy.ok():
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                result.terminal_outcome = 'cancelled'
                return result
            
            # Placeholder for actual tracking logic
            break

        goal_handle.succeed()
        result.terminal_outcome = 'reached'
        return result

def main(args=None):
    rclpy.init(args=args)
    node = RLLocalPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
