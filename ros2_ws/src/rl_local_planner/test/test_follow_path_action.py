import unittest
import rclpy
from rclpy.action import ActionClient
from nav_interfaces.action import FollowPath
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from rl_local_planner.rl_local_planner_node import RLLocalPlanner
import threading

class TestFollowPathAction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.node = RLLocalPlanner()
        self.client = ActionClient(self.node, FollowPath, '/don/rl_local_planner/follow_path')
        
        self.executor = rclpy.executors.SingleThreadedExecutor()
        self.executor.add_node(self.node)
        
        self.spin_thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.spin_thread.start()

    def tearDown(self):
        self.executor.shutdown()
        self.node.destroy_node()
        self.spin_thread.join()

    def test_goal_rejection_empty_path(self):
        # Wait for server
        self.assertTrue(self.client.wait_for_server(timeout_sec=5.0))
        
        goal_msg = FollowPath.Goal()
        # reference_path is empty by default
        
        future = self.client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=5.0)
        
        goal_handle = future.result()
        self.assertFalse(goal_handle.accepted)

    def test_goal_acceptance_valid_path(self):
        # Wait for server
        self.assertTrue(self.client.wait_for_server(timeout_sec=5.0))
        
        goal_msg = FollowPath.Goal()
        pose = PoseStamped()
        goal_msg.reference_path.poses.append(pose)
        
        future = self.client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=5.0)
        
        goal_handle = future.result()
        self.assertTrue(goal_handle.accepted)

    def test_feedback_published(self):
        """Test that feedback is published during execution."""
        # Wait for server
        self.assertTrue(self.client.wait_for_server(timeout_sec=5.0))
        
        goal_msg = FollowPath.Goal()
        pose = PoseStamped()
        goal_msg.reference_path.poses.append(pose)
        
        feedback_received = []
        def feedback_callback(feedback):
            feedback_received.append(feedback)

        send_goal_future = self.client.send_goal_async(
            goal_msg, 
            feedback_callback=feedback_callback
        )
        rclpy.spin_until_future_complete(self.node, send_goal_future, timeout_sec=5.0)
        
        goal_handle = send_goal_future.result()
        self.assertTrue(goal_handle.accepted)
        
        get_result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self.node, get_result_future, timeout_sec=5.0)
        
        # This should fail with current implementation because it returns immediately without publishing feedback
        self.assertGreater(len(feedback_received), 0, "No feedback received during execution")

if __name__ == '__main__':
    unittest.main()
