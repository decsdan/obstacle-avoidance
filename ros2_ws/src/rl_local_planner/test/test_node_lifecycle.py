import unittest
import rclpy
from rl_local_planner.rl_local_planner_node import RLLocalPlanner

class TestRLLocalPlannerLifecycle(unittest.TestCase):
    def setUp(self):
        rclpy.init()
        self.node = RLLocalPlanner()

    def tearDown(self):
        self.node.destroy_node()
        rclpy.shutdown()

    def test_action_server_exists(self):
        # This is a basic test to see if the action server was initialized
        self.assertIsNotNone(self.node._action_server)

if __name__ == '__main__':
    unittest.main()
