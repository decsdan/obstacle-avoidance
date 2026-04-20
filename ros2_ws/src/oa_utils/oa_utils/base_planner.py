import time
import rclpy
from rclpy.node import Node
from nav_interfaces.srv import PlanPath
from nav_msgs.msg import Path


class BaseGlobalPlanner(Node):
    # Inherit from this for all global planners to avoid re-writing service boilerplate.
    # Subclasses implement _handle_plan_path(self, request, response).

    def __init__(self, node_name: str, planner_name: str):
        super().__init__(node_name)

        # Rely on standard ROS 2 namespace remapping
        self.declare_parameter('namespace', '')
        self.ns = self.get_parameter('namespace').value

        self.declare_parameter('max_path_waypoints', 50)
        self.declare_parameter('tight_space_radius', 5)
        self.declare_parameter('max_search_nodes', 100000)
        self.declare_parameter('simplification_epsilon', 0.1)
        self.declare_parameter('occ_threshold_log_odds', 0.4)

        self.max_path_waypoints = self.get_parameter('max_path_waypoints').value
        self.tight_space_radius = self.get_parameter('tight_space_radius').value
        self.max_search_nodes = self.get_parameter('max_search_nodes').value
        self.simplification_epsilon = self.get_parameter('simplification_epsilon').value
        self.occ_threshold = self.get_parameter('occ_threshold_log_odds').value

        # Note: if namespace is empty, this becomes e.g. `/a_star/plan_path`
        # If set to '/don', it becomes `/don/a_star/plan_path`
        prefix = f"{self.ns}/" if self.ns else ""
        srv_name = f"{prefix}{planner_name}/plan_path"

        self._srv = self.create_service(
            PlanPath, srv_name, self._handle_plan_path)

        self.get_logger().info(f'{planner_name} planner ready on {srv_name}')

    def _handle_plan_path(self, request, response):
        """Must be implemented by subclasses."""
        raise NotImplementedError

    def _fail(self, response: PlanPath.Response, reason: str, t0: float, msg: str) -> PlanPath.Response:
        response.success = False
        response.failure_reason = reason
        response.compute_time = time.monotonic() - t0
        response.nodes_expanded = 0
        response.cached = False
        response.path = Path()
        response.path.header.frame_id = 'map'
        response.path.header.stamp = self.get_clock().now().to_msg()
        self.get_logger().error(f'PlanPath {reason}: {msg}')
        return response
