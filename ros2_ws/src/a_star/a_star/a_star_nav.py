#!/usr/bin/env python3
import heapq
import math
import time
from typing import Tuple, List

import numpy as np

import rclpy
from nav_interfaces.srv import PlanPath
from nav_msgs.msg import Path

from oa_utils.base_planner import BaseGlobalPlanner
from oa_utils.conversions import *
from oa_utils.pathing import simplify_path


class AStarPlanner(BaseGlobalPlanner):

    def __init__(self):
        super().__init__('a_star_planner', 'a_star')

    def _handle_plan_path(self, request, response: PlanPath.Response) -> PlanPath.Response:
        t0 = time.monotonic()

        grid_msg = request.grid_snapshot
        if grid_msg.info.width == 0 or grid_msg.info.height == 0:
            return self._fail(response, 'BAD_INPUT', t0, 'Empty grid_snapshot')

        grid = grid_from_msg(grid_msg, self.occ_threshold)
        res = grid_msg.info.resolution
        origin = (grid_msg.info.origin.position.x, grid_msg.info.origin.position.y)

        start_x, start_y = request.start.pose.position.x, request.start.pose.position.y
        goal_x, goal_y = request.goal.pose.position.x, request.goal.pose.position.y

        start_cell = world_to_grid(start_x, start_y, res, origin)
        goal_cell = world_to_grid(goal_x, goal_y, res, origin)

        rows, cols = grid.shape
        if not (0 <= start_cell[0] < cols and 0 <= start_cell[1] < rows):
            return self._fail(response, 'BAD_INPUT', t0, f'Start {start_cell} out of bounds')
        if not (0 <= goal_cell[0] < cols and 0 <= goal_cell[1] < rows):
            return self._fail(response, 'BAD_INPUT', t0, f'Goal {goal_cell} out of bounds')
        if grid[goal_cell[1], goal_cell[0]] != 0:
            return self._fail(response, 'BAD_INPUT', t0, 'Goal is an obstacle')

        budget = int(request.budget) if request.budget > 0 else self.max_search_nodes
        timeout = request.timeout if request.timeout > 0 else float('inf')

        path_cells, nodes_expanded, reason = self._astar(
            grid, start_cell, goal_cell, budget, timeout, t0)

        compute_time = time.monotonic() - t0

        if not path_cells:
            response.success = False
            response.failure_reason = reason or 'NO_PATH'
            response.compute_time = compute_time
            response.nodes_expanded = nodes_expanded
            response.cached = False
            response.path = Path()
            response.path.header.frame_id = grid_msg.header.frame_id or 'map'
            response.path.header.stamp = self.get_clock().now().to_msg()
            return response

        path_world = [grid_to_world(gx, gy, res, origin) for gx, gy in path_cells]
        simplified = simplify_path(path_world, self.max_path_waypoints, self.simplification_epsilon)

        response.path = build_path_msg(simplified, grid_msg.header.frame_id or 'map', self.get_clock().now().to_msg())
        response.success = True
        response.failure_reason = ''
        response.compute_time = compute_time
        response.nodes_expanded = nodes_expanded
        response.cached = False

        self.get_logger().info(f'Path found with {len(simplified)} points.')
        return response

    # TODO(dan): try octile distance, might reduce expanded nodes on open maps
    def _astar(
        self,
        grid: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        budget: int,
        timeout: float,
        t_start: float
    ) -> Tuple[List[Tuple[int, int]], int, str]:
        rows, cols = grid.shape

        def h_dist(a: Tuple[int, int], b: Tuple[int, int]) -> float:
            return math.hypot(a[0] - b[0], a[1] - b[1])

        def in_bounds(cell: Tuple[int, int]) -> bool:
            return 0 <= cell[0] < cols and 0 <= cell[1] < rows

        def is_free(cell: Tuple[int, int]) -> bool:
            if not in_bounds(cell):
                return False
            dist_start = abs(cell[0] - start[0]) + abs(cell[1] - start[1])
            if dist_start <= self.tight_space_radius:
                return True
            return grid[cell[1], cell[0]] == 0

        open_set = []
        heapq.heappush(open_set, (0.0, start))
        came_from = {}
        g_score = {start: 0.0}
        expanded = 0

        moves = [(-1, 0), (1, 0), (0, -1), (0, 1),
                 (-1, -1), (-1, 1), (1, -1), (1, 1)]

        while open_set:
            _, current = heapq.heappop(open_set)
            expanded += 1

            if expanded >= budget:
                return [], expanded, 'TIMEOUT'
            if time.monotonic() - t_start > timeout:
                return [], expanded, 'TIMEOUT'

            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return path[::-1], expanded, ''

            for dx, dy in moves:
                neighbor = (current[0] + dx, current[1] + dy)
                if not is_free(neighbor):
                    continue
                move_cost = 1.4142135 if dx != 0 and dy != 0 else 1.0
                tentative_g = g_score[current] + move_cost
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + h_dist(neighbor, goal)
                    heapq.heappush(open_set, (f, neighbor))

        return [], expanded, 'NO_PATH'


def main(args=None):
    rclpy.init(args=args)
    node = AStarPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
