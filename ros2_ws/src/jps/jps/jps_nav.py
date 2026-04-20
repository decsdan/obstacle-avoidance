#!/usr/bin/env python3
import heapq
import math
import time
from typing import Tuple, List, Optional

import numpy as np

import rclpy
from nav_interfaces.srv import PlanPath
from nav_msgs.msg import Path

from oa_utils.base_planner import BaseGlobalPlanner
from oa_utils.conversions import *
from oa_utils.pathing import simplify_path


class JPSPlanner(BaseGlobalPlanner):

    def __init__(self):
        super().__init__('jps_planner', 'jps')
        self.declare_parameter('max_jump_depth', 500)
        self.max_jump_depth = self.get_parameter('max_jump_depth').value
        self._grid = None
        self._start = None
        self._deadline = None

    def _handle_plan_path(self, request, response: PlanPath.Response) -> PlanPath.Response:
        t0 = time.monotonic()

        grid_msg = request.grid_snapshot
        if grid_msg.info.width == 0 or grid_msg.info.height == 0:
            return self._fail(response, 'BAD_INPUT', t0, 'Empty grid_snapshot')

        self._grid = grid_from_msg(grid_msg, self.occ_threshold)
        res = grid_msg.info.resolution
        origin = (grid_msg.info.origin.position.x, grid_msg.info.origin.position.y)

        start_cell = world_to_grid(
            request.start.pose.position.x,
            request.start.pose.position.y,
            res, origin)
        goal_cell = world_to_grid(
            request.goal.pose.position.x,
            request.goal.pose.position.y,
            res, origin)

        rows, cols = self._grid.shape
        if not (0 <= start_cell[0] < cols and 0 <= start_cell[1] < rows):
            return self._fail(response, 'BAD_INPUT', t0, f'Start {start_cell} out of bounds')
        if not (0 <= goal_cell[0] < cols and 0 <= goal_cell[1] < rows):
            return self._fail(response, 'BAD_INPUT', t0, f'Goal {goal_cell} out of bounds')
        if self._grid[goal_cell[1], goal_cell[0]] != 0:
            return self._fail(response, 'BAD_INPUT', t0, 'Goal cell is blocked')

        self._start = start_cell
        budget = int(request.budget) if request.budget > 0 else self.max_search_nodes
        self._deadline = t0 + (request.timeout if request.timeout > 0 else 1e9)

        path_cells, nodes_expanded, reason = self._jps(start_cell, goal_cell, budget)
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
            self.get_logger().warn(f'Planning failed: {response.failure_reason}')
            return response

        path_world = [grid_to_world(gx, gy, res, origin) for gx, gy in path_cells]
        waypoints = simplify_path(path_world, self.max_path_waypoints, self.simplification_epsilon)

        response.path = build_path_msg(waypoints, grid_msg.header.frame_id or 'map', self.get_clock().now().to_msg())
        response.success = True
        response.failure_reason = ''
        response.compute_time = compute_time
        response.nodes_expanded = nodes_expanded
        response.cached = False

        self.get_logger().info(f'Path found with {len(waypoints)} points')
        return response

    def _jps(self, start: Tuple[int, int], goal: Tuple[int, int], budget: int) -> Tuple[List[Tuple[int, int]], int, str]:
        def heuristic(a, b):
            return math.hypot(a[0] - b[0], a[1] - b[1])

        open_set = []
        heapq.heappush(open_set, (heuristic(start, goal), start))
        came_from = {}
        g_score = {start: 0.0}
        expanded = 0

        while open_set:
            _, current = heapq.heappop(open_set)
            expanded += 1

            if expanded >= budget or time.monotonic() > self._deadline:
                return [], expanded, 'TIMEOUT'

            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return path[::-1], expanded, ''

            for successor in self._get_successors(current, goal):
                dx, dy = successor[0] - current[0], successor[1] - current[1]
                tentative_g = g_score[current] + math.hypot(dx, dy)

                if successor not in g_score or tentative_g < g_score[successor]:
                    came_from[successor] = current
                    g_score[successor] = tentative_g
                    f = tentative_g + heuristic(successor, goal)
                    heapq.heappush(open_set, (f, successor))

        return [], expanded, 'NO_PATH'

    def _get_successors(self, node: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        successors = []
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0),
                       (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            jump_point = self._jump(node, (dx, dy), goal, depth=0)
            if jump_point is not None:
                successors.append(jump_point)
        return successors

    def _jump(
        self, node: Tuple[int, int], direction: Tuple[int, int], goal: Tuple[int, int], depth: int
    ) -> Optional[Tuple[int, int]]:
        if depth > self.max_jump_depth:
            return None

        dx, dy = direction
        next_pt = (node[0] + dx, node[1] + dy)

        if not self._is_free(next_pt):
            return None

        if next_pt == goal:
            return next_pt

        if dx != 0 and dy != 0:
            if (self._is_free((next_pt[0] - dx, next_pt[1])) and
                    not self._is_free((next_pt[0] - dx, next_pt[1] + dy))):
                return next_pt
            if (self._is_free((next_pt[0], next_pt[1] - dy)) and
                    not self._is_free((next_pt[0] + dx, next_pt[1] - dy))):
                return next_pt

            if self._jump(next_pt, (dx, 0), goal, depth + 1) is not None:
                return next_pt
            if self._jump(next_pt, (0, dy), goal, depth + 1) is not None:
                return next_pt
        else:
            if dx != 0:
                if (self._is_free((next_pt[0], next_pt[1] + 1)) and
                        not self._is_free((next_pt[0] + dx, next_pt[1] + 1))):
                    return next_pt
                if (self._is_free((next_pt[0], next_pt[1] - 1)) and
                        not self._is_free((next_pt[0] + dx, next_pt[1] - 1))):
                    return next_pt
            else:
                if (self._is_free((next_pt[0] + 1, next_pt[1])) and
                        not self._is_free((next_pt[0] + 1, next_pt[1] + dy))):
                    return next_pt
                if (self._is_free((next_pt[0] - 1, next_pt[1])) and
                        not self._is_free((next_pt[0] - 1, next_pt[1] + dy))):
                    return next_pt

        return self._jump(next_pt, direction, goal, depth + 1)

    def _is_free(self, cell: Tuple[int, int]) -> bool:
        gx, gy = cell
        rows, cols = self._grid.shape
        if not (0 <= gx < cols and 0 <= gy < rows):
            return False
        dist_start = abs(gx - self._start[0]) + abs(gy - self._start[1])
        if dist_start <= self.tight_space_radius:
            return True
        return self._grid[gy, gx] == 0


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(JPSPlanner())
    rclpy.shutdown()


if __name__ == '__main__':
    main()
