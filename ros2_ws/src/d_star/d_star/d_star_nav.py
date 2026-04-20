#!/usr/bin/env python3
# Originally authored by Devin Dennis as part of the 2025 Carleton Senior
# Capstone Project (see AUTHORS.md). Substantially rewritten by
# Daniel Scheider, 2026.
"""D* Lite global path planner."""
import heapq
import math
import time
from collections import defaultdict
from typing import Tuple, List, Optional

import numpy as np

import rclpy
from nav_interfaces.srv import PlanPath
from nav_msgs.msg import Path

from oa_utils.base_planner import BaseGlobalPlanner
from oa_utils.conversions import grid_from_msg, world_to_grid, grid_to_world, build_path_msg
from oa_utils.pathing import simplify_path


class DStarLite:
    # Incremental search over a 2D occupancy grid.

    def __init__(self, grid, start, goal, tight_space_radius=0, logger=None):
        self.grid = grid
        self.rows, self.cols = grid.shape
        self.start = start
        self.goal = goal
        self.s_last = start
        self.k_m = 0.0
        self.tight_space_radius = tight_space_radius
        self.logger = logger

        self.g = defaultdict(lambda: float('inf'))
        self.rhs = defaultdict(lambda: float('inf'))

        self.open_list = []
        self._open_valid = {}

        self.rhs[self.goal] = 0.0
        heapq.heappush(self.open_list, (self.calculate_key(self.goal), self.goal))
        self._open_valid[self.goal] = self.calculate_key(self.goal)


    @staticmethod
    def heuristic(s1, s2):
        return math.hypot(s1[0] - s2[0], s1[1] - s2[1])

    def calculate_key(self, s):
        g_rhs = min(self.g[s], self.rhs[s])
        return (g_rhs + self.heuristic(self.start, s) + self.k_m, g_rhs)

    def is_valid(self, pos):
        x, y = pos
        if x < 0 or x >= self.cols or y < 0 or y >= self.rows:
            return False
        distance_from_start = abs(x - self.start[0]) + abs(y - self.start[1])
        if distance_from_start <= self.tight_space_radius:
            return True
        return self.grid[y, x] == 0

    def get_neighbors(self, s):
        result = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1),
                       (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            neighbor = (s[0] + dx, s[1] + dy)
            if self.is_valid(neighbor):
                result.append(neighbor)
        return result

    def cost(self, s1, s2):
        if not self.is_valid(s1) or not self.is_valid(s2):
            return float('inf')
        dx = abs(s1[0] - s2[0])
        dy = abs(s1[1] - s2[1])
        return 1.4142135 if (dx + dy) == 2 else 1.0

    def _open_push(self, key, u):
        heapq.heappush(self.open_list, (key, u))
        self._open_valid[u] = key

    def _open_remove(self, u):
        self._open_valid.pop(u, None)

    def _open_pop(self):
        while self.open_list:
            key, u = heapq.heappop(self.open_list)
            if self._open_valid.get(u) == key:
                del self._open_valid[u]
                return key, u
        return None, None

    def _open_top(self):
        while self.open_list:
            key, u = self.open_list[0]
            if self._open_valid.get(u) == key:
                return key, u
            heapq.heappop(self.open_list)
        return None, None

    @staticmethod
    def compare_keys(k1, k2):
        return k1[0] < k2[0] or (k1[0] == k2[0] and k1[1] < k2[1])

    def update_vertex(self, u):
        if u != self.goal:
            min_cost = float('inf')
            for s_prime in self.get_neighbors(u):
                c = self.cost(u, s_prime) + self.g[s_prime]
                if c < min_cost:
                    min_cost = c
            self.rhs[u] = min_cost

        self._open_remove(u)

        if self.g[u] != self.rhs[u]:
            self._open_push(self.calculate_key(u), u)

    def compute_shortest_path(self, budget, deadline):
        """Process states until start is locally consistent or budget hit."""
        iterations = 0

        while True:
            top_key, _ = self._open_top()
            if top_key is None:
                break

            start_key = self.calculate_key(self.start)
            if not (self.compare_keys(top_key, start_key) or
                    self.rhs[self.start] != self.g[self.start]):
                break

            iterations += 1
            if iterations > budget:
                return False, iterations, 'TIMEOUT'
            if time.monotonic() > deadline:
                return False, iterations, 'TIMEOUT'

            k_old, u = self._open_pop()
            if u is None:
                break

            k_new = self.calculate_key(u)

            if self.compare_keys(k_old, k_new):
                self._open_push(k_new, u)
            elif self.g[u] > self.rhs[u]:
                self.g[u] = self.rhs[u]
                for s in self.get_neighbors(u):
                    self.update_vertex(s)
            else:
                self.g[u] = float('inf')
                self.update_vertex(u)
                for s in self.get_neighbors(u):
                    self.update_vertex(s)

        return True, iterations, ''

    def extract_path(self):
        """Walk from start to goal by descending g-values."""
        if self.g[self.start] == float('inf'):
            return []

        # old greedy version looped on maze_02 -- switched to visited-set approach
        # path = [self.start]
        # while path[-1] != self.goal:
        #     nbs = self.get_neighbors(path[-1])
        #     path.append(min(nbs, key=lambda s: self.g[s]))

        path = [self.start]
        current = self.start
        visited = {self.start}
        limit = self.rows * self.cols

        while current != self.goal:
            if len(path) > limit:
                return []

            neighbors = self.get_neighbors(current)
            candidates = [(n, self.g[n]) for n in neighbors
                          if self.g[n] != float('inf') and n not in visited]
            if not candidates:
                return []

            next_state, min_g = min(candidates, key=lambda x: x[1])
            if min_g > self.g[current] + 0.01 and next_state != self.goal:
                return []

            visited.add(next_state)
            path.append(next_state)
            current = next_state

        return path


class _Session:
    # per-request cache carried across PlanPath calls

    __slots__ = ('dstar', 'grid', 'goal',
                 'width', 'height', 'resolution', 'origin')

    def __init__(self, dstar, grid, goal, width, height, resolution, origin):
        self.dstar = dstar
        self.grid = grid
        self.goal = goal
        self.width = width
        self.height = height
        self.resolution = resolution
        self.origin = origin


class DStarPlanner(BaseGlobalPlanner):
    """Incremental replanner — caches search state between calls."""

    def __init__(self):
        super().__init__('d_star_planner', 'd_star')
        self.declare_parameter('max_iterations', 10_000_000)
        self.max_iterations = self.get_parameter('max_iterations').value
        self._session = None

    def _handle_plan_path(self, request, response: PlanPath.Response) -> PlanPath.Response:
        t0 = time.monotonic()

        grid_msg = request.grid_snapshot
        if grid_msg.info.width == 0 or grid_msg.info.height == 0:
            return self._fail(response, 'BAD_INPUT', t0, 'Empty grid_snapshot')

        grid = grid_from_msg(grid_msg, self.occ_threshold)
        resolution = grid_msg.info.resolution
        origin = (grid_msg.info.origin.position.x, grid_msg.info.origin.position.y)
        width = grid_msg.info.width
        height = grid_msg.info.height

        start_cell = world_to_grid(
            request.start.pose.position.x,
            request.start.pose.position.y,
            resolution, origin)
        goal_cell = world_to_grid(
            request.goal.pose.position.x,
            request.goal.pose.position.y,
            resolution, origin)

        rows, cols = grid.shape
        if not (0 <= start_cell[0] < cols and 0 <= start_cell[1] < rows):
            return self._fail(response, 'BAD_INPUT', t0, f'Start {start_cell} out of grid bounds')
        if not (0 <= goal_cell[0] < cols and 0 <= goal_cell[1] < rows):
            return self._fail(response, 'BAD_INPUT', t0, f'Goal {goal_cell} out of grid bounds')
        if grid[goal_cell[1], goal_cell[0]] != 0:
            return self._fail(response, 'BAD_INPUT', t0, 'Goal cell is in an obstacle')

        budget = int(request.budget) if request.budget > 0 else self.max_iterations
        deadline = t0 + (request.timeout if request.timeout > 0 else 1e9)

        cached = self._can_reuse(goal_cell, width, height, resolution, origin)
        if cached:
            dstar = self._reuse_session(grid, start_cell)
        else:
            dstar = DStarLite(
                grid, start_cell, goal_cell,
                tight_space_radius=self.tight_space_radius,
                logger=self.get_logger())
            self._session = _Session(
                dstar=dstar, grid=grid.copy(), goal=goal_cell,
                width=width, height=height,
                resolution=resolution, origin=origin)

        ok, iterations, reason = dstar.compute_shortest_path(budget, deadline)
        if not ok:
            # Note: _fail in base class returns response, but we might want to add cached/nodes_expanded fields manually
            response = self._fail(response, reason or 'TIMEOUT', t0,
                                  f'compute_shortest_path bailed after {iterations} iterations')
            response.nodes_expanded = iterations
            response.cached = cached
            return response

        path_cells = dstar.extract_path()
        compute_time = time.monotonic() - t0

        if not path_cells:
            response.success = False
            response.failure_reason = 'NO_PATH'
            response.compute_time = compute_time
            response.nodes_expanded = iterations
            response.cached = cached
            response.path = Path()
            response.path.header.frame_id = grid_msg.header.frame_id or 'map'
            response.path.header.stamp = self.get_clock().now().to_msg()
            self.get_logger().warn(
                f'PlanPath NO_PATH (cached={cached}, iter {iterations}, '
                f'{compute_time*1000:.1f} ms)')
            return response

        path_world = [grid_to_world(gx, gy, resolution, origin) for gx, gy in path_cells]
        waypoints = simplify_path(path_world, self.max_path_waypoints, self.simplification_epsilon)

        response.path = build_path_msg(waypoints, grid_msg.header.frame_id or 'map', self.get_clock().now().to_msg())
        response.success = True
        response.failure_reason = ''
        response.compute_time = compute_time
        response.nodes_expanded = iterations
        response.cached = cached

        self.get_logger().info(
            f'PlanPath OK (cached={cached}): {len(waypoints)} pts, '
            f'iterations {iterations}, {compute_time*1000:.1f} ms')
        return response

    def _can_reuse(self, goal_cell, width, height, resolution, origin):
        s = self._session
        if s is None:
            return False
        if s.goal != goal_cell:
            return False
        if s.width != width or s.height != height:
            return False
        if abs(s.resolution - resolution) > 1e-6:
            return False
        if (abs(s.origin[0] - origin[0]) > 1e-6 or
                abs(s.origin[1] - origin[1]) > 1e-6):
            return False
        return True

    def _reuse_session(self, grid, start_cell):
        s = self._session
        dstar = s.dstar
        old_grid = s.grid
        old_start = dstar.start

        dstar.k_m += DStarLite.heuristic(dstar.s_last, start_cell)
        dstar.s_last = start_cell
        dstar.start = start_cell
        dstar.grid = grid

        for cell in self._changed_cells(old_grid, grid, old_start, start_cell):
            dstar.update_vertex(cell)

        s.grid = grid.copy()
        return dstar

    def _changed_cells(self, old_grid, new_grid, old_start, new_start):
        rows, cols = new_grid.shape
        touched = set()

        diff = np.argwhere(old_grid != new_grid)
        for y, x in diff:
            touched.add((int(x), int(y)))

        r = self.tight_space_radius
        if r > 0:
            for sx, sy in (old_start, new_start):
                for dx in range(-r, r + 1):
                    for dy in range(-r, r + 1):
                        if abs(dx) + abs(dy) <= r:
                            touched.add((sx + dx, sy + dy))

        expanded = set()
        for (x, y) in touched:
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < cols and 0 <= ny < rows:
                        expanded.add((nx, ny))
        return expanded


def main(args=None):
    rclpy.init(args=args)
    node = DStarPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
