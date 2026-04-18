#!/usr/bin/env python3
# Originally authored by Devin Dennis as part of the 2025 Carleton Senior
# Capstone Project (see AUTHORS.md). Substantially rewritten by
# Daniel Scheider, 2026.
"""D* Lite global path planner.

Planner only, the navigation server calls this node with a grid and a
start/goal, and it hands back a path. The D* Lite algorithm class is kept;
for now each call runs fresh, and caching between calls can be added later.
"""

import heapq
import math
import time
from collections import defaultdict

import numpy as np

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_interfaces.srv import PlanPath
from nav_msgs.msg import Path
from rclpy.node import Node


class PlannerConstants:
    """Default configuration constants for the D* Lite planner."""

    MAX_PATH_ITERATIONS = 10_000_000
    MAX_WAYPOINTS = 50
    TIGHT_SPACE_RADIUS = 5
    DEFAULT_NAMESPACE = '/don'


class DStarLite:
    """D* Lite incremental search over a 2D occupancy grid."""

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

    # ------------------------------------------------------------------
    # Algorithm primitives
    # ------------------------------------------------------------------

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
        """Process states until start is locally consistent or budget hit.

        Returns:
            (ok, iterations, reason).  reason is 'TIMEOUT' if budget or wall
            deadline was hit, '' when the path is consistent.
        """
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


class DStarPlanner(Node):
    """D* Lite PlanPath service server."""

    def __init__(self):
        super().__init__('d_star_planner')

        self.declare_parameter('namespace', PlannerConstants.DEFAULT_NAMESPACE)
        self.ns = self.get_parameter('namespace').value

        self.declare_parameter(
            'max_waypoints', PlannerConstants.MAX_WAYPOINTS)
        self.declare_parameter(
            'tight_space_radius', PlannerConstants.TIGHT_SPACE_RADIUS)
        self.declare_parameter(
            'max_iterations', PlannerConstants.MAX_PATH_ITERATIONS)

        self.max_waypoints = self.get_parameter('max_waypoints').value
        self.tight_space_radius = self.get_parameter('tight_space_radius').value
        self.max_iterations = self.get_parameter('max_iterations').value

        self._srv = self.create_service(
            PlanPath, f'{self.ns}/d_star/plan_path', self._handle_plan_path)

        self.get_logger().info(
            f'd_star planner ready on {self.ns}/d_star/plan_path')

    # ------------------------------------------------------------------
    # Service entry point
    # ------------------------------------------------------------------

    def _handle_plan_path(self, request, response):
        t0 = time.monotonic()

        grid_msg = request.grid_snapshot
        if grid_msg.info.width == 0 or grid_msg.info.height == 0:
            return self._fail(response, 'BAD_INPUT', t0, 'Empty grid_snapshot')

        grid = self._grid_from_msg(grid_msg)
        resolution = grid_msg.info.resolution
        origin = (grid_msg.info.origin.position.x,
                  grid_msg.info.origin.position.y)

        start_cell = self._world_to_grid(
            request.start.pose.position.x,
            request.start.pose.position.y,
            resolution, origin)
        goal_cell = self._world_to_grid(
            request.goal.pose.position.x,
            request.goal.pose.position.y,
            resolution, origin)

        rows, cols = grid.shape
        if not (0 <= start_cell[0] < cols and 0 <= start_cell[1] < rows):
            return self._fail(response, 'BAD_INPUT', t0,
                              f'Start {start_cell} out of grid bounds')
        if not (0 <= goal_cell[0] < cols and 0 <= goal_cell[1] < rows):
            return self._fail(response, 'BAD_INPUT', t0,
                              f'Goal {goal_cell} out of grid bounds')
        if grid[goal_cell[1], goal_cell[0]] != 0:
            return self._fail(response, 'BAD_INPUT', t0,
                              'Goal cell is in an obstacle')

        budget = int(request.budget) if request.budget > 0 else self.max_iterations
        deadline = t0 + (request.timeout if request.timeout > 0 else 1e9)

        dstar = DStarLite(
            grid, start_cell, goal_cell,
            tight_space_radius=self.tight_space_radius,
            logger=self.get_logger())
        ok, iterations, reason = dstar.compute_shortest_path(budget, deadline)
        if not ok:
            return self._fail(response, reason or 'TIMEOUT', t0,
                              f'compute_shortest_path bailed after {iterations} iterations',
                              nodes_expanded=iterations)

        path_cells = dstar.extract_path()
        compute_time = time.monotonic() - t0

        if not path_cells:
            response.success = False
            response.failure_reason = 'NO_PATH'
            response.compute_time = compute_time
            response.nodes_expanded = iterations
            response.cached = False
            response.path = Path()
            response.path.header.frame_id = grid_msg.header.frame_id or 'map'
            response.path.header.stamp = self.get_clock().now().to_msg()
            self.get_logger().warn(
                f'PlanPath NO_PATH (iter {iterations}, '
                f'{compute_time*1000:.1f} ms)')
            return response

        path_world = [
            self._grid_to_world(gx, gy, resolution, origin)
            for gx, gy in path_cells
        ]
        waypoints = self._decimate(path_world, self.max_waypoints)

        response.path = self._path_msg(waypoints, grid_msg.header.frame_id or 'map')
        response.success = True
        response.failure_reason = ''
        response.compute_time = compute_time
        response.nodes_expanded = iterations
        response.cached = False

        self.get_logger().info(
            f'PlanPath OK: {len(waypoints)} pts, '
            f'iterations {iterations}, {compute_time*1000:.1f} ms')
        return response

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _grid_from_msg(grid_msg):
        width = grid_msg.info.width
        height = grid_msg.info.height
        data = np.array(grid_msg.data, dtype=np.int16).reshape((height, width))
        grid = np.zeros_like(data, dtype=np.uint8)
        grid[(data >= 50) | (data < 0)] = 1
        return grid

    @staticmethod
    def _world_to_grid(x, y, resolution, origin):
        return (int((x - origin[0]) / resolution),
                int((y - origin[1]) / resolution))

    @staticmethod
    def _grid_to_world(gx, gy, resolution, origin):
        return ((gx + 0.5) * resolution + origin[0],
                (gy + 0.5) * resolution + origin[1])

    @staticmethod
    def _decimate(path, max_points):
        """Evenly decimate a dense path to at most ``max_points`` waypoints."""
        if len(path) <= max_points:
            return list(path)
        indices = np.linspace(0, len(path) - 1, max_points).astype(int)
        out = [path[i] for i in indices]
        out[0] = path[0]
        out[-1] = path[-1]
        return out

    def _path_msg(self, waypoints, frame_id):
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = frame_id

        yaw = 0.0
        for i, (wx, wy) in enumerate(waypoints):
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = float(wx)
            pose.pose.position.y = float(wy)
            pose.pose.position.z = 0.0
            if i + 1 < len(waypoints):
                nx, ny = waypoints[i + 1]
                yaw = math.atan2(ny - wy, nx - wx)
            pose.pose.orientation.z = math.sin(yaw / 2.0)
            pose.pose.orientation.w = math.cos(yaw / 2.0)
            path_msg.poses.append(pose)
        return path_msg

    def _fail(self, response, reason, t_start, log_msg, nodes_expanded=0):
        response.success = False
        response.failure_reason = reason
        response.compute_time = time.monotonic() - t_start
        response.nodes_expanded = nodes_expanded
        response.cached = False
        response.path = Path()
        response.path.header.frame_id = 'map'
        response.path.header.stamp = self.get_clock().now().to_msg()
        self.get_logger().warn(f'PlanPath {reason}: {log_msg}')
        return response


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
