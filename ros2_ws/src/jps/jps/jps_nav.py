#!/usr/bin/env python3
# Originally authored by the 2025 Carleton Senior Capstone Project
# (see AUTHORS.md). Substantially rewritten by Daniel Scheider, 2026.
"""Jump Point Search global path planner.

Planner only, the navigation server calls this node with a grid and a
start/goal, and it hands back a path. The JPS search, RDP simplification,
and spline smoothing are kept; all the goal-subscription, TF, replan, and
cmd_vel wiring is gone.
"""

import heapq
import math
import time

import numpy as np
from scipy.interpolate import CubicSpline

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_interfaces.srv import PlanPath
from nav_msgs.msg import Path
from rclpy.node import Node


class NavigatorConstants:
    """Default configuration constants for the JPS planner."""

    MAX_PATH_WAYPOINTS = 50
    TIGHT_SPACE_RADIUS = 5
    SIMPLIFICATION_EPSILON = 0.1
    MAX_JUMP_DEPTH = 500
    MAX_SEARCH_NODES = 100_000

    DEFAULT_NAMESPACE = '/don'


class JPSPlanner(Node):
    """JPS PlanPath service server."""

    def __init__(self):
        super().__init__('jps_planner')

        self.declare_parameter('namespace', NavigatorConstants.DEFAULT_NAMESPACE)
        self.ns = self.get_parameter('namespace').value

        self.declare_parameter(
            'max_path_waypoints', NavigatorConstants.MAX_PATH_WAYPOINTS)
        self.declare_parameter(
            'tight_space_radius', NavigatorConstants.TIGHT_SPACE_RADIUS)
        self.declare_parameter(
            'simplification_epsilon', NavigatorConstants.SIMPLIFICATION_EPSILON)
        self.declare_parameter(
            'max_jump_depth', NavigatorConstants.MAX_JUMP_DEPTH)
        self.declare_parameter(
            'max_search_nodes', NavigatorConstants.MAX_SEARCH_NODES)

        self.max_path_waypoints = self.get_parameter('max_path_waypoints').value
        self.tight_space_radius = self.get_parameter('tight_space_radius').value
        self.simplification_epsilon = self.get_parameter(
            'simplification_epsilon').value
        self.max_jump_depth = self.get_parameter('max_jump_depth').value
        self.max_search_nodes = self.get_parameter('max_search_nodes').value

        # Per-request state; set at the top of _handle_plan_path.
        self._grid = None
        self._start = None
        self._deadline = None

        self._srv = self.create_service(
            PlanPath, f'{self.ns}/jps/plan_path', self._handle_plan_path)

        self.get_logger().info(
            f'jps planner ready on {self.ns}/jps/plan_path')

    # ------------------------------------------------------------------
    # Service entry point
    # ------------------------------------------------------------------

    def _handle_plan_path(self, request, response):
        t0 = time.monotonic()

        grid_msg = request.grid_snapshot
        if grid_msg.info.width == 0 or grid_msg.info.height == 0:
            return self._fail(response, 'BAD_INPUT', t0, 'Empty grid_snapshot')

        self._grid = self._grid_from_msg(grid_msg)
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

        rows, cols = self._grid.shape
        if not (0 <= start_cell[0] < cols and 0 <= start_cell[1] < rows):
            return self._fail(response, 'BAD_INPUT', t0,
                              f'Start {start_cell} out of grid bounds')
        if not (0 <= goal_cell[0] < cols and 0 <= goal_cell[1] < rows):
            return self._fail(response, 'BAD_INPUT', t0,
                              f'Goal {goal_cell} out of grid bounds')
        if self._grid[goal_cell[1], goal_cell[0]] != 0:
            return self._fail(response, 'BAD_INPUT', t0,
                              'Goal cell is in an obstacle')

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
            self.get_logger().warn(
                f'PlanPath {response.failure_reason} '
                f'(expanded {nodes_expanded}, {compute_time*1000:.1f} ms)')
            return response

        path_world = [
            self._grid_to_world(gx, gy, resolution, origin)
            for gx, gy in path_cells
        ]
        waypoints = self._simplify_path(path_world, self.max_path_waypoints)

        response.path = self._path_msg(waypoints, grid_msg.header.frame_id or 'map')
        response.success = True
        response.failure_reason = ''
        response.compute_time = compute_time
        response.nodes_expanded = nodes_expanded
        response.cached = False

        self.get_logger().info(
            f'PlanPath OK: {len(waypoints)} pts, '
            f'expanded {nodes_expanded}, {compute_time*1000:.1f} ms')
        return response

    # ------------------------------------------------------------------
    # JPS core
    # ------------------------------------------------------------------

    def _jps(self, start, goal, budget):
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

            if expanded >= budget:
                return [], expanded, 'TIMEOUT'
            if time.monotonic() > self._deadline:
                return [], expanded, 'TIMEOUT'

            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return path[::-1], expanded, ''

            for successor in self._get_successors(current, goal):
                dx = successor[0] - current[0]
                dy = successor[1] - current[1]
                distance = math.hypot(dx, dy)
                tentative_g = g_score[current] + distance

                if successor not in g_score or tentative_g < g_score[successor]:
                    came_from[successor] = current
                    g_score[successor] = tentative_g
                    f = tentative_g + heuristic(successor, goal)
                    heapq.heappush(open_set, (f, successor))

        return [], expanded, 'NO_PATH'

    def _get_successors(self, node, goal):
        successors = []
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0),
                       (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            jump_point = self._jump(node, (dx, dy), goal, depth=0)
            if jump_point is not None:
                successors.append(jump_point)
        return successors

    def _jump(self, node, direction, goal, depth):
        """Jump recursively until a jump point is found or the ray leaves the grid."""
        if depth > self.max_jump_depth:
            return None

        dx, dy = direction
        nxt = (node[0] + dx, node[1] + dy)

        if not self._is_free(nxt):
            return None

        if nxt == goal:
            return nxt

        if dx != 0 and dy != 0:
            # Diagonal: forced neighbours on blocked cardinals
            if (self._is_free((nxt[0] - dx, nxt[1])) and
                    not self._is_free((nxt[0] - dx, nxt[1] + dy))):
                return nxt
            if (self._is_free((nxt[0], nxt[1] - dy)) and
                    not self._is_free((nxt[0] + dx, nxt[1] - dy))):
                return nxt

            if self._jump(nxt, (dx, 0), goal, depth + 1) is not None:
                return nxt
            if self._jump(nxt, (0, dy), goal, depth + 1) is not None:
                return nxt
        else:
            if dx != 0:
                if (self._is_free((nxt[0], nxt[1] + 1)) and
                        not self._is_free((nxt[0] + dx, nxt[1] + 1))):
                    return nxt
                if (self._is_free((nxt[0], nxt[1] - 1)) and
                        not self._is_free((nxt[0] + dx, nxt[1] - 1))):
                    return nxt
            else:
                if (self._is_free((nxt[0] + 1, nxt[1])) and
                        not self._is_free((nxt[0] + 1, nxt[1] + dy))):
                    return nxt
                if (self._is_free((nxt[0] - 1, nxt[1])) and
                        not self._is_free((nxt[0] - 1, nxt[1] + dy))):
                    return nxt

        return self._jump(nxt, direction, goal, depth + 1)

    def _is_free(self, cell):
        """True if ``cell`` is in bounds and free.

        Cells within ``tight_space_radius`` of the start are treated as
        free regardless of inflation so the robot can escape its own halo.
        """
        gx, gy = cell
        rows, cols = self._grid.shape
        if gx < 0 or gx >= cols or gy < 0 or gy >= rows:
            return False
        distance_from_start = abs(gx - self._start[0]) + abs(gy - self._start[1])
        if distance_from_start <= self.tight_space_radius:
            return True
        return self._grid[gy, gx] == 0

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def _simplify_path(self, path, max_points):
        if len(path) <= 2:
            return path

        rdp_path = self._rdp(path, self.simplification_epsilon)
        if len(rdp_path) >= 4:
            smoothed = self._smooth(rdp_path, max_points)
        else:
            smoothed = list(rdp_path)

        smoothed[0] = path[0]
        smoothed[-1] = path[-1]
        return smoothed

    @staticmethod
    def _rdp(points, epsilon):
        if len(points) <= 2:
            return list(points)

        start = np.array(points[0])
        end = np.array(points[-1])
        line_vec = end - start
        line_len = np.linalg.norm(line_vec)

        if line_len < 1e-10:
            return [points[0], points[-1]]

        line_unit = line_vec / line_len
        max_dist = 0.0
        max_idx = 0

        for i in range(1, len(points) - 1):
            pt = np.array(points[i])
            proj = np.clip(np.dot(pt - start, line_unit), 0, line_len)
            closest = start + proj * line_unit
            dist = np.linalg.norm(pt - closest)
            if dist > max_dist:
                max_dist = dist
                max_idx = i

        if max_dist > epsilon:
            left = JPSPlanner._rdp(points[:max_idx + 1], epsilon)
            right = JPSPlanner._rdp(points[max_idx:], epsilon)
            return left[:-1] + right
        return [points[0], points[-1]]

    @staticmethod
    def _smooth(waypoints, max_points):
        pts = np.array(waypoints)
        diffs = np.diff(pts, axis=0)
        chord_lengths = np.sqrt(np.sum(diffs ** 2, axis=1))
        t = np.concatenate([[0], np.cumsum(chord_lengths)])
        total_length = t[-1]

        if total_length < 1e-10:
            return list(waypoints)

        cs_x = CubicSpline(t, pts[:, 0])
        cs_y = CubicSpline(t, pts[:, 1])

        n_out = min(max_points, max(len(waypoints), 10))
        t_new = np.linspace(0, total_length, n_out)
        return list(zip(cs_x(t_new).tolist(), cs_y(t_new).tolist()))

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

    def _fail(self, response, reason, t_start, log_msg):
        response.success = False
        response.failure_reason = reason
        response.compute_time = time.monotonic() - t_start
        response.nodes_expanded = 0
        response.cached = False
        response.path = Path()
        response.path.header.frame_id = 'map'
        response.path.header.stamp = self.get_clock().now().to_msg()
        self.get_logger().warn(f'PlanPath {reason}: {log_msg}')
        return response


def main(args=None):
    rclpy.init(args=args)
    node = JPSPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
