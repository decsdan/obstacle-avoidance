"""Unit tests for the A* heuristic admissibility / consistency.

These tests check the two properties: admissibility (h never overestimates true
cost on an open grid) and consistency (the triangle inequality holds
between adjacent cells, with ``sqrt(2)`` cost used by the planner).
"""

import math

import pytest


def h(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def step_cost(a, b):
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return math.sqrt(2.0) if (dx + dy) == 2 else 1.0


def test_h_zero_at_goal():
    assert h((3, 4), (3, 4)) == 0.0


def test_h_admissible_against_8_connected_chebyshev():
    """Euclidean h <= true min-cost path on an obstacle-free 8-connected grid.

    True cost = max(|dx|, |dy|) * sqrt(2) + abs(|dx| - |dy|) * 1; Euclidean
    is always <= that.
    """
    for ax, ay in [(0, 0), (1, 2), (5, 0)]:
        for bx, by in [(3, 4), (7, 1), (10, 10)]:
            dx = abs(ax - bx)
            dy = abs(ay - by)
            true_cost = max(dx, dy) * math.sqrt(2.0) + abs(dx - dy) * (
                1.0 - math.sqrt(2.0) / 2.0)
            # Use the plain optimal-on-8-grid formula directly:
            true_cost = math.sqrt(2.0) * min(dx, dy) + abs(dx - dy)
            assert h((ax, ay), (bx, by)) <= true_cost + 1e-9


def test_h_consistent_under_step_cost():
    """h(n) <= cost(n, n') + h(n') for every neighbor n' of n."""
    goal = (5, 5)
    for nx in range(8):
        for ny in range(8):
            n = (nx, ny)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    np_ = (n[0] + dx, n[1] + dy)
                    assert h(n, goal) <= step_cost(n, np_) + h(np_, goal) + 1e-9


def test_h_symmetry():
    assert h((0, 0), (3, 4)) == pytest.approx(h((3, 4), (0, 0)))


def test_h_triangle_inequality():
    a, b, c = (0, 0), (5, 0), (5, 5)
    assert h(a, c) <= h(a, b) + h(b, c) + 1e-9
