"""Unit tests for the DStarLite incremental search class."""

import math

import numpy as np
import pytest

try:
    from d_star.d_star_nav import DStarLite
except ImportError:
    pytest.skip('d_star_nav requires ROS imports to load',
                allow_module_level=True)


def _solve(grid, start, goal, budget=10_000):
    ds = DStarLite(grid, start, goal)
    ok, _iters, reason = ds.compute_shortest_path(
        budget=budget, deadline=math.inf)
    return ok, reason, ds.extract_path()


def test_open_grid_finds_straight_path():
    grid = np.zeros((5, 5), dtype=np.int8)
    ok, reason, path = _solve(grid, (0, 0), (4, 4))
    assert ok and reason == ''
    assert path[0] == (0, 0)
    assert path[-1] == (4, 4)


def test_walled_grid_finds_no_path():
    grid = np.zeros((5, 5), dtype=np.int8)
    grid[:, 2] = 100  # full vertical wall
    ds = DStarLite(grid, (0, 0), (4, 4))
    ok, _iters, _reason = ds.compute_shortest_path(
        budget=10_000, deadline=math.inf)
    # Search may return ok if it exhausts the open list; path must be empty.
    assert ds.extract_path() == []


def test_path_avoids_obstacles():
    grid = np.zeros((5, 5), dtype=np.int8)
    grid[2, 2] = 100
    ok, _reason, path = _solve(grid, (0, 0), (4, 4))
    assert ok
    assert (2, 2) not in path


def test_heuristic_is_euclidean():
    assert DStarLite.heuristic((0, 0), (3, 4)) == pytest.approx(5.0)
    assert DStarLite.heuristic((1, 1), (1, 1)) == 0.0


def test_diagonal_cost_matches_sqrt2():
    grid = np.zeros((3, 3), dtype=np.int8)
    ds = DStarLite(grid, (0, 0), (2, 2))
    assert ds.cost((0, 0), (1, 1)) == pytest.approx(math.sqrt(2.0), rel=1e-4)
    assert ds.cost((0, 0), (1, 0)) == pytest.approx(1.0)


def test_compare_keys_lex_order():
    assert DStarLite.compare_keys((1.0, 0.0), (1.5, 0.0)) is True
    assert DStarLite.compare_keys((1.0, 0.0), (1.0, 0.5)) is True
    assert DStarLite.compare_keys((1.0, 0.0), (1.0, 0.0)) is False
    assert DStarLite.compare_keys((2.0, 0.0), (1.0, 9.0)) is False
