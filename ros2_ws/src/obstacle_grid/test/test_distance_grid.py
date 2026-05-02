"""Unit tests for inflation and distance-transform helpers."""

import numpy as np
import pytest

from obstacle_grid.distance_grid import distance_field, inflate_binary


def test_zero_inflation_returns_copy():
    grid = np.array([[0, 100], [0, 0]], dtype=np.int8)
    out = inflate_binary(grid, 0)
    assert np.array_equal(out, grid)
    assert out is not grid  # must be a copy


def test_inflation_grows_obstacle_by_radius():
    grid = np.zeros((5, 5), dtype=np.int8)
    grid[2, 2] = 100
    out = inflate_binary(grid, inflation_cells=1)
    expected = np.array([
        [0, 0, 0, 0, 0],
        [0, 100, 100, 100, 0],
        [0, 100, 100, 100, 0],
        [0, 100, 100, 100, 0],
        [0, 0, 0, 0, 0],
    ], dtype=np.int8)
    assert np.array_equal(out, expected)


def test_distance_field_zero_at_obstacle():
    grid = np.zeros((3, 3), dtype=np.int8)
    grid[1, 1] = 100
    field = distance_field(grid, resolution=0.5)
    assert field[1, 1] == pytest.approx(0.0)


def test_distance_field_scales_with_resolution():
    grid = np.zeros((1, 5), dtype=np.int8)
    grid[0, 0] = 100
    field_low = distance_field(grid, resolution=0.1)
    field_high = distance_field(grid, resolution=1.0)
    np.testing.assert_allclose(field_high / 10.0, field_low, rtol=1e-5)


def test_inflated_obstacle_count_monotonic_in_radius():
    grid = np.zeros((11, 11), dtype=np.int8)
    grid[5, 5] = 100
    counts = [int((inflate_binary(grid, r) >= 100).sum())
              for r in range(0, 4)]
    assert counts == sorted(counts)
