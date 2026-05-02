"""Unit tests for world<->grid coordinate conversions."""

import pytest

try:
    from oa_utils.conversions import grid_to_world, world_to_grid
except ImportError:
    pytest.skip('oa_utils.conversions requires geometry_msgs/nav_msgs',
                allow_module_level=True)


def test_world_to_grid_round_trip_at_cell_center():
    res = 0.05
    origin = (-5.0, -5.0)
    gx, gy = world_to_grid(0.0, 0.0, res, origin)
    wx, wy = grid_to_world(gx, gy, res, origin)
    # Cell-center round trip lands within a half-cell of the input.
    assert abs(wx - 0.0) <= res
    assert abs(wy - 0.0) <= res


def test_world_to_grid_origin_maps_to_zero():
    res = 0.1
    gx, gy = world_to_grid(0.0, 0.0, res, (0.0, 0.0))
    assert (gx, gy) == (0, 0)


def test_grid_to_world_returns_cell_center():
    res = 0.5
    origin = (0.0, 0.0)
    wx, wy = grid_to_world(2, 3, res, origin)
    assert wx == pytest.approx(2.5 * res)
    assert wy == pytest.approx(3.5 * res)


def test_world_to_grid_negative_origin_offset():
    res = 1.0
    origin = (-2.0, -2.0)
    assert world_to_grid(0.0, 0.0, res, origin) == (2, 2)
    assert world_to_grid(-2.0, -2.0, res, origin) == (0, 0)
