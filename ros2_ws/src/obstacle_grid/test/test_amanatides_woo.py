"""Unit tests for the Amanatides-Woo voxel traversal."""

import math

import pytest

from obstacle_grid.amanatides_woo import trace_ray


def _cells(x0, y0, x1, y1, origin=(0.0, 0.0), res=1.0):
    return [(gx, gy) for gx, gy, _t0, _t1 in trace_ray(
        x0, y0, x1, y1, origin[0], origin[1], res)]


def test_zero_length_ray_yields_single_cell():
    cells = _cells(0.5, 0.5, 0.5, 0.5)
    assert cells == [(0, 0)]


def test_horizontal_ray_visits_each_column_once():
    cells = _cells(0.5, 0.5, 3.5, 0.5)
    assert cells == [(0, 0), (1, 0), (2, 0), (3, 0)]


def test_vertical_ray_visits_each_row_once():
    cells = _cells(0.5, 0.5, 0.5, 3.5)
    assert cells == [(0, 0), (0, 1), (0, 2), (0, 3)]


def test_diagonal_ray_visits_endpoint_cell():
    cells = _cells(0.5, 0.5, 3.5, 3.5)
    assert cells[0] == (0, 0)
    assert cells[-1] == (3, 3)


def test_negative_direction_steps_correctly():
    cells = _cells(3.5, 0.5, 0.5, 0.5)
    assert cells == [(3, 0), (2, 0), (1, 0), (0, 0)]


def test_t_values_cover_unit_interval():
    spans = list(trace_ray(0.5, 0.5, 3.5, 0.5, 0.0, 0.0, 1.0))
    assert spans[0][2] == pytest.approx(0.0)
    assert spans[-1][3] == pytest.approx(1.0)
    for prev, curr in zip(spans, spans[1:]):
        assert curr[2] == pytest.approx(prev[3])


def test_origin_offset_shifts_grid_indices():
    cells = _cells(0.5, 0.5, 2.5, 0.5, origin=(-1.0, -1.0), res=1.0)
    assert cells == [(1, 1), (2, 1), (3, 1)]


def test_resolution_scales_coordinates():
    cells = _cells(0.0, 0.0, 0.4, 0.0, origin=(0.0, 0.0), res=0.1)
    assert cells[0] == (0, 0)
    assert cells[-1] == (4, 0)
