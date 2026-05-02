"""Unit tests for shared path-simplification helpers."""

import math

import pytest

from oa_utils.pathing import rdp, simplify_path, smooth_path


def test_rdp_collinear_collapses_to_endpoints():
    pts = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)]
    out = rdp(pts, epsilon=0.01)
    assert out == [(0.0, 0.0), (3.0, 0.0)]


def test_rdp_preserves_corner():
    pts = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
    out = rdp(pts, epsilon=0.01)
    assert (1.0, 0.0) in out
    assert out[0] == (0.0, 0.0)
    assert out[-1] == (1.0, 1.0)


def test_rdp_short_path_unchanged():
    pts = [(0.0, 0.0), (1.0, 1.0)]
    assert rdp(pts, epsilon=0.5) == pts


def test_rdp_zero_length_path():
    pts = [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]
    out = rdp(pts, epsilon=0.5)
    assert out == [(0.0, 0.0), (0.0, 0.0)]


def test_simplify_path_keeps_endpoints():
    pts = [(float(i), math.sin(i)) for i in range(10)]
    out = simplify_path(pts, max_points=20, epsilon=0.05)
    assert out[0] == pts[0]
    assert out[-1] == pts[-1]


def test_simplify_short_path_returns_input():
    pts = [(0.0, 0.0), (1.0, 1.0)]
    assert simplify_path(pts, max_points=10, epsilon=0.1) == pts


def test_smooth_path_returns_max_points():
    pts = [(float(i), 0.0) for i in range(5)]
    out = smooth_path(pts, max_points=8)
    assert len(out) == 8
    assert out[0][0] == pytest.approx(0.0)
    assert out[-1][0] == pytest.approx(4.0)
