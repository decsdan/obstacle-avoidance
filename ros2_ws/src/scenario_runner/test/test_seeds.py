
import pytest

from scenario_runner.seeds import derive


def test_derive_matches_prd_formulas():
    s = derive(42)
    assert s.master == 42
    assert s.sensor_noise == 42 * 3 + 1
    assert s.planner_rng == 42 * 3 + 2
    assert s.world_layout == 42 * 3 + 3


def test_derive_is_pure_function():
    a = derive(7)
    b = derive(7)
    assert a == b


def test_derive_distinct_subordinates():
    s = derive(123456)
    values = {s.master, s.sensor_noise, s.planner_rng, s.world_layout}
    assert len(values) == 4


def test_derive_handles_large_seed():
    big = (1 << 64) - 1
    s = derive(big)
    # All fields should be 64-bit clamped.
    for v in (s.master, s.sensor_noise, s.planner_rng, s.world_layout):
        assert 0 <= v < (1 << 64)
