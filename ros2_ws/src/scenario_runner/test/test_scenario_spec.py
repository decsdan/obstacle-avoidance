"""Unit tests for scenario YAML loader and manifest expander."""

import os
import textwrap

import pytest

from scenario_runner.scenario_spec import (
    ScenarioSchemaError, load_manifest, load_scenario,
)


_VALID_SCENARIO = textwrap.dedent("""
    id: test_01
    world: W1_indoor_office
    seed: 42
    description: "test"
    start_pose: {x: 0.0, y: 0.0, theta: 0.0}
    goal_pose: {x: 1.0, y: 2.0, theta: 0.0}
    strategy: {global: a_star, local: dwa}
    termination:
      goal_tolerance: 0.2
      timeout_sec: 60.0
      stuck_window_sec: 5.0
      stuck_distance_m: 0.05
""").strip()


def test_load_scenario_round_trip(tmp_path):
    p = tmp_path / 'scen.yaml'
    p.write_text(_VALID_SCENARIO)
    s = load_scenario(str(p))
    assert s.id == 'test_01'
    assert s.world == 'W1_indoor_office'
    assert s.seed == 42
    assert s.start_pose.x == 0.0
    assert s.goal_pose.y == 2.0
    assert s.strategy.global_planner == 'a_star'
    assert s.strategy.local_planner == 'dwa'
    assert s.termination.goal_tolerance == pytest.approx(0.2)
    # Defaults applied for omitted blocks.
    assert s.randomization.lidar_range_sigma == pytest.approx(0.01)
    assert s.logging.include_raw_scan is True


def test_missing_required_field_raises(tmp_path):
    bad = textwrap.dedent("""
        id: test_02
        seed: 7
        start_pose: {x: 0.0, y: 0.0, theta: 0.0}
        goal_pose: {x: 1.0, y: 0.0, theta: 0.0}
        strategy: {global: a_star, local: dwa}
        termination:
          goal_tolerance: 0.2
          timeout_sec: 60.0
          stuck_window_sec: 5.0
          stuck_distance_m: 0.05
    """).strip()
    p = tmp_path / 'bad.yaml'
    p.write_text(bad)
    with pytest.raises(ScenarioSchemaError, match='world'):
        load_scenario(str(p))


def test_manifest_repeat_expands_with_seed_offset(tmp_path):
    manifest = textwrap.dedent("""
        version: 1
        scenarios:
          - id: a
          - id: b
            repeat: 3
            seed_offset: 100
    """).strip()
    p = tmp_path / 'manifest.yaml'
    p.write_text(manifest)
    m = load_manifest(str(p))
    ids = [e.scenario_id for e in m.entries]
    seeds = [e.seed_override for e in m.entries]
    assert ids == ['a', 'b', 'b', 'b']
    assert seeds == [None, 100, 101, 102]


def test_manifest_rejects_non_list(tmp_path):
    p = tmp_path / 'manifest.yaml'
    p.write_text('version: 1\nscenarios: not_a_list\n')
    with pytest.raises(ScenarioSchemaError):
        load_manifest(str(p))
