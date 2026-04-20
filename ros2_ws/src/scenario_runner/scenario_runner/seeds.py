"""Deterministic subordinate-seed derivation from a scenario master seed."""
from dataclasses import dataclass


@dataclass(frozen=True)
class SubordinateSeeds:
    """Per-episode subordinate seeds, all derived from one master seed."""

    master: int
    sensor_noise: int
    planner_rng: int
    world_layout: int


def derive(master_seed: int) -> SubordinateSeeds:
    # per PRD §4.5
    master = int(master_seed) & 0xFFFFFFFFFFFFFFFF
    return SubordinateSeeds(
        master=master,
        sensor_noise=(master * 3 + 1) & 0xFFFFFFFFFFFFFFFF,
        planner_rng=(master * 3 + 2) & 0xFFFFFFFFFFFFFFFF,
        world_layout=(master * 3 + 3) & 0xFFFFFFFFFFFFFFFF,
    )
