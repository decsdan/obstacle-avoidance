"""Deterministic subordinate-seed derivation from a scenario master seed.

One master seed per episode fans out to distinct subordinate seeds so the
sensor-noise RNG, the planner RNG, and the world-layout RNG never share
state. The derivation is a pure function of the master seed so two runs
with the same scenario YAML produce byte-identical HDF5 sidecars
(PRD \u00a74.5). New subordinate seeds are added here, not ad hoc at call
sites.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SubordinateSeeds:
    """Per-episode subordinate seeds, all derived from one master seed."""

    master: int
    sensor_noise: int
    planner_rng: int
    world_layout: int


def derive(master_seed: int) -> SubordinateSeeds:
    """Return the subordinate-seed bundle for one episode.

    The formulas match PRD \u00a74.5 exactly. Any additional subordinate seed
    must extend the dataclass and this function in lockstep; never compute
    a seed at a call site.
    """
    master = int(master_seed) & 0xFFFFFFFFFFFFFFFF
    return SubordinateSeeds(
        master=master,
        sensor_noise=(master * 3 + 1) & 0xFFFFFFFFFFFFFFFF,
        planner_rng=(master * 3 + 2) & 0xFFFFFFFFFFFFFFFF,
        world_layout=(master * 3 + 3) & 0xFFFFFFFFFFFFFFFF,
    )
