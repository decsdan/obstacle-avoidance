"""Scenario YAML loader, schema validator, and manifest expander."""
from dataclasses import asdict, dataclass, field
from typing import Any

import yaml


class ScenarioSchemaError(ValueError):
    """Raised when a scenario or manifest YAML fails schema validation."""


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    theta: float


@dataclass(frozen=True)
class Strategy:
    global_planner: str
    local_planner: str


@dataclass(frozen=True)
class Termination:
    goal_tolerance: float
    timeout_sec: float
    stuck_window_sec: float
    stuck_distance_m: float


@dataclass(frozen=True)
class Randomization:
    lidar_range_sigma: float = 0.01
    lidar_dropout_rate: float = 0.0
    odom_linear_sigma: float = 0.005
    odom_angular_sigma: float = 0.01
    lighting_variation: float = 0.0
    obstacle_jitter_sigma: float = 0.0
    imu_noise_scale: float = 1.0


@dataclass(frozen=True)
class LoggingConfig:
    include_raw_scan: bool = True
    include_tf_tree: bool = True
    include_debug_markers: bool = True
    downsample_hz: float | None = None


@dataclass(frozen=True)
class Scenario:
    id: str
    world: str
    seed: int
    description: str
    start_pose: Pose2D
    goal_pose: Pose2D
    strategy: Strategy
    termination: Termination
    randomization: Randomization
    logging: LoggingConfig

    def as_dict(self) -> dict:
        """Nested dict for HDF5 attribute serialization."""
        return asdict(self)


@dataclass(frozen=True)
class ManifestEntry:
    """One manifest row after ``repeat`` expansion."""

    scenario_id: str
    seed_override: int | None


@dataclass(frozen=True)
class Manifest:
    version: int
    entries: tuple[ManifestEntry, ...]


def _require(mapping: dict, key: str, context: str) -> Any:
    if key not in mapping:
        raise ScenarioSchemaError(f'{context}: missing required field "{key}"')
    return mapping[key]


def _parse_pose(raw: dict, context: str) -> Pose2D:
    return Pose2D(
        x=float(_require(raw, 'x', context)),
        y=float(_require(raw, 'y', context)),
        theta=float(_require(raw, 'theta', context)),
    )


def load_scenario(path: str) -> Scenario:
    """Parse and validate one scenario YAML."""
    with open(path, 'r') as handle:
        raw = yaml.safe_load(handle) or {}

    strategy_raw = _require(raw, 'strategy', path)
    term_raw = _require(raw, 'termination', path)
    rand_raw = raw.get('randomization', {}) or {}
    log_raw = raw.get('logging', {}) or {}

    return Scenario(
        id=str(_require(raw, 'id', path)),
        world=str(_require(raw, 'world', path)),
        seed=int(_require(raw, 'seed', path)),
        description=str(raw.get('description', '')),
        start_pose=_parse_pose(
            _require(raw, 'start_pose', path), f'{path}:start_pose'),
        goal_pose=_parse_pose(
            _require(raw, 'goal_pose', path), f'{path}:goal_pose'),
        strategy=Strategy(
            global_planner=str(_require(strategy_raw, 'global', path)),
            local_planner=str(_require(strategy_raw, 'local', path)),
        ),
        termination=Termination(
            goal_tolerance=float(_require(term_raw, 'goal_tolerance', path)),
            timeout_sec=float(_require(term_raw, 'timeout_sec', path)),
            stuck_window_sec=float(
                _require(term_raw, 'stuck_window_sec', path)),
            stuck_distance_m=float(
                _require(term_raw, 'stuck_distance_m', path)),
        ),
        randomization=Randomization(**{
            k: float(v) for k, v in rand_raw.items()
            if k in Randomization.__dataclass_fields__
        }),
        logging=LoggingConfig(
            include_raw_scan=bool(log_raw.get('include_raw_scan', True)),
            include_tf_tree=bool(log_raw.get('include_tf_tree', True)),
            include_debug_markers=bool(
                log_raw.get('include_debug_markers', True)),
            downsample_hz=log_raw.get('downsample_hz', None),
        ),
    )


def load_manifest(path: str) -> Manifest:
    """Parse and validate a manifest YAML, expanding ``repeat`` entries."""
    with open(path, 'r') as handle:
        raw = yaml.safe_load(handle) or {}

    version = int(raw.get('version', 1))
    scenarios = raw.get('scenarios', []) or []
    if not isinstance(scenarios, list):
        raise ScenarioSchemaError(f'{path}: "scenarios" must be a list')

    entries: list[ManifestEntry] = []
    for row in scenarios:
        scenario_id = str(_require(row, 'id', path))
        repeat = int(row.get('repeat', 1))
        seed_offset = row.get('seed_offset', None)
        for i in range(repeat):
            override = None
            if seed_offset is not None:
                override = int(seed_offset) + i
            entries.append(ManifestEntry(
                scenario_id=scenario_id,
                seed_override=override,
            ))

    return Manifest(version=version, entries=tuple(entries))
