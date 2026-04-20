"""Per-episode driver tying the runner to one Navigate goal."""
import math
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import yaml

from scenario_runner.dataset_writer import HDF5SidecarWriter, TickRecord
from scenario_runner.scenario_spec import Scenario
from scenario_runner.seeds import SubordinateSeeds
from scenario_runner.terminal_detector import TerminalConfig, TerminalDetector
from scenario_runner.versioning import EnvVersions


@dataclass
class EpisodePaths:
    bag_dir: str
    hdf5_path: str

    @classmethod
    def for_episode(
        cls,
        root: str,
        scenario_id: str,
        seed: int,
    ) -> 'EpisodePaths':
        base = os.path.join(root, scenario_id, str(seed))
        os.makedirs(base, exist_ok=True)
        return cls(
            bag_dir=os.path.join(base, 'bag'),
            hdf5_path=os.path.join(base, 'episode.h5'),
        )


class Episode:
    """State for one scenario run; one instance per episode."""

    def __init__(
        self,
        scenario: Scenario,
        seed: int,
        paths: EpisodePaths,
        tick_hz: float,
        env_versions: EnvVersions,
        scenario_yaml_text: str,
    ):
        self.scenario = scenario
        self.seed = seed
        self.paths = paths
        self.tick_hz = tick_hz
        self.env_versions = env_versions
        self.scenario_yaml_text = scenario_yaml_text

        from scenario_runner.seeds import derive
        self.subordinate_seeds: SubordinateSeeds = derive(seed)

        self.detector = TerminalDetector(
            TerminalConfig(
                goal_tolerance=scenario.termination.goal_tolerance,
                timeout_sec=scenario.termination.timeout_sec,
                stuck_window_sec=scenario.termination.stuck_window_sec,
                stuck_distance_m=scenario.termination.stuck_distance_m,
            ),
            tick_hz=tick_hz,
        )

        self.writer = HDF5SidecarWriter(
            path=paths.hdf5_path,
            scenario_id=scenario.id,
            strategy_global=scenario.strategy.global_planner,
            strategy_local=scenario.strategy.local_planner,
            include_raw_scan=scenario.logging.include_raw_scan,
        )
        self._writer_open = False

        self.tick_index = 0
        self.total_distance = 0.0
        self._last_pose: Optional[tuple[float, float]] = None
        self.start_sim_time: Optional[float] = None
        self.last_sim_time: float = 0.0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._writer_open:
            if exc_type is not None:
                self.close(terminal_outcome='aborted_ungracefully')
            else:
                self.close(terminal_outcome='closed_without_outcome')

    def ensure_writer_open(self, grid_shape: tuple[int, int], scan_len: int) -> None:
        if self._writer_open:
            return
        self.writer.open(grid_shape, scan_len)
        self._writer_open = True

    def record_tick(self, record: TickRecord) -> None:
        if not self._writer_open:
            raise RuntimeError('Episode writer not opened; call ensure_writer_open first')
        self.writer.append(record)
        self.tick_index += 1
        self.last_sim_time = record.timestamp_sim
        if self.start_sim_time is None:
            self.start_sim_time = record.timestamp_sim
        if self._last_pose is not None:
            dx = record.pose[0] - self._last_pose[0]
            dy = record.pose[1] - self._last_pose[1]
            self.total_distance += math.hypot(dx, dy)
        self._last_pose = (record.pose[0], record.pose[1])

    def close(self, terminal_outcome: str) -> None:
        """Finalize the HDF5 file. Idempotent."""
        if not self._writer_open:
            return
        total_sim_time = 0.0
        if self.start_sim_time is not None:
            total_sim_time = max(0.0, self.last_sim_time - self.start_sim_time)
        subordinate = {
            'master': self.subordinate_seeds.master,
            'sensor_noise': self.subordinate_seeds.sensor_noise,
            'planner_rng': self.subordinate_seeds.planner_rng,
            'world_layout': self.subordinate_seeds.world_layout,
        }
        self.writer.close(
            terminal_outcome=terminal_outcome,
            total_sim_time=total_sim_time,
            total_distance=self.total_distance,
            scenario_yaml_text=self.scenario_yaml_text,
            subordinate_seeds=subordinate,
            env_versions=self.env_versions.as_dict(),
        )
        self._writer_open = False


def pose_from_transform(translation, rotation) -> tuple[float, float, float]:
    """Map-frame ``(x, y, theta)`` from a TF2 translation + quaternion."""
    yaw = math.atan2(
        2.0 * (rotation.w * rotation.z + rotation.x * rotation.y),
        1.0 - 2.0 * (rotation.y * rotation.y + rotation.z * rotation.z),
    )
    return (float(translation.x), float(translation.y), yaw)
