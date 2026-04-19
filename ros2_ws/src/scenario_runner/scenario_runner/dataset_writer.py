"""Matched rosbag2 + HDF5 sidecar episode writer.

Two artifacts per episode (PRD \u00a74.4):

1. ``bag/`` -- full rosbag2 stream for fidelity replay and debugging.
2. ``episode.h5`` -- per-tick aligned arrays for ML training.

The HDF5 writer allocates resizable datasets for each mandatory field
in PRD \u00a74.3 and extends them one row at a time. Scenario YAML, resolved
subordinate seeds, and environment versions are attached as file-level
attributes on close. Close is idempotent so terminal-detection paths
and exception paths can both invoke it without branching.
"""

from dataclasses import dataclass
from typing import Any, Optional

import h5py
import numpy as np
import yaml

try:
    import rosbag2_py
    _ROSBAG2_AVAILABLE = True
except ImportError:  # allow unit tests to import without rosbag2_py
    _ROSBAG2_AVAILABLE = False


@dataclass
class TickRecord:
    """One control-cycle record matching PRD \u00a74.3 mandatory fields."""

    timestamp_sim: float
    timestamp_wall: float
    tick_index: int
    pose: tuple[float, float, float]
    velocity: tuple[float, float]
    goal: tuple[float, float, float]
    cmd_vel: tuple[float, float]
    path_waypoint_idx: int
    obstacle_grid: np.ndarray  # (H, W) uint8 log-odds clipped
    grid_origin: tuple[float, float, float]
    grid_resolution: float
    global_path: np.ndarray  # (N, 3) float32; zero-row means no path
    scan_ranges: Optional[np.ndarray] = None


class HDF5SidecarWriter:
    """Resizable per-tick HDF5 dataset writer."""

    _INITIAL_CAPACITY = 512

    def __init__(
        self,
        path: str,
        scenario_id: str,
        strategy_global: str,
        strategy_local: str,
        include_raw_scan: bool,
    ):
        self._path = path
        self._scenario_id = scenario_id
        self._strategy_global = strategy_global
        self._strategy_local = strategy_local
        self._include_raw_scan = include_raw_scan
        self._file: Optional[h5py.File] = None
        self._datasets: dict[str, h5py.Dataset] = {}
        self._size = 0
        self._capacity = self._INITIAL_CAPACITY
        self._closed = False

    def open(self, grid_shape: tuple[int, int], scan_len: int) -> None:
        """Allocate resizable datasets sized from the first observation."""
        self._file = h5py.File(self._path, 'w')
        h, w = grid_shape
        f = self._file

        self._make('timestamp_sim', (self._capacity,), np.float64)
        self._make('timestamp_wall', (self._capacity,), np.float64)
        self._make('tick_index', (self._capacity,), np.uint64)
        self._make('pose', (self._capacity, 3), np.float32)
        self._make('velocity', (self._capacity, 2), np.float32)
        self._make('goal', (self._capacity, 3), np.float32)
        self._make('cmd_vel', (self._capacity, 2), np.float32)
        self._make('path_waypoint_idx', (self._capacity,), np.int32)
        self._make('grid_origin', (self._capacity, 3), np.float32)
        self._make('grid_resolution', (self._capacity,), np.float32)
        self._make(
            'obstacle_grid',
            (self._capacity, h, w),
            np.uint8,
            chunks=(1, h, w),
        )
        # Variable-length path per tick, stored as an HDF5 vlen of (N*3) floats.
        vlen = h5py.vlen_dtype(np.dtype('float32'))
        f.create_dataset(
            'global_path',
            shape=(self._capacity,),
            maxshape=(None,),
            dtype=vlen,
        )
        self._datasets['global_path'] = f['global_path']

        if self._include_raw_scan and scan_len > 0:
            self._make(
                'scan_ranges',
                (self._capacity, scan_len),
                np.float32,
                chunks=(1, scan_len),
            )

        f.attrs['scenario_id'] = self._scenario_id
        f.attrs['strategy_global'] = self._strategy_global
        f.attrs['strategy_local'] = self._strategy_local

    def append(self, record: TickRecord) -> None:
        if self._closed or self._file is None:
            raise RuntimeError('HDF5SidecarWriter is closed or not opened')
        if self._size >= self._capacity:
            self._grow()

        i = self._size
        self._datasets['timestamp_sim'][i] = record.timestamp_sim
        self._datasets['timestamp_wall'][i] = record.timestamp_wall
        self._datasets['tick_index'][i] = record.tick_index
        self._datasets['pose'][i] = np.asarray(record.pose, dtype=np.float32)
        self._datasets['velocity'][i] = np.asarray(
            record.velocity, dtype=np.float32)
        self._datasets['goal'][i] = np.asarray(record.goal, dtype=np.float32)
        self._datasets['cmd_vel'][i] = np.asarray(
            record.cmd_vel, dtype=np.float32)
        self._datasets['path_waypoint_idx'][i] = record.path_waypoint_idx
        self._datasets['grid_origin'][i] = np.asarray(
            record.grid_origin, dtype=np.float32)
        self._datasets['grid_resolution'][i] = record.grid_resolution
        self._datasets['obstacle_grid'][i] = record.obstacle_grid.astype(
            np.uint8, copy=False)
        if record.global_path.size == 0:
            self._datasets['global_path'][i] = np.empty((0,), dtype=np.float32)
        else:
            self._datasets['global_path'][i] = record.global_path.astype(
                np.float32, copy=False).reshape(-1)
        if 'scan_ranges' in self._datasets and record.scan_ranges is not None:
            self._datasets['scan_ranges'][i] = record.scan_ranges.astype(
                np.float32, copy=False)
        self._size += 1

    def close(
        self,
        terminal_outcome: str,
        total_sim_time: float,
        total_distance: float,
        scenario_yaml_text: str,
        subordinate_seeds: dict,
        env_versions: dict,
    ) -> None:
        if self._closed:
            return
        self._closed = True
        if self._file is None:
            return

        for name, ds in self._datasets.items():
            ds.resize((self._size,) + ds.shape[1:])

        f = self._file
        f.attrs['terminal_outcome'] = terminal_outcome
        f.attrs['total_ticks'] = np.uint64(self._size)
        f.attrs['total_sim_time'] = np.float64(total_sim_time)
        f.attrs['total_distance'] = np.float32(total_distance)
        f.attrs['scenario_yaml'] = scenario_yaml_text
        f.attrs['subordinate_seeds'] = yaml.safe_dump(
            subordinate_seeds, sort_keys=True)
        f.attrs['env_versions'] = yaml.safe_dump(
            env_versions, sort_keys=True)

        f.close()
        self._file = None

    def _make(
        self,
        name: str,
        shape: tuple[int, ...],
        dtype: Any,
        chunks: Optional[tuple[int, ...]] = None,
    ) -> None:
        f = self._file
        ds = f.create_dataset(
            name,
            shape=shape,
            maxshape=(None,) + shape[1:],
            dtype=dtype,
            chunks=chunks,
        )
        self._datasets[name] = ds

    def _grow(self) -> None:
        new_capacity = self._capacity * 2
        for name, ds in self._datasets.items():
            ds.resize((new_capacity,) + ds.shape[1:])
        self._capacity = new_capacity


class BagRecorder:
    """Thin rosbag2_py wrapper scoped to one episode.

    Records the full topic stream for fidelity replay. Topic-type mapping
    is supplied by the caller so tests can construct it without requiring
    rosbag2_py to be importable.
    """

    def __init__(self, uri: str, storage_id: str = 'sqlite3'):
        self._uri = uri
        self._storage_id = storage_id
        self._writer = None

    def open(self, topics: list[tuple[str, str]]) -> None:
        """Open the bag with ``(topic_name, type_name)`` pairs."""
        if not _ROSBAG2_AVAILABLE:
            raise RuntimeError('rosbag2_py is unavailable in this environment')
        storage = rosbag2_py.StorageOptions(
            uri=self._uri, storage_id=self._storage_id)
        converter = rosbag2_py.ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr',
        )
        self._writer = rosbag2_py.SequentialWriter()
        self._writer.open(storage, converter)
        for topic, type_name in topics:
            self._writer.create_topic(rosbag2_py.TopicMetadata(
                name=topic,
                type=type_name,
                serialization_format='cdr',
            ))

    def write(self, topic: str, serialized_msg: bytes, stamp_ns: int) -> None:
        if self._writer is None:
            raise RuntimeError('BagRecorder is not open')
        self._writer.write(topic, serialized_msg, stamp_ns)

    def close(self) -> None:
        self._writer = None
