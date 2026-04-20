"""Terminal-outcome classifier for one episode."""
import math
from collections import deque
from dataclasses import dataclass
from typing import Optional


@dataclass
class TerminalConfig:
    goal_tolerance: float
    timeout_sec: float
    stuck_window_sec: float
    stuck_distance_m: float


class TerminalDetector:
    """Per-episode terminal-condition state machine."""

    def __init__(self, cfg: TerminalConfig, tick_hz: float):
        self._cfg = cfg
        self._tick_hz = tick_hz
        self._window_len = max(1, int(cfg.stuck_window_sec * tick_hz))
        self._history: deque[tuple[float, float]] = deque(
            maxlen=self._window_len)
        self._start_sim_time: Optional[float] = None
        self._collision = False
        self._cancelled = False
        self.terminal: Optional[str] = None

    def note_collision(self) -> None:
        """Latch a collision; next ``update`` call will classify terminal."""
        self._collision = True

    def note_cancel(self) -> None:
        """Latch an external cancel; next ``update`` will classify terminal."""
        self._cancelled = True

    def update(
        self,
        sim_time: float,
        pose_xy: tuple[float, float],
        goal_xy: tuple[float, float],
    ) -> Optional[str]:
        """Advance the state machine. Returns the terminal once latched."""
        if self.terminal is not None:
            return self.terminal

        if self._start_sim_time is None:
            self._start_sim_time = sim_time

        if self._collision:
            self.terminal = 'collision'
            return self.terminal

        if self._cancelled:
            self.terminal = 'cancelled'
            return self.terminal

        dx = pose_xy[0] - goal_xy[0]
        dy = pose_xy[1] - goal_xy[1]
        if math.hypot(dx, dy) <= self._cfg.goal_tolerance:
            self.terminal = 'reached'
            return self.terminal

        if sim_time - self._start_sim_time >= self._cfg.timeout_sec:
            self.terminal = 'timeout'
            return self.terminal

        self._history.append((pose_xy[0], pose_xy[1]))
        if len(self._history) >= self._window_len:
            hx0, hy0 = self._history[0]
            if math.hypot(pose_xy[0] - hx0, pose_xy[1] - hy0) \
                    < self._cfg.stuck_distance_m:
                self.terminal = 'stuck'
                return self.terminal

        return None
