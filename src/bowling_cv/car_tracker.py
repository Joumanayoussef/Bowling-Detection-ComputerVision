"""Car tracking: best detection per frame, EMA smoothing, short gap prediction, path."""
from __future__ import annotations

import math
from typing import Optional

from .config import Config
from .detector import Detection
from .geometry import Box, center


class CarTracker:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.box: Optional[Box] = None          # current (detected or predicted) box
        self.detected: bool = False             # True if box comes from this frame's detection
        self.position: Optional[tuple[float, float]] = None  # smoothed center
        self.path: list[tuple[int, int]] = []
        self._velocity = (0.0, 0.0)
        self._last_measured: Optional[tuple[float, float]] = None
        self._missed = 0

    def update(self, detections: list[Detection]) -> Optional[Box]:
        cars = [d for d in detections if d.label == self.cfg.car_label]
        best = max(cars, key=lambda d: d.conf) if cars else None

        if best is not None:
            cx, cy = center(best.box)
            if self._last_measured is not None:
                self._velocity = (cx - self._last_measured[0], cy - self._last_measured[1])
            self._last_measured = (cx, cy)
            self._missed = 0
            a = self.cfg.car_ema_alpha
            if self.position is None:
                self.position = (cx, cy)
            else:
                self.position = (a * cx + (1 - a) * self.position[0],
                                 a * cy + (1 - a) * self.position[1])
            self.box = best.box
            self.detected = True
        else:
            self._missed += 1
            self.detected = False
            if self.box is not None and self._missed <= self.cfg.car_max_pred_frames:
                # Short, damped constant-velocity prediction to bridge detector gaps.
                dx, dy = self._velocity[0] * 0.5, self._velocity[1] * 0.5
                self.box = (self.box[0] + dx, self.box[1] + dy, self.box[2] + dx, self.box[3] + dy)
                self.position = (self.position[0] + dx, self.position[1] + dy)
            else:
                self.box = None
                self.position = None

        if self.position is not None:
            pt = (int(self.position[0]), int(self.position[1]))
            if not self.path or math.dist(pt, self.path[-1]) > self.cfg.car_path_min_step_px:
                self.path.append(pt)
        return self.box
