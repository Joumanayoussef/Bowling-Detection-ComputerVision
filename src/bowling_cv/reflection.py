"""Reflective-floor filter.

A shiny floor mirrors each pin just below its base. YOLO can pick the mirror
image up as an extra "standing pin" or as a "fallen pin", which leads to extra
locked pins or false falls. Two defences:

* Spatial (this module): reject a pin detection that sits directly under a pin
  that is standing in the same frame, is horizontally aligned with it, has a
  similar width and (nearly) touches its bottom edge, i.e. looks like its mirror
  image. When the locked pins form a row, also reject detections whose center
  is well below the row's floor line.
* Temporal (in ``pin_tracker``): a fall signal must persist for
  ``Config.confirm_frames`` consecutive frames. Reflections flicker and rarely do.

Every rejected detection is recorded with its reason.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

from .config import Config
from .geometry import Box, center, height, size, width, x_overlap_ratio

log = logging.getLogger(__name__)

MIRROR = "mirror_below_pin"
BELOW_FLOOR = "below_floor_line"


@dataclass
class Rejection:
    frame: int
    time_s: float
    label: str
    conf: float
    box: Box
    reason: str
    detail: str

    def to_dict(self) -> dict:
        return {
            "frame": self.frame,
            "time_s": round(self.time_s, 3),
            "label": self.label,
            "conf": round(self.conf, 3),
            "box": [round(v, 1) for v in self.box],
            "reason": self.reason,
            "detail": self.detail,
        }


def is_mirror_of(det: Box, pin: Box, cfg: Config) -> bool:
    """True if ``det`` looks like the floor reflection of ``pin``."""
    h_det = height(det)
    if h_det <= 0 or height(pin) <= 0:
        return False
    pin_bottom = pin[3]
    below = max(0.0, det[3] - max(det[1], pin_bottom)) / h_det
    if below < cfg.mirror_min_below_frac:
        return False
    if x_overlap_ratio(det, pin) < cfg.mirror_min_x_overlap:
        return False
    lo, hi = cfg.mirror_width_ratio
    ratio = width(det) / max(width(pin), 1e-6)
    if not lo <= ratio <= hi:
        return False
    gap = det[1] - pin_bottom  # negative when the boxes overlap vertically
    return gap <= cfg.mirror_max_gap_frac * height(pin)


class FloorLine:
    """Line y = a*x + b through the bottom edges of the locked pins."""

    def __init__(self, a: float, b: float, pin_size: float):
        self.a, self.b, self.pin_size = a, b, pin_size

    def y_at(self, x: float) -> float:
        return self.a * x + self.b

    @classmethod
    def fit(cls, pin_boxes: list[Box], cfg: Config) -> tuple[Optional["FloorLine"], str]:
        if not pin_boxes:
            return None, "no locked pins"
        pin_size = float(np.median([size(b) for b in pin_boxes]))
        xs = np.array([center(b)[0] for b in pin_boxes])
        ys = np.array([b[3] for b in pin_boxes])
        if len(pin_boxes) == 1 or np.ptp(xs) < 1e-6:
            a, b = 0.0, float(ys.max())
        else:
            a, b = np.polyfit(xs, ys, 1)
        resid = np.abs(ys - (a * xs + b)).max()
        if resid > cfg.floor_row_max_residual_frac * pin_size:
            return None, (f"pins do not form a row (max residual {resid:.1f}px > "
                          f"{cfg.floor_row_max_residual_frac} x pin size {pin_size:.1f}px)")
        return cls(float(a), float(b), pin_size), f"y = {a:.3f}*x + {b:.1f}"


class ReflectionFilter:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.enabled = cfg.reflection_filter
        self.floor: Optional[FloorLine] = None
        self.rejections: list[Rejection] = []

    # ── Setup phase ─────────────────────────────────────────
    def filter_locked_candidates(self, boxes: list[Box]) -> tuple[list[int], list[tuple[int, str]]]:
        """Drop setup-phase pin candidates that are mirror images of another candidate.

        Returns (kept indices, [(rejected index, detail)]). Also fits the floor line.
        """
        if not self.enabled:
            return list(range(len(boxes))), []
        rejected: list[tuple[int, str]] = []
        for i, bi in enumerate(boxes):
            for j, bj in enumerate(boxes):
                if i != j and is_mirror_of(bi, bj, self.cfg):
                    rejected.append((i, f"mirror of candidate {j}"))
                    break
        rej_idx = {i for i, _ in rejected}
        kept = [i for i in range(len(boxes)) if i not in rej_idx]
        self.floor, msg = FloorLine.fit([boxes[i] for i in kept], self.cfg)
        log.info("Floor line: %s", msg if self.floor else f"disabled - {msg}")
        return kept, rejected

    def refresh_floor(self, pin_boxes: list[Box]) -> None:
        """Re-fit the floor line to the pins' current reference boxes (camera drift).

        Only when the setup-phase fit found a row; that decision is not revisited.
        """
        if self.floor is not None and pin_boxes:
            line, _ = FloorLine.fit(pin_boxes, replace(self.cfg, floor_row_max_residual_frac=float("inf")))
            if line is not None:
                self.floor = line

    # ── Game phase ──────────────────────────────────────────
    def check(self, box: Box, standing_now: dict[int, Box]) -> Optional[tuple[str, str]]:
        """Return (reason, detail) if the detection should be rejected, else None.

        ``standing_now`` maps pin id → box of the standing detection matched to
        that pin in the current frame.
        """
        if not self.enabled:
            return None
        for pid, pbox in standing_now.items():
            if is_mirror_of(box, pbox, self.cfg):
                return MIRROR, f"under standing pin {pid}"
        if self.floor is not None:
            cx, cy = center(box)
            limit = self.floor.y_at(cx) + self.cfg.floor_margin_frac * self.floor.pin_size
            if cy > limit:
                return BELOW_FLOOR, f"center y={cy:.0f} > floor+margin {limit:.0f}"
        return None

    def reject(self, frame: int, time_s: float, label: str, conf: float, box: Box,
               reason: str, detail: str) -> None:
        r = Rejection(frame, time_s, label, conf, box, reason, detail)
        self.rejections.append(r)
        log.debug("reject frame=%d t=%.2f %s conf=%.2f reason=%s (%s)",
                  frame, time_s, label, conf, reason, detail)

    def counts_by_reason(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in self.rejections:
            out[r.reason] = out.get(r.reason, 0) + 1
        return out

    def rejections_in_frame(self, frame: int) -> list[Rejection]:
        out = []
        for r in reversed(self.rejections):
            if r.frame != frame:
                break
            out.append(r)
        return out
