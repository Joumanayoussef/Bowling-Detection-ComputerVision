"""Per-frame pipeline shared by analyze_video.py and live_camera.py."""
from __future__ import annotations

import numpy as np

from .car_tracker import CarTracker
from .config import Config
from .detector import Detection, Detector
from .pin_tracker import FallEvent, PinTracker
from .reflection import ReflectionFilter
from .render import draw_hud, draw_scene


class Pipeline:
    def __init__(self, detector: Detector, cfg: Config, fps: float):
        self.detector = detector
        self.cfg = cfg
        self.fps = fps
        self.reflection = ReflectionFilter(cfg)
        self.car = CarTracker(cfg)
        self.pins = PinTracker(cfg, fps, self.reflection)
        self.last_detections: list[Detection] = []

    def process(self, frame: np.ndarray, frame_idx: int, annotate: bool = True) -> list[FallEvent]:
        """Detect, track and (optionally) draw on ``frame`` in place."""
        dets = self.detector.detect(frame)
        self.last_detections = dets
        car_box = self.car.update(dets)
        events = self.pins.update(frame_idx, dets, car_box)
        if annotate:
            t = frame_idx / self.fps
            draw_scene(frame, t, dets, self.pins, self.car, self.reflection.rejections_in_frame(frame_idx))
            draw_hud(frame, t, self.pins, self.cfg.reflection_filter)
        return events

    def report(self, **meta) -> dict:
        events = self.pins.events_by_time()
        return {
            **meta,
            "fps": self.fps,
            "reflection_filter": self.cfg.reflection_filter,
            "confirm_frames": self.cfg.confirm_frames,
            "pins_locked": [
                {"pin_id": p.id, "box": [round(v, 1) for v in p.ref_box]} for p in self.pins.pins.values()
            ],
            "total_falls": len(events),
            "events": [e.to_dict() for e in events],
            "rejected_reflection_count": len(self.reflection.rejections),
            "rejected_by_reason": self.reflection.counts_by_reason(),
            "unmatched_pin_detections": self.pins.unmatched_detections,
            "rejected_detections": [r.to_dict() for r in self.reflection.rejections],
        }
