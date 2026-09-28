"""YOLO wrapper. Class names always come from ``model.names``."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Config
from .geometry import Box


def normalize_label(name: str) -> str:
    return str(name).strip().lower().replace("_", " ")


@dataclass(frozen=True)
class Detection:
    label: str      # normalized class name from model.names
    conf: float
    box: Box


class Detector:
    def __init__(self, weights: str, cfg: Config):
        from ultralytics import YOLO  # heavy import, keep it local

        self.cfg = cfg
        self.model = YOLO(weights)
        self.names: dict[int, str] = {int(i): normalize_label(n) for i, n in self.model.names.items()}
        required = {cfg.standing_label, cfg.fallen_label, cfg.car_label}
        missing = required - set(self.names.values())
        if missing:
            raise ValueError(
                f"Model classes {sorted(self.names.values())} do not include {sorted(missing)}; "
                "set the *_label fields in Config to match your model."
            )

    def detect(self, frame: np.ndarray) -> list[Detection]:
        cfg = self.cfg
        min_conf = min(cfg.conf, cfg.car_conf)
        result = self.model.predict(frame, imgsz=cfg.imgsz, conf=min_conf, iou=cfg.nms_iou,
                                    verbose=False)[0]
        dets: list[Detection] = []
        if result.boxes is None:
            return dets
        xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        clss = result.boxes.cls.cpu().numpy().astype(int)
        for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clss):
            label = self.names.get(int(k), str(k))
            threshold = cfg.car_conf if label == cfg.car_label else cfg.conf
            if c >= threshold:
                dets.append(Detection(label, float(c), (float(x1), float(y1), float(x2), float(y2))))
        return dets
