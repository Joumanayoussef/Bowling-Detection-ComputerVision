"""Drawing: detections, locked pins, car path, HUD and the final summary screen."""
from __future__ import annotations

import math
from typing import Iterable, Optional

import cv2
import numpy as np

from .car_tracker import CarTracker
from .detector import Detection
from .pin_tracker import FallEvent, PinTracker
from .reflection import Rejection

FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (255, 255, 255)
GREEN = (80, 200, 80)
CAR_BLUE = (255, 100, 50)
BALL_ORANGE = (0, 165, 255)
PATH_CYAN = (212, 188, 0)
REJECT_MAGENTA = (255, 0, 255)
GREY = (170, 170, 170)

SIGNAL_TEXT = {
    "class_transition": "class change",
    "car_contact": "car contact",
    "proximity_disappearance": "near + vanished",
    "chain_reaction": "chain reaction",
    "timeout": "timeout",
}


def fmt_time(t: float) -> str:
    """mm:ss.s"""
    m, s = divmod(max(t, 0.0), 60)
    return f"{int(m):02d}:{s:04.1f}"


def ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suf = "th"
    else:
        suf = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


def _scale(frame: np.ndarray) -> float:
    return max(0.45, min(frame.shape[:2]) / 600.0)


def _box(frame, b, color, thick):
    x1, y1, x2, y2 = map(int, b)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)


def _label(frame, text, org, color, s, bg=(0, 0, 0)):
    fs, th = 0.5 * s, max(1, int(round(1.5 * s)))
    (tw, tht), _ = cv2.getTextSize(text, FONT, fs, th)
    x, y = int(org[0]), int(max(org[1], tht + 4))
    cv2.rectangle(frame, (x, y - tht - 4), (x + tw + 4, y + 3), bg, -1)
    cv2.putText(frame, text, (x + 2, y), FONT, fs, color, th, cv2.LINE_AA)


def _dashed_box(frame, b, color, thick, dash=6):
    x1, y1, x2, y2 = map(int, b)
    for (xa, ya, xb, yb) in ((x1, y1, x2, y1), (x2, y1, x2, y2), (x2, y2, x1, y2), (x1, y2, x1, y1)):
        length = int(math.hypot(xb - xa, yb - ya))
        for s in range(0, length, dash * 2):
            e = min(s + dash, length)
            p = (int(xa + (xb - xa) * s / max(length, 1)), int(ya + (yb - ya) * s / max(length, 1)))
            q = (int(xa + (xb - xa) * e / max(length, 1)), int(ya + (yb - ya) * e / max(length, 1)))
            cv2.line(frame, p, q, color, thick)


def draw_scene(frame: np.ndarray, t: float, detections: list[Detection], pins: PinTracker,
               car: CarTracker, rejections: Iterable[Rejection] = ()) -> None:
    s = _scale(frame)
    th = max(1, int(round(2 * s)))
    cfg = pins.cfg

    # Car path (dotted) with start marker.
    path = car.path
    for a, b in zip(path, path[1:]):
        d = math.dist(a, b)
        for i in range(max(int(d / (14 * s)), 1)):
            u = i / max(int(d / (14 * s)), 1)
            cv2.circle(frame, (int(a[0] + u * (b[0] - a[0])), int(a[1] + u * (b[1] - a[1]))),
                       max(2, int(3 * s)), PATH_CYAN, -1)
    if path:
        cv2.circle(frame, path[0], int(7 * s), (71, 160, 67), -1)

    for d in detections:
        if d.label == cfg.ball_label:
            _box(frame, d.box, BALL_ORANGE, th)
            _label(frame, f"ball {d.conf:.2f}", (d.box[0], d.box[1] - 4), BALL_ORANGE, s)

    if car.box is not None:
        _box(frame, car.box, CAR_BLUE, th if car.detected else max(1, th - 1))
        _label(frame, "car" if car.detected else "car (pred)", (car.box[0], car.box[1] - 4), CAR_BLUE, s)

    for r in rejections:
        _dashed_box(frame, r.box, REJECT_MAGENTA, max(1, th - 1))
        text = "rejected (mirror rule)" if r.reason.startswith("mirror") else "rejected (below floor)"
        _label(frame, text, (r.box[0], r.box[3] + 16 * s), REJECT_MAGENTA, s * 0.8)

    if not pins.locked:
        for d in detections:
            if d.label == cfg.standing_label:
                _box(frame, d.box, GREY, max(1, th - 1))
        return

    fallen_by_pin = {e.pin_id: e for e in pins.events}
    for p in pins.pins.values():
        if p.fallen:
            e = fallen_by_pin[p.id]
            _box(frame, p.box, GREEN, th + 1)
            _label(frame, f"#{p.id} fell {e.time_s:.2f}s", (p.box[0], p.box[1] - 4), (0, 0, 0), s, bg=GREEN)
        else:
            color = WHITE if p.standing_seen else GREY
            _box(frame, p.ref_box, color, th)
            _label(frame, f"pin {p.id}", (p.ref_box[0], p.ref_box[1] - 4), color, s)


def draw_hud(frame: np.ndarray, t: float, pins: PinTracker, reflection_on: bool) -> None:
    s = _scale(frame)
    fs = 0.5 * s
    thick = max(1, int(round(1.5 * s)))
    line_h = int(24 * s)
    events = pins.events
    lines = [f"{ordinal(i)}  pin {e.pin_id}  {fmt_time(e.time_s)}  {SIGNAL_TEXT.get(e.signal, e.signal)}"
             for i, e in enumerate(sorted(events, key=lambda e: e.time_s), start=1)]
    w = int(max([300 * s] + [cv2.getTextSize(l, FONT, fs, thick)[0][0] + 20 for l in lines]))
    h = int(46 * s) + line_h * len(lines) + int(8 * s)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv2.putText(frame, f"Score: {pins.score}", (int(10 * s), int(34 * s)), FONT, 1.0 * s, GREEN,
                max(2, int(2 * s)), cv2.LINE_AA)
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (int(10 * s), int(46 * s) + line_h * (i + 1) - int(6 * s)), FONT, fs,
                    WHITE, thick, cv2.LINE_AA)

    # Clock + status (top right).
    status = "locking pins..." if not pins.locked else f"{len(pins.pins)} pins locked"
    status += "" if reflection_on else "  | reflection filter OFF"
    clock = fmt_time(t)
    for i, (text, scale) in enumerate(((clock, 0.8 * s), (status, 0.45 * s))):
        (tw, tht), _ = cv2.getTextSize(text, FONT, scale, thick)
        x = frame.shape[1] - tw - int(10 * s)
        y = int(30 * s) + i * int(24 * s)
        cv2.rectangle(frame, (x - 6, y - tht - 6), (x + tw + 6, y + 6), (0, 0, 0), -1)
        cv2.putText(frame, text, (x, y), FONT, scale, WHITE, thick, cv2.LINE_AA)


def summary_frame(width: int, height: int, events: list[FallEvent], n_pins: int,
                  rejected: Optional[int] = None) -> np.ndarray:
    """Black end screen: 'FINAL SCORE' and '1st pin fell at X.XX seconds' lines."""
    img = np.zeros((height, width, 3), np.uint8)
    s = max(0.5, min(width, height) / 600.0)
    thick = max(1, int(round(1.5 * s)))

    def centered(text, y, scale, color, t=thick):
        tw = cv2.getTextSize(text, FONT, scale, t)[0][0]
        cv2.putText(img, text, ((width - tw) // 2, int(y)), FONT, scale, color, t, cv2.LINE_AA)

    y = height * 0.12
    centered("FINAL SCORE", y, 1.4 * s, GREEN, max(2, int(2.5 * s)))
    y += 60 * s
    centered(f"Total pins knocked: {len(events)}", y, 0.8 * s, WHITE)
    y += 26 * s
    centered(f"({n_pins} pins tracked)", y, 0.45 * s, GREY)
    y += 22 * s
    cv2.line(img, (int(width * 0.12), int(y)), (int(width * 0.88), int(y)), GREY, 1)
    y += 40 * s
    for i, e in enumerate(sorted(events, key=lambda e: e.time_s), start=1):
        centered(f"{ordinal(i)} pin fell at {e.time_s:.2f} seconds", y, 0.65 * s, (200, 255, 200))
        y += 22 * s
        centered(f"(pin {e.pin_id}, {SIGNAL_TEXT.get(e.signal, e.signal)})", y, 0.45 * s, GREY)
        y += 34 * s
    if not events:
        centered("No falls detected", y, 0.65 * s, GREY)
    if rejected is not None:
        centered(f"Detections rejected by reflection filter: {rejected}", height - 30 * s, 0.45 * s, GREY)
    return img
