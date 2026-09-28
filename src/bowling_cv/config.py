"""All tunable thresholds in one place.

Distances ending in ``_frac`` are multiples of a locked pin's box, so the same
config works at any video resolution: "width" is the shorter side of the box,
"length" the longer side. Pins stand side by side, so separating neighbours
uses width; how far a falling pin or the car reaches uses length.
"""
from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass
class Config:
    # ── Detector ────────────────────────────────────────────
    imgsz: int = 320                  # the model was trained at 320
    conf: float = 0.30                # min confidence for pins / ball
    car_conf: float = 0.20            # min confidence for the car
    nms_iou: float = 0.5
    # Role → class name. The index of each class is read from model.names.
    standing_label: str = "standing pin"
    fallen_label: str = "fallen pin"
    car_label: str = "car"
    ball_label: str = "ball"

    # ── Setup phase: lock pin positions ─────────────────────
    setup_seconds: float = 1.0        # collect standing-pin detections this long
    max_setup_seconds: float = 3.0    # extend setup if no pin has been locked yet
    setup_min_presence: float = 0.3   # a pin must be seen in >=30% of setup frames
    merge_dist_frac: float = 0.5      # x width: detections closer than this are one pin

    # ── Matching detections to locked pins ──────────────────
    match_standing_frac: float = 0.5  # x width: a standing pin does not move
    match_standing_inside: float = 0.8  # ...or the detection lies >=80% inside the locked box
    match_fallen_frac: float = 0.75   # x length: a fallen pin lies away from where it stood
    ref_follow_alpha: float = 0.05    # locked box moves toward each matched standing detection...
    ref_follow_shape_tol: float = 0.2 # ...if its width and height are within 20% of the locked box

    # ── Fall signals (distances x length) ───────────────────
    confirm_frames: int = 8           # k: a signal must persist k consecutive frames (~0.27 s at 30 fps)
    car_pad_frac: float = 0.15        # car box padding for contact (signal 2)
    near_dist_frac: float = 1.5       # "car near the pin" edge distance (signal 3)
    recent_car_seconds: float = 1.0   # car near within this long before disappearance
    missing_frames: int = 8           # N: standing pin undetected this many frames
    chain_window_seconds: float = 1.0 # fallen neighbour touched the pin within this
    timeout_seconds: float = 2.0      # T: missing this long after the car passed (signal 5)
    timeout_near_frac: float = 4.0    # "car passed" = came within this distance

    # ── Reflective-floor filter ─────────────────────────────
    reflection_filter: bool = True
    mirror_min_below_frac: float = 0.6    # share of the detection below the pin's bottom edge
    mirror_min_x_overlap: float = 0.5     # horizontal overlap / narrower width
    mirror_max_gap_frac: float = 0.25     # vertical gap to the pin's bottom edge
    mirror_width_ratio: tuple[float, float] = (0.5, 2.0)
    floor_margin_frac: float = 1.0        # center this far below the floor line → reject
    floor_row_max_residual_frac: float = 0.5  # pins must form a row for the floor-line test

    # ── Car smoothing / path ────────────────────────────────
    car_ema_alpha: float = 0.45
    car_max_pred_frames: int = 2
    car_path_min_step_px: float = 6.0

    # ── Output ──────────────────────────────────────────────
    summary_seconds: float = 3.0

    def without_reflection_filter(self) -> "Config":
        """Spatial filter off and no temporal persistence (k = 1)."""
        return replace(self, reflection_filter=False, confirm_frames=1)
