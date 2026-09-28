"""Synthetic-detection tests: one test per fall signal plus identity/once-only rules.

Scene: fps=10, setup 1 s (frames 0-9). Three pins 20x40 px standing on y=140,
at x = 100, 200, 300 (size = 40 px).
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bowling_cv.config import Config  # noqa: E402
from bowling_cv.detector import Detection  # noqa: E402
from bowling_cv.pin_tracker import (CAR_CONTACT, CHAIN, CLASS_TRANSITION, PROXIMITY,  # noqa: E402
                                    TIMEOUT, PinTracker)

FPS = 10
PINS = {1: (90, 100, 110, 140), 2: (190, 100, 210, 140), 3: (290, 100, 310, 140)}


def S(box):
    return Detection("standing pin", 0.9, box)


def F(box):
    return Detection("fallen pin", 0.9, box)


def car_at(box):
    return Detection("car", 0.9, box)


def lying(pin_box, dx=0, dy=0):
    """Fallen-pin box next to where the pin stood (lying sideways)."""
    x1, y1, x2, y2 = pin_box
    return (x1 - 10 + dx, y2 - 20 + dy, x2 + 10 + dx, y2 + dy)


def make(cfg=None):
    cfg = cfg or Config(confirm_frames=3)
    t = PinTracker(cfg, FPS)
    for f in range(10):  # setup: all pins standing
        t.update(f, [S(b) for b in PINS.values()], None)
    assert t.locked
    return t


def run(t, start, frames, dets_fn, car_fn=lambda f: None):
    for f in range(start, start + frames):
        car = car_fn(f)
        dets = dets_fn(f) + ([car_at(car)] if car else [])
        t.update(f, dets, car)
    return start + frames


def standing_except(*gone):
    return lambda f: [S(b) for pid, b in PINS.items() if pid not in gone]


# ── Identity ────────────────────────────────────────────────
def test_setup_locks_pins_left_to_right_and_merges_duplicates():
    t = PinTracker(Config(confirm_frames=3), FPS)
    for f in range(10):
        dets = [S(b) for b in PINS.values()]
        dets.append(S((92, 101, 111, 141)))  # duplicate of pin 1
        t.update(f, dets, None)
    assert sorted(t.pins) == [1, 2, 3]
    assert t.pins[1].ref_box[0] < t.pins[2].ref_box[0] < t.pins[3].ref_box[0]


def test_setup_starts_at_first_standing_detection():
    t = PinTracker(Config(confirm_frames=3), FPS)
    for f in range(5):  # black intro
        t.update(f, [], None)
    for f in range(5, 14):
        t.update(f, [S(b) for b in PINS.values()], None)
    assert not t.locked
    t.update(14, [S(b) for b in PINS.values()], None)
    assert t.locked and len(t.pins) == 3


def test_no_new_pin_ids_mid_video():
    t = make()
    run(t, 10, 30, lambda f: [S(b) for b in PINS.values()] + [S((500, 100, 520, 140))])
    assert sorted(t.pins) == [1, 2, 3]
    assert t.score == 0
    assert t.unmatched_detections == 30


def test_unstable_setup_candidate_is_not_locked():
    t = PinTracker(Config(confirm_frames=3), FPS)
    for f in range(10):
        dets = [S(b) for b in PINS.values()]
        if f == 4:
            dets.append(S((500, 100, 520, 140)))  # seen in 1/10 setup frames
        t.update(f, dets, None)
    assert sorted(t.pins) == [1, 2, 3]


# ── Signal 1: class transition ──────────────────────────────
def test_class_transition_needs_k_consecutive_frames():
    t = make()
    fallen = lambda f: standing_except(2)(f) + [F(lying(PINS[2]))]
    run(t, 10, 2, fallen)  # k-1 frames
    assert t.score == 0
    run(t, 12, 1, standing_except())  # back to standing: streak resets
    run(t, 13, 3, fallen)
    assert t.score == 1
    e = t.events[0]
    assert (e.pin_id, e.signal, e.frame) == (2, CLASS_TRANSITION, 13)
    assert e.time_s == pytest.approx(1.3)


# ── Signal 2: car contact ───────────────────────────────────
def test_car_contact():
    t = make()
    car = (200, 110, 260, 150)  # overlaps pin 2
    run(t, 10, 3, standing_except(), lambda f: car)
    assert [(e.pin_id, e.signal) for e in t.events] == [(2, CAR_CONTACT)]


def test_single_frame_car_contact_is_not_enough():
    t = make()
    run(t, 10, 1, standing_except(), lambda f: (200, 110, 260, 150))
    run(t, 11, 10, standing_except())
    assert t.score == 0


# ── Signal 3: proximity + disappearance ─────────────────────
def test_proximity_then_disappearance():
    cfg = Config(confirm_frames=3)
    t = make(cfg)
    near_car = (225, 100, 265, 140)  # 15 px from pin 2: near, but no padded contact (pad 6 px)
    f = run(t, 10, 3, standing_except(), lambda f: near_car)
    run(t, f, cfg.missing_frames, standing_except(2))
    assert [(e.pin_id, e.signal, e.frame) for e in t.events] == [(2, PROXIMITY, f)]


def test_disappearance_without_car_is_not_a_fall():
    t = make()
    run(t, 10, 15, standing_except(2))
    assert t.score == 0


def test_missing_while_car_covers_pin_is_not_counted():
    t = make(Config(confirm_frames=100))  # large k: isolate the missing counter from contact
    cover = (180, 90, 220, 150)  # covers pin 2 completely
    run(t, 10, 20, standing_except(2), lambda f: cover)
    assert t.pins[2].missing == 0
    assert t.score == 0


# ── Signal 4: chain reaction ────────────────────────────────
def test_chain_reaction():
    a, b = (90, 100, 110, 140), (125, 100, 145, 140)  # two neighbouring pins, 15 px apart
    t = PinTracker(Config(confirm_frames=3), FPS)
    for f in range(10):
        t.update(f, [S(a), S(b)], None)
    # Pin 1 falls to the right; its body lies across pin 2's base. The body is
    # closer to pin 2's center, but pin 2 is seen standing so it goes to pin 1.
    body = (100, 115, 140, 135)
    f = run(t, 10, 3, lambda f: [S(b), F(body)])
    assert [(e.pin_id, e.signal) for e in t.events] == [(1, CLASS_TRANSITION)]
    run(t, f, t.cfg.missing_frames, lambda f: [F(body)])  # pin 2 then vanishes
    assert [(e.pin_id, e.signal) for e in t.events] == [(1, CLASS_TRANSITION), (2, CHAIN)]


# ── Signal 5: timeout ───────────────────────────────────────
def test_timeout_after_car_passed():
    cfg = Config(confirm_frames=3)
    t = make(cfg)
    passing = (190, 220, 210, 240)  # 80 px below pin 2: passed (<4x size) but not near (>1.5x)
    f = run(t, 10, 3, standing_except(), lambda f: passing)
    frames_needed = int(cfg.timeout_seconds * FPS)
    run(t, f, frames_needed - 1, standing_except(2))
    assert t.score == 0
    run(t, f + frames_needed - 1, 1, standing_except(2))
    assert [(e.pin_id, e.signal, e.frame) for e in t.events] == [(2, TIMEOUT, f)]


# ── Once only ───────────────────────────────────────────────
def test_pin_falls_once_only():
    t = make()
    fallen = lambda f: standing_except(2)(f) + [F(lying(PINS[2]))]
    run(t, 10, 40, fallen, lambda f: (200, 110, 260, 150) if f % 2 else None)
    assert t.score == 1
    assert sum(1 for e in t.events if e.pin_id == 2) == 1


def test_events_by_time_renumbers_by_onset():
    t = make()
    t.events.clear()
    from bowling_cv.pin_tracker import FallEvent
    t.events += [FallEvent(1, 1, TIMEOUT, 30, 3.0, 50, 5.0), FallEvent(2, 2, CAR_CONTACT, 20, 2.0, 22, 2.2)]
    assert [(e.pin_id, e.order) for e in t.events_by_time()] == [(2, 1), (1, 2)]


# ── Drift following ─────────────────────────────────────────
def test_locked_box_follows_slow_camera_drift():
    t = make()
    for f in range(10, 110):  # the whole scene drifts 0.5 px/frame to the left: 50 px total
        dx = -(f - 9) * 0.5
        t.update(f, [S((b[0] + dx, b[1], b[2] + dx, b[3])) for b in PINS.values()], None)
    assert all(p.standing_seen for p in t.pins.values())
    assert t.pins[1].ref_box[0] < 90 - 40


def test_shape_change_does_not_drag_locked_box():
    t = make()
    wide = (80, 100, 125, 140)  # pin 1 tilting: box 45 px wide instead of 20
    run(t, 10, 20, lambda f: [S(wide)] + standing_except(1)(f))
    assert t.pins[1].ref_box == PINS[1]
