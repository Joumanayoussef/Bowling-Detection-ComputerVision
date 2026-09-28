"""Reflection filter: spatial rules, logging, and effect on false falls."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bowling_cv.config import Config  # noqa: E402
from bowling_cv.detector import Detection  # noqa: E402
from bowling_cv.pin_tracker import PinTracker  # noqa: E402
from bowling_cv.reflection import BELOW_FLOOR, MIRROR, FloorLine, is_mirror_of  # noqa: E402

FPS = 10
PINS = [(90, 100, 110, 140), (190, 100, 210, 140), (290, 100, 310, 140)]


def mirror(b, gap=1):
    """Reflection of pin b: same x-range, directly under its base."""
    h = b[3] - b[1]
    return (b[0], b[3] + gap, b[2], b[3] + gap + h)


# Short, wide "fallen pin" false detection on the shiny floor just under pin 2's base.
REFLECTION_BLOB = (185, 138, 215, 158)


def S(b):
    return Detection("standing pin", 0.9, b)


def F(b):
    return Detection("fallen pin", 0.6, b)


def test_is_mirror_of_geometry():
    cfg = Config()
    pin = PINS[0]
    assert is_mirror_of(mirror(pin), pin, cfg)
    assert not is_mirror_of(mirror(pin, gap=30), pin, cfg)                 # not touching
    assert not is_mirror_of((150, 141, 170, 181), pin, cfg)                # not aligned
    assert not is_mirror_of((90, 60, 110, 100), pin, cfg)                  # above, not below
    assert not is_mirror_of((60, 141, 140, 181), pin, cfg)                 # much wider


def test_floor_line_row_and_scatter():
    cfg = Config()
    line, _ = FloorLine.fit(PINS, cfg)
    assert line is not None and abs(line.y_at(200) - 140) < 1e-6
    scattered = [(90, 100, 110, 140), (190, 300, 210, 340), (290, 20, 310, 60)]
    line, msg = FloorLine.fit(scattered, cfg)
    assert line is None and "row" in msg


def _run_with_mirror_fallen(cfg):
    """Pins stand still the whole time; a 'fallen pin' reflection flickers under pin 2."""
    t = PinTracker(cfg, FPS)
    for f in range(10):
        t.update(f, [S(b) for b in PINS], None)
    for f in range(10, 40):
        dets = [S(b) for b in PINS]
        if f % 3 != 0:  # flickers: present 2 of every 3 frames
            dets.append(F(REFLECTION_BLOB))
        if f in (20, 21):  # the real pin detection drops for two frames
            dets = [d for d in dets if d.box != PINS[1]]
        t.update(f, dets, None)
    return t


def test_filter_rejects_mirror_and_prevents_false_fall():
    t = _run_with_mirror_fallen(Config())
    assert t.score == 0
    reasons = {r.reason for r in t.reflection.rejections}
    assert MIRROR in reasons
    assert all(r.detail for r in t.reflection.rejections)


def test_without_filter_the_mirror_causes_a_false_fall():
    t = _run_with_mirror_fallen(Config().without_reflection_filter())
    assert t.score == 1
    assert t.reflection.rejections == []


def test_mirror_is_not_locked_as_pin():
    t = PinTracker(Config(), FPS)
    for f in range(10):
        t.update(f, [S(b) for b in PINS] + [S(mirror(PINS[0]))], None)
    assert len(t.pins) == 3
    assert [r.reason for r in t.reflection.rejections] == ["mirror_at_lock"]

    t = PinTracker(Config().without_reflection_filter(), FPS)
    for f in range(10):
        t.update(f, [S(b) for b in PINS] + [S(mirror(PINS[0]))], None)
    assert len(t.pins) == 4


def test_below_floor_rejected_in_row_layout():
    t = PinTracker(Config(), FPS)
    for f in range(10):
        t.update(f, [S(b) for b in PINS], None)
    far_below = (180, 230, 220, 250)  # center 100 px under the floor line (margin 40 px)
    t.update(10, [S(b) for b in PINS] + [F(far_below)], None)
    assert [r.reason for r in t.reflection.rejections] == [BELOW_FLOOR]


def test_real_fallen_pin_below_its_base_is_kept_once_pin_is_gone():
    """The mirror rule needs the pin above to be standing in the same frame,
    so a genuinely fallen pin lying where its mirror would be is not rejected."""
    t = PinTracker(Config(), FPS)
    for f in range(10):
        t.update(f, [S(b) for b in PINS], None)
    body = (180, 125, 220, 145)  # pin 2 lying across its own base
    for f in range(10, 13):
        t.update(f, [S(PINS[0]), S(PINS[2]), F(body)], None)
    assert [e.pin_id for e in t.events] == [2]
    assert t.reflection.rejections == []
