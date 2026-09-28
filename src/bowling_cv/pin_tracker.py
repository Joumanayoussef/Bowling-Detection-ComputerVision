"""Pin identity and fall detection.

Pins get fixed IDs during a short setup phase and are never created afterwards,
so a detector that drops and re-finds a pin cannot make one pin count twice
(the bug in the old tracker-ID based script).

Setup phase (``setup_seconds`` from the first frame with a standing pin)
    Collect "standing pin" detections, cluster them by center distance, keep
    clusters seen in enough setup frames, merge duplicates, drop mirror images
    (reflection filter) and assign IDs left to right.

Tracking phase
    Each frame, standing detections are matched to the nearest locked pin
    (greedy, by center distance). Remaining pin detections go through the
    reflection filter, then fallen detections are matched to the nearest pin.
    Five signals can mark a pin fallen, once only:

    1. class_transition        matched "fallen pin" (and no standing match) for k frames
    2. car_contact             padded car box overlaps the pin for k frames
    3. proximity_disappearance standing pin missing N frames, car was near it recently
    4. chain_reaction          a fallen pin overlapped it recently, then it went
                               missing N frames or was seen fallen for k frames
    5. timeout                 standing pin missing > T seconds after the car passed

    If several are satisfied in the same frame the first in that order wins:
    car_contact, chain_reaction, class_transition, proximity_disappearance, timeout.
    The event time is the onset of the evidence (first frame of the streak).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from .config import Config
from .detector import Detection
from .geometry import (Box, area, center, center_dist, contains_point, edge_dist, intersection, median_box,
                       overlaps, pad, short_side, size)
from .reflection import ReflectionFilter

log = logging.getLogger(__name__)

CLASS_TRANSITION = "class_transition"
CAR_CONTACT = "car_contact"
PROXIMITY = "proximity_disappearance"
CHAIN = "chain_reaction"
TIMEOUT = "timeout"
SIGNAL_PRIORITY = (CAR_CONTACT, CHAIN, CLASS_TRANSITION, PROXIMITY, TIMEOUT)


@dataclass
class Pin:
    id: int
    ref_box: Box                 # locked position from the setup phase
    width: float                 # shorter side of the locked box
    length: float                # longer side of the locked box
    box: Box                     # latest box (standing or fallen detection)
    fallen: bool = False
    # evidence (frame indices / streak lengths)
    standing_seen: bool = False
    fallen_streak: int = 0
    fallen_onset: Optional[int] = None
    contact_streak: int = 0
    contact_onset: Optional[int] = None
    missing: int = 0
    missing_onset: Optional[int] = None
    last_car_near: Optional[int] = None
    car_ever_near: Optional[int] = None
    last_chain_contact: Optional[int] = None
    chain_source: Optional[int] = None
    last_box_frame: Optional[int] = None  # last frame a detection was matched to this pin


@dataclass
class FallEvent:
    pin_id: int
    order: int
    signal: str
    frame: int                 # onset frame of the evidence
    time_s: float
    confirm_frame: int         # frame in which the signal was confirmed
    confirm_time_s: float
    also_satisfied: list[str] = field(default_factory=list)
    detail: str = ""

    def to_dict(self) -> dict:
        return {
            "pin_id": self.pin_id,
            "order": self.order,
            "time_s": round(self.time_s, 3),
            "signal": self.signal,
            "confirm_time_s": round(self.confirm_time_s, 3),
            "also_satisfied": self.also_satisfied,
            "detail": self.detail,
        }


def _greedy_match(dets: list[Detection], targets: dict[int, tuple[Box, float]],
                  inside_frac: Optional[float] = None,
                  must_overlap: frozenset[int] | set[int] = frozenset()) -> dict[int, int]:
    """Assign each target at most one detection, closest pairs first.

    ``targets`` maps pin id → (reference box, max center distance). With
    ``inside_frac``, a detection that lies mostly inside the reference box also
    matches (a partly occluded pin yields a smaller box with a shifted center).
    Targets in ``must_overlap`` only accept detections overlapping their box.
    Returns pin id → index into ``dets``.
    """
    pairs = []
    for di, d in enumerate(dets):
        for pid, (ref, radius) in targets.items():
            if pid in must_overlap and not overlaps(d.box, ref):
                continue
            dist = center_dist(d.box, ref)
            inside = inside_frac is not None and area(d.box) > 0 \
                and intersection(d.box, ref) / area(d.box) >= inside_frac
            if dist <= radius or inside:
                pairs.append((dist, di, pid))
    pairs.sort()
    out: dict[int, int] = {}
    used: set[int] = set()
    for _, di, pid in pairs:
        if pid in out or di in used:
            continue
        out[pid] = di
        used.add(di)
    return out


class PinTracker:
    def __init__(self, cfg: Config, fps: float, reflection: Optional[ReflectionFilter] = None):
        self.cfg = cfg
        self.fps = fps
        self.reflection = reflection or ReflectionFilter(cfg)
        self.locked = False
        self.pins: dict[int, Pin] = {}
        self.events: list[FallEvent] = []
        self.unmatched_detections = 0
        self._setup_boxes: list[tuple[int, Box]] = []
        self._setup_frames = 0
        self._setup_start: Optional[int] = None

    # ── Public API ──────────────────────────────────────────
    @property
    def score(self) -> int:
        return len(self.events)

    def events_by_time(self) -> list[FallEvent]:
        """Events sorted by onset time and renumbered (a slow signal such as
        timeout can confirm a fall whose onset precedes an earlier confirmation)."""
        out = sorted(self.events, key=lambda e: (e.time_s, e.pin_id))
        for i, e in enumerate(out, start=1):
            e.order = i
        return out

    def update(self, frame: int, detections: list[Detection], car_box: Optional[Box]) -> list[FallEvent]:
        """Process one frame. Returns the falls confirmed in this frame."""
        t = frame / self.fps
        if not self.locked:
            self._collect_setup(frame, t, detections)
            return []
        return self._track(frame, t, detections, car_box)

    # ── Setup ───────────────────────────────────────────────
    def _collect_setup(self, frame: int, t: float, detections: list[Detection]) -> None:
        standing = [d.box for d in detections if d.label == self.cfg.standing_label]
        if self._setup_start is None:
            if not standing:
                return  # e.g. black intro frames: setup starts at the first standing pin
            self._setup_start = frame
            log.info("Setup starts at %.2fs (first standing-pin detection)", t)
        self._setup_frames += 1
        self._setup_boxes.extend((frame, b) for b in standing)
        elapsed = (frame + 1 - self._setup_start) / self.fps  # setup time covered incl. this frame
        if elapsed + 1e-9 >= self.cfg.setup_seconds:
            self._lock(frame, t, force=elapsed + 1e-9 >= self.cfg.max_setup_seconds)

    def _lock(self, frame: int, t: float, force: bool) -> None:
        cfg = self.cfg
        clusters: list[dict] = []  # {"boxes": [...], "frames": set(), "cx", "cy"}
        for f, box in self._setup_boxes:
            cx, cy = center(box)
            best, best_d = None, None
            for c in clusters:
                d = ((c["cx"] - cx) ** 2 + (c["cy"] - cy) ** 2) ** 0.5
                if d <= cfg.merge_dist_frac * short_side(box) and (best_d is None or d < best_d):
                    best, best_d = c, d
            if best is None:
                clusters.append({"boxes": [box], "frames": {f}, "cx": cx, "cy": cy})
            else:
                best["boxes"].append(box)
                best["frames"].add(f)
                n = len(best["boxes"])
                best["cx"] += (cx - best["cx"]) / n
                best["cy"] += (cy - best["cy"]) / n

        min_frames = max(1, int(round(cfg.setup_min_presence * self._setup_frames)))
        stable = [c for c in clusters if len(c["frames"]) >= min_frames]
        for c in clusters:
            if c not in stable:
                log.info("Setup: dropped unstable pin candidate at (%.0f, %.0f), seen in %d/%d frames",
                         c["cx"], c["cy"], len(c["frames"]), self._setup_frames)

        # Merge duplicates whose median boxes are still close.
        merged: list[dict] = []
        for c in sorted(stable, key=lambda c: -len(c["frames"])):
            mb = median_box(c["boxes"])
            dup = next((m for m in merged if center_dist(m["box"], mb) <= cfg.merge_dist_frac * short_side(mb)), None)
            if dup is not None:
                log.info("Setup: merged duplicate pin candidate at (%.0f, %.0f)", *center(mb))
                continue
            merged.append({"box": mb, "frames": len(c["frames"])})

        boxes = [m["box"] for m in merged]
        kept, rejected = self.reflection.filter_locked_candidates(boxes)
        for i, detail in rejected:
            self.reflection.reject(frame, t, cfg.standing_label, 0.0, boxes[i], "mirror_at_lock", detail)
            log.info("Setup: rejected pin candidate at (%.0f, %.0f) as reflection (%s)", *center(boxes[i]), detail)

        if not kept and not force:
            return  # keep collecting until max_setup_seconds
        for new_id, i in enumerate(sorted(kept, key=lambda i: center(boxes[i])[0]), start=1):
            b = boxes[i]
            self.pins[new_id] = Pin(id=new_id, ref_box=b, width=short_side(b), length=size(b), box=b)
            log.info("Setup: locked pin %d at (%.0f, %.0f) %.0fx%.0fpx", new_id, *center(b), short_side(b), size(b))
        self.locked = True
        if not self.pins:
            log.warning("Setup ended at %.2fs without any locked pin; no falls can be detected", t)
        else:
            log.info("Setup done at %.2fs: %d pins locked from %d frames", t, len(self.pins), self._setup_frames)

    # ── Tracking ────────────────────────────────────────────
    def _track(self, frame: int, t: float, detections: list[Detection],
               car_box: Optional[Box]) -> list[FallEvent]:
        cfg = self.cfg
        standing = [d for d in detections if d.label == cfg.standing_label]
        fallen = [d for d in detections if d.label == cfg.fallen_label]

        # 1. Standing detections → standing pins at their locked position.
        targets = {p.id: (p.ref_box, cfg.match_standing_frac * p.width)
                   for p in self.pins.values() if not p.fallen}
        st_match = _greedy_match(standing, targets, inside_frac=cfg.match_standing_inside)
        standing_now = {pid: standing[di].box for pid, di in st_match.items()}
        used_st = set(st_match.values())

        # 2. Reflection filter on every other pin detection.
        leftovers = [d for i, d in enumerate(standing) if i not in used_st]
        fallen_ok: list[Detection] = []
        for d in leftovers + fallen:
            verdict = self.reflection.check(d.box, standing_now)
            if verdict is not None:
                self.reflection.reject(frame, t, d.label, d.conf, d.box, *verdict)
            elif d.label == cfg.fallen_label:
                fallen_ok.append(d)
            else:
                self.unmatched_detections += 1

        # 3. Fallen detections → nearest pin not seen standing in this frame.
        #    A pin that is still standing must be overlapped (it falls across its
        #    own base); already-fallen pins absorb detections of their body.
        targets = {p.id: (p.box if p.fallen else p.ref_box, cfg.match_fallen_frac * p.length)
                   for p in self.pins.values() if p.id not in standing_now}
        must_overlap = {p.id for p in self.pins.values() if not p.fallen}
        fa_match = _greedy_match(fallen_ok, targets, must_overlap=must_overlap)
        self.unmatched_detections += len(fallen_ok) - len(fa_match)

        for p in self.pins.values():
            if p.fallen:
                if p.id in fa_match:
                    p.box = fallen_ok[fa_match[p.id]].box
                    p.last_box_frame = frame
                continue
            self._update_evidence(p, frame, standing_now.get(p.id),
                                  fallen_ok[fa_match[p.id]].box if p.id in fa_match else None, car_box)

        # 4. Evaluate signals.
        new_events = []
        candidates = []
        for p in self.pins.values():
            if p.fallen:
                continue
            sat = self.satisfied_signals(p, frame)
            if sat:
                candidates.append((min(o for _, o, _ in sat), p, sat))
        for _, p, sat in sorted(candidates, key=lambda c: (c[0], c[1].id)):
            sat.sort(key=lambda s: SIGNAL_PRIORITY.index(s[0]))
            signal, onset, detail = sat[0]
            new_events.append(self._mark_fallen(p, signal, onset, frame, [s for s, _, _ in sat[1:]], detail))
        return new_events

    def _update_evidence(self, p: Pin, frame: int, st_box: Optional[Box], fa_box: Optional[Box],
                         car_box: Optional[Box]) -> None:
        cfg = self.cfg
        p.standing_seen = st_box is not None
        if st_box is not None:
            p.box = st_box
            p.last_box_frame = frame

        # Signal 1 evidence: seen as fallen and not as standing.
        if fa_box is not None and st_box is None:
            if p.fallen_streak == 0:
                p.fallen_onset = frame
                log.debug("pin %d: fallen-class streak starts at frame %d", p.id, frame)
            p.fallen_streak += 1
            p.box = fa_box
            p.last_box_frame = frame
        else:
            p.fallen_streak = 0

        # Signals 2, 3, 5 evidence: car contact / proximity.
        if car_box is not None and overlaps(pad(car_box, cfg.car_pad_frac * p.length), p.ref_box):
            if p.contact_streak == 0:
                p.contact_onset = frame
                log.debug("pin %d: car contact starts at frame %d", p.id, frame)
            p.contact_streak += 1
        else:
            p.contact_streak = 0
        if car_box is not None:
            d = edge_dist(car_box, p.ref_box)
            if d <= cfg.near_dist_frac * p.length:
                p.last_car_near = frame
            if d <= cfg.timeout_near_frac * p.length:
                p.car_ever_near = frame

        # Missing standing detection (not counted while the car covers the pin).
        if st_box is not None:
            p.missing = 0
            p.missing_onset = None
        elif car_box is not None and contains_point(car_box, center(p.ref_box)):
            pass  # occluded by the car: neither evidence of standing nor of falling
        else:
            if p.missing == 0:
                p.missing_onset = frame
            p.missing += 1

        # Signal 4 evidence: a fallen pin currently overlaps this pin.
        for q in self.pins.values():
            if q.fallen and q.last_box_frame is not None and frame - q.last_box_frame <= 1 \
                    and overlaps(q.box, p.ref_box):
                p.last_chain_contact = frame
                p.chain_source = q.id

    def satisfied_signals(self, p: Pin, frame: int) -> list[tuple[str, int, str]]:
        """All fall signals currently satisfied for pin ``p``: (signal, onset frame, detail)."""
        cfg, fps = self.cfg, self.fps
        k, n = cfg.confirm_frames, cfg.missing_frames
        out = []
        missing_ok = p.missing >= n and p.missing_onset is not None
        fallen_ok = p.fallen_streak >= k and p.fallen_onset is not None

        if p.contact_streak >= k and p.contact_onset is not None:
            out.append((CAR_CONTACT, p.contact_onset, f"car overlapped pin for {p.contact_streak} frames"))
        if p.last_chain_contact is not None and frame - p.last_chain_contact <= cfg.chain_window_seconds * fps \
                and (missing_ok or fallen_ok):
            onsets = [o for ok, o in ((missing_ok, p.missing_onset), (fallen_ok, p.fallen_onset)) if ok]
            out.append((CHAIN, min(onsets), f"fallen pin {p.chain_source} overlapped it"))
        if fallen_ok:
            out.append((CLASS_TRANSITION, p.fallen_onset, f"detected as fallen for {p.fallen_streak} frames"))
        if missing_ok and p.last_car_near is not None \
                and p.missing_onset - p.last_car_near <= cfg.recent_car_seconds * fps:
            out.append((PROXIMITY, p.missing_onset, f"missing {p.missing} frames after car was near"))
        if p.missing >= cfg.timeout_seconds * fps and p.missing_onset is not None \
                and p.car_ever_near is not None:
            out.append((TIMEOUT, p.missing_onset, f"missing {p.missing / fps:.1f}s after car passed"))
        return out

    def _mark_fallen(self, p: Pin, signal: str, onset: int, frame: int,
                     also: list[str], detail: str) -> FallEvent:
        p.fallen = True
        ev = FallEvent(pin_id=p.id, order=len(self.events) + 1, signal=signal,
                       frame=onset, time_s=onset / self.fps,
                       confirm_frame=frame, confirm_time_s=frame / self.fps,
                       also_satisfied=also, detail=detail)
        self.events.append(ev)
        log.info("FALL #%d: pin %d at %.2fs via %s (%s; confirmed %.2fs%s)",
                 ev.order, p.id, ev.time_s, signal, detail, ev.confirm_time_s,
                 f"; also {', '.join(also)}" if also else "")
        return ev
