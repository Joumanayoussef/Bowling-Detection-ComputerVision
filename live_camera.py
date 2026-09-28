"""Live webcam version of the analyzer (same modules as analyze_video.py).

Keys:  Q quit   R reset (re-lock pins)   S save screenshot
Timestamps use wall-clock time since the last reset, so fall times stay correct
even when inference is slower than the camera frame rate.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from bowling_cv.config import Config  # noqa: E402
from bowling_cv.detector import Detector  # noqa: E402
from bowling_cv.pipeline import Pipeline  # noqa: E402
from bowling_cv.render import summary_frame  # noqa: E402

log = logging.getLogger("live_camera")


class _ClockPipeline(Pipeline):
    """Pipeline whose frame index is derived from elapsed time at a nominal fps."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.t_start = time.perf_counter()
        self._last_idx = -1

    def process_now(self, frame):
        frame_idx = int(round((time.perf_counter() - self.t_start) * self.fps))
        frame_idx = max(frame_idx, self._last_idx + 1)  # indices must increase
        self._last_idx = frame_idx
        return self.process(frame, frame_idx)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--model", default="models/best.pt")
    p.add_argument("--conf", type=float, default=Config.conf)
    p.add_argument("--no-reflection-filter", action="store_true")
    p.add_argument("--fps", type=float, default=30.0, help="nominal rate used to express time in frames")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    logging.getLogger("ultralytics").setLevel(logging.WARNING)

    cfg = Config(conf=args.conf)
    if args.no_reflection_filter:
        cfg = cfg.without_reflection_filter()
    detector = Detector(args.model, cfg)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {args.camera}")

    window = "BowlingCV live (Q quit, R reset, S screenshot)"
    pipe = _ClockPipeline(detector, cfg, args.fps)
    shot = 0
    h, w = 480, 640
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        pipe.process_now(frame)
        cv2.imshow(window, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            pipe = _ClockPipeline(detector, cfg, args.fps)
            log.info("Reset: locking pins again")
        elif key == ord("s"):
            shot += 1
            name = f"screenshot_{shot}.jpg"
            cv2.imwrite(name, frame)
            log.info("Saved %s", name)
    cap.release()

    end = summary_frame(w, h, pipe.pins.events_by_time(), len(pipe.pins.pins),
                        len(pipe.reflection.rejections))
    cv2.imshow(window, end)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
