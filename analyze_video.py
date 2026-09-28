"""Analyze a bowling video: lock pins, detect falls, write an annotated video + JSON events.

Example:
    python analyze_video.py --video clip.mp4 --out outputs
    python analyze_video.py --video clip.mp4 --no-reflection-filter
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from bowling_cv.config import Config  # noqa: E402
from bowling_cv.detector import Detector  # noqa: E402
from bowling_cv.pipeline import Pipeline  # noqa: E402
from bowling_cv.render import ordinal, summary_frame  # noqa: E402

log = logging.getLogger("analyze_video")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True, help="input video file")
    p.add_argument("--model", default="models/best.pt", help="YOLO weights (default: models/best.pt)")
    p.add_argument("--conf", type=float, default=Config.conf, help=f"pin/ball confidence (default {Config.conf})")
    p.add_argument("--out", default="outputs", help="output directory (default: outputs)")
    p.add_argument("--show", action="store_true", help="show frames while processing (Q to stop)")
    p.add_argument("--no-reflection-filter", action="store_true",
                   help="disable the spatial reflection filter and temporal persistence (k=1)")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    return p.parse_args(argv)


def setup_logging(level: str, log_path: Path) -> None:
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(fmt)
    file = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file.setLevel(logging.DEBUG)
    file.setFormatter(fmt)
    root.addHandler(console)
    root.addHandler(file)
    for noisy in ("ultralytics", "PIL", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def analyze(video: Path, model: str, cfg: Config, out_dir: Path, show: bool = False) -> dict:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    log.info("Video %s: %dx%d @ %.3f fps, %d frames (%.2fs)", video.name, w, h, fps, total, total / fps)

    tag = "" if cfg.reflection_filter else "_nofilter"
    out_video = out_dir / f"{video.stem}{tag}_annotated.mp4"
    out_json = out_dir / f"{video.stem}{tag}_events.json"
    # Written at the source fps so playback speed matches the original.
    writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    pipe = Pipeline(Detector(model, cfg), cfg, fps)
    frame_idx = 0
    t0 = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        pipe.process(frame, frame_idx)
        writer.write(frame)
        if show:
            cv2.imshow("BowlingCV (Q to stop)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        frame_idx += 1
        if frame_idx % 50 == 0:
            log.debug("frame %d/%d", frame_idx, total)
    cap.release()

    report = pipe.report(video=str(video), frames=frame_idx, duration_s=round(frame_idx / fps, 3))
    end = summary_frame(w, h, pipe.pins.events_by_time(), len(pipe.pins.pins), report["rejected_reflection_count"])
    for _ in range(int(round(cfg.summary_seconds * fps))):
        writer.write(end)
    writer.release()
    if show:
        cv2.imshow("BowlingCV (Q to stop)", end)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()

    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info("Processed %d frames in %.1fs", frame_idx, time.perf_counter() - t0)
    log.info("Pins locked: %d | falls: %d | reflection rejections: %d %s",
             len(report["pins_locked"]), report["total_falls"], report["rejected_reflection_count"],
             report["rejected_by_reason"])
    for e in report["events"]:
        log.info("  %s pin fell at %.2f seconds (pin %d, %s)", ordinal(e["order"]), e["time_s"],
                 e["pin_id"], e["signal"])
    log.info("Wrote %s and %s", out_video, out_json)
    return report


def main(argv=None) -> None:
    args = parse_args(argv)
    cfg = Config(conf=args.conf)
    if args.no_reflection_filter:
        cfg = cfg.without_reflection_filter()
    video = Path(args.video.strip().strip('"'))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "" if cfg.reflection_filter else "_nofilter"
    setup_logging(args.log_level, out_dir / f"{video.stem}{tag}.log")
    analyze(video, args.model, cfg, out_dir, show=args.show)


if __name__ == "__main__":
    main()
