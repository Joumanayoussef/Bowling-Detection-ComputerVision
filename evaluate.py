"""Run every ground-truth video with and without the reflection filter and score it.

    python evaluate.py --videos-dir D:/bowling-cv-videos

Writes annotated videos + event JSON to --out, and eval/results.json +
eval/results.md (the table used in the README).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from analyze_video import analyze, setup_logging  # noqa: E402
from bowling_cv.config import Config  # noqa: E402
from bowling_cv.evaluation import score  # noqa: E402

ROOT = Path(__file__).resolve().parent


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--videos-dir", required=True)
    p.add_argument("--gt", default=str(ROOT / "eval" / "ground_truth.json"))
    p.add_argument("--model", default=str(ROOT / "models" / "best.pt"))
    p.add_argument("--out", default=str(ROOT / "outputs"))
    args = p.parse_args()

    gt = json.loads(Path(args.gt).read_text(encoding="utf-8"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for v in gt["videos"]:
        video = Path(args.videos_dir) / v["file"]
        row = {"file": v["file"], "view": v["view"]}
        for mode, cfg in (("filter", Config()), ("nofilter", Config().without_reflection_filter())):
            tag = "" if mode == "filter" else "_nofilter"
            setup_logging("WARNING", out / f"{video.stem}{tag}.log")
            report = analyze(video, args.model, cfg, out)
            row[mode] = score(report, v, gt["tolerance_s"])
            row[mode]["rejected_reflection_count"] = report["rejected_reflection_count"]
        results.append(row)
        f, n = row["filter"], row["nofilter"]
        print(f"{v['file']}: true {f['true_falls']} | filter: det {f['detected']} ok {f['correct']} "
              f"false {f['false']} | no filter: det {n['detected']} ok {n['correct']} false {n['false']}")

    (ROOT / "eval" / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    lines = [
        "| Video | Setup | True falls | Filter ON: detected / correct / **false** / missed | "
        "Filter OFF: detected / correct / **false** / missed | Spatial-filter rejections |",
        "|---|---|---|---|---|---|",
    ]
    tot = {k: 0 for k in ("t", "fd", "fc", "ff", "fm", "nd", "nc", "nf", "nm")}
    for r in results:
        f, n = r["filter"], r["nofilter"]
        lines.append(f"| `{r['file']}` | {r['view']} | {f['true_falls']} | "
                     f"{f['detected']} / {f['correct']} / **{f['false']}** / {f['missed']} | "
                     f"{n['detected']} / {n['correct']} / **{n['false']}** / {n['missed']} | "
                     f"{f['rejected_reflection_count']} |")
        for key, src, field in (("t", f, "true_falls"), ("fd", f, "detected"), ("fc", f, "correct"),
                                ("ff", f, "false"), ("fm", f, "missed"), ("nd", n, "detected"),
                                ("nc", n, "correct"), ("nf", n, "false"), ("nm", n, "missed")):
            tot[key] += src[field]
    lines.append(f"| **Total** | | {tot['t']} | {tot['fd']} / {tot['fc']} / **{tot['ff']}** / {tot['fm']} | "
                 f"{tot['nd']} / {tot['nc']} / **{tot['nf']}** / {tot['nm']} | |")
    lines.append("")
    lines.append("Per-event detail (filter ON):")
    lines.append("")
    lines.append("| Video | Pin | Detected (s) | True (s) | Signal | Verdict |")
    lines.append("|---|---|---|---|---|---|")
    for r in results:
        for e in r["filter"]["events"]:
            true_s = "-" if e["true_s"] is None else f"{e['true_s']:.2f}"
            lines.append(f"| `{r['file']}` | {e['gt_pin']} | {e['time_s']:.2f} | {true_s} | "
                         f"{e['signal']} | {e['verdict']} |")
    (ROOT / "eval" / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    logging.getLogger().handlers.clear()
    print("Wrote eval/results.json and eval/results.md")


if __name__ == "__main__":
    main()
