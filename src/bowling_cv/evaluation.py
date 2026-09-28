"""Score detected falls against hand-annotated ground truth.

Locked pins are paired with ground-truth pins by position (greedy, nearest
first). A detected fall is *correct* when its pin really falls and the time
is within ``tolerance_s`` of the true time; every other detected fall is
*false*. True falls without a correct detection are *missed*.
"""
from __future__ import annotations

import math


def score(report: dict, gt_video: dict, tolerance_s: float) -> dict:
    locked = {p["pin_id"]: p["box"] for p in report["pins_locked"]}
    centers = {pid: ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for pid, b in locked.items()}
    gt = gt_video["pins"]

    pairs = sorted((math.dist(c, g["xy"]), pid, gi) for pid, c in centers.items() for gi, g in enumerate(gt))
    pin_to_gt: dict[int, int] = {}
    used: set[int] = set()
    for dist, pid, gi in pairs:
        if pid in pin_to_gt or gi in used:
            continue
        size = max(locked[pid][2] - locked[pid][0], locked[pid][3] - locked[pid][1])
        if dist <= size:
            pin_to_gt[pid] = gi
            used.add(gi)

    rows = []
    correct_gt = set()
    for e in report["events"]:
        gi = pin_to_gt.get(e["pin_id"])
        truth = gt[gi]["fall_s"] if gi is not None else None
        ok = truth is not None and abs(e["time_s"] - truth) <= tolerance_s
        if ok:
            correct_gt.add(gi)
        rows.append({
            "pin_id": e["pin_id"],
            "gt_pin": gt[gi]["name"] if gi is not None else None,
            "time_s": e["time_s"],
            "true_s": truth,
            "signal": e["signal"],
            "verdict": "correct" if ok else "false",
        })
    true_falls = [i for i, g in enumerate(gt) if g["fall_s"] is not None]
    errors = [abs(r["time_s"] - r["true_s"]) for r in rows if r["verdict"] == "correct"]
    return {
        "true_falls": len(true_falls),
        "detected": len(rows),
        "correct": sum(r["verdict"] == "correct" for r in rows),
        "false": sum(r["verdict"] == "false" for r in rows),
        "missed": len([i for i in true_falls if i not in correct_gt]),
        "never_locked": [gt[i]["name"] for i in range(len(gt)) if i not in used],
        "mean_abs_time_error_s": round(sum(errors) / len(errors), 2) if errors else None,
        "events": rows,
    }
