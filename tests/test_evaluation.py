import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bowling_cv.evaluation import score  # noqa: E402

GT = {"pins": [{"name": "a", "xy": [100, 120], "fall_s": 2.0},
               {"name": "b", "xy": [200, 120], "fall_s": None},
               {"name": "c", "xy": [300, 120], "fall_s": 5.0}]}
REPORT = {
    "pins_locked": [{"pin_id": 1, "box": [90, 100, 110, 140]}, {"pin_id": 2, "box": [190, 100, 210, 140]}],
    "events": [{"pin_id": 1, "time_s": 2.4, "signal": "car_contact"},
               {"pin_id": 2, "time_s": 3.0, "signal": "timeout"}],
}


def test_score_counts_correct_false_missed():
    s = score(REPORT, GT, tolerance_s=1.0)
    assert (s["true_falls"], s["detected"], s["correct"], s["false"], s["missed"]) == (2, 2, 1, 1, 1)
    assert s["never_locked"] == ["c"]


def test_wrong_time_is_false():
    s = score(REPORT, GT, tolerance_s=0.2)
    assert s["correct"] == 0 and s["false"] == 2
