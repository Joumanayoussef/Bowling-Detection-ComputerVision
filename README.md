# BowlingCV

Detects which pins fall, and when, in videos of a toy bowling setup (an RC car or a ball knocking over plastic pins). A YOLOv8n detector finds pins, the car and the ball. A tracker locks each pin's identity at the start, and five fall signals decide when a pin has gone down. The output is an annotated video, a JSON event log and a final score screen. An Android app runs the same detector offline with TFLite.

![demo](docs/demo.gif)

*Low side view, hand-held camera: the car tips the yellow pin (car contact), then the red pin (car nearby, then the pin disappears). The clip ends on the summary screen.*

## Pipeline

```mermaid
flowchart LR
    A[Video frame] --> B[YOLOv8n detector<br/>ball / car / fallen pin / standing pin]
    B --> C{Setup phase?<br/>first ~1 s with pins}
    C -- yes --> D[Cluster standing-pin boxes<br/>merge duplicates, drop mirror images<br/>lock fixed pin IDs]
    C -- no --> E[Match standing detections<br/>to locked pins]
    E --> F[Reflection filter<br/>on remaining pin detections]
    F --> G[Match fallen detections<br/>to pins]
    B --> H[Car tracker<br/>EMA smoothing + path]
    G --> I[Five fall signals<br/>k-frame persistence]
    H --> I
    I --> J[Events: pin, time, signal]
    J --> K[Annotated video + HUD<br/>JSON + summary screen]
```

### Pin identity (the over-counting fix)

The old script used YOLO tracker IDs. When a detection dropped out, the pin came back with a new ID and was counted again; it showed a score of 3 on a two-pin video. Now:

- **Setup** (1 s from the first frame with a standing pin): standing-pin boxes are clustered by center distance. A pin must appear in at least 30% of setup frames. Duplicates are merged, and IDs are assigned left to right.
- **Afterwards** every detection is matched to the nearest *locked* pin. Standing detections match within half a pin width, or when they lie ≥80% inside the locked box (a partly occluded pin). **No pin IDs are created after setup.**
- A matched standing detection of the *same shape* pulls the locked box slowly toward it (EMA, α = 0.05). This follows a drifting hand-held camera without letting a tipping pin drag its anchor.
- A pin can be marked fallen **once**.

### The five fall signals

Each signal is implemented in [`pin_tracker.py`](src/bowling_cv/pin_tracker.py), logged when it fires, and covered by a unit test in [`tests/`](tests/).

| # | Signal (`signal` in JSON) | Fires when |
|---|---|---|
| 1 | `class_transition` | a "fallen pin" box overlapping the pin, with no standing detection of it, for k consecutive frames |
| 2 | `car_contact` | the padded car box overlaps the pin for k consecutive frames |
| 3 | `proximity_disappearance` | the standing pin is missing for N frames and the car was near it within the last 1 s |
| 4 | `chain_reaction` | an already-fallen pin overlapped it within the last 1 s, then it went missing for N frames or was seen fallen for k frames |
| 5 | `timeout` | the standing pin is missing for more than T = 2 s after the car came within 4 pin lengths |

- **Tie-breaking:** if several signals fire in the same frame, the first in the order 2, 4, 1, 3, 5 is recorded. The others are listed in `also_satisfied`.
- **Timing:** `time_s` is the *onset* of the evidence (the first frame of the streak). `confirm_time_s` is when the signal was confirmed, which is also when the fall appears on the HUD.
- **Occlusion:** while the car box covers a pin's center, that pin is not counted as missing.
- **Settings:** k = 8 frames and N = 8 frames. All thresholds live in [`config.py`](src/bowling_cv/config.py) and scale with pin size, so they work at any resolution.

### Reflective-floor filter

[`reflection.py`](src/bowling_cv/reflection.py) has two parts:

- **Spatial.** A pin detection is rejected as a *mirror image* if it:
  - lies mostly (≥60%) below the bottom edge of a pin that is detected standing in the same frame;
  - overlaps that pin horizontally (≥50%);
  - has a similar width;
  - touches or nearly touches the pin's base.

  Setup candidates that are mirrors of another candidate are never locked. When the locked pins form a row, a floor line is fitted through their bottom edges, and detections whose center is more than one pin length below it are rejected. Every rejection is logged with its reason and written to the JSON.
- **Temporal.** A fall signal must persist for k consecutive frames. Flickering detections fail this.

`--no-reflection-filter` turns off **both** parts (spatial rules off, k = 1).

## Results

### Detector (YOLOv8n, 320 px, run stopped after 81 of 100 epochs)

From [`training/results_bowling_v4.csv`](training/results_bowling_v4.csv), best epoch (58). The per-class values were reproduced with `training/validate.py`:

| | mAP50 | mAP50-95 | Precision | Recall |
|---|---|---|---|---|
| all classes | **0.666** | 0.457 | 0.699 | 0.628 |

| Class | AP50 | Val instances |
|---|---|---|
| fallen pin | 0.768 | 88 |
| standing pin | 0.695 | 149 |
| ball | 0.645 | 4 |
| car | 0.558 | 8 |

**Caveats:**

- The dataset's `data.yaml` sets `test` = `val`, so these are validation numbers. There is no separate held-out test set.
- The validation split is only 28 images, with 4 ball and 8 car instances, so the ball and car AP values are very noisy.

### Fall detection on real videos

Three raw videos were annotated by hand, frame by frame; see [`eval/ground_truth.json`](eval/ground_truth.json). The videos themselves are not in the repo. [`evaluate.py`](evaluate.py) runs the full pipeline with and without the filter.

A detected fall counts as **correct** if that pin really fell and the time is within ±1 s. Every other detection is **false**, and a true fall with no correct detection is **missed**.

| Video | Setup | True falls | Filter ON: detected / correct / **false** / missed | Filter OFF: detected / correct / **false** / missed | Spatial-filter rejections |
|---|---|---|---|---|---|
| `first_video.mp4` | top-down, RC car, 4 pins | 3 | 4 / 2 / **2** / 1 | 4 / 1 / **3** / 2 | 0 |
| `whatsapp_2026-05-13_2.53.32PM.mp4` | low side view, RC car, 2 pins, hand-held camera drifts | 2 | 2 / 2 / **0** / 0 | 2 / 2 / **0** / 0 | 6 |
| `videobowling.mp4` | side view, reflective floor, thrown ball (no car), 6 pins, 3 rolls | 6 | 5 / 5 / **0** / 1 | 5 / 4 / **1** / 2 | 71 |
| **Total** | | 11 | 11 / 9 / **2** / 2 | 11 / 7 / **4** / 4 | |

Per-event detail is in [`eval/results.md`](eval/results.md).

**What the filter actually did.** Turning the filter on cut false falls from 4 to 2. An ablation on the same cached detections shows that **all of that improvement comes from temporal persistence, not from the spatial reflection rules:**

| Configuration | Correct / false (first / whatsapp / videobowling) |
|---|---|
| spatial + temporal (default) | 2/2, 2/0, 5/0 |
| temporal only (k = 8, no spatial rules) | 2/2, 2/0, 5/0 |
| spatial only (k = 1) | 1/3, 2/0, 4/1 |
| neither | 1/3, 2/0, 4/1 |

I inspected the spatial rejections frame by frame. In these videos YOLO never detected an actual floor reflection as a pin, at conf 0.30. The rejected boxes were:

- the ball, mislabelled "standing pin" (most of the 71 in `videobowling.mp4`; see [screenshot](docs/rejected_ball_mirror_rule.jpg));
- a fallen pin lying in front of a standing one (already counted, so harmless here);
- the car, mislabelled "fallen pin" (the 6 in the WhatsApp video).

So **the spatial reflection filter did not reduce false falls on any of these videos.** It is implemented and unit-tested on synthetic mirror detections, but it has not been shown to help on real footage. It can also reject a real object lying directly in front of a standing pin.

The value k = 8 was chosen on these same three videos, so the "filter ON" numbers are tuned, not held-out.

**Failure case: top-down view.** ![top-down](docs/top_down_view.jpg)

The detector's standing/fallen classes do not transfer to a camera looking straight down. There, an upright pin is a round blob and a lying pin looks like a side-view silhouette. In `first_video.mp4` YOLO labels the upright blue pin "fallen" for over a second (false `class_transition` at 1.40 s; it really falls at 3.1 s). The car also brushes the yellow pin without knocking it over (false `car_contact` at 1.47 s).

Note: an earlier version of this project reported 4 falls at 1.50, 2.00, 2.57 and 3.47 s for this video. Frame-by-frame inspection shows 3 falls: green ~2.0 s, blue ~3.1 s, red ~5.7 s. The yellow pin is pushed but stays upright.

**Other observations.**

- `videobowling.mp4`: one of the two red pins is never detected separately from the other (their boxes overlap), so it is never locked and its fall is always missed.
- Chain-reaction onsets can be early by up to ~0.75 s. The "missing" streak starts as soon as the ball occludes a pin.

| Side view, reflective floor | Final summary screen |
|---|---|
| ![side view](docs/side_view_reflective_floor.jpg) | ![summary](docs/final_summary.jpg) |

## Output

`analyze_video.py` writes to `--out` (default `outputs/`):

- `<video>_annotated.mp4` at the **source frame rate**. It shows locked pins (white = seen standing, grey = not seen this frame), fallen pins (green, with fall time), the car box and its dotted path, and spatially rejected detections (dashed magenta). The HUD lists each fall with `mm:ss.s` and its signal. A final summary screen ("1st pin fell at X.XX seconds …") is appended for 3 s.
- `<video>_events.json` contains:
  - `events` (`pin_id`, `time_s`, `signal`, `confirm_time_s`, …);
  - `rejected_reflection_count`, `rejected_by_reason` and each rejected detection;
  - locked pin boxes.
- `<video>.log`, a full DEBUG log, including every rejection and signal streak.

## How to run

Tested with Python 3.11 on Windows, CPU only.

```bash
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

python analyze_video.py --video path/to/clip.mp4                 # outputs/clip_annotated.mp4 + clip_events.json
python analyze_video.py --video path/to/clip.mp4 --no-reflection-filter --show
python live_camera.py --camera 0                                 # webcam; Q quit, R re-lock pins, S screenshot
python evaluate.py --videos-dir path/to/videos                   # videos listed in eval/ground_truth.json
python -m pytest                                                 # 25 unit tests
```

`analyze_video.py` options: `--video`, `--conf` (pin/ball confidence, default 0.30), `--out`, `--show`, `--no-reflection-filter`, `--model` (default `models/best.pt`), `--log-level`.

On this laptop CPU the analyzer runs at roughly real time for 30 fps, ~500-px-wide video.

### Training

```bash
echo ROBOFLOW_API_KEY=your_key > .env        # never committed (.gitignore)
python training/download_dataset.py          # -> training/dataset/
python training/train.py                     # YOLOv8n, 320 px, bowling_v4 settings
python training/validate.py --weights models/best.pt
python training/export_tflite.py             # -> exports/bowling_model.tflite + labels.txt
```

`models/best.pt` (24.5 MB) is the `bowling_v4` checkpoint used for all results above.

## Android app

[`android-app/`](android-app/) is a Kotlin app (minSdk 26):

- **Detection:** `TFLiteDetector.kt` runs `bowling_model.tflite` **offline**. It takes 320×320 float32 input, decodes the YOLOv8 output and applies per-class NMS.
- **Live camera:** a **CameraX** `ImageAnalysis` pipeline (`CameraAnalyzer.kt`) draws an overlay.
- **Video files:** `VideoAnalyzer.kt` analyzes a picked video and shows the result.

Limitations of the app:

- It still uses its **original** Kotlin tracker (`PinTracker.kt`, `ScoreManager.kt`). The Python tracker, fall signals and reflection filter described above have **not** been ported.
- Only the Gradle build files and wrapper properties are included; there are no `gradlew` scripts or wrapper jar. Open it in Android Studio, or run `gradle wrapper` first.
- The app was not rebuilt or tested as part of this cleanup.

## Limitations

- **Small dataset.** 535 images in the Roboflow export, including 4× augmented copies of the training images. Validation has 28 images, with only 4 balls and 8 cars. Validation = test.
- **Toy setup.** Plastic pins, a small RC car or a light ball, indoor floors. Only one lane-like arrangement per video, and only three evaluation videos with 11 true falls. That is far too few to generalise from.
- **Detection errors drive most mistakes.** Examples: standing and fallen confused in top-down views, the ball labelled as a pin, overlapping pins merged into one box. The tracker cannot recover a pin that is never detected during setup.
- **Heuristics.** Car contact assumes a touched pin falls, which is false when the car only nudges it. Timing is the onset of the evidence and can lead the visible fall by up to ~0.75 s. Setup assumes the pins are standing and undisturbed in the first second in which they are detected.
- **Reflection filter.** Implemented, but it showed no measurable benefit on the available footage (see above).
- **Webcam script.** `live_camera.py` shares all modules with the video analyzer. Its clock-based pipeline was exercised on video frames, but it has not been run against a live camera.
