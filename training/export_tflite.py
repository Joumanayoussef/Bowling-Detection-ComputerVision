"""Export weights to float32 TFLite for the Android app.

    python training/export_tflite.py --weights models/best.pt

Writes exports/bowling_model.tflite, exports/labels.txt (class order from
model.names) and exports/metadata.json. Copy the first two into
android-app/app/src/main/assets/. Ultralytics installs the TensorFlow export
dependencies on first use.
"""
import argparse
import glob
import json
import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", default=str(ROOT / "models" / "best.pt"))
    p.add_argument("--imgsz", type=int, default=320)
    p.add_argument("--out", default=str(ROOT / "exports"))
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)
    names = [model.names[i] for i in sorted(model.names)]

    path = model.export(format="tflite", imgsz=args.imgsz, int8=False, nms=False)
    candidates = glob.glob(str(Path(path).parent / "**" / "*float32.tflite"), recursive=True) or [str(path)]
    dest = out / "bowling_model.tflite"
    shutil.copy(candidates[0], dest)
    (out / "labels.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    (out / "metadata.json").write_text(json.dumps(
        {"input_size": args.imgsz, "classes": names, "nc": len(names), "format": "tflite_float32"}, indent=2))
    print(f"Model : {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    print(f"Labels: {out / 'labels.txt'} -> {names}")
    print("Copy both into android-app/app/src/main/assets/")


if __name__ == "__main__":
    main()
