"""Validate weights on the dataset's val split and print overall + per-class metrics.

Note: this dataset's data.yaml sets test = val (valid/images), so there is no
separate held-out test set; these numbers are validation numbers.

    python training/validate.py --weights models/best.pt
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", default=str(HERE.parent / "models" / "best.pt"))
    p.add_argument("--data", default=str(HERE / "dataset" / "data.yaml"))
    p.add_argument("--imgsz", type=int, default=320)
    args = p.parse_args()

    data = Path(args.data).resolve()
    if not data.exists():
        raise SystemExit(f"{data} not found. Run training/download_dataset.py first.")
    model = YOLO(args.weights)
    m = model.val(data=str(data), imgsz=args.imgsz, split="val", plots=False, verbose=False)

    print("\n" + "=" * 50)
    print(f"Weights: {args.weights}")
    print(f"  mAP50     : {m.box.map50:.3f}")
    print(f"  mAP50-95  : {m.box.map:.3f}")
    print(f"  Precision : {m.box.mp:.3f}")
    print(f"  Recall    : {m.box.mr:.3f}")
    print("Per-class AP50:")
    for i, cls_idx in enumerate(m.box.ap_class_index):
        print(f"  {model.names[int(cls_idx)]:15s} {m.box.ap50[i]:.3f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
