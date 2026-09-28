"""Train YOLOv8n on the bowling dataset (settings of the bowling_v4 run).

    python training/train.py                       # uses training/dataset/data.yaml
    python training/train.py --data path/to/data.yaml --epochs 100
"""
import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(HERE / "dataset" / "data.yaml"))
    p.add_argument("--model", default="yolov8n.pt", help="pretrained starting weights")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=320)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--name", default="bowling_v4")
    args = p.parse_args()

    data = Path(args.data).resolve()
    if not data.exists():
        raise SystemExit(f"{data} not found. Run training/download_dataset.py first.")
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Data: {data} | epochs: {args.epochs} | device: {device}")

    YOLO(args.model).train(
        data=str(data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(HERE / "runs"),
        name=args.name,
        device=device,
        patience=25,
        # learning rate
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=5,
        # augmentation (Roboflow already applied flip/rotation/brightness/exposure/blur)
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
        fliplr=0.5,
        flipud=0.0,
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.15,
        degrees=10,
        translate=0.1,
        scale=0.5,
        # loss weights
        box=7.5,
        cls=0.5,
        dfl=1.5,
    )
    print(f"Best weights: {HERE / 'runs' / args.name / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
