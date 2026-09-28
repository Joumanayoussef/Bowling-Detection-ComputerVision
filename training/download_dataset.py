"""Download the Roboflow dataset (YOLOv8 format) into training/dataset/.

Needs ROBOFLOW_API_KEY in the environment or in a .env file (never commit it).
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from roboflow import Roboflow

load_dotenv()

API_KEY   = os.getenv("ROBOFLOW_API_KEY")
WORKSPACE = "joumanas-workspace"
PROJECT   = "bowling-cv-project-rqaik"
VERSION   = 4
TARGET    = Path(__file__).resolve().parent / "dataset"


def download():
    if not API_KEY:
        print("ERROR: ROBOFLOW_API_KEY is not set.")
        print("Create a .env file with: ROBOFLOW_API_KEY=your_key")
        print("Find it: roboflow.com → Settings → API Keys")
        sys.exit(1)

    print("Connecting to Roboflow...")
    rf      = Roboflow(api_key=API_KEY)
    project = rf.workspace(WORKSPACE).project(PROJECT)
    version = project.version(VERSION)

    print(f"Downloading version {VERSION} to {TARGET} ...")
    dataset = version.download("yolov8", location=str(TARGET))

    print(f"\nDone! Dataset saved to: {dataset.location}")
    print("Next: python training/train.py")
    return dataset.location


if __name__ == "__main__":
    download()
