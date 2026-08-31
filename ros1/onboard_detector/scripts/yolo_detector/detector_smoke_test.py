#!/usr/bin/env python3
"""Load the bundled YOLO detector and run one CPU image inference."""

import argparse
import json
import time
from pathlib import Path

import cv2
import torch

from module.detector import Detector
from utils.tool import handle_preds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.65)
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    weight = base / "weights" / "weight_AP05_0.253207_280-epoch.pth"
    image_path = base / "test_data" / "0.jpg"
    if not weight.is_file() or not image_path.is_file():
        raise SystemExit("YOLO weight or test image is missing")
    image = cv2.imread(str(image_path))
    if image is None:
        raise SystemExit("OpenCV could not read {}".format(image_path))
    model = Detector(80, True).to("cpu")
    model.load_state_dict(torch.load(weight, map_location="cpu"))
    model.eval()
    tensor = torch.from_numpy(cv2.resize(image, (352, 352)).transpose(2, 0, 1)).unsqueeze(0).float() / 255.0
    with torch.no_grad():
        started = time.perf_counter()
        predictions = model(tensor)
        forward_ms = (time.perf_counter() - started) * 1000.0
        detections = handle_preds(predictions, "cpu", args.threshold)
    if not torch.isfinite(predictions).all():
        raise SystemExit("YOLO produced non-finite predictions")
    report = {
        "kind": "yolo_detector_smoke",
        "result": "passed",
        "device": "cpu",
        "forward_wall_ms": round(forward_ms, 4),
        "batch_detections": int(len(detections[0])),
        "input_shape": list(tensor.shape),
        "weight": str(weight),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
