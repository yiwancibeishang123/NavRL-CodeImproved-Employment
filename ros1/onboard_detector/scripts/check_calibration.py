#!/usr/bin/env python3
"""Validate camera intrinsics/extrinsics before loading ROS parameters."""
import argparse
import math
from pathlib import Path

import yaml


def _matrix(values, key):
    if not isinstance(values, list) or len(values) != 16:
        raise ValueError("{} must contain 16 values".format(key))
    matrix = [float(value) for value in values]
    if not all(math.isfinite(value) for value in matrix):
        raise ValueError("{} contains NaN/Inf".format(key))
    if any(abs(matrix[index]) > 1e-4 for index in (12, 13, 14)) or abs(matrix[15] - 1.0) > 1e-4:
        raise ValueError("{} must be a 4x4 homogeneous transform".format(key))
    return matrix


def _validate_rotation(matrix, key):
    rotation = [matrix[0:3], matrix[4:7], matrix[8:11]]
    for row in rotation:
        norm = math.sqrt(sum(value * value for value in row))
        if abs(norm - 1.0) > 1e-3:
            raise ValueError("{} rotation rows must have unit norm".format(key))
    for first, second in ((0, 1), (0, 2), (1, 2)):
        dot = sum(rotation[first][index] * rotation[second][index] for index in range(3))
        if abs(dot) > 1e-3:
            raise ValueError("{} rotation rows must be orthogonal".format(key))
    determinant = (
        rotation[0][0] * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1])
        - rotation[0][1] * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0])
        + rotation[0][2] * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0])
    )
    if abs(determinant - 1.0) > 1e-3:
        raise ValueError("{} rotation determinant must be +1".format(key))


def validate_calibration(data):
    """Validate intrinsic bounds and rigid camera-to-body transforms.

    This confirms mathematical consistency. It cannot prove that values came
    from an on-aircraft calibration procedure.
    """
    image_cols = int(data.get("image_cols", 0))
    image_rows = int(data.get("image_rows", 0))
    if image_cols <= 0 or image_rows <= 0:
        raise ValueError("image_cols and image_rows must be positive")
    for key in ("depth_intrinsics", "color_intrinsics"):
        values = data.get(key)
        if not isinstance(values, list) or len(values) != 4 or any(float(value) <= 0 for value in values):
            raise ValueError("{} must contain four positive values".format(key))
        _, _, cx, cy = (float(value) for value in values)
        if not (0.0 <= cx < image_cols and 0.0 <= cy < image_rows):
            raise ValueError("{} principal point must lie inside the image".format(key))
    for key in ("body_to_camera", "body_to_camera_color"):
        matrix = _matrix(data.get(key), key)
        _validate_rotation(matrix, key)
    if float(data.get("depth_scale_factor", 0)) <= 0:
        raise ValueError("depth_scale_factor must be positive")
    return {
        "image_size": [image_cols, image_rows],
        "depth_scale_factor": float(data["depth_scale_factor"]),
    }


def load_and_validate(path):
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("calibration YAML must contain a mapping")
    return validate_calibration(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    try:
        load_and_validate(args.config)
    except (TypeError, ValueError) as exc:
        raise SystemExit(str(exc))
    print("camera calibration format passed:", args.config)


if __name__ == "__main__":
    main()
