#!/usr/bin/env bash
# Build the Noetic image, compile the ROS workspace, and run offline checks.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_TAG="${NAVRL_NOETIC_IMAGE:-navrl-noetic:20260825}"
DOCKER=(sudo docker)
mkdir -p "${ROOT_DIR}/benchmark_results"
if ! "${DOCKER[@]}" image inspect "${IMAGE_TAG}" >/dev/null 2>&1; then
  "${DOCKER[@]}" build -f "${ROOT_DIR}/docker/Dockerfile.noetic" -t "${IMAGE_TAG}" "${ROOT_DIR}"
fi

"${DOCKER[@]}" run --rm \
  -v "${ROOT_DIR}:/workspace/NavRL-decoder-deployment:ro" \
  -v "${ROOT_DIR}/benchmark_results:/tmp/navrl_results" \
  "${IMAGE_TAG}" bash -lc '
    # ROS Noetic setup scripts reference optional variables, so nounset would
    # abort before the workspace is even configured.
    set -eo pipefail
    source /opt/ros/noetic/setup.bash
    python3 -m pip install --no-cache-dir --no-build-isolation "setuptools<66" "importlib-metadata>=4.8,<7" zipp
    rm -rf /tmp/navrl_third_party
    mkdir -p /tmp/navrl_third_party
    cp -a /workspace/NavRL-decoder-deployment/isaac-training/third_party/tensordict /tmp/navrl_third_party/
    cp -a /workspace/NavRL-decoder-deployment/isaac-training/third_party/rl /tmp/navrl_third_party/
    python3 -m pip install --no-build-isolation -e /tmp/navrl_third_party/tensordict
    python3 -m pip install --no-build-isolation -e /tmp/navrl_third_party/rl
    mkdir -p /tmp/navrl_ws/src
    cp -a /workspace/NavRL-decoder-deployment/ros1/. /tmp/navrl_ws/src/
    cd /tmp/navrl_ws
    catkin_make -j2
    source devel/setup.bash
    python3 src/navigation_runner/scripts/validate_deployment.py
    python3 src/onboard_detector/scripts/check_calibration.py \
      src/navigation_runner/cfg/detection/real/real_detector_px4.yaml
    python3 src/navigation_runner/scripts/deployment_acceptance.py \
      --project-root /workspace/NavRL-decoder-deployment \
      --out /tmp/navrl_results/deployment_acceptance.json
    roscore >/tmp/navrl_roscore.log 2>&1 &
    roscore_pid=$!
    cleanup_roscore() { kill "$roscore_pid" 2>/dev/null || true; }
    trap cleanup_roscore EXIT
    sleep 1
    python3 src/navigation_runner/scripts/px4_mavros_mock_test.py \
      --out /tmp/navrl_results/px4_mavros_mock_acceptance.json
    python3 src/navigation_runner/scripts/benchmark_policy.py \
      --checkpoint src/navigation_runner/scripts/ckpts/step3-80new.pt \
      --device cpu --warmup 20 --samples 100 \
      --out /tmp/navrl_results/policy_profile_cpu_noetic.json
    cat /tmp/navrl_results/policy_profile_cpu_noetic.json
  '
