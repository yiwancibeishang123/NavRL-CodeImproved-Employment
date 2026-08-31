#!/usr/bin/env bash
# Validate recovery-aware visual protection and the independent LiDAR escape path.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_TAG="${NAVRL_NOETIC_IMAGE:-navrl-noetic:20260825}"
RESULT_DIR="${NAVRL_SAFETY_RESULTS:-${ROOT_DIR}/benchmark_results/perception_safety}"
RESULT_DIR="$(realpath -m "${RESULT_DIR}")"
DOCKER=(sudo docker)
mkdir -p "${RESULT_DIR}"

if ! "${DOCKER[@]}" image inspect "${IMAGE_TAG}" >/dev/null 2>&1; then
  "${DOCKER[@]}" build -f "${ROOT_DIR}/docker/Dockerfile.noetic" -t "${IMAGE_TAG}" "${ROOT_DIR}"
fi

"${DOCKER[@]}" run --rm \
  -v "${ROOT_DIR}:/workspace/NavRL-decoder-deployment:ro" \
  -v "${RESULT_DIR}:/tmp/navrl_results" \
  "${IMAGE_TAG}" bash -lc '
    set -eo pipefail
    set +u; source /opt/ros/noetic/setup.bash; set -u
    python3 -m pip install --no-cache-dir --no-build-isolation "setuptools<66" "importlib-metadata>=4.8,<7" zipp
    mkdir -p /tmp/navrl_third_party /tmp/navrl_ws/src
    cp -a /workspace/NavRL-decoder-deployment/isaac-training/third_party/tensordict /tmp/navrl_third_party/
    cp -a /workspace/NavRL-decoder-deployment/isaac-training/third_party/rl /tmp/navrl_third_party/
    python3 -m pip install --no-build-isolation -e /tmp/navrl_third_party/tensordict
    python3 -m pip install --no-build-isolation -e /tmp/navrl_third_party/rl
    cp -a /workspace/NavRL-decoder-deployment/ros1/. /tmp/navrl_ws/src/
    cd /tmp/navrl_ws && catkin_make -j2 && source devel/setup.bash
    export DISPLAY=:99 LIBGL_ALWAYS_SOFTWARE=1
    export GAZEBO_MODEL_PATH="/tmp/navrl_ws/src/uav_simulator/models:${GAZEBO_MODEL_PATH:-}"
    sim_pid=""; nav_pid=""; challenge_pid=""; xvfb_pid=""
    stop_child() { test -z "$1" && return 0; kill -INT "$1" 2>/dev/null || true; sleep 1; kill -TERM "$1" 2>/dev/null || true; }
    cleanup() { stop_child "$nav_pid"; stop_child "$challenge_pid"; stop_child "$sim_pid"; test -z "$xvfb_pid" || kill "$xvfb_pid" 2>/dev/null || true; }
    trap cleanup EXIT
    Xvfb :99 -screen 0 1280x1024x24 -nolisten tcp >/tmp/navrl_safety_xvfb.log 2>&1 & xvfb_pid=$!
    roslaunch uav_simulator start.launch gui:=false keyboard_control:=false world_name:=/tmp/navrl_ws/src/uav_simulator/worlds/validation/perception_hifi_person.world >/tmp/navrl_safety_gazebo.log 2>&1 & sim_pid=$!
    for _ in $(seq 1 60); do rostopic info /camera/depth/image_raw 2>/dev/null | grep -q Publishers: && break; sleep 1; done
    python3 src/onboard_detector/scripts/perception_challenge_node.py _mode:=combined _input_topic:=/camera/color/image_raw _output_topic:=/camera/challenge/color/image_raw >/tmp/navrl_safety_challenge.log 2>&1 & challenge_pid=$!
    roslaunch navigation_runner navrl_deployment.launch use_real_detector:=true use_px4:=false rviz:=false device:=cpu auto_takeoff:=false \
      color_topic:=/camera/challenge/color/image_raw perception_require_detection:=true perception_watchdog_required:=true \
      perception_lidar_fallback_enabled:=true metrics_log:=/tmp/navrl_results/lidar_fallback_metrics.jsonl run_label:=lidar_fallback_combined \
      >/tmp/navrl_safety_navigation.log 2>&1 & nav_pid=$!
    for _ in $(seq 1 60); do rosservice list | grep -qx /gazebo/set_model_state && rostopic info /yolo_detector/perception_degraded 2>/dev/null | grep -q Publishers: && break; sleep 1; done
    python3 src/navigation_runner/scripts/lidar_fallback_eval.py --output /tmp/navrl_results/lidar_fallback_acceptance.json
  '

echo "Perception safety evidence: ${RESULT_DIR}"
