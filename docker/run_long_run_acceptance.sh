#!/usr/bin/env bash
# Twenty-minute wall-clock Gazebo + ROS + YOLO + PPO stability acceptance.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_TAG="${NAVRL_NOETIC_IMAGE:-navrl-noetic:20260825}"
RESULT_DIR="${NAVRL_LONG_RUN_RESULTS:-${ROOT_DIR}/benchmark_results/long_run}"
DURATION_S="${NAVRL_LONG_RUN_SECONDS:-1200}"
RESULT_DIR="$(realpath -m "${RESULT_DIR}")"
DOCKER=(sudo docker)
mkdir -p "${RESULT_DIR}"

if ! "${DOCKER[@]}" image inspect "${IMAGE_TAG}" >/dev/null 2>&1; then
  "${DOCKER[@]}" build -f "${ROOT_DIR}/docker/Dockerfile.noetic" -t "${IMAGE_TAG}" "${ROOT_DIR}"
fi

"${DOCKER[@]}" run --rm -e "NAVRL_LONG_RUN_SECONDS=${DURATION_S}" \
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
    sim_pid=""; nav_pid=""; monitor_pid=""; xvfb_pid=""
    stop_child() { test -z "$1" && return 0; kill -INT "$1" 2>/dev/null || true; sleep 1; kill -TERM "$1" 2>/dev/null || true; }
    cleanup() { stop_child "$monitor_pid"; stop_child "$nav_pid"; stop_child "$sim_pid"; test -z "$xvfb_pid" || kill "$xvfb_pid" 2>/dev/null || true; }
    trap cleanup EXIT
    Xvfb :99 -screen 0 1280x1024x24 -nolisten tcp >/tmp/navrl_long_xvfb.log 2>&1 & xvfb_pid=$!
    roslaunch uav_simulator start.launch gui:=false keyboard_control:=false world_name:=/tmp/navrl_ws/src/uav_simulator/worlds/validation/perception_hifi_person.world >/tmp/navrl_long_gazebo.log 2>&1 & sim_pid=$!
    for _ in $(seq 1 60); do rostopic info /camera/depth/image_raw 2>/dev/null | grep -q Publishers: && break; sleep 1; done
    roslaunch navigation_runner navrl_deployment.launch use_real_detector:=true use_px4:=false rviz:=false device:=cpu auto_takeoff:=false \
      perception_require_detection:=false perception_watchdog_required:=false metrics_resource_stride:=20 \
      metrics_log:=/tmp/navrl_results/long_run_navigation_metrics.jsonl run_label:=gazebo_yolo_ppo_soak \
      >/tmp/navrl_long_navigation.log 2>&1 & nav_pid=$!
    sleep 20
    python3 src/navigation_runner/scripts/long_run_monitor.py --duration "$NAVRL_LONG_RUN_SECONDS" --period 1 \
      --output /tmp/navrl_results/long_run_health_raw.json & monitor_pid=$!
    python3 src/navigation_runner/scripts/mission_patrol.py --duration "$NAVRL_LONG_RUN_SECONDS" --period 30
    wait "$monitor_pid"; monitor_pid=""
    stop_child "$nav_pid"; nav_pid=""
    python3 src/navigation_runner/scripts/long_run_analyze.py \
      --navigation /tmp/navrl_results/long_run_navigation_metrics.jsonl \
      --health /tmp/navrl_results/long_run_health_raw.json \
      --out-dir /tmp/navrl_results --minimum-duration "$NAVRL_LONG_RUN_SECONDS"
  '

echo "Long-run acceptance evidence: ${RESULT_DIR}"
