#!/usr/bin/env bash
# Fresh Gazebo RGB-D + YOLO + tracker + calibration acceptance run.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_TAG="${NAVRL_NOETIC_IMAGE:-navrl-noetic:20260825}"
RESULT_DIR="${NAVRL_PERCEPTION_RESULTS:-${ROOT_DIR}/benchmark_results/hifi_perception}"
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
    # ROS Noetic setup reads optional variables before assigning them.
    set +u
    source /opt/ros/noetic/setup.bash
    set -u
    python3 -m pip install --no-cache-dir --no-build-isolation "setuptools<66" "importlib-metadata>=4.8,<7" zipp
    mkdir -p /tmp/navrl_third_party /tmp/navrl_ws/src
    cp -a /workspace/NavRL-decoder-deployment/isaac-training/third_party/tensordict /tmp/navrl_third_party/
    cp -a /workspace/NavRL-decoder-deployment/isaac-training/third_party/rl /tmp/navrl_third_party/
    python3 -m pip install --no-build-isolation -e /tmp/navrl_third_party/tensordict
    python3 -m pip install --no-build-isolation -e /tmp/navrl_third_party/rl
    cp -a /workspace/NavRL-decoder-deployment/ros1/. /tmp/navrl_ws/src/
    cd /tmp/navrl_ws
    catkin_make -j2
    source devel/setup.bash
    export DISPLAY=:99 LIBGL_ALWAYS_SOFTWARE=1
    export GAZEBO_MODEL_PATH="/tmp/navrl_ws/src/uav_simulator/models:${GAZEBO_MODEL_PATH:-}"

    sim_pid=""; detector_pid=""; yolo_pid=""; challenge_pid=""; watchdog_pid=""; xvfb_pid=""
    stop_child() {
      child_pid="$1"
      test -z "$child_pid" && return 0
      kill -INT "$child_pid" 2>/dev/null || true
      for _ in $(seq 1 8); do
        kill -0 "$child_pid" 2>/dev/null || return 0
        sleep 1
      done
      kill -TERM "$child_pid" 2>/dev/null || true
      sleep 1
      kill -KILL "$child_pid" 2>/dev/null || true
    }
    cleanup() {
      stop_child "$yolo_pid"
      stop_child "$watchdog_pid"
      stop_child "$challenge_pid"
      stop_child "$detector_pid"
      stop_child "$sim_pid"
      test -z "$xvfb_pid" || kill "$xvfb_pid" 2>/dev/null || true
    }
    trap cleanup EXIT
    Xvfb :99 -screen 0 1280x1024x24 -nolisten tcp >/tmp/navrl_hifi_xvfb.log 2>&1 & xvfb_pid=$!
    start_chain() {
      local mode="$1"
      local image_topic="/camera/color/image_raw"
      if test "$mode" != clean; then
        image_topic="/camera/challenge/color/image_raw"
        python3 /tmp/navrl_ws/src/onboard_detector/scripts/perception_challenge_node.py \
          _mode:="$mode" _input_topic:=/camera/color/image_raw _output_topic:="$image_topic" \
          >/tmp/navrl_hifi_challenge_"$mode".log 2>&1 & challenge_pid=$!
      fi
      rosparam load /tmp/navrl_ws/src/onboard_detector/cfg/detector_param.yaml /onboard_detector
      rosparam set /onboard_detector/color_image_topic "$image_topic"
      rosrun onboard_detector detector_node >/tmp/navrl_hifi_detector_"$mode".log 2>&1 & detector_pid=$!
      python3 /tmp/navrl_ws/src/onboard_detector/scripts/yolo_detector/yolo_detector_node.py \
        _threshold:=0.35 _device:=cpu _image_topic:="$image_topic" \
        >/tmp/navrl_hifi_yolo_"$mode".log 2>&1 & yolo_pid=$!
      python3 /tmp/navrl_ws/src/onboard_detector/scripts/perception_watchdog.py \
        _require_detection:=$(test "$mode" = clean && echo false || echo true) \
        >/tmp/navrl_hifi_watchdog_"$mode".log 2>&1 & watchdog_pid=$!
      for _ in $(seq 1 40); do
        if rostopic info /yolo_detector/detected_bounding_boxes 2>/dev/null | grep -q "Publishers:" \
          && rosservice list | grep -qx /onboard_detector/get_dynamic_obstacles; then break; fi
        sleep 1
      done
    }
    stop_chain() {
      stop_child "$yolo_pid"; yolo_pid=""
      stop_child "$watchdog_pid"; watchdog_pid=""
      stop_child "$challenge_pid"; challenge_pid=""
      stop_child "$detector_pid"; detector_pid=""
    }
    start_sim() {
      local world="$1"
      roslaunch uav_simulator start.launch gui:=false keyboard_control:=false world_name:="$world" >/tmp/navrl_hifi_gazebo_"$(basename "$world" .world)".log 2>&1 & sim_pid=$!
      for _ in $(seq 1 50); do
        if rosnode list >/dev/null 2>&1 && rostopic info /camera/depth/image_raw 2>/dev/null | grep -q "Publishers:"; then return 0; fi
        sleep 1
      done
      echo "Gazebo/ROS master did not become ready for $world" >&2
      return 1
    }

    start_sim /tmp/navrl_ws/src/uav_simulator/worlds/validation/perception_hifi_person.world
    start_chain clean
    python3 src/onboard_detector/scripts/sim_calibration_eval.py --output /tmp/navrl_results/hifi_calibration_acceptance.json --target-model person1_0.5_0.5_1.8
    python3 src/onboard_detector/scripts/capture_camera_frame.py --output /tmp/navrl_results/hifi_camera_target.png --timeout 15
    python3 src/onboard_detector/scripts/high_fidelity_perception_eval.py --output /tmp/navrl_results/hifi_perception_acceptance.json \
      --target-model person1_0.5_0.5_1.8 --challenge-mode clean --color-topic /camera/color/image_raw --duration 35 --min-samples 5

    # Entry and exit of the perception protection state are checked in one
    # uninterrupted RGB-D/YOLO/watchdog process. The challenge node changes
    # from a deliberately unserviceable composite domain to clean imagery;
    # it must clear its own latched degraded flag after positive detections.
    stop_chain
    start_chain combined
    python3 src/onboard_detector/scripts/perception_watchdog_recovery_eval.py \
      --output /tmp/navrl_results/hifi_perception_watchdog_recovery.json \
      --combined-duration 8 --clean-duration 12

    # Adverse camera-domain suite. The same moving target, depth stream and
    # tracker are retained; only RGB is corrupted before YOLO.
    for mode in partial_occlusion fog rain backlight combined; do
      stop_chain
      start_chain "$mode"
      python3 src/onboard_detector/scripts/capture_camera_frame.py \
        --topic /camera/challenge/color/image_raw \
        --output /tmp/navrl_results/hifi_camera_challenge_"$mode".png --timeout 15
      python3 src/onboard_detector/scripts/high_fidelity_perception_eval.py \
        --output /tmp/navrl_results/hifi_perception_challenge_"$mode".json \
        --target-model person1_0.5_0.5_1.8 --challenge-mode "$mode" \
        --color-topic /camera/challenge/color/image_raw --duration 20 --min-samples 5 \
        --allow-zero-positives
    done
    stop_chain
    python3 src/onboard_detector/scripts/summarize_perception_challenges.py \
      --input-dir /tmp/navrl_results --output /tmp/navrl_results/hifi_perception_challenge_suite.json \
      --min-eligible-frames 40

    # Negative control: restart the same camera/YOLO chain in an empty scene
    # so positive detector frames have a measurable false-positive denominator.
    stop_chain
    stop_child "$sim_pid"
    yolo_pid=""; watchdog_pid=""; detector_pid=""; sim_pid=""
    sleep 2
    start_sim /tmp/navrl_ws/src/uav_simulator/worlds/validation/perception_hifi_empty.world
    start_chain clean
    python3 src/onboard_detector/scripts/high_fidelity_perception_eval.py --output /tmp/navrl_results/hifi_perception_negative_control.json --target-model "" --detector-only --duration 20
  '

echo "High-fidelity RGB-D perception and negative-control evidence: ${RESULT_DIR}"
