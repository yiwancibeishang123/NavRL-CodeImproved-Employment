#!/usr/bin/env bash
# Headless Gazebo clean/stress run with repeatable logs.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_TAG="${NAVRL_NOETIC_IMAGE:-navrl-noetic:20260825}"
RESULT_DIR="${NAVRL_SIM_RESULTS:-${ROOT_DIR}/benchmark_results/sim_latest}"
RESULT_DIR="$(realpath -m "${RESULT_DIR}")"
MISSION_SECONDS="${NAVRL_SIM_MISSION_SECONDS:-10}"
REPEATS="${NAVRL_SIM_REPEATS:-5}"
DOCKER=(sudo docker)

mkdir -p "${RESULT_DIR}"
if ! "${DOCKER[@]}" image inspect "${IMAGE_TAG}" >/dev/null 2>&1; then
  "${DOCKER[@]}" build -f "${ROOT_DIR}/docker/Dockerfile.noetic" -t "${IMAGE_TAG}" "${ROOT_DIR}"
fi

"${DOCKER[@]}" run --rm \
  -v "${ROOT_DIR}:/workspace/NavRL-decoder-deployment:ro" \
  -v "${RESULT_DIR}:/tmp/navrl_results" \
  -e "MISSION_SECONDS=${MISSION_SECONDS}" \
  -e "REPEATS=${REPEATS}" \
  "${IMAGE_TAG}" bash -lc '
    set -eo pipefail
    source /opt/ros/noetic/setup.bash
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

    nav_pid=""
    sim_pid=""
    xvfb_pid=""
    cleanup() {
      test -z "$nav_pid" || kill -INT "$nav_pid" 2>/dev/null || true
      test -z "$sim_pid" || kill -INT "$sim_pid" 2>/dev/null || true
      test -z "$xvfb_pid" || kill "$xvfb_pid" 2>/dev/null || true
    }
    trap cleanup EXIT

    Xvfb :99 -screen 0 1280x1024x24 -nolisten tcp >/tmp/navrl_xvfb.log 2>&1 &
    xvfb_pid=$!
    sleep 2
    start_sim() {
      roslaunch uav_simulator start.launch gui:=false keyboard_control:=false >/tmp/navrl_gazebo.log 2>&1 &
      sim_pid=$!
      for _ in $(seq 1 45); do
        if rostopic info /camera/depth/image_raw 2>/dev/null | grep -q "Publishers:"; then break; fi
        sleep 1
      done
      timeout 20 rostopic echo -n 1 /camera/depth/image_raw >/tmp/navrl_depth_once.txt
    }

    stop_sim() {
      test -z "$sim_pid" || kill -INT "$sim_pid" 2>/dev/null || true
      test -z "$sim_pid" || wait "$sim_pid" 2>/dev/null || true
      sim_pid=""
      sleep 2
    }

    run_case() {
      local label="$1"
      local metrics="$2"
      local trace="$3"
      shift 3
      roslaunch navigation_runner sim_real_gap.launch device:=cpu rviz:=false \
        run_label:="$label" metrics_log:="$metrics" observation_trace_log:="$trace" \
        metrics_resource_stride:=10 sensor_timeout:=2.0 obstacle_timeout:=2.0 \
        perception_timeout:=2.0 "$@" >/tmp/navrl_navigation_"$label".log 2>&1 &
      nav_pid=$!
      # 起飞完成后再发目标，避免软件渲染下固定等待时间不够。
      for _ in $(seq 1 60); do
        if grep -q "take off completed" /tmp/navrl_navigation_"$label".log 2>/dev/null; then
          break
        fi
        sleep 1
      done
      # Allow the first valid raycast/obstacle sample to reach the controller.
      sleep 5
      rostopic pub -1 /move_base_simple/goal geometry_msgs/PoseStamped \
        "{header: {frame_id: map}, pose: {position: {x: 2.0, y: 0.0, z: 1.0}, orientation: {w: 1.0}}}"
      sleep "${MISSION_SECONDS}"
      kill -INT "$nav_pid" 2>/dev/null || true
      wait "$nav_pid" 2>/dev/null || true
      nav_pid=""
      python3 src/navigation_runner/scripts/mission_metrics_analyze.py \
        --input "$metrics" --output "/tmp/navrl_results/${label}_mission_metrics.json"
    }

    reset_vehicle() {
      # reset_world 不会清掉速度控制留下的状态，这里把模型位姿和速度一起归零。
      timeout 2 rostopic pub -r 20 /CERLAB/quadcopter/cmd_vel geometry_msgs/TwistStamped \
        "{twist: {linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}}" >/dev/null 2>&1 || true
      rosservice call /gazebo/reset_world "{}" >/dev/null 2>&1 || true
      rosservice call /gazebo/pause_physics "{}" >/dev/null 2>&1 || true
      for _ in $(seq 1 10); do
        if rosservice call /gazebo/set_model_state \
          "model_state: {model_name: 'quadcopter', pose: {position: {x: 0.0, y: 0.0, z: 0.1}, orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}, twist: {linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}, reference_frame: 'world'}" >/dev/null 2>&1; then
          break
        fi
        sleep 1
      done
      rosservice call /gazebo/unpause_physics "{}" >/dev/null 2>&1 || true
      sleep 2
      timeout 10 rostopic echo -n 1 /CERLAB/quadcopter/odom >/dev/null
    }

    clean_files=()
    stress_files=()
    for repeat in $(seq 1 "${REPEATS}"); do
      start_sim
      clean_label="clean_sim_${repeat}"
      clean_metrics="/tmp/navrl_results/${clean_label}_metrics.jsonl"
      clean_trace="/tmp/navrl_results/${clean_label}_observations.jsonl"
      run_case "$clean_label" "$clean_metrics" "$clean_trace"
      clean_files+=("/tmp/navrl_results/${clean_label}_mission_metrics.json")
      stop_sim
      start_sim
      stress_label="stress_example_not_calibrated_${repeat}"
      stress_metrics="/tmp/navrl_results/${stress_label}_metrics.jsonl"
      stress_trace="/tmp/navrl_results/${stress_label}_observations.jsonl"
      run_case "$stress_label" "$stress_metrics" "$stress_trace" \
        sim_real_perturbation_enabled:=true sim_real_perturbation_seed:=7 \
        lidar_noise_std:=0.01 lidar_dropout_ratio:=0.01 \
        obstacle_position_noise_std:=0.02 obstacle_velocity_noise_std:=0.02 \
        obstacle_dropout_ratio:=0.02 state_noise_std:=0.005 observation_delay_steps:=1 \
        sensor_timeout:=2.0 obstacle_timeout:=2.0 perception_timeout:=2.0
      stress_files+=("/tmp/navrl_results/${stress_label}_mission_metrics.json")
      stop_sim
    done
    python3 src/navigation_runner/scripts/repeated_mission_analyze.py --group clean_sim --input "${clean_files[@]}" --output /tmp/navrl_results/clean_sim_task_metrics_repeated.json
    python3 src/navigation_runner/scripts/repeated_mission_analyze.py --group pressure_sim --input "${stress_files[@]}" --output /tmp/navrl_results/pressure_sim_task_metrics_repeated.json
    cp /tmp/navrl_results/clean_sim_1_metrics.jsonl /tmp/navrl_results/clean_sim_metrics.jsonl
    cp /tmp/navrl_results/clean_sim_1_observations.jsonl /tmp/navrl_results/clean_sim_observations.jsonl
    cp /tmp/navrl_results/stress_example_not_calibrated_1_metrics.jsonl /tmp/navrl_results/perturbed_sim_metrics.jsonl
    cp /tmp/navrl_results/stress_example_not_calibrated_1_observations.jsonl /tmp/navrl_results/perturbed_sim_observations.jsonl
    cp /tmp/navrl_results/clean_sim_1_mission_metrics.json /tmp/navrl_results/clean_sim_mission_metrics.json
    cp /tmp/navrl_results/stress_example_not_calibrated_1_mission_metrics.json /tmp/navrl_results/stress_example_not_calibrated_mission_metrics.json
    python3 src/navigation_runner/scripts/sim_real_gap_analyze.py \
      --clean /tmp/navrl_results/clean_sim_metrics.jsonl \
      --perturbed /tmp/navrl_results/perturbed_sim_metrics.jsonl \
      --clean-task-summary /tmp/navrl_results/clean_sim_mission_metrics.json \
      --perturbed-task-summary /tmp/navrl_results/stress_example_not_calibrated_mission_metrics.json \
      --out-dir /tmp/navrl_results
    python3 - <<"PY"
import json
report = json.load(open("/tmp/navrl_results/sim_real_gap_summary.json"))
for label in ("clean_sim", "perturbed_sim"):
    summary = report["groups"][label]["summary"]
    if summary["policy_inference_records"] <= 0 or summary["command_records"] <= 0:
        raise SystemExit("{} did not reach PPO command control".format(label))
    print(label, "policy calls:", summary["policy_inference_records"])
PY
  '

echo "Headless Gazebo acceptance evidence: ${RESULT_DIR}"
