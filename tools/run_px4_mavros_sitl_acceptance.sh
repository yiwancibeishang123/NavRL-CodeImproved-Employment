#!/usr/bin/env bash
# Isolated PX4 SITL + Gazebo Garden + ROS 2 MAVROS acceptance run.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PX4_DIR="${PX4_DIR:-/home/hzw/NMPC-QUA/.deps/PX4-Autopilot}"
RESULT_DIR="${NAVRL_PX4_RESULTS:-${ROOT_DIR}/benchmark_results/hifi_px4_mavros}"
mkdir -p "${RESULT_DIR}"

if [[ ! -x "${PX4_DIR}/build/px4_sitl_default/bin/px4" ]]; then
  echo "PX4 SITL binary is missing: ${PX4_DIR}/build/px4_sitl_default/bin/px4" >&2
  exit 1
fi

set +u
source /opt/ros/humble/setup.bash
set -u
export HEADLESS=1
export PX4_SIM_MODEL=gz_x500
export PX4_GZ_WORLD=lawn
# Start above the ground contact surface.  Starting at z=0 lets the model
# settle through a contact impulse and keeps PX4's vertical-velocity
# innovation check unhealthy during the short OFFBOARD acceptance flight.
export PX4_GZ_MODEL_POSE="${PX4_GZ_MODEL_POSE:-0,0,1,0,0,0}"
export PX4_SIM_SPEED_FACTOR=1
export PX4_HOME_LAT=47.397742
export PX4_HOME_LON=8.545594
export PX4_HOME_ALT=488
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"

px4_pid=""
mavros_pid=""
sitl_rootfs="$(mktemp -d /tmp/navrl-px4-sitl-rootfs.XXXXXX)"
cp -a "${PX4_DIR}/build/px4_sitl_default/rootfs/." "${sitl_rootfs}/"
# The source rootfs keeps etc as a symlink.  Make the temporary copy fully
# private so PX4 parameter writes cannot affect an existing developer SITL.
rm -f "${sitl_rootfs}/etc"
cp -a "${PX4_DIR}/build/px4_sitl_default/etc" "${sitl_rootfs}/etc"
rm -f "${sitl_rootfs}/parameters.bson" "${sitl_rootfs}/parameters_backup.bson"
cleanup() {
  if [[ -n "${mavros_pid}" ]]; then
    kill -INT "-${mavros_pid}" 2>/dev/null || true
    for _ in $(seq 1 5); do
      kill -0 "${mavros_pid}" 2>/dev/null || break
      sleep 1
    done
    # ROS 2 launch can leave its node running after SIGINT from a non-TTY.
    # Both processes are in the dedicated session below, so TERM is scoped.
    kill -TERM "-${mavros_pid}" 2>/dev/null || true
  fi
  [[ -z "${px4_pid}" ]] || kill -INT "-${px4_pid}" 2>/dev/null || true
  wait "${mavros_pid}" 2>/dev/null || true
  wait "${px4_pid}" 2>/dev/null || true
  rm -rf "${sitl_rootfs}"
}
trap cleanup EXIT INT TERM

(
  cd "${sitl_rootfs}"
  exec setsid "${PX4_DIR}/build/px4_sitl_default/bin/px4" "${sitl_rootfs}/etc" -s etc/init.d-posix/rcS -d
) >"${RESULT_DIR}/px4_sitl.log" 2>&1 &
px4_pid=$!

for _ in $(seq 1 60); do
  if grep -q "INFO  \[mavlink\] mode: Onboard" "${RESULT_DIR}/px4_sitl.log" 2>/dev/null; then break; fi
  sleep 1
done

# PX4 v1.15 SITL starts its onboard MAVLink endpoint on 14580 and sends to
# MAVROS on 14540.  MAVROS therefore listens on 14540 and replies to 14580.
setsid ros2 launch mavros px4.launch fcu_url:=udp://:14540@localhost:14580 >"${RESULT_DIR}/mavros.log" 2>&1 &
mavros_pid=$!
python3 "${ROOT_DIR}/tools/px4_mavros_sitl_acceptance.py" --output "${RESULT_DIR}/px4_mavros_sitl_acceptance.json"
