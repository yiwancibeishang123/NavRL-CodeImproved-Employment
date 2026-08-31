#!/usr/bin/env bash
# Prepare an Ubuntu 20.04 / ROS Noetic workspace from this delivery archive.
# ROS Noetic setup scripts reference optional environment variables.  ``-u``
# makes a clean shell fail while sourcing them, before installation starts.
set -eo pipefail

WITH_APT=false
WORKSPACE="${NAVRL_WORKSPACE:-$HOME/navrl_ws}"
for argument in "$@"; do
  case "$argument" in
    --with-apt) WITH_APT=true ;;
    --workspace=*) WORKSPACE="${argument#*=}" ;;
    *) echo "Unknown option: $argument" >&2; exit 2 ;;
  esac
done

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -d "$ROOT_DIR/ros1/navigation_runner" || ! -f "$ROOT_DIR/isaac-training/setup_deployment_ubuntu.sh" ]]; then
  echo "Run this script from the extracted NavRL-decoder-deployment directory." >&2
  exit 1
fi
if ! command -v conda >/dev/null; then
  echo "Conda was not found. Install Miniconda/Anaconda first, then rerun this script." >&2
  exit 1
fi

if [[ "$WITH_APT" == true ]]; then
  sudo apt-get update
  sudo apt-get install -y build-essential cmake python3-catkin-tools \
    libpcl-dev libeigen3-dev libprotobuf-dev libboost-system-dev \
    qtbase5-dev libgazebo11-dev \
    ros-noetic-ros-base ros-noetic-mavros ros-noetic-mavros-extras ros-noetic-cv-bridge \
    ros-noetic-image-transport ros-noetic-vision-msgs ros-noetic-visualization-msgs \
    ros-noetic-pcl-conversions ros-noetic-nav-msgs ros-noetic-sensor-msgs \
    ros-noetic-geometry-msgs ros-noetic-std-srvs \
    ros-noetic-pcl-ros ros-noetic-tf ros-noetic-tf2-ros ros-noetic-tf2-geometry-msgs \
    ros-noetic-message-filters ros-noetic-gazebo-ros ros-noetic-gazebo-plugins
fi

if [[ ! -f /opt/ros/noetic/setup.bash ]]; then
  echo "ROS Noetic was not found. Install ROS Noetic first, or rerun with --with-apt after adding the ROS repository." >&2
  exit 1
fi
if [[ -e "$WORKSPACE/src/navigation_runner" ]]; then
  echo "Refusing to overwrite existing $WORKSPACE/src/navigation_runner. Choose another --workspace path." >&2
  exit 1
fi

source /opt/ros/noetic/setup.bash
bash "$ROOT_DIR/isaac-training/setup_deployment_ubuntu.sh"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${NAVRL_ENV_NAME:-NavRL}"

mkdir -p "$WORKSPACE/src"
cp -a "$ROOT_DIR/ros1/." "$WORKSPACE/src/"
cd "$WORKSPACE"
catkin_make

echo
echo "Workspace ready: $WORKSPACE"
echo "Next shell: source /opt/ros/noetic/setup.bash && source $WORKSPACE/devel/setup.bash && conda activate ${NAVRL_ENV_NAME:-NavRL}"
echo "Then run: python3 $WORKSPACE/src/navigation_runner/scripts/validate_deployment.py"
