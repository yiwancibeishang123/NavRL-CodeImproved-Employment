#!/usr/bin/env bash

dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
workspace="$(cd "${dir}/../.." && pwd)"
export GAZEBO_MODEL_PATH="${dir}/models:${GAZEBO_MODEL_PATH:-}"
export GAZEBO_RESOURCE_PATH="${dir}/urdf:${GAZEBO_RESOURCE_PATH:-}"
# Plugins are emitted by Catkin, never into a source checkout.  Retain both
# devel and install layouts so the helper works for development and release.
export GAZEBO_PLUGIN_PATH="${workspace}/devel/lib:${workspace}/install/lib:${GAZEBO_PLUGIN_PATH:-}"
