#!/usr/bin/env bash

# This script installs all the steps mentioned in the README.md to setup the project for door state estimation and tracking.

# if any command fail exit the whole script
set -e

# get directory for the baselines
# assumes the structure: catkin_ws/src/alma_handle_detection
ESTIMATOR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

pushd "$ESTIMATOR_DIR" || return

# install python dependencies
pip install -r requirements.txt

# exit the directory
popd || true

echo "--------------------------------------------------"
echo "[INFO] Estimator depedencies installation successful."
