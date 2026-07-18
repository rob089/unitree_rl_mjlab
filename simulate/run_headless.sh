#!/usr/bin/env bash
# Launch the headless simulator with a working EGL vendor.
#
# This machine has no X server and the NVIDIA EGL driver is not registered in
# /usr/share/glvnd/egl_vendor.d, so libglvnd falls back to Mesa software
# rendering (~1 fps on the G1 scene). Point it at the host's NVIDIA ICD instead;
# both variables are scoped to this process, nothing system-wide changes.
#
# Usage: ./run_headless.sh [-p 8080] [--paused] [extra unitree_mujoco args]

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="${HERE}/build/unitree_mujoco_headless"

NVIDIA_LIB_DIR="/var/lib/snapd/hostfs/usr/lib/x86_64-linux-gnu"
NVIDIA_ICD="/var/lib/snapd/hostfs/usr/share/glvnd/egl_vendor.d/10_nvidia.json"

if [[ -f "${NVIDIA_LIB_DIR}/libEGL_nvidia.so.0" && -f "${NVIDIA_ICD}" ]]; then
  export LD_LIBRARY_PATH="${NVIDIA_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  export __EGL_VENDOR_LIBRARY_FILENAMES="${NVIDIA_ICD}"
else
  echo "NVIDIA EGL not found; falling back to software rendering." >&2
fi

exec "${BIN}" "$@"
