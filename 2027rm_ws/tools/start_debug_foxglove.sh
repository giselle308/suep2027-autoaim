#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="${WORKSPACE_ROOT}/2027rm_ws"
STATE_DIR="/tmp/rm_debug_foxglove"
mkdir -p "${STATE_DIR}"

log() {
  printf '[start_debug_foxglove] %s\n' "$*"
}

start_bg() {
  local name="$1"
  local cmd="$2"
  local log_file="${STATE_DIR}/${name}.log"
  local pid_file="${STATE_DIR}/${name}.pid"

  if [[ -f "${pid_file}" ]] && kill -0 "$(cat "${pid_file}")" 2>/dev/null; then
    log "${name} already running, pid=$(cat "${pid_file}")"
    return
  fi

  # Run each service in its own session/process-group so stop script can kill the whole tree.
  nohup setsid bash -lc "${cmd}" >"${log_file}" 2>&1 &
  echo $! >"${pid_file}"
  log "started ${name}, pid=$!, log=${log_file}"
}

if ! command -v ros2 >/dev/null 2>&1; then
  log "ros2 command not found. Please install ROS2 Humble first."
  exit 1
fi

if ! bash -lc 'source /opt/ros/humble/setup.bash && ros2 pkg list | grep -q "^rosbridge_server$"'; then
  log "rosbridge_server not installed. Run: sudo apt install ros-humble-rosbridge-server"
  exit 1
fi

start_bg "rosbridge" "source /opt/ros/humble/setup.bash && ros2 launch rosbridge_server rosbridge_websocket_launch.xml"
start_bg "publisher" "source /opt/ros/humble/setup.bash && python3 \"${PROJECT_ROOT}/tools/foxglove_ros2_publisher.py\""
start_bg "detector" "cd \"${WORKSPACE_ROOT}\" && RM_DEBUG_FOXGLOVE=1 cmake --build build --target run_cgraph_camera_detector"

log "all processes started"
log "foxglove websocket: ws://127.0.0.1:9090"
log "image topic: /debug/image/compressed"
log "stop all: ${PROJECT_ROOT}/tools/stop_debug_foxglove.sh"
