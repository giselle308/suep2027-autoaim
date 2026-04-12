#!/usr/bin/env bash
set -euo pipefail

STATE_DIR="/tmp/rm_debug_foxglove"

log() {
  printf '[stop_debug_foxglove] %s\n' "$*"
}

stop_one() {
  local name="$1"
  local pid_file="${STATE_DIR}/${name}.pid"

  if [[ ! -f "${pid_file}" ]]; then
    log "${name}: pid file not found"
    return
  fi

  local pid
  pid="$(cat "${pid_file}")"
  local pgid
  pgid="$(ps -o pgid= -p "${pid}" 2>/dev/null | tr -d ' ' || true)"

  if kill -0 "${pid}" 2>/dev/null; then
    if [[ -n "${pgid}" ]]; then
      # Kill the whole process group so spawned children also exit.
      kill -TERM -- "-${pgid}" 2>/dev/null || true
      sleep 0.3
      if ps -o pid= -g "${pgid}" 2>/dev/null | grep -q .; then
        kill -KILL -- "-${pgid}" 2>/dev/null || true
      fi
    else
      kill "${pid}" || true
      sleep 0.2
      if kill -0 "${pid}" 2>/dev/null; then
        kill -9 "${pid}" || true
      fi
    fi
    log "stopped ${name}, pid=${pid}, pgid=${pgid:-unknown}"
  else
    log "${name}: process not running"
  fi

  rm -f "${pid_file}"
}

cleanup_orphans() {
  pkill -f 'rosbridge_websocket_launch.xml' 2>/dev/null || true
  pkill -f 'foxglove_ros2_publisher.py' 2>/dev/null || true
  pkill -f 'run_cgraph_camera_detector' 2>/dev/null || true
  pkill -f '/build/cgraph_camera_detector model/last.xml' 2>/dev/null || true
}

stop_one "detector"
stop_one "publisher"
stop_one "rosbridge"
cleanup_orphans

log "done"
