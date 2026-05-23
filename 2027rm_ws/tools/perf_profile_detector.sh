#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BUILD_DIR="${WORKSPACE_ROOT}/2027rm_ws/build-perf"
OUT_DIR="${WORKSPACE_ROOT}/2027rm_ws/perf"
TARGET="${BUILD_DIR}/cgraph_camera_detector"
DURATION_SEC="${1:-20}"

mkdir -p "${OUT_DIR}"

cmake -S "${WORKSPACE_ROOT}" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCAMERAAPP_PERF_SYMBOLS=ON
cmake --build "${BUILD_DIR}" --target cgraph_camera_detector -j

cd "${WORKSPACE_ROOT}"

if [[ -r /proc/sys/kernel/perf_event_paranoid ]]; then
  PARANOID="$(cat /proc/sys/kernel/perf_event_paranoid)"
  if [[ "${PARANOID}" -gt 1 ]]; then
    echo "[perf] current perf_event_paranoid=${PARANOID}; perf may be blocked for normal users."
    echo "[perf] temporary fix:"
    echo "  sudo sysctl kernel.perf_event_paranoid=1"
    echo "[perf] stronger profiling access if still blocked:"
    echo "  sudo sysctl kernel.perf_event_paranoid=-1"
    echo
  fi
fi

echo "[perf] running perf stat for ${DURATION_SEC}s"
timeout --signal=INT "${DURATION_SEC}" \
  perf stat \
    -e cycles,instructions,cache-references,cache-misses,branches,branch-misses,context-switches,cpu-migrations,page-faults \
    "${TARGET}" 2>&1 | tee "${OUT_DIR}/perf-stat.txt" || true

echo "[perf] recording call graph for ${DURATION_SEC}s"
timeout --signal=INT "${DURATION_SEC}" \
  perf record -F 199 -g -o "${OUT_DIR}/perf.data" -- "${TARGET}" || true

echo "[perf] writing text report"
perf report -i "${OUT_DIR}/perf.data" --stdio --no-children --sort comm,dso,symbol > "${OUT_DIR}/perf-report.txt" || true

echo "[perf] writing SVG flamegraph"
python3 "${SCRIPT_DIR}/make_perf_flamegraph.py" \
  -i "${OUT_DIR}/perf.data" \
  -o "${OUT_DIR}/flamegraph.svg" || true

echo "[perf] outputs:"
echo "  ${OUT_DIR}/perf-stat.txt"
echo "  ${OUT_DIR}/perf.data"
echo "  ${OUT_DIR}/perf-report.txt"
echo "  ${OUT_DIR}/flamegraph.svg"
