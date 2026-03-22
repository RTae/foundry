#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_nsys_profile.sh [options] -- <python command and args>

Options:
  --output NAME         Nsight output prefix (default: rfd3_profile)
  --trace true|false    Enable NVTX tracing ranges (default: true)
  --sync true|false     Enable cuda synchronize at key boundaries (default: false)
  --capture all|none    Nsight cudaProfilerApi capture range (default: none)
  --low-mem 0|1         Set RFD3_LOW_MEMORY_MODE (default: unchanged)
  -h, --help            Show this help

Examples:
  run_nsys_profile.sh --output rfd3_profile -- python inference_script.py
  run_nsys_profile.sh --trace false -- python inference_script.py
EOF
}

OUTPUT="rfd3_profile"
TRACE_NVTX="true"
PROFILE_SYNC="false"
CAPTURE_RANGE="none"
LOW_MEM=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output)
      OUTPUT="$2"
      shift 2
      ;;
    --trace)
      TRACE_NVTX="$2"
      shift 2
      ;;
    --sync)
      PROFILE_SYNC="$2"
      shift 2
      ;;
    --capture)
      CAPTURE_RANGE="$2"
      shift 2
      ;;
    --low-mem)
      LOW_MEM="$2"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ $# -eq 0 ]]; then
  echo "No command provided after --" >&2
  usage
  exit 1
fi

export RFD3_TRACE_NVTX="$TRACE_NVTX"
export RFD3_PROFILE_SYNC="$PROFILE_SYNC"
if [[ -n "$LOW_MEM" ]]; then
  export RFD3_LOW_MEMORY_MODE="$LOW_MEM"
fi

NSYS_ARGS=(
  profile
  -o "$OUTPUT"
  --force-overwrite=true
  --sample=none
  --trace=cuda,nvtx,osrt
  --cuda-memory-usage=true
)

if [[ "$CAPTURE_RANGE" == "all" ]]; then
  NSYS_ARGS+=(--capture-range=cudaProfilerApi)
fi

echo "RFD3_TRACE_NVTX=$RFD3_TRACE_NVTX"
echo "RFD3_PROFILE_SYNC=$RFD3_PROFILE_SYNC"
if [[ -n "${RFD3_LOW_MEMORY_MODE:-}" ]]; then
  echo "RFD3_LOW_MEMORY_MODE=$RFD3_LOW_MEMORY_MODE"
fi

echo "Running: nsys ${NSYS_ARGS[*]} $*"
nsys "${NSYS_ARGS[@]}" "$@"
