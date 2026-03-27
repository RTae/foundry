#!/usr/bin/env bash
# Compare standard vs low-memory (compiled) inference outputs.
# Uses a fixed seed so both runs get identical noise trajectories.
set -euo pipefail

SEED="${1:-42}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT="models/rfd3/docs/examples/common_simulate.json"
STD_DIR="logs/inference_outs/compare_standard/0"
LM_DIR="logs/inference_outs/compare_lowmem/0"
NSYS_STD="logs/nsys/rfd3_compare_standard"
NSYS_LM="logs/nsys/rfd3_compare_lowmem"

export RFD3_TRACE_NVTX=1
export RFD3_PROFILE_SYNC=1

echo "=== Cleaning old outputs ==="
rm -rf "$STD_DIR" "$LM_DIR"
mkdir -p "$STD_DIR" "$LM_DIR"

echo ""
echo "=== Running STANDARD inference (seed=$SEED) ==="
nsys profile \
  -o "$NSYS_STD" \
  --force-overwrite=true \
  --trace=cuda,nvtx,osrt,cublas,cudnn \
  --cuda-memory-usage=true \
  rfd3 design \
    out_dir="$STD_DIR" \
    inputs="$INPUT" \
    seed="$SEED" \
    diffusion_batch_size=1 n_batches=1 \
    skip_existing=False dump_trajectories=False prevalidate_inputs=False

echo ""
echo "=== Running LOW-MEM (compiled) inference (seed=$SEED) ==="
nsys profile \
  -o "$NSYS_LM" \
  --force-overwrite=true \
  --trace=cuda,nvtx,osrt,cublas,cudnn \
  --cuda-memory-usage=true \
  rfd3 design \
    out_dir="$LM_DIR" \
    inputs="$INPUT" \
    low_memory_mode=True \
    seed="$SEED" \
    diffusion_batch_size=1 n_batches=1 \
    skip_existing=False dump_trajectories=False prevalidate_inputs=False

echo ""
echo "=== Comparing outputs ==="
python3 "$SCRIPT_DIR/compare_inference_outputs.py" "$STD_DIR" "$LM_DIR"
