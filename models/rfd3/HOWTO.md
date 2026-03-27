# RFD3 HOWTO

## Nsight Profiling

Detailed tracing docs: [models/rfd3/docs/nsight_tracing.md](docs/nsight_tracing.md)

### Canonical Commands

Default config (uses `diffusion_batch_size` from config):

```bash
RFD3_TRACE_NVTX=1 RFD3_PROFILE_SYNC=1 \
nsys profile -o ./logs/nsys/rfd3_common_sim \
  --force-overwrite=true \
  --trace=cuda,nvtx,osrt,cublas,cudnn \
  --cuda-memory-usage=true \
  rfd3 design \
    out_dir=logs/inference_outs/common_sim/0 \
    inputs=models/rfd3/docs/examples/common_simulate.json \
    skip_existing=False dump_trajectories=False prevalidate_inputs=False
```

Single-model (override batch settings):

```bash
RFD3_TRACE_NVTX=1 RFD3_PROFILE_SYNC=1 \
nsys profile -o ./logs/nsys/rfd3_common_sim_single_model \
  --force-overwrite=true \
  --trace=cuda,nvtx,osrt,cublas,cudnn \
  --cuda-memory-usage=true \
  rfd3 design \
    out_dir=logs/inference_outs/common_sim_single_model/0 \
    inputs=models/rfd3/docs/examples/common_simulate.json \
    diffusion_batch_size=1 n_batches=1 \
    skip_existing=False dump_trajectories=False prevalidate_inputs=False
```

Low-memory mode (sparse/compiled path):

```bash
RFD3_TRACE_NVTX=1 RFD3_PROFILE_SYNC=1 \
nsys profile -o ./logs/nsys/rfd3_common_sim_lowmem \
  --force-overwrite=true \
  --trace=cuda,nvtx,osrt,cublas,cudnn \
  --cuda-memory-usage=true \
  rfd3 design \
    out_dir=logs/inference_outs/common_sim_lowmem/0 \
    inputs=models/rfd3/docs/examples/common_simulate.json \
    low_memory_mode=True \
    diffusion_batch_size=1 n_batches=1 \
    skip_existing=False dump_trajectories=False prevalidate_inputs=False
```

### Helper Script

The helper script wraps nsys with the same default flags:

```bash
bash models/rfd3/scripts/run_nsys_profile.sh \
  --output ./logs/nsys/rfd3_common_sim \
  --sync true \
  -- \
  rfd3 design \
    out_dir=logs/inference_outs/common_sim/0 \
    inputs=models/rfd3/docs/examples/common_simulate.json \
    skip_existing=False dump_trajectories=False prevalidate_inputs=False
```

## Correctness Comparison (Standard vs Low-Memory)

The comparison script runs both standard and low-memory inference under nsys,
then compares the outputs side by side to verify correctness:

```bash
bash models/rfd3/scripts/compare_standard_vs_lowmem.sh        # default seed=42
bash models/rfd3/scripts/compare_standard_vs_lowmem.sh 123     # custom seed
```

What it does:
1. Cleans old outputs under `logs/inference_outs/compare_{standard,lowmem}/`
2. Runs standard inference with nsys → `logs/nsys/rfd3_compare_standard.nsys-rep`
3. Runs low-memory inference with nsys → `logs/nsys/rfd3_compare_lowmem.nsys-rep`
4. Calls `compare_inference_outputs.py` to diff metrics and structures

Both runs use the same seed so noise trajectories are identical, isolating
any differences to the compiled chunked-pairwise path.

## Notes

- Prefer `low_memory_mode=True` on `rfd3 design` over manually setting `RFD3_LOW_MEMORY_MODE`.
- Use output names like `rfd3_L2048_B1_rec2` or `rfd3_L4096_B1_lowmem` to track input-size scaling.