# NVIDIA Nsight Tracing for RFD3

This guide explains how to profile RFD3 with hierarchical NVTX ranges.

## What Is Traced

The instrumentation covers:

- Top-level model flow: token initialization, diffusion module, CFG, sampler.
- Diffusion module stages: time embedding, position scaling, atom encoder, token encoder, 18-block diffusion transformer, decoder, output heads, recycling iterations.
- Layer internals: attention QKV projection, pair-bias, softmax, weighted sum, gating, output projection.
- Block internals: attention path, transition path, residual updates for atom/token transformer blocks.
- Pairformer blocks in both TokenInitializer and DiffusionTokenEncoder.
- Memory-critical paths: chunked pairwise embedding and cache setup.

## Hierarchical Range Names

Ranges follow a hierarchical pattern such as:

- `RFD3/TokenInitializer`
- `RFD3/DiffusionModule/AtomEncoder/Block_0/Attention`
- `RFD3/DiffusionModule/DiffusionTransformer/Block_5/Attention`
- `RFD3/DiffusionModule/DiffusionTransformer/Block_5/Transition`
- `RFD3/DiffusionModule/Decoder/Block_2/Upcast`
- `RFD3/ClassifierFreeGuidance`

## Enable/Disable Profiling

NVTX annotation overhead is minimized through environment gating in tracing helper:

- `RFD3_TRACE_NVTX=1|0` (or true/false) controls NVTX ranges.
- `RFD3_PROFILE_SYNC=1|0` (or true/false) enables optional CUDA synchronization where used.

When `RFD3_TRACE_NVTX=0`, trace contexts become no-ops.

## Quick Start

Canonical command (default config):

```bash
RFD3_TRACE_NVTX=1 RFD3_PROFILE_SYNC=1 \
nsys profile -o ./logs/nsys/rfd3_common_sim \
--force-overwrite=true \
--trace=cuda,nvtx,osrt,cublas,cudnn \
--cuda-memory-usage=true \
rfd3 design out_dir=logs/inference_outs/common_sim/0 \
inputs=models/rfd3/docs/examples/common_simulate.json \
skip_existing=False dump_trajectories=False prevalidate_inputs=False
```

Single-model command (runtime batch overrides):

```bash
RFD3_TRACE_NVTX=1 RFD3_PROFILE_SYNC=1 \
nsys profile -o ./logs/nsys/rfd3_common_sim_single_model \
--force-overwrite=true \
--trace=cuda,nvtx,osrt,cublas,cudnn \
--cuda-memory-usage=true \
rfd3 design out_dir=logs/inference_outs/common_sim_single_model/0 \
inputs=models/rfd3/docs/examples/common_simulate.json \
diffusion_batch_size=1 n_batches=1 \
skip_existing=False dump_trajectories=False prevalidate_inputs=False
```

Optional helper script (for quick ad hoc profiling):

```bash
bash models/rfd3/scripts/run_nsys_profile.sh --output rfd3_profile -- python inference_script.py
```

Helper script integrated with canonical workflow:

```bash
bash models/rfd3/scripts/run_nsys_profile.sh \
	--output ./logs/nsys/rfd3_common_sim \
	--sync true \
	-- \
	rfd3 design out_dir=logs/inference_outs/common_sim/0 \
	inputs=models/rfd3/docs/examples/common_simulate.json \
	skip_existing=False dump_trajectories=False prevalidate_inputs=False
```

The script uses the same Nsight flags as canonical docs by default:

- --force-overwrite=true
- --trace=cuda,nvtx,osrt,cublas,cudnn
- --cuda-memory-usage=true

Low-memory mode from CLI config override (recommended):

```bash
RFD3_TRACE_NVTX=1 RFD3_PROFILE_SYNC=1 \
nsys profile -o ./logs/nsys/rfd3_common_sim_lowmem \
--force-overwrite=true \
--trace=cuda,nvtx,osrt,cublas,cudnn \
--cuda-memory-usage=true \
rfd3 design out_dir=logs/inference_outs/common_sim_lowmem/0 \
inputs=models/rfd3/docs/examples/common_simulate.json \
low_memory_mode=True diffusion_batch_size=1 n_batches=1 \
skip_existing=False dump_trajectories=False prevalidate_inputs=False
```

Low-memory mode from CLI config override (recommended):

```bash
RFD3_TRACE_NVTX=1 RFD3_PROFILE_SYNC=1 \
nsys profile -o ./logs/nsys/rfd3_common_sim_lowmem \
--force-overwrite=true \
--trace=cuda,nvtx,osrt,cublas,cudnn \
--cuda-memory-usage=true \
rfd3 design out_dir=logs/inference_outs/common_sim_lowmem/0 \
inputs=models/rfd3/docs/examples/common_simulate.json \
low_memory_mode=True diffusion_batch_size=1 n_batches=1 \
skip_existing=False dump_trajectories=False prevalidate_inputs=False
```

## Reading Results In Nsight Systems

Look at NVTX lanes and group by range name prefix:

- Compare `RFD3/DiffusionModule/DiffusionTransformer/Block_i` durations across `i=0..17` to find slowest blocks.
- Within each block, compare `Attention` vs `Transition` to get the time breakdown.
- Inspect `Upcast` and `Downcast` ranges for cross-attention overhead.
- Compare recycling iterations under `RFD3/DiffusionModule/Recycling/Recycle_i`.
- Correlate CUDA kernels and memory transfer rows with active NVTX ranges.

## Guidance

- Gradient checkpointing: place NVTX ranges inside checkpointed functions for accurate per-op attribution during recomputation.
- Dynamic shapes: include shape context in profiling metadata outside NVTX names (for example in logs), rather than exploding range cardinality.
- Granularity: start with module/block/layer ranges first. Add matmul-level ranges only when a specific hotspot requires deeper decomposition.
- Input-size correlation: track per-run stats (`L`, token count, atom count, batch size) next to the report filename.

## Suggested Naming For Size Correlation

Use output names like:

- `rfd3_L2048_B1_rec2`
- `rfd3_L4096_B1_lowmem`

This helps compare timelines for scaling behavior.
