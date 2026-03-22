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

- `RFD3_TRACE_NVTX=true|false` controls NVTX ranges.
- `RFD3_PROFILE_SYNC=true|false` enables optional CUDA synchronization where used.

When `RFD3_TRACE_NVTX=false`, trace contexts become no-ops.

## Quick Start

Use helper script:

```bash
bash models/rfd3/scripts/run_nsys_profile.sh --output rfd3_profile -- python inference_script.py
```

Disable ranges while keeping Nsight capture:

```bash
bash models/rfd3/scripts/run_nsys_profile.sh --trace false --output rfd3_no_nvtx -- python inference_script.py
```

Enable low-memory mode during profiling:

```bash
bash models/rfd3/scripts/run_nsys_profile.sh --low-mem 1 --output rfd3_lowmem -- python inference_script.py
```

Manual command:

```bash
RFD3_TRACE_NVTX=true \
RFD3_PROFILE_SYNC=false \
nsys profile -o rfd3_profile --trace=cuda,nvtx,osrt python inference_script.py
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
