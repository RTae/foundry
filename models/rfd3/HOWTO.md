RFD3 Nsight profiling commands (canonical):

Default config (for example diffusion_batch_size from config):

```bash
RFD3_TRACE_NVTX=1 RFD3_PROFILE_SYNC=1 \
nsys profile -o ./logs/nsys/rfd3_common_sim \
--force-overwrite=true \
--trace=cuda,nvtx,osrt,cublas,cudnn \
--cuda-memory-usage=true rfd3 design out_dir=logs/inference_outs/common_sim/0 inputs=models/rfd3/docs/examples/common_simulate.json skip_existing=False dump_trajectories=False prevalidate_inputs=False
```

Single-model command (override runtime batch settings):

```bash
RFD3_TRACE_NVTX=1 RFD3_PROFILE_SYNC=1 \
nsys profile -o ./logs/nsys/rfd3_common_sim_single_model \
--force-overwrite=true \
--trace=cuda,nvtx,osrt,cublas,cudnn \
--cuda-memory-usage=true \
rfd3 design out_dir=logs/inference_outs/common_sim_single_model/0 \
inputs=models/rfd3/docs/examples/common_simulate.json diffusion_batch_size=1 n_batches=1 skip_existing=False dump_trajectories=False prevalidate_inputs=False
```

Notes:

- Keep these commands as the source of truth for Nsight profiling in this repo.
- Equivalent guidance is documented in models/rfd3/docs/nsight_tracing.md.