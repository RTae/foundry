```bash
RFD3_TRACE_NVTX=1 RFD3_PROFILE_SYNC=1 \
nsys profile -o ./logs/nsys/rfd3_common_sim \
--force-overwrite=true \
--trace=cuda,nvtx,osrt,cublas,cudnn \
--cuda-memory-usage=true rfd3 design out_dir=logs/inference_outs/common_sim/0 inputs=models/rfd3/docs/examples/common_simulate.json skip_existing=False dump_trajectories=False prevalidate_inputs=False
```