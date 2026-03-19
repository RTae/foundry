import os
from contextlib import contextmanager

import torch


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def tracing_enabled() -> bool:
    return _env_flag("RFD3_TRACE_NVTX", default=True)


def should_sync() -> bool:
    return _env_flag("RFD3_PROFILE_SYNC", default=False)


def cuda_ready() -> bool:
    return torch.cuda.is_available()


def maybe_sync_cuda() -> None:
    if should_sync() and cuda_ready():
        torch.cuda.synchronize()


@contextmanager
def trace_range(name: str):
    if tracing_enabled() and cuda_ready():
        with torch.cuda.nvtx.range(name):
            yield
        return
    yield
