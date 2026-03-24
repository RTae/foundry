import os
import threading
from contextlib import contextmanager

import torch


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def tracing_enabled() -> bool:
    return _env_flag("RFD3_TRACE_NVTX", default=True)


def memory_tracking_enabled() -> bool:
    return _env_flag("RFD3_TRACK_MEMORY", default=False)


def should_sync() -> bool:
    return _env_flag("RFD3_PROFILE_SYNC", default=False)


def cuda_ready() -> bool:
    return torch.cuda.is_available()


def maybe_sync_cuda() -> None:
    if should_sync() and cuda_ready():
        torch.cuda.synchronize()


# ── GPU Memory Tracker ──────────────────────────────────────────────────────
# Enabled via RFD3_TRACK_MEMORY=1.  Records (nvtx_range, alloc_before, alloc_after,
# delta, reserved, peak) at the entry and exit of every trace_range.  Results are
# written to a JSON file at the end of the run via dump_memory_log().

_memory_log: list[dict] = []
_memory_log_lock = threading.Lock()


def _record_memory(name: str, phase: str) -> dict | None:
    if not cuda_ready():
        return None
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    peak = torch.cuda.max_memory_allocated()
    entry = {
        "range": name,
        "phase": phase,
        "allocated_mb": round(allocated / 1024**2, 2),
        "reserved_mb": round(reserved / 1024**2, 2),
        "peak_mb": round(peak / 1024**2, 2),
    }
    with _memory_log_lock:
        _memory_log.append(entry)
    return entry


def dump_memory_log(path: str | os.PathLike | None = None) -> list[dict]:
    """Write the accumulated memory log to a JSON file and return it."""
    import json
    from pathlib import Path

    with _memory_log_lock:
        log = list(_memory_log)

    if path is None:
        path = os.getenv("RFD3_MEMORY_LOG_PATH", "rfd3_memory_log.json")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(log, f, indent=2)
    return log


def get_memory_log() -> list[dict]:
    """Return a copy of the memory log collected so far."""
    with _memory_log_lock:
        return list(_memory_log)


@contextmanager
def trace_range(name: str):
    if tracing_enabled() and cuda_ready():
        if memory_tracking_enabled():
            _record_memory(name, "enter")
        torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()
            if memory_tracking_enabled():
                _record_memory(name, "exit")
        return
    yield
