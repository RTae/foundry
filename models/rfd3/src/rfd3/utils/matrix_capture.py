from __future__ import annotations

import os
from pathlib import Path

import torch


_WRITTEN_TAGS: set[str] = set()


def _capture_dir() -> Path | None:
    value = os.getenv("RFD3_CAPTURE_MATRIX_DIR")
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    path = Path(value)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_once(tag: str, payload: dict) -> None:
    root = _capture_dir()
    if root is None or tag in _WRITTEN_TAGS:
        return
    torch.save(payload, root / f"{tag}.pt")
    _WRITTEN_TAGS.add(tag)


def capture_attention_indices(indices: torch.Tensor) -> None:
    indices_cpu = indices.detach().to("cpu", dtype=torch.long)
    _write_once(
        "attention_indices",
        {
            "indices": indices_cpu,
            "shape": tuple(indices_cpu.shape),
        },
    )


def capture_pairwise_initializer_structure(
    *,
    p_ll: torch.Tensor,
    motif_valid_mask: torch.Tensor,
    ref_valid_mask: torch.Tensor,
    tok_idx: torch.Tensor,
) -> None:
    p_ll_cpu = p_ll.detach().to("cpu")
    channel_norm = torch.linalg.norm(p_ll_cpu, dim=-1)
    payload = {
        "shape": tuple(p_ll_cpu.shape),
        "token_index": tok_idx.detach().to("cpu", dtype=torch.long),
        "motif_valid_mask": motif_valid_mask.detach().to("cpu"),
        "ref_valid_mask": ref_valid_mask.detach().to("cpu"),
        "pair_energy": channel_norm,
        "pair_nonzero_mask": channel_norm > 0,
    }
    _write_once("pairwise_initializer", payload)


def capture_initializer_outputs(outputs: dict) -> None:
    """Capture the full initializer_outputs dict from low_memory=False path.

    Saves every tensor with its shape/dtype, plus a summary file.
    Controlled by RFD3_CAPTURE_MATRIX_DIR env var (same as other captures).
    """
    root = _capture_dir()
    if root is None or "initializer_outputs" in _WRITTEN_TAGS:
        return

    payload = {}
    summary = {}
    for key, val in outputs.items():
        if isinstance(val, torch.Tensor):
            cpu_val = val.detach().to("cpu")
            payload[key] = cpu_val
            summary[key] = {
                "shape": list(cpu_val.shape),
                "dtype": str(cpu_val.dtype),
                "numel": cpu_val.numel(),
                "size_mb": round(cpu_val.numel() * cpu_val.element_size() / 1e6, 2),
            }
        else:
            summary[key] = {"type": type(val).__name__}

    payload["__summary__"] = summary
    torch.save(payload, root / "initializer_outputs.pt")
    _WRITTEN_TAGS.add("initializer_outputs")
