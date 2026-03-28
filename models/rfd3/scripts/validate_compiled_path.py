"""Validate vectorized chunked-pairwise path against looped path during real inference.

Monkey-patches forward_chunked to run BOTH looped and vectorized paths on the same
inputs, comparing outputs. Tests the uncompiled vectorized function directly
(torch.compile is a separate JIT layer that can reorder bf16 ops).

Usage (via rfd3 CLI):
    PYTHONPATH=models/rfd3/scripts:$PYTHONPATH python -c "
    import validate_compiled_path
    from rfd3.cli import app; app()
    " design out_dir=logs/inference_outs/validate_compiled/0 \
      inputs=models/rfd3/docs/examples/common_simulate.json \
      low_memory_mode=True diffusion_batch_size=1 n_batches=1 \
      skip_existing=False dump_trajectories=False prevalidate_inputs=False
"""

import atexit
import torch
from rfd3.model.layers.chunked_pairwise import (
    ChunkedPairwiseEmbedder, _chunked_pairwise_compute, _get_compiled_compute,
)

_call_log = []
_THRESHOLD = 1e-4


def _run_vectorized_uncompiled(self, f, indices, C_L, Z_init_II, tok_idx):
    """Call _chunked_pairwise_compute directly (no torch.compile) + pair_mlp."""
    from rfd3.utils.tracing import trace_range

    if C_L.dim() == 2:
        C_L = C_L.unsqueeze(0)

    has_motif = self.motif_pos_embedder is not None and "motif_pos" in f
    has_ref = self.ref_pos_embedder is not None and "ref_pos" in f
    B, L, k = indices.shape
    device = indices.device
    dummy3 = torch.zeros(1, 3, device=device, dtype=C_L.dtype)
    dummy1 = torch.zeros(1, dtype=torch.bool, device=device)

    motif_pos = f.get("motif_pos", dummy3) if has_motif else dummy3
    is_motif = f.get("is_motif_atom_with_fixed_coord", dummy1) if has_motif else dummy1
    if has_motif:
        motif_out_w = self.motif_pos_embedder.output_proj.weight
        motif_vm_w = self.motif_pos_embedder.process_valid_mask.weight
        motif_n_freqs = self.motif_pos_embedder.n_freqs
    else:
        motif_out_w = torch.zeros(self.c_atompair, 1, device=device, dtype=C_L.dtype)
        motif_vm_w = torch.zeros(self.c_atompair, 1, device=device, dtype=C_L.dtype)
        motif_n_freqs = 1

    ref_pos = f.get("ref_pos", dummy3) if has_ref else dummy3
    ref_space_uid = f.get("ref_space_uid", dummy1.long()) if has_ref else dummy1.long()
    is_motif_seq = f.get("is_motif_atom_with_fixed_seq", dummy1) if has_ref else dummy1
    if has_ref:
        ref_embed_frame = self.ref_pos_embedder.embed_frame
        ref_d_w = (self.ref_pos_embedder.process_d.weight
                   if ref_embed_frame and hasattr(self.ref_pos_embedder, 'process_d')
                   else torch.zeros(self.c_atompair, 3, device=device, dtype=C_L.dtype))
        ref_inv_w = self.ref_pos_embedder.process_inverse_dist.weight
        ref_vm_w = self.ref_pos_embedder.process_valid_mask.weight
    else:
        ref_embed_frame = True
        ref_d_w = torch.zeros(self.c_atompair, 3, device=device, dtype=C_L.dtype)
        ref_inv_w = torch.zeros(self.c_atompair, 1, device=device, dtype=C_L.dtype)
        ref_vm_w = torch.zeros(self.c_atompair, 1, device=device, dtype=C_L.dtype)

    sl = self._sl_cached if self._sl_cached is not None else self.process_single_l(C_L.squeeze(0))
    sm = self._sm_cached if self._sm_cached is not None else self.process_single_m(C_L.squeeze(0))
    Z_proc = self._Z_proc_cached if self._Z_proc_cached is not None else self.process_z(Z_init_II)

    P_LL_sparse = _chunked_pairwise_compute(
        indices, C_L, self.c_atompair,
        motif_pos, is_motif, has_motif,
        motif_out_w, motif_vm_w, motif_n_freqs,
        ref_pos, ref_space_uid, is_motif_seq, has_ref,
        ref_embed_frame, ref_d_w, ref_inv_w, ref_vm_w,
        sl, sm, tok_idx, Z_proc,
    )
    P_LL_sparse = P_LL_sparse + self.pair_mlp(P_LL_sparse)
    return P_LL_sparse.contiguous()


def _patched_forward_chunked(self, f, indices, C_L, Z_init_II, tok_idx, use_loop=True):
    """Run looped, vectorized-uncompiled, and compiled paths; compare all three."""

    # Run looped (reference)
    out_loop = self._original_forward_chunked(
        f, indices, C_L, Z_init_II, tok_idx, use_loop=True
    )
    # Run vectorized uncompiled via production code path (use_loop=False, use_compile=False)
    out_vec = self._original_forward_chunked(
        f, indices, C_L, Z_init_II, tok_idx, use_loop=False, use_compile=False
    )
    # Run torch.compiled path (use_loop=False, use_compile=True)
    out_compiled = self._original_forward_chunked(
        f, indices, C_L, Z_init_II, tok_idx, use_loop=False, use_compile=True
    )

    vec_diff = (out_loop - out_vec).abs().max().item()
    compile_diff = (out_loop - out_compiled).abs().max().item()
    rel_vec = vec_diff / (out_loop.abs().max().item() + 1e-12)
    call_num = len(_call_log) + 1
    _call_log.append({"call": call_num, "shape": list(out_loop.shape),
                       "vec_abs_diff": vec_diff, "compile_abs_diff": compile_diff})

    print(f"  call {call_num:3d}: shape={list(out_loop.shape)}  "
          f"vec={vec_diff:.2e}  compile={compile_diff:.2e}  "
          f"dtypes: C_L={C_L.dtype} out={out_loop.dtype}")

    if vec_diff > _THRESHOLD and call_num == 1:
        # === Deep diagnostic on first vectorized failure ===
        print(f"\n  === DIAGNOSTIC (call {call_num}) ===")
        print(f"    C_L: {C_L.dtype}, indices: {indices.dtype}")
        if self._sl_cached is not None:
            print(f"    sl_cached: {self._sl_cached.dtype}")
        if self._Z_proc_cached is not None:
            print(f"    Z_proc_cached: {self._Z_proc_cached.dtype}")

        has_motif = self.motif_pos_embedder is not None and "motif_pos" in f
        has_ref = self.ref_pos_embedder is not None and "ref_pos" in f
        if has_motif:
            print(f"    motif_pos: {f['motif_pos'].dtype}")
            print(f"    output_proj.weight: {self.motif_pos_embedder.output_proj.weight.dtype}")
        if has_ref:
            print(f"    ref_pos: {f['ref_pos'].dtype}")
            print(f"    embed_frame: {self.ref_pos_embedder.embed_frame}")

        print(f"    vec_diff (looped vs vectorized-uncompiled): {vec_diff:.2e}")
        print(f"    compile_diff (looped vs torch.compiled):    {compile_diff:.2e}")

        print_summary()
        raise RuntimeError(
            f"FAIL — vectorized diff {vec_diff:.2e} exceeds threshold {_THRESHOLD:.0e} "
            f"on call {call_num}")

    return out_loop


def print_summary():
    if not _call_log:
        print("\nNo forward_chunked calls were captured.")
        return
    max_vec = max(c["vec_abs_diff"] for c in _call_log)
    max_compile = max(c["compile_abs_diff"] for c in _call_log)
    n = len(_call_log)
    print(f"\n{'='*70}")
    print(f"Validation summary: {n} forward_chunked calls")
    print(f"  Worst vectorized diff (looped vs uncompiled):  {max_vec:.2e}")
    print(f"  Worst compile diff (looped vs torch.compiled): {max_compile:.2e}")
    if max_vec < _THRESHOLD:
        print(f"  PASS — vectorized path matches looped path")
    else:
        print(f"  FAIL — vectorized path diverges from looped path")
    if max_compile > _THRESHOLD:
        print(f"  NOTE — torch.compile introduces {max_compile:.2e} diff (expected under bf16 autocast)")
    print(f"{'='*70}")


# Install patch at import time
ChunkedPairwiseEmbedder._original_forward_chunked = ChunkedPairwiseEmbedder.forward_chunked
ChunkedPairwiseEmbedder.forward_chunked = _patched_forward_chunked
print("Installed validation patch on ChunkedPairwiseEmbedder.forward_chunked")
atexit.register(print_summary)
