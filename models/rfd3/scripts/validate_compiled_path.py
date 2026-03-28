"""Validate compiled chunked-pairwise path against looped path during real inference.

Monkey-patches forward_chunked to run BOTH looped and compiled paths on the same
inputs, comparing outputs. Stops on first threshold breach with diagnostics.

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
from rfd3.model.layers.chunked_pairwise import ChunkedPairwiseEmbedder, _chunked_pairwise_compute

_call_log = []
_THRESHOLD = 1e-4


def _patched_forward_chunked(self, f, indices, C_L, Z_init_II, tok_idx, use_loop=True):
    """Run both paths on same inputs, compare, stop on threshold breach."""

    # Run looped (reference)
    out_loop = self._original_forward_chunked(
        f, indices, C_L, Z_init_II, tok_idx, use_loop=True
    )
    # Run compiled
    out_compiled = self._original_forward_chunked(
        f, indices, C_L, Z_init_II, tok_idx, use_loop=False
    )

    max_diff = (out_loop - out_compiled).abs().max().item()
    rel_diff = max_diff / (out_loop.abs().max().item() + 1e-12)
    call_num = len(_call_log) + 1
    _call_log.append({"call": call_num, "shape": list(out_loop.shape),
                       "max_abs_diff": max_diff, "max_rel_diff": rel_diff})

    print(f"  call {call_num:3d}: shape={list(out_loop.shape)}  "
          f"abs={max_diff:.2e}  rel={rel_diff:.2e}  "
          f"dtypes: C_L={C_L.dtype} out={out_loop.dtype}")

    if max_diff > _THRESHOLD and call_num == 1:
        # === Deep diagnostic on first failure ===
        print(f"\n  === DIAGNOSTIC (call {call_num}) ===")

        # 1. Check dtypes
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
            if hasattr(self.ref_pos_embedder, 'process_d'):
                print(f"    process_d.weight: {self.ref_pos_embedder.process_d.weight.dtype}")

        # 2. Run _chunked_pairwise_compute directly (uncompiled) with fp32 upcast
        if C_L.dim() == 2:
            C_L_b = C_L.unsqueeze(0)
        else:
            C_L_b = C_L

        B, L, k = indices.shape
        device = indices.device
        dummy3 = torch.zeros(1, 3, device=device, dtype=torch.float32)
        dummy1 = torch.zeros(1, dtype=torch.bool, device=device)

        motif_pos = f.get("motif_pos", dummy3) if has_motif else dummy3
        is_motif = f.get("is_motif_atom_with_fixed_coord", dummy1) if has_motif else dummy1
        if has_motif:
            motif_out_w = self.motif_pos_embedder.output_proj.weight.t().float()
            motif_vm_w = self.motif_pos_embedder.process_valid_mask.weight.t().float()
            motif_n_freqs = self.motif_pos_embedder.n_freqs
        else:
            motif_out_w = torch.zeros(1, self.c_atompair, device=device)
            motif_vm_w = torch.zeros(1, self.c_atompair, device=device)
            motif_n_freqs = 1

        ref_pos = f.get("ref_pos", dummy3) if has_ref else dummy3
        ref_space_uid = f.get("ref_space_uid", dummy1.long()) if has_ref else dummy1.long()
        is_motif_seq = f.get("is_motif_atom_with_fixed_seq", dummy1) if has_ref else dummy1
        if has_ref:
            ref_embed_frame = self.ref_pos_embedder.embed_frame
            ref_d_w = (self.ref_pos_embedder.process_d.weight.t().float()
                       if ref_embed_frame and hasattr(self.ref_pos_embedder, 'process_d')
                       else torch.zeros(3, self.c_atompair, device=device))
            ref_inv_w = self.ref_pos_embedder.process_inverse_dist.weight.t().float()
            ref_vm_w = self.ref_pos_embedder.process_valid_mask.weight.t().float()
        else:
            ref_embed_frame = True
            ref_d_w = torch.zeros(3, self.c_atompair, device=device)
            ref_inv_w = torch.zeros(1, self.c_atompair, device=device)
            ref_vm_w = torch.zeros(1, self.c_atompair, device=device)

        sl = self._sl_cached.float() if self._sl_cached is not None else None
        sm = self._sm_cached.float() if self._sm_cached is not None else None
        Z_proc = (self._Z_proc_cached.float() if self._Z_proc_cached is not None
                  else self.process_z(Z_init_II).float())

        # Call uncompiled in fp32
        P_fp32 = _chunked_pairwise_compute(
            indices, C_L_b.float(), self.c_atompair,
            motif_pos.float(), is_motif, has_motif,
            motif_out_w, motif_vm_w, motif_n_freqs,
            ref_pos.float(), ref_space_uid, is_motif_seq, has_ref,
            ref_embed_frame, ref_d_w, ref_inv_w, ref_vm_w,
            sl, sm, tok_idx, Z_proc,
        )
        P_fp32_final = P_fp32 + self.pair_mlp(P_fp32.to(out_loop.dtype)).float()

        # Call uncompiled in native dtype (same as _forward_chunked_compiled uses)
        sl_native = self._sl_cached if self._sl_cached is not None else None
        sm_native = self._sm_cached if self._sm_cached is not None else None
        Z_native = self._Z_proc_cached if self._Z_proc_cached is not None else self.process_z(Z_init_II)

        if has_motif:
            mot_out_n = self.motif_pos_embedder.output_proj.weight.t()
            mot_vm_n = self.motif_pos_embedder.process_valid_mask.weight.t()
        else:
            mot_out_n = torch.zeros(1, self.c_atompair, device=device, dtype=C_L_b.dtype)
            mot_vm_n = torch.zeros(1, self.c_atompair, device=device, dtype=C_L_b.dtype)

        if has_ref:
            ref_d_n = (self.ref_pos_embedder.process_d.weight.t()
                       if ref_embed_frame and hasattr(self.ref_pos_embedder, 'process_d')
                       else torch.zeros(3, self.c_atompair, device=device, dtype=C_L_b.dtype))
            ref_inv_n = self.ref_pos_embedder.process_inverse_dist.weight.t()
            ref_vm_n = self.ref_pos_embedder.process_valid_mask.weight.t()
        else:
            ref_d_n = torch.zeros(3, self.c_atompair, device=device, dtype=C_L_b.dtype)
            ref_inv_n = torch.zeros(1, self.c_atompair, device=device, dtype=C_L_b.dtype)
            ref_vm_n = torch.zeros(1, self.c_atompair, device=device, dtype=C_L_b.dtype)

        P_native = _chunked_pairwise_compute(
            indices, C_L_b, self.c_atompair,
            motif_pos if has_motif else dummy3.to(C_L_b.dtype),
            is_motif, has_motif,
            mot_out_n, mot_vm_n, motif_n_freqs,
            ref_pos if has_ref else dummy3.to(C_L_b.dtype),
            ref_space_uid, is_motif_seq, has_ref,
            ref_embed_frame, ref_d_n, ref_inv_n, ref_vm_n,
            sl_native, sm_native, tok_idx, Z_native,
        )
        P_native_final = P_native + self.pair_mlp(P_native)

        d1 = (P_fp32_final.to(out_loop.dtype) - out_loop).abs().max().item()
        d2 = (P_native_final - out_loop).abs().max().item()
        d3 = (P_native_final - out_compiled).abs().max().item()
        d4 = (P_native - P_fp32.to(P_native.dtype)).abs().max().item()

        print(f"    fp32_uncompiled   vs looped:   {d1:.2e}")
        print(f"    native_uncompiled vs looped:   {d2:.2e}")
        print(f"    native_uncompiled vs compiled:  {d3:.2e}")
        print(f"    native_pre_mlp vs fp32_pre_mlp: {d4:.2e}")
        print(f"    P_native dtype: {P_native.dtype}")
        print(f"    P_fp32   dtype: {P_fp32.dtype}")

        print_summary()
        raise RuntimeError(
            f"FAIL — max_abs_diff {max_diff:.2e} exceeds threshold {_THRESHOLD:.0e} "
            f"on call {call_num}")

    return out_loop


def print_summary():
    if not _call_log:
        print("\nNo forward_chunked calls were captured.")
        return
    max_abs = max(c["max_abs_diff"] for c in _call_log)
    max_rel = max(c["max_rel_diff"] for c in _call_log)
    n = len(_call_log)
    print(f"\n{'='*70}")
    print(f"Validation summary: {n} forward_chunked calls")
    print(f"  Worst absolute diff: {max_abs:.2e}")
    print(f"  Worst relative diff: {max_rel:.2e}")
    if max_abs < _THRESHOLD:
        print(f"  PASS — compiled path matches looped path")
    else:
        print(f"  FAIL — compiled path diverges from looped path")
    print(f"{'='*70}")


# Install patch at import time
ChunkedPairwiseEmbedder._original_forward_chunked = ChunkedPairwiseEmbedder.forward_chunked
ChunkedPairwiseEmbedder.forward_chunked = _patched_forward_chunked
print("Installed validation patch on ChunkedPairwiseEmbedder.forward_chunked")
atexit.register(print_summary)
