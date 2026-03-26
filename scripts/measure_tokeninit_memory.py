"""
Measure memory usage of TokenInitializer outputs in RFD3.

Shows FOUR views of memory:
  1. Output tensors only (what stays in memory after forward pass)
  2. Peak intermediate memory (temporary tensors during computation)
  3. Nsight log parsing (from actual profiling runs)
  4. Scaling analysis table
  5. LIVE GPU measurement (runs TokenInitializer on GPU and captures real memory)

Usage:
    # All analytical modes (no model needed):
    python3 scripts/measure_tokeninit_memory.py

    # Custom protein size:
    python3 scripts/measure_tokeninit_memory.py --n-tokens 200 --n-atoms 2800

    # Parse Nsight memory log:
    python3 scripts/measure_tokeninit_memory.py --nsight-log logs/inference_outs/memory_tracked/0/rfd3_memory_log.json

    # Live GPU measurement (actually runs TokenInitializer):
    python3 scripts/measure_tokeninit_memory.py --live
    python3 scripts/measure_tokeninit_memory.py --live --n-tokens 200 --n-atoms 2800
"""

import argparse
import json
import os
import sys


# ── RFD3 default config values ──
C_S = 384
C_Z = 128
C_ATOM = 128
C_ATOMPAIR = 16
N_FREQS = 32  # sinusoidal frequency bands


def format_bytes(n_bytes):
    """Format bytes into human-readable string."""
    if n_bytes >= 1024**3:
        return f"{n_bytes / 1024**3:.2f} GB"
    elif n_bytes >= 1024**2:
        return f"{n_bytes / 1024**2:.1f} MB"
    elif n_bytes >= 1024:
        return f"{n_bytes / 1024:.1f} KB"
    return f"{n_bytes} B"


def _tensor_bytes(*dims):
    """Calculate float32 tensor memory from dimensions."""
    n = 1
    for d in dims:
        n *= d
    return n * 4


# ───────────────────────────────────────────────────────────────────
# 1. Output tensors only
# ───────────────────────────────────────────────────────────────────
def analytical_outputs(I, L, c_s=C_S, c_z=C_Z, c_atom=C_ATOM, c_atompair=C_ATOMPAIR):
    """Memory of the 5 tensors that STAY in memory after TokenInitializer."""
    tensors = {
        "P_LL":     (L, L, c_atompair),
        "Z_II":     (I, I, c_z),
        "Q_L_init": (L, c_atom),
        "C_L":      (L, c_atom),
        "S_I":      (I, c_s),
    }

    print(f"\n{'='*70}")
    print(f"  1. OUTPUT TENSORS  (I={I} tokens, L={L} atoms)")
    print(f"     What stays in GPU memory after TokenInitializer.forward()")
    print(f"{'='*70}\n")

    total_bytes = 0
    rows = []
    for name, shape in tensors.items():
        mem = _tensor_bytes(*shape)
        total_bytes += mem
        rows.append((name, shape, mem))

    print(f"  {'Tensor':<12} {'Shape':<25} {'Memory':>12} {'%':>7}")
    print(f"  {'-'*12} {'-'*25} {'-'*12} {'-'*7}")
    for name, shape, mem in rows:
        pct = 100.0 * mem / total_bytes
        print(f"  {name:<12} {str(list(shape)):<25} {format_bytes(mem):>12} {pct:>6.1f}%")

    print(f"  {'-'*12} {'-'*25} {'-'*12} {'-'*7}")
    print(f"  {'TOTAL':<12} {'':<25} {format_bytes(total_bytes):>12} {'100.0%':>7}")
    print()
    print(f"  >> P_LL is {100.0 * rows[0][2] / total_bytes:.1f}% of retained memory")
    print()

    return total_bytes


# ───────────────────────────────────────────────────────────────────
# 2. Peak intermediate memory during computation
# ───────────────────────────────────────────────────────────────────
def analytical_peak(I, L, c_s=C_S, c_z=C_Z, c_atom=C_ATOM, c_atompair=C_ATOMPAIR, n_freqs=N_FREQS):
    """
    Estimate peak GPU memory DURING TokenInitializer computation.

    The peak occurs in SinusoidalDistEmbed.forward() (MotifPosEmbed)
    when multiple [L, L, ...] intermediate tensors exist simultaneously.
    """
    print(f"\n{'='*70}")
    print(f"  2. PEAK INTERMEDIATE MEMORY  (I={I} tokens, L={L} atoms)")
    print(f"     Temporary tensors alive during MotifPosEmbed computation")
    print(f"{'='*70}\n")

    # ── SinusoidalDistEmbed.forward() intermediates ──
    # These tensors co-exist during the forward pass:
    intermediates = {
        # Step: D_LL = pos.unsqueeze(-2) - pos.unsqueeze(-3)
        "D_LL (distance vectors)":   (L, L, 3),
        # Step: dist_matrix = torch.linalg.norm(D_LL, dim=-1)
        "dist_matrix":               (L, L),
        # Step: angles = dist_matrix.unsqueeze(-1) * freq
        "angles":                    (L, L, n_freqs),
        # Step: sin_embed, cos_embed (before cat)
        "sin_embed":                 (L, L, n_freqs),
        "cos_embed":                 (L, L, n_freqs),
        # Step: sincos_embed = cat([sin, cos]) — replaces sin+cos
        "sincos_embed":              (L, L, 2 * n_freqs),
        # Step: P_LL = output_proj(sincos_embed)
        "P_LL (building)":           (L, L, c_atompair),
        # valid_mask stays alive throughout
        "valid_mask":                (L, L, 1),
    }

    print(f"  {'Tensor':<30} {'Shape':<20} {'Memory':>12}")
    print(f"  {'-'*30} {'-'*20} {'-'*12}")

    total = 0
    for name, shape in intermediates.items():
        mem = _tensor_bytes(*shape)
        total += mem
        print(f"  {name:<30} {str(list(shape)):<20} {format_bytes(mem):>12}")

    print(f"  {'-'*30} {'-'*20} {'-'*12}")

    # Peak: not all alive at same time. The worst moment is when
    # sincos_embed [L,L,64] + P_LL [L,L,16] + D_LL [L,L,3] + dist [L,L]
    # + valid_mask [L,L,1] co-exist (sin/cos freed after cat)
    peak_tensors = {
        "sincos_embed":   _tensor_bytes(L, L, 2 * n_freqs),
        "P_LL (output)":  _tensor_bytes(L, L, c_atompair),
        "D_LL":           _tensor_bytes(L, L, 3),
        "dist_matrix":    _tensor_bytes(L, L),
        "valid_mask":     _tensor_bytes(L, L, 1),
    }
    peak = sum(peak_tensors.values())

    # Also: PositionPairDistEmbedder adds another [L,L,3] + [L,L,1] + [L,L,16]
    ref_pos_extra = _tensor_bytes(L, L, 3) + _tensor_bytes(L, L, 1) + _tensor_bytes(L, L, c_atompair)

    # SingleProjections: C_L outer product [L,L,16] x2
    single_proj = _tensor_bytes(L, L, c_atompair) * 2

    # ZProjection: Z_II broadcast to [L,L,16]
    z_proj = _tensor_bytes(L, L, c_atompair)

    # pair_mlp: 3x (ReLU + Linear), each keeps [L,L,16]
    pair_mlp = _tensor_bytes(L, L, c_atompair) * 3

    print()
    print(f"  PEAK ESTIMATE (worst-case co-existing tensors):")
    print()
    print(f"    SinusoidalDistEmbed peak:    {format_bytes(peak):>12}")
    print(f"    + PositionPairDistEmbedder:  {format_bytes(ref_pos_extra):>12}")
    print(f"    + SingleProjections (x2):    {format_bytes(single_proj):>12}")
    print(f"    + ZProjection:               {format_bytes(z_proj):>12}")
    print(f"    + pair_mlp intermediates:     {format_bytes(pair_mlp):>12}")
    total_peak = peak + ref_pos_extra + single_proj + z_proj + pair_mlp
    print(f"    {'─'*40}")
    print(f"    TOTAL PEAK (intermediates):   {format_bytes(total_peak):>12}")
    print()

    # Compare with output
    output_mem = _tensor_bytes(L, L, c_atompair) + _tensor_bytes(I, I, c_z) + \
                 _tensor_bytes(L, c_atom) * 2 + _tensor_bytes(I, c_s)
    print(f"    For comparison:")
    print(f"      Final outputs retained:    {format_bytes(output_mem):>12}")
    print(f"      Peak / Output ratio:       {total_peak / output_mem:>11.1f}x")
    print(f"      Peak is {format_bytes(total_peak - output_mem)} MORE than outputs")
    print()

    return total_peak


# ───────────────────────────────────────────────────────────────────
# 3. Parse Nsight memory log
# ───────────────────────────────────────────────────────────────────
def parse_nsight_log(log_path):
    """Parse and display memory breakdown from RFD3 memory log JSON."""
    with open(log_path) as f:
        data = json.load(f)

    print(f"\n{'='*70}")
    print(f"  3. NSIGHT MEMORY LOG: {os.path.basename(log_path)}")
    print(f"{'='*70}\n")

    # Find TokenInitializer baseline
    baseline = None
    for entry in data:
        if entry["range"] == "RFD3/TokenInitializer" and entry["phase"] == "enter":
            baseline = entry["allocated_mb"]
            break

    if baseline is None:
        print("  No RFD3/TokenInitializer entry found in log.")
        return

    print(f"  Baseline (model weights): {baseline:.1f} MB\n")
    print(f"  {'Phase':<48} {'Alloc':>10} {'Peak':>10}")
    print(f"  {'-'*48} {'-'*10} {'-'*10}")

    # Track the overall peak
    overall_peak = 0
    seen = set()

    for entry in data:
        name = entry.get("range", "")
        if "TokenInit" not in name:
            continue
        if entry["phase"] != "exit":
            continue

        short = name.replace("RFD3/TokenInitializer/", "").replace("RFD3/TokenInitializer", ">>> TOTAL")

        # Skip Pairformer sub-sub-steps for readability
        if ("Pairformer/Block_0/" in name or "Pairformer/Block_1/" in name):
            continue

        # Deduplicate (activation checkpointing causes double exit)
        key = short + str(entry["allocated_mb"])
        if key in seen:
            continue
        seen.add(key)

        alloc_delta = entry["allocated_mb"] - baseline
        peak_delta = entry["peak_mb"] - baseline
        overall_peak = max(overall_peak, peak_delta)

        marker = ""
        if peak_delta > 1000:
            marker = " <<<< SPIKE"

        print(f"  {short:<48} {alloc_delta:>+9.1f}M {peak_delta:>+9.1f}M{marker}")

    print()
    print(f"  PEAK MEMORY SPIKE: +{overall_peak:.0f} MB above baseline")
    print(f"  TOTAL at peak:     {baseline + overall_peak:.0f} MB")
    print()
    print(f"  The spike happens in MotifPosEmbed (SinusoidalDistEmbed):")
    print(f"  - sincos_embed [L, L, 64] is the largest temporary tensor")
    print(f"  - After projection to [L, L, 16], the temps are freed")
    print()



# ───────────────────────────────────────────────────────────────────
# 4. Scaling table
# ───────────────────────────────────────────────────────────────────
def scaling_analysis():
    """Show how memory scales with protein size — outputs AND peak."""
    print(f"\n{'='*70}")
    print(f"  4. SCALING ANALYSIS: Output vs Peak Memory")
    print(f"{'='*70}\n")
    print(f"  {'I':>5} {'L':>6} {'Outputs':>12} {'Peak':>12} {'Peak/Out':>10} {'Bottleneck':>12}")
    print(f"  {'-'*5} {'-'*6} {'-'*12} {'-'*12} {'-'*10} {'-'*12}")

    for I, L in [(50, 700), (100, 1400), (150, 2100), (200, 2800), (300, 4200), (500, 7000)]:
        # Outputs
        p_ll = _tensor_bytes(L, L, C_ATOMPAIR)
        z_ii = _tensor_bytes(I, I, C_Z)
        rest = _tensor_bytes(L, C_ATOM) * 2 + _tensor_bytes(I, C_S)
        output_total = p_ll + z_ii + rest

        # Peak intermediates (SinusoidalDistEmbed)
        sincos = _tensor_bytes(L, L, 2 * N_FREQS)  # [L,L,64]
        d_ll = _tensor_bytes(L, L, 3)
        dist = _tensor_bytes(L, L)
        mask = _tensor_bytes(L, L, 1)
        ref_extra = _tensor_bytes(L, L, 3) + _tensor_bytes(L, L, 1) + _tensor_bytes(L, L, C_ATOMPAIR)
        single_proj = _tensor_bytes(L, L, C_ATOMPAIR) * 2
        z_proj = _tensor_bytes(L, L, C_ATOMPAIR)
        mlp_inter = _tensor_bytes(L, L, C_ATOMPAIR) * 3
        peak_total = sincos + p_ll + d_ll + dist + mask + ref_extra + single_proj + z_proj + mlp_inter

        ratio = peak_total / output_total
        print(f"  {I:>5} {L:>6} {format_bytes(output_total):>12} {format_bytes(peak_total):>12} {ratio:>9.1f}x {'sincos_emb':>12}")

    print()
    print("  'Outputs'  = tensors retained after forward pass (P_LL + Z_II + Q_L + C_L + S_I)")
    print("  'Peak'     = max temporary memory during SinusoidalDistEmbed computation")
    print("  'Peak/Out' = how much MORE memory is needed during computation vs just storing results")
    print()
    print("  The sincos_embed [L,L,64] tensor is 4x larger than P_LL [L,L,16].")
    print("  This is why Nsight shows ~5 GB even though final outputs are only ~126 MB.")
    print()


# ───────────────────────────────────────────────────────────────────
# 5. Live GPU measurement
# ───────────────────────────────────────────────────────────────────
def _build_synthetic_features(I, L, device):
    """Build a synthetic feature dict with the right shapes/dtypes for TokenInitializer."""
    import torch

    # atom_to_token_map: each token has ~L/I atoms
    atoms_per_token = L // I
    tok_idx = torch.arange(I, device=device).repeat_interleave(atoms_per_token)
    # Adjust length to exactly L
    if len(tok_idx) < L:
        tok_idx = torch.cat([tok_idx, tok_idx.new_full((L - len(tok_idx),), I - 1)])
    tok_idx = tok_idx[:L]

    f = {
        "atom_to_token_map": tok_idx,
        "restype": torch.randint(0, 20, (I,), device=device),
        "ref_atom_name_chars": torch.randint(0, 2, (L, 4, 64), device=device),
        "token_bonds": torch.zeros(I, I, dtype=torch.bool, device=device),
        "ref_space_uid": tok_idx.clone(),
        "is_ca": torch.zeros(L, dtype=torch.bool, device=device),
        "ref_pos": torch.randn(L, 3, device=device),
        "motif_pos": torch.randn(L, 3, device=device),
        "asym_id": torch.zeros(I, dtype=torch.long, device=device),
        "entity_id": torch.zeros(I, dtype=torch.long, device=device),
        "residue_index": torch.arange(I, device=device),
        "token_index": torch.arange(I, device=device),
        "sym_id": torch.zeros(I, dtype=torch.long, device=device),
        "unindexing_pair_mask": torch.zeros(I, I, dtype=torch.bool, device=device),
        "is_motif_atom_with_fixed_coord": torch.ones(L, dtype=torch.bool, device=device),
        "is_motif_atom_with_fixed_seq": torch.ones(L, dtype=torch.bool, device=device),
        # 1D token features (from config: token_1d_features)
        "ref_motif_token_type": torch.zeros(I, 3, device=device),
        "restype": torch.nn.functional.one_hot(
            torch.randint(0, 20, (I,), device=device), 32
        ).float(),
        "ref_plddt": torch.ones(I, 1, device=device),
        "is_non_loopy": torch.ones(I, 1, device=device),
        # 1D atom features (from config: atom_1d_features)
        "ref_element": torch.nn.functional.one_hot(
            torch.randint(0, 4, (L,), device=device), 128
        ).float(),
        "ref_charge": torch.zeros(L, 1, device=device),
        "ref_mask": torch.ones(L, 1, device=device),
        "ref_is_motif_atom_with_fixed_coord": torch.ones(L, 1, device=device),
        "ref_is_motif_atom_unindexed": torch.zeros(L, 1, device=device),
        "has_zero_occupancy": torch.zeros(L, 1, device=device),
        "ref_atomwise_rasa": torch.zeros(L, 3, device=device),
        "active_donor": torch.zeros(L, 1, device=device),
        "active_acceptor": torch.zeros(L, 1, device=device),
        "is_atom_level_hotspot": torch.zeros(L, 1, device=device),
    }
    # Mark one atom per token as CA
    ca_indices = torch.arange(0, L, atoms_per_token, device=device)[:I]
    f["is_ca"][ca_indices] = True

    return f


def measure_live(n_tokens=None, n_atoms=None):
    """
    Actually run TokenInitializer.forward() on GPU and capture memory
    at each computation phase using the built-in trace_range hooks.

    Uses synthetic features (no test data files needed).
    """
    import torch
    from omegaconf import OmegaConf

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. Live measurement requires a GPU.")
        return

    # Enable memory tracking via the built-in trace_range mechanism
    os.environ["RFD3_TRACK_MEMORY"] = "1"
    os.environ["RFD3_TRACE_NVTX"] = "1"

    from rfd3.model.layers.encoders import TokenInitializer
    from rfd3.utils.tracing import get_memory_log, _memory_log, _memory_log_lock

    I = n_tokens or 100
    L = n_atoms or I * 14  # ~14 atoms per token (standard protein)
    device = torch.device("cuda")

    print(f"\n{'='*70}")
    print(f"  5. LIVE GPU MEMORY MEASUREMENT")
    print(f"     Running TokenInitializer.forward() on GPU")
    print(f"{'='*70}")
    print(f"\n  Protein size: I={I} tokens, L={L} atoms")

    # ── Token initializer config (matches rfd3_net.yaml) ──
    ti_cfg = {
        "relative_position_encoding": {"r_max": 32, "s_max": 2},
        "n_pairformer_blocks": 2,
        "pairformer_block": {
            "use_triangle_attn": False,
            "use_triangle_mult": False,
            "attention_pair_bias": {"n_head": 16, "kq_norm": True},
        },
        "token_1d_features": {
            "ref_motif_token_type": 3,
            "restype": 32,
            "ref_plddt": 1,
            "is_non_loopy": 1,
        },
        "downcast": {
            "method": "cross_attention",
            "cross_attention_block": {
                "n_head": 4,
                "c_model": 128,
                "dropout": 0.0,
                "kq_norm": True,
            },
        },
        "atom_1d_features": {
            "ref_atom_name_chars": 256,
            "ref_element": 128,
            "ref_charge": 1,
            "ref_mask": 1,
            "ref_is_motif_atom_with_fixed_coord": 1,
            "ref_is_motif_atom_unindexed": 1,
            "has_zero_occupancy": 1,
            "ref_pos": 3,
            "ref_atomwise_rasa": 3,
            "active_donor": 1,
            "active_acceptor": 1,
            "is_atom_level_hotspot": 1,
        },
        "atom_transformer": {
            "n_blocks": 0,
            "atom_transformer_block": {
                "n_head": 4,
                "kq_norm": True,
                "no_residual_connection_between_attention_and_transition": False,
                "dropout": 0.0,
                "n_attn_seq_neighbours": 4,
                "n_attn_keys": 128,
            },
        },
    }

    # ── Instantiate TokenInitializer with random weights ──
    print("  Instantiating TokenInitializer (random weights)...")
    token_init = TokenInitializer(
        c_s=C_S, c_z=C_Z, c_atom=C_ATOM, c_atompair=C_ATOMPAIR,
        **ti_cfg,
    ).to(device)

    # ── Build synthetic features ──
    print(f"  Building synthetic features...")
    f = _build_synthetic_features(I, L, device)

    # ── Clear memory log and reset stats ──
    with _memory_log_lock:
        _memory_log.clear()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    baseline_bytes = torch.cuda.memory_allocated()
    print(f"  Baseline memory (model weights + features): {format_bytes(baseline_bytes)}")
    print(f"\n  Running forward pass...\n")

    # ── Run forward ──
    with torch.no_grad():
        result = token_init(f)

    torch.cuda.synchronize()
    post_bytes = torch.cuda.memory_allocated()
    peak_bytes = torch.cuda.max_memory_allocated()

    # ── Read back memory log ──
    log = get_memory_log()

    # Display per-phase breakdown
    baseline_mb = baseline_bytes / 1024**2

    print(f"  {'Phase':<52} {'Alloc':>10} {'Peak':>10}")
    print(f"  {'-'*52} {'-'*10} {'-'*10}")

    overall_peak_delta = 0
    for entry in log:
        if entry["phase"] != "exit":
            continue
        name = entry["range"]
        # Skip non-TokenInitializer ranges (e.g. from sub-layers)
        if "TokenInitializer" not in name and "RFD3/" not in name:
            continue

        short = name.replace("RFD3/TokenInitializer/", "").replace("RFD3/TokenInitializer", ">>> TOTAL")

        # Skip Pairformer sub-sub-steps for readability
        if "Pairformer/Block_0/" in name or "Pairformer/Block_1/" in name:
            continue

        alloc_delta = entry["allocated_mb"] - baseline_mb
        peak_delta = entry["peak_mb"] - baseline_mb
        overall_peak_delta = max(overall_peak_delta, peak_delta)

        marker = ""
        if peak_delta > 100:
            marker = " <<<< SPIKE"

        print(f"  {short:<52} {alloc_delta:>+9.1f}M {peak_delta:>+9.1f}M{marker}")

    print()
    print(f"  {'─'*72}")
    print(f"  BASELINE (model weights):  {format_bytes(baseline_bytes):>12}")
    print(f"  POST-FORWARD (retained):   {format_bytes(post_bytes):>12}  (+{format_bytes(post_bytes - baseline_bytes)})")
    print(f"  PEAK (during forward):     {format_bytes(peak_bytes):>12}  (+{format_bytes(peak_bytes - baseline_bytes)})")
    print()

    # ── Output tensor sizes ──
    print(f"  Output tensor breakdown:")
    total_out = 0
    for name, tensor in result.items():
        if isinstance(tensor, torch.Tensor):
            mem = tensor.element_size() * tensor.nelement()
            total_out += mem
            print(f"    {name:<12} {str(list(tensor.shape)):<25} {format_bytes(mem):>12}")
    print(f"    {'TOTAL':<12} {'':<25} {format_bytes(total_out):>12}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Measure TokenInitializer memory: outputs, peak intermediates, Nsight logs, and live GPU",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 scripts/measure_tokeninit_memory.py                           # All analytical
  python3 scripts/measure_tokeninit_memory.py --n-tokens 200 --n-atoms 2800   # Custom size
  python3 scripts/measure_tokeninit_memory.py --nsight-log logs/inference_outs/memory_tracked/0/rfd3_memory_log.json
  python3 scripts/measure_tokeninit_memory.py --live                    # Run on GPU (I=100, L=1400)
  python3 scripts/measure_tokeninit_memory.py --live --n-tokens 200 --n-atoms 2800  # Custom size
        """,
    )
    parser.add_argument("--n-tokens", type=int, default=100, help="Number of tokens/residues I (default: 100)")
    parser.add_argument("--n-atoms", type=int, default=1400, help="Number of atoms L (default: 1400)")
    parser.add_argument("--nsight-log", type=str, default=None, help="Path to rfd3_memory_log.json from a profiling run")
    parser.add_argument("--outputs-only", action="store_true", help="Show only output tensor sizes")
    parser.add_argument("--peak-only", action="store_true", help="Show only peak intermediate analysis")
    parser.add_argument("--scaling-only", action="store_true", help="Show only scaling table")
    parser.add_argument("--live", action="store_true", help="Run TokenInitializer on GPU and measure real memory")
    args = parser.parse_args()

    if args.live:
        measure_live(n_tokens=args.n_tokens, n_atoms=args.n_atoms)
        return

    show_all = not any([args.outputs_only, args.peak_only, args.scaling_only, args.nsight_log])

    if show_all or args.outputs_only:
        analytical_outputs(I=args.n_tokens, L=args.n_atoms)

    if show_all or args.peak_only:
        analytical_peak(I=args.n_tokens, L=args.n_atoms)

    if show_all or args.scaling_only:
        scaling_analysis()

    if args.nsight_log:
        parse_nsight_log(args.nsight_log)


if __name__ == "__main__":
    main()
