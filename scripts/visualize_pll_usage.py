"""
Visualize why low-memory mode uses less memory than high-memory mode.

High-memory: computes full L×L P_LL matrix (all pairs)
Low-memory:  computes P_LL only for the k neighbors each atom attends to

Usage:
  Step 1: Capture real data from inference pipeline
    python scripts/visualize_pll_usage.py --capture

  Step 2: Visualize (uses captured data, or falls back to mock if not found)
    python scripts/visualize_pll_usage.py
"""

import argparse
import sys
sys.path.insert(0, "/root/foundry/models/rfd3/src")
sys.path.insert(0, "/root/foundry/src")

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

CAPTURE_PATH = "/root/foundry/logs/pll_captured_data.pt"


def capture_from_inference():
    """Run one inference step and capture the real attention indices + positions."""
    from rfd3.model.layers import block_utils

    captured = {}
    _original_fn = block_utils.create_attention_indices

    def _hook(f, n_attn_keys, n_attn_seq_neighbours, X_L=None, tok_idx=None):
        result = _original_fn(f, n_attn_keys, n_attn_seq_neighbours, X_L=X_L, tok_idx=tok_idx)
        if not captured:  # Only capture the first call
            tok_idx_actual = f["atom_to_token_map"] if tok_idx is None else tok_idx
            captured["indices"] = result.detach().cpu()
            captured["tok_idx"] = tok_idx_actual.detach().cpu()
            if X_L is not None:
                captured["X_L"] = X_L.detach().cpu()
            captured["n_attn_keys"] = n_attn_keys
            captured["n_attn_seq_neighbours"] = n_attn_seq_neighbours
            L = len(tok_idx_actual)
            captured["L"] = L
            print(f"\n[CAPTURE] Got real data: L={L}, indices={result.shape}, "
                  f"k={n_attn_keys}, n_seq={n_attn_seq_neighbours}")
        return result

    # Monkey-patch
    block_utils.create_attention_indices = _hook
    # Also patch the imported reference in RFD3_diffusion_module
    from rfd3.model import RFD3_diffusion_module
    RFD3_diffusion_module.create_attention_indices = _hook

    # Run inference using the same approach as rfd3 CLI
    from hydra import compose, initialize_config_dir
    from pathlib import Path
    config_path = str(Path("/root/foundry/models/rfd3/configs"))
    overrides = [
        "inputs=models/rfd3/docs/examples/common_simulate.json",
        "out_dir=logs/inference_outs/pll_capture",
        "diffusion_batch_size=1",
        "n_batches=1",
        "skip_existing=False",
        "dump_trajectories=False",
        "prevalidate_inputs=False",
    ]

    print("Running RFD3 inference to capture real attention indices...")
    with initialize_config_dir(config_dir=config_path, version_base="1.3"):
        cfg = compose(config_name="inference", overrides=overrides)
        from foundry.utils.logging import suppress_warnings
        from rfd3.run_inference import run_inference
        with suppress_warnings(is_inference=True):
            run_inference(cfg)

    if not captured:
        raise RuntimeError("Hook never fired — create_attention_indices was not called")

    torch.save(captured, CAPTURE_PATH)
    print(f"\n[CAPTURE] Saved real data to {CAPTURE_PATH}")
    print(f"  L={captured['L']}, indices shape={captured['indices'].shape}")
    return captured


def load_captured_data():
    """Load previously captured data, or return None."""
    import os
    if os.path.exists(CAPTURE_PATH):
        data = torch.load(CAPTURE_PATH, map_location="cpu", weights_only=True)
        print(f"Loaded captured data: L={data['L']}, indices={data['indices'].shape}")
        return data
    return None


def classify_neighbors(indices, tok_idx):
    """Classify neighbors as window vs KNN using actual token distances."""
    L, k = indices.shape[-2], indices.shape[-1]
    if indices.ndim == 3:
        indices = indices.squeeze(0)

    query_tok = tok_idx.unsqueeze(1).expand(L, k)  # [L, k]
    neighbor_tok = tok_idx[indices]  # [L, k]
    tok_diff = (query_tok - neighbor_tok).abs()

    # Window = within n_seq_neighbours tokens (token-level, not atom-level)
    # Default n_seq_neighbours=2, so window threshold = 2 tokens
    n_seq = 2
    is_window = tok_diff <= n_seq

    return is_window


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", action="store_true",
                        help="Run inference to capture real data before visualizing")
    args = parser.parse_args()

    if args.capture:
        captured = capture_from_inference()
    else:
        captured = load_captured_data()

    if captured is not None:
        indices = captured["indices"]
        if indices.ndim == 3:
            indices = indices.squeeze(0)
        tok_idx = captured["tok_idx"]
        L = captured["L"]
        k = indices.shape[-1]
        is_window = classify_neighbors(indices, tok_idx)
        source = "real inference"
    else:
        print("No captured data found. Run with --capture first.")
        return

    print(f"Data source: {source}")
    print(f"Indices shape: {indices.shape}")
    
    # --- Statistics ---
    total_dense = L * L
    total_sparse = L * k
    usage_pct = total_sparse / total_dense * 100
    waste_pct = 100 - usage_pct
    
    dense_memory_mb = total_dense * 128 * 2 / (1024**2)   # c_atompair=128, bf16
    sparse_memory_mb = total_sparse * 128 * 2 / (1024**2)
    
    print(f"\nDense P_LL:  {L}×{L} = {total_dense:,} pairs  ({dense_memory_mb:.1f} MB at c=128, bf16)")
    print(f"Sparse P_LL: {L}×{k} = {total_sparse:,} pairs ({sparse_memory_mb:.1f} MB at c=128, bf16)")
    print(f"Usage: {usage_pct:.1f}% | Waste: {waste_pct:.1f}%")
    
    n_window = is_window.sum().item()
    n_knn = (~is_window).sum().item()
    window_pct = n_window / (L * k) * 100
    print(f"Window neighbors: {window_pct:.0f}% | KNN neighbors: {100-window_pct:.0f}%")
    
    ratio = dense_memory_mb / sparse_memory_mb

    # === FIGURE: Two-row layout ===
    # Row 1: Flow diagrams explaining WHY
    # Row 2: Heatmaps showing the result
    fig = plt.figure(figsize=(16, 12))
    
    # --- Row 1: Flow diagrams ---
    ax_flow = fig.add_axes([0.05, 0.55, 0.90, 0.40])  # [left, bottom, width, height]
    ax_flow.set_xlim(0, 10)
    ax_flow.set_ylim(0, 5)
    ax_flow.axis("off")
    
    # Title
    ax_flow.text(5, 4.8, "Why does high memory mode compute pairs it never uses?",
                 ha="center", va="top", fontsize=16, fontweight="bold")

    # --- HIGH MEMORY flow (top half) ---
    y_high = 3.5
    
    # Box 1: Pre-compute
    ax_flow.add_patch(plt.Rectangle((0.3, y_high-0.4), 2.2, 0.8, 
                      facecolor="#FFEBEE", edgecolor="#D32F2F", linewidth=2, zorder=2))
    ax_flow.text(1.4, y_high, "Step 0: Pre-compute\nfull L×L P_LL", 
                 ha="center", va="center", fontsize=9, fontweight="bold", zorder=3)
    
    # Arrow
    ax_flow.annotate("", xy=(3.3, y_high), xytext=(2.6, y_high),
                     arrowprops=dict(arrowstyle="->", lw=2, color="#333"))
    
    # Box 2: Diffusion loop
    ax_flow.add_patch(plt.Rectangle((3.3, y_high-0.4), 2.6, 0.8,
                      facecolor="#FFF3E0", edgecolor="#E65100", linewidth=2, 
                      linestyle="--", zorder=2))
    ax_flow.text(4.6, y_high+0.15, "Steps 1-200: each step", 
                 ha="center", va="center", fontsize=9, fontweight="bold", zorder=3)
    ax_flow.text(4.6, y_high-0.15, "new indices from X_noisy\n(positions change!)", 
                 ha="center", va="center", fontsize=8, zorder=3)
    
    # Arrow
    ax_flow.annotate("", xy=(6.7, y_high), xytext=(6.0, y_high),
                     arrowprops=dict(arrowstyle="->", lw=2, color="#333"))
    
    # Box 3: Lookup
    ax_flow.add_patch(plt.Rectangle((6.7, y_high-0.4), 2.8, 0.8,
                      facecolor="#FFCDD2", edgecolor="#D32F2F", linewidth=2, zorder=2))
    ax_flow.text(8.1, y_high+0.15, "Look up k pairs from", 
                 ha="center", va="center", fontsize=9, fontweight="bold", zorder=3)
    ax_flow.text(8.1, y_high-0.15, "pre-computed L×L → most wasted", 
                 ha="center", va="center", fontsize=9, color="#D32F2F", fontweight="bold", zorder=3)
    
    # Label
    ax_flow.text(0.15, y_high, "HIGH\nMEM", ha="center", va="center", 
                 fontsize=10, fontweight="bold", color="#D32F2F")

    # --- LOW MEMORY flow (bottom half) ---
    y_low = 1.8
    
    # Box 1: No pre-compute
    ax_flow.add_patch(plt.Rectangle((0.3, y_low-0.4), 2.2, 0.8,
                      facecolor="#E8F5E9", edgecolor="#2E7D32", linewidth=2, zorder=2))
    ax_flow.text(1.4, y_low, "Step 0: Cache only\nstatic MLP weights\n(tiny)", 
                 ha="center", va="center", fontsize=9, fontweight="bold", zorder=3)
    
    # Arrow
    ax_flow.annotate("", xy=(3.3, y_low), xytext=(2.6, y_low),
                     arrowprops=dict(arrowstyle="->", lw=2, color="#333"))
    
    # Box 2: Create indices
    ax_flow.add_patch(plt.Rectangle((3.3, y_low-0.4), 2.6, 0.8,
                      facecolor="#E8F5E9", edgecolor="#2E7D32", linewidth=2,
                      linestyle="--", zorder=2))
    ax_flow.text(4.6, y_low+0.15, "Steps 1-200: each step", 
                 ha="center", va="center", fontsize=9, fontweight="bold", zorder=3)
    ax_flow.text(4.6, y_low-0.15, "create indices from X_noisy", 
                 ha="center", va="center", fontsize=8, zorder=3)
    
    # Arrow
    ax_flow.annotate("", xy=(6.7, y_low), xytext=(6.0, y_low),
                     arrowprops=dict(arrowstyle="->", lw=2, color="#333"))
    
    # Box 3: Compute only needed
    ax_flow.add_patch(plt.Rectangle((6.7, y_low-0.4), 2.8, 0.8,
                      facecolor="#C8E6C9", edgecolor="#2E7D32", linewidth=2, zorder=2))
    ax_flow.text(8.1, y_low+0.15, "Compute P_LL only for", 
                 ha="center", va="center", fontsize=9, fontweight="bold", zorder=3)
    ax_flow.text(8.1, y_low-0.15, "k needed pairs → all used", 
                 ha="center", va="center", fontsize=9, color="#2E7D32", fontweight="bold", zorder=3)
    
    # Label
    ax_flow.text(0.15, y_low, "LOW\nMEM", ha="center", va="center", 
                 fontsize=10, fontweight="bold", color="#2E7D32")
    
    # Key insight annotation
    ax_flow.text(5, 0.6, 
                 "Key: Attention indices change every step because atom positions evolve during diffusion.\n"
                 "High memory must pre-compute all L×L pairs since it cannot know future indices.\n"
                 "Low memory computes P_LL after indices are known, so it only computes what is needed.",
                 ha="center", va="center", fontsize=10, style="italic",
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="#F5F5F5", 
                           edgecolor="#999", alpha=0.9))

    # --- Row 2: Heatmaps ---
    # For visualization, downsample if L is large
    L_viz = min(L, 200)
    if L_viz < L:
        step = L // L_viz
        indices_viz = indices[::step]
        is_window_viz = is_window[::step]
        L_viz = indices_viz.shape[0]
    else:
        indices_viz = indices
        is_window_viz = is_window
        L_viz = L

    from matplotlib.colors import ListedColormap
    # Teal for window, bright magenta for KNN — maximally contrasting
    cmap = ListedColormap(["#F0F0F0", "#4CAF50", "#FF9800"])  # 0=unused, 1=KNN(green), 2=window(orange)
    
    ax1 = fig.add_axes([0.05, 0.06, 0.40, 0.42])
    ax2 = fig.add_axes([0.52, 0.06, 0.46, 0.42])
    
    # --- Panel 1: Full L×L matrix (downsampled to L_viz×L_viz) ---
    L_col = L_viz  # downsample columns to match rows
    full_map = np.zeros((L_viz, L_col), dtype=np.uint8)
    for i in range(L_viz):
        for j_idx in range(k):
            j = indices_viz[i, j_idx].item()
            j_ds = j // step if L_viz < L else j
            if 0 <= j_ds < L_col:
                val = 2 if is_window_viz[i, j_idx] else 1
                if full_map[i, j_ds] < val:  # window over KNN over unused
                    full_map[i, j_ds] = val
    
    ax1.imshow(full_map, cmap=cmap, aspect="auto", interpolation="nearest", vmin=0, vmax=2)
    ax1.set_title("High Memory: full L×L P_LL", 
                  fontsize=12, fontweight="bold", color="#D32F2F", pad=8)
    ax1.set_xlabel("Key atom index", fontsize=10)
    ax1.set_ylabel("Query atom index", fontsize=10)
    
    patches1 = [
        mpatches.Patch(color="#F0F0F0", label="Computed but unused"),
        mpatches.Patch(color="#FF9800", label="Window neighbors"),
        mpatches.Patch(color="#4CAF50", label="KNN neighbors"),
    ]
    ax1.legend(handles=patches1, loc="lower right", fontsize=7, framealpha=0.9)
    
    # --- Panel 2: Sparse [L, k] — zoomed to a representative block ---
    # Find a region where KNN proportion is close to average
    n_knn_per_row = (~is_window).sum(dim=1)  # [L] full resolution
    avg_knn = n_knn_per_row.float().mean().item()
    best_start = L // 2
    best_diff = float("inf")
    block_size = 80
    for start in range(L // 4, 3 * L // 4):
        if start + block_size > L:
            break
        block_avg = n_knn_per_row[start:start+block_size].float().mean().item()
        diff = abs(block_avg - avg_knn)
        if diff < best_diff:
            best_diff = diff
            best_start = start
    
    # Extract block at full resolution (no downsampling)
    zoom_is_window = is_window[best_start:best_start+block_size]
    
    # Sort within each row: window first, then KNN
    zoom_sort_key = (~zoom_is_window).int()
    zoom_sorted_order = zoom_sort_key.argsort(dim=1, stable=True)
    zoom_is_window_sorted = zoom_is_window.gather(1, zoom_sorted_order)
    
    zoom_map = np.zeros((block_size, k), dtype=np.uint8)
    for i in range(block_size):
        for j_idx in range(k):
            zoom_map[i, j_idx] = 2 if zoom_is_window_sorted[i, j_idx] else 1
    
    ax2.imshow(zoom_map, cmap=cmap, aspect="auto", interpolation="none", vmin=0, vmax=2)
    ax2.set_title("Low Memory: sparse L×k P_LL (representative atoms)", 
                  fontsize=12, fontweight="bold", color="#2E7D32", pad=8)
    ax2.set_xlabel("Neighbor index (k)", fontsize=10)
    ax2.set_ylabel("Query atom index", fontsize=10)
    
    # Draw boundary line where window ends and KNN begins
    zoom_n_window = zoom_is_window_sorted.sum(dim=1).numpy()
    ax2.plot(zoom_n_window, np.arange(block_size), color="black", linewidth=2, alpha=0.8)
    
    patches2 = [
        mpatches.Patch(color="#FF9800", label="Window neighbors"),
        mpatches.Patch(color="#4CAF50", label="KNN neighbors"),
    ]
    ax2.legend(handles=patches2, loc="lower right", fontsize=7, framealpha=0.9)
    
    # Bottom annotation
    fig.text(0.5, 0.01,
             "From real inference (common_simulate)  |  "
             "Only a small fraction of dense P_LL is used by attention",
             ha="center", fontsize=9, style="italic", color="#555")
    
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    
    out_path = "/root/foundry/logs/pll_high_vs_low_memory.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"\nSaved: {out_path}")
    plt.close()
    
    # Print summary for slide
    print(f"\n{'='*60}")
    print(f"SUMMARY FOR SLIDE (from {source})")
    print(f"{'='*60}")
    print(f"  L={L}, k={k}")
    print(f"  Dense P_LL:  {L}×{L} = {total_dense:,} pairs = {dense_memory_mb:.0f} MB")
    print(f"  Sparse P_LL: {L}×{k} = {total_sparse:,} pairs = {sparse_memory_mb:.0f} MB")
    print(f"  Memory ratio: {ratio:.0f}×")
    print(f"  Attention uses only {usage_pct:.1f}% of the dense matrix")
    print(f"  {waste_pct:.1f}% of dense P_LL computation is wasted")
    print(f"  Window neighbors: {window_pct:.0f}% | KNN neighbors: {100-window_pct:.0f}%")


if __name__ == "__main__":
    main()
