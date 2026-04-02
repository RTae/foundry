"""
Measure and visualize the window vs KNN neighbor ratio per atom
from real captured inference data.

Usage:
  python scripts/measure_window_knn_ratio.py
"""

import sys
sys.path.insert(0, "/root/foundry/models/rfd3/src")
sys.path.insert(0, "/root/foundry/src")

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CAPTURE_PATH = "/root/foundry/logs/pll_captured_data.pt"


def main():
    data = torch.load(CAPTURE_PATH, map_location="cpu", weights_only=True)
    indices = data["indices"].squeeze(0)  # [L, k]
    tok_idx = data["tok_idx"]
    n_seq = int(data["n_attn_seq_neighbours"])
    L, k = indices.shape

    # Classify each neighbor as window or KNN
    query_tok = tok_idx.unsqueeze(1).expand(L, k)
    neighbor_tok = tok_idx[indices]
    tok_diff = (query_tok - neighbor_tok).abs()
    is_window = tok_diff <= n_seq

    n_window = is_window.sum(dim=1).float().numpy()  # [L]
    n_knn = k - n_window
    window_pct = n_window / k * 100
    knn_pct = n_knn / k * 100

    # Print stats
    print(f"L={L}, k={k}, n_seq_neighbours={n_seq}")
    print(f"Window: mean={window_pct.mean():.1f}%, median={np.median(window_pct):.1f}%, range={window_pct.min():.1f}-{window_pct.max():.1f}%")
    print(f"KNN:    mean={knn_pct.mean():.1f}%, median={np.median(knn_pct):.1f}%, range={knn_pct.min():.1f}-{knn_pct.max():.1f}%")

    # --- Figure: per-atom stacked bar showing window vs KNN counts ---
    fig, (ax_bar, ax_hist) = plt.subplots(1, 2, figsize=(14, 5),
                                           gridspec_kw={"width_ratios": [2, 1]})

    # Sort atoms by window count for a cleaner visual
    sort_idx = np.argsort(n_window)
    win_sorted = n_window[sort_idx]
    knn_sorted = n_knn[sort_idx]

    # Downsample to at most 200 atoms for readability
    max_bars = 200
    if L > max_bars:
        step = L // max_bars
        win_sorted = win_sorted[::step]
        knn_sorted = knn_sorted[::step]
    L_viz = len(win_sorted)
    x = np.arange(L_viz)

    # Panel 1: stacked bar — each atom shows its window (bottom) + KNN (top)
    ax_bar.bar(x, win_sorted, width=1.0, color="#FF9800", label="Window")
    ax_bar.bar(x, knn_sorted, width=1.0, bottom=win_sorted, color="#4CAF50", label="KNN")
    ax_bar.axhline(k, color="black", linewidth=0.5, alpha=0.3)

    mean_win = n_window.mean()
    ax_bar.axhline(mean_win, color="black", linestyle="--", linewidth=1.5,
                   label=f"Mean window = {mean_win:.0f} ({mean_win/k*100:.0f}%)")

    ax_bar.set_xlabel(f"Atoms (sorted by window count, {L_viz} of {L} shown)", fontsize=11)
    ax_bar.set_ylabel(f"Number of neighbors (k={k})", fontsize=11)
    ax_bar.set_title("Per atom neighbor composition", fontsize=13, fontweight="bold")
    ax_bar.legend(fontsize=9, loc="upper left")
    ax_bar.set_xlim(0, L_viz)
    ax_bar.set_ylim(0, k + 5)

    # Panel 2: histogram of window %
    mean_val = window_pct.mean()
    median_val = np.median(window_pct)

    bins = np.arange(0, 101, 5)
    ax_hist.hist(window_pct, bins=bins, color="#607D8B", edgecolor="white", linewidth=0.8)
    ax_hist.axvline(mean_val, color="black", linestyle="--", linewidth=2,
                    label=f"Mean: {mean_val:.1f}%")
    ax_hist.axvline(median_val, color="#999", linestyle=":", linewidth=2,
                    label=f"Median: {median_val:.1f}%")

    ax_hist.set_xlabel("Window %", fontsize=11)
    ax_hist.set_ylabel("Number of atoms", fontsize=11)
    ax_hist.set_title("Distribution of window %", fontsize=13, fontweight="bold")
    ax_hist.legend(fontsize=9)
    ax_hist.set_xlim(0, 100)

    fig.suptitle(f"Window vs KNN neighbor ratio  (L={L}, k={k}, n_seq_neighbours={n_seq})",
                 fontsize=14, fontweight="bold", y=1.02)

    plt.tight_layout()
    out_path = "/root/foundry/logs/window_knn_distribution.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"\nSaved: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
