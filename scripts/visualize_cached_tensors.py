#!/usr/bin/env python3
"""Visualize cached tensor data from a low_memory=False run.

Usage:
    # First, run inference with capture enabled:
    #   RFD3_CAPTURE_MATRIX_DIR=logs/rfd3_matrix_capture  <your inference command>
    #
    # Then visualize:
    python scripts/visualize_cached_tensors.py --capture-dir logs/rfd3_matrix_capture

This loads the .pt files saved by the matrix_capture utility and produces:
  1. A text summary of every tensor (shape, dtype, memory)
  2. Heatmap of P_LL pair-energy (channel-norm of atom-pair features)
  3. Spy plot of attention_indices sparsity pattern
  4. Per-tensor memory bar chart
"""

import argparse
import sys
from pathlib import Path

import torch

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
except ImportError:
    print("matplotlib is required: pip install matplotlib", file=sys.stderr)
    sys.exit(1)


def load_pt(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def print_summary(capture_dir: Path):
    """Print a structured summary of every captured .pt file."""
    pt_files = sorted(capture_dir.glob("*.pt"))
    if not pt_files:
        print(f"No .pt files found in {capture_dir}")
        return

    print("=" * 80)
    print(f"  Captured tensors in: {capture_dir}")
    print("=" * 80)

    for pt_file in pt_files:
        data = load_pt(pt_file)
        print(f"\n--- {pt_file.name} ---")

        # If there is a __summary__ key, use it
        if "__summary__" in data:
            summary = data["__summary__"]
            for key, info in summary.items():
                if "shape" in info:
                    print(
                        f"  {key:20s}  shape={info['shape']:<30s}  "
                        f"dtype={info['dtype']:<15s}  {info['size_mb']:>8.2f} MB"
                    )
                else:
                    print(f"  {key:20s}  type={info.get('type', '?')}")
        else:
            for key, val in data.items():
                if isinstance(val, torch.Tensor):
                    size_mb = val.numel() * val.element_size() / 1e6
                    print(
                        f"  {key:20s}  shape={str(list(val.shape)):<30s}  "
                        f"dtype={str(val.dtype):<15s}  {size_mb:>8.2f} MB"
                    )
                else:
                    print(f"  {key:20s}  {val}")
    print()


def plot_pair_energy(capture_dir: Path, out_dir: Path):
    """Heatmap of P_LL channel-norm (pair_energy)."""
    pw_path = capture_dir / "pairwise_initializer.pt"
    if not pw_path.exists():
        print(f"Skipping pair-energy plot ({pw_path} not found)")
        return

    data = load_pt(pw_path)
    pair_energy = data["pair_energy"].float().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Full heatmap
    im0 = axes[0].imshow(pair_energy, aspect="auto", cmap="viridis")
    axes[0].set_title("P_LL pair energy (channel norm)")
    axes[0].set_xlabel("atom j")
    axes[0].set_ylabel("atom i")
    fig.colorbar(im0, ax=axes[0], shrink=0.8)

    # Log-scale
    pe_pos = pair_energy.copy()
    pe_pos[pe_pos <= 0] = pe_pos[pe_pos > 0].min() if (pe_pos > 0).any() else 1e-6
    im1 = axes[1].imshow(pe_pos, aspect="auto", cmap="inferno", norm=LogNorm())
    axes[1].set_title("P_LL pair energy (log scale)")
    axes[1].set_xlabel("atom j")
    axes[1].set_ylabel("atom i")
    fig.colorbar(im1, ax=axes[1], shrink=0.8)

    # Non-zero mask
    nonzero = data["pair_nonzero_mask"].numpy().astype(float)
    axes[2].imshow(nonzero, aspect="auto", cmap="gray_r")
    axes[2].set_title(f"P_LL non-zero mask ({nonzero.sum():.0f}/{nonzero.size} entries)")
    axes[2].set_xlabel("atom j")
    axes[2].set_ylabel("atom i")

    fig.tight_layout()
    out_path = out_dir / "pll_pair_energy.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_attention_sparsity(capture_dir: Path, out_dir: Path):
    """Spy plot showing which keys each query attends to."""
    idx_path = capture_dir / "attention_indices.pt"
    if not idx_path.exists():
        print(f"Skipping attention sparsity plot ({idx_path} not found)")
        return

    data = load_pt(idx_path)
    indices = data["indices"]  # (D, L, k)
    if indices.ndim == 3:
        indices = indices[0]  # take first batch element -> (L, k)

    L, k = indices.shape
    print(f"Attention indices: L={L}, k={k} (each query attends to {k} keys)")

    # Build sparse boolean mask
    mask = torch.zeros(L, L, dtype=torch.bool)
    for i in range(L):
        mask[i, indices[i]] = True

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.spy(mask.numpy(), markersize=0.2, aspect="auto", color="steelblue")
    ax.set_title(f"Attention sparsity pattern  (L={L}, k={k})")
    ax.set_xlabel("key position")
    ax.set_ylabel("query position")

    out_path = out_dir / "attention_sparsity.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_memory_breakdown(capture_dir: Path, out_dir: Path):
    """Bar chart of memory used by each cached tensor."""
    init_path = capture_dir / "initializer_outputs.pt"
    if not init_path.exists():
        print(f"Skipping memory breakdown ({init_path} not found)")
        return

    data = load_pt(init_path)
    summary = data.get("__summary__", {})

    names = []
    sizes = []
    for key, info in summary.items():
        if "size_mb" in info:
            names.append(key)
            sizes.append(info["size_mb"])

    if not names:
        return

    # Sort by size descending
    order = sorted(range(len(sizes)), key=lambda i: sizes[i], reverse=True)
    names = [names[i] for i in order]
    sizes = [sizes[i] for i in order]

    fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.6)))
    colors = plt.cm.Set2([i / len(names) for i in range(len(names))])
    bars = ax.barh(names, sizes, color=colors)
    ax.set_xlabel("Memory (MB)")
    ax.set_title("Cached tensor memory breakdown (low_memory=False)")
    for bar, sz in zip(bars, sizes):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2, f"{sz:.1f} MB", va="center")
    ax.invert_yaxis()
    fig.tight_layout()

    out_path = out_dir / "memory_breakdown.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_initializer_tensor_stats(capture_dir: Path, out_dir: Path):
    """Distribution plots for each initializer tensor."""
    init_path = capture_dir / "initializer_outputs.pt"
    if not init_path.exists():
        print(f"Skipping tensor stats plot ({init_path} not found)")
        return

    data = load_pt(init_path)
    tensor_keys = [k for k in data if isinstance(data[k], torch.Tensor)]
    if not tensor_keys:
        return

    n = len(tensor_keys)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, key in zip(axes, tensor_keys):
        t = data[key].float().flatten()
        ax.hist(t.numpy(), bins=100, alpha=0.75, edgecolor="black", linewidth=0.3)
        ax.set_title(f"{key}\nshape={list(data[key].shape)}")
        ax.set_xlabel("value")
        ax.set_ylabel("count")
        ax.axvline(t.mean().item(), color="red", linestyle="--", label=f"mean={t.mean():.3f}")
        ax.axvline(t.median().item(), color="orange", linestyle="--", label=f"median={t.median():.3f}")
        ax.legend(fontsize=7)

    fig.suptitle("Initializer tensor value distributions", fontsize=14)
    fig.tight_layout()

    out_path = out_dir / "tensor_distributions.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--capture-dir",
        type=Path,
        default=Path("logs/rfd3_matrix_capture"),
        help="Directory containing captured .pt files",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Where to save plots (defaults to capture-dir)",
    )
    args = parser.parse_args()

    if not args.capture_dir.exists():
        print(f"Capture directory does not exist: {args.capture_dir}", file=sys.stderr)
        sys.exit(1)

    out_dir = args.out_dir or args.capture_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Text summary
    print_summary(args.capture_dir)

    # 2. Plots
    plot_pair_energy(args.capture_dir, out_dir)
    plot_attention_sparsity(args.capture_dir, out_dir)
    plot_memory_breakdown(args.capture_dir, out_dir)
    plot_initializer_tensor_stats(args.capture_dir, out_dir)

    print(f"\nAll plots saved to: {out_dir}")


if __name__ == "__main__":
    main()
