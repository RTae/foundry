#!/usr/bin/env python3
"""Measure peak GPU memory as protein size scales.

Runs RFD3 inference at multiple protein lengths and records peak GPU memory
for both standard and low-memory modes.  Results are printed as a table and
optionally saved to a JSON file.

Usage:
    # Default sizes (100, 200, 400, 600, 800):
    python3 scripts/measure_peak_memory_scaling.py

    # Custom sizes:
    python3 scripts/measure_peak_memory_scaling.py --sizes 100 200 500 1000

    # Only low-memory mode:
    python3 scripts/measure_peak_memory_scaling.py --mode lowmem

    # Save results to JSON:
    python3 scripts/measure_peak_memory_scaling.py --out logs/memory_scaling.json

    # Skip standard mode for large sizes (OOM-safe):
    python3 scripts/measure_peak_memory_scaling.py --sizes 100 200 400 800 1600 --oom-skip
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _format_mb(mb: float) -> str:
    if mb >= 1024:
        return f"{mb / 1024:.2f} GB"
    return f"{mb:.0f} MB"


def _make_input_json(length: int, tmp_dir: str) -> str:
    """Create a minimal unconditional design input JSON for the given length."""
    spec = {
        f"L{length}": {
            "length": f"{length}-{length}",
            "is_non_loopy": True,
        }
    }
    path = os.path.join(tmp_dir, f"input_L{length}.json")
    with open(path, "w") as f:
        json.dump(spec, f)
    return path


def _run_single(
    length: int,
    low_memory_mode: bool,
    tmp_dir: str,
    seed: int = 42,
    diffusion_batch_size: int = 1,
    n_batches: int = 1,
) -> dict:
    """Run inference for a single protein length, return memory stats."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    mode_label = "lowmem" if low_memory_mode else "standard"
    print(f"\n  [{mode_label}] L={length} — running inference...", flush=True)

    # Import here to avoid loading model at module level
    from dataclasses import asdict
    from rfd3.engine import RFD3InferenceConfig, RFD3InferenceEngine

    input_json = _make_input_json(length, tmp_dir)
    out_dir = os.path.join(tmp_dir, f"out_L{length}_{mode_label}")

    # Clear GPU state
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    mem_before = torch.cuda.memory_allocated()

    t0 = time.perf_counter()
    engine = None

    try:
        config = RFD3InferenceConfig(
            low_memory_mode=low_memory_mode,
            diffusion_batch_size=diffusion_batch_size,
            seed=seed,
            skip_existing=False,
            dump_trajectories=False,
            prevalidate_inputs=False,
        )
        engine = RFD3InferenceEngine(**asdict(config))
        engine.run(
            inputs=input_json,
            n_batches=n_batches,
            out_dir=out_dir,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        peak_bytes = torch.cuda.max_memory_allocated()
        peak_mb = peak_bytes / 1024**2
        reserved_mb = torch.cuda.memory_reserved() / 1024**2

        result = {
            "length": length,
            "mode": mode_label,
            "peak_allocated_mb": round(peak_mb, 1),
            "peak_reserved_mb": round(reserved_mb, 1),
            "elapsed_s": round(elapsed, 1),
            "status": "ok",
        }
        print(
            f"  [{mode_label}] L={length} — peak: {_format_mb(peak_mb)}, "
            f"reserved: {_format_mb(reserved_mb)}, time: {elapsed:.1f}s"
        )

    except torch.cuda.OutOfMemoryError:
        elapsed = time.perf_counter() - t0
        result = {
            "length": length,
            "mode": mode_label,
            "peak_allocated_mb": None,
            "peak_reserved_mb": None,
            "elapsed_s": round(elapsed, 1),
            "status": "OOM",
        }
        print(f"  [{mode_label}] L={length} — OOM!")

    finally:
        # Clean up to free GPU memory for the next run
        if engine is not None:
            del engine
        gc.collect()
        torch.cuda.empty_cache()

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Measure peak GPU memory scaling with protein size",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[100, 200, 400, 600, 800],
        help="Protein lengths to test (default: 100 200 400 600 800)",
    )
    parser.add_argument(
        "--mode",
        choices=["both", "standard", "lowmem"],
        default="both",
        help="Which mode(s) to test (default: both)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Path to save results JSON (optional)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--oom-skip",
        action="store_true",
        help="If a size OOMs, skip all larger sizes for that mode",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default="logs/memory_scaling.png",
        help="Path to save the plot PNG (default: logs/memory_scaling.png, use --no-plot to disable)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable plot generation",
    )
    args = parser.parse_args()

    sizes = sorted(args.sizes)
    modes = []
    if args.mode in ("both", "standard"):
        modes.append(False)  # low_memory_mode=False
    if args.mode in ("both", "lowmem"):
        modes.append(True)  # low_memory_mode=True

    print("=" * 72)
    print("  Peak GPU Memory Scaling")
    print(f"  Sizes: {sizes}")
    print(f"  Modes: {['lowmem' if m else 'standard' for m in modes]}")
    print(f"  Seed:  {args.seed}")
    print("=" * 72)

    results = []

    with tempfile.TemporaryDirectory(prefix="rfd3_memscale_") as tmp_dir:
        for low_mem in modes:
            mode_label = "lowmem" if low_mem else "standard"
            oom_hit = False

            for length in sizes:
                if oom_hit and args.oom_skip:
                    results.append({
                        "length": length,
                        "mode": mode_label,
                        "peak_allocated_mb": None,
                        "peak_reserved_mb": None,
                        "elapsed_s": None,
                        "status": "skipped (prior OOM)",
                    })
                    print(f"  [{mode_label}] L={length} — skipped (prior OOM)")
                    continue

                r = _run_single(
                    length=length,
                    low_memory_mode=low_mem,
                    tmp_dir=tmp_dir,
                    seed=args.seed,
                )
                results.append(r)
                if r["status"] == "OOM":
                    oom_hit = True

    # ── Summary table ──
    print(f"\n{'=' * 72}")
    print("  RESULTS SUMMARY")
    print(f"{'=' * 72}\n")

    # Group by mode
    for low_mem in modes:
        mode_label = "lowmem" if low_mem else "standard"
        mode_results = [r for r in results if r["mode"] == mode_label]

        print(f"  Mode: {mode_label}")
        print(f"  {'Length':>8} {'Peak Alloc':>12} {'Peak Rsvd':>12} {'Time':>8} {'Status':>10}")
        print(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*8} {'-'*10}")

        for r in mode_results:
            peak_a = _format_mb(r["peak_allocated_mb"]) if r["peak_allocated_mb"] else "—"
            peak_r = _format_mb(r["peak_reserved_mb"]) if r["peak_reserved_mb"] else "—"
            time_s = f"{r['elapsed_s']:.1f}s" if r["elapsed_s"] else "—"
            print(f"  {r['length']:>8} {peak_a:>12} {peak_r:>12} {time_s:>8} {r['status']:>10}")
        print()

    # ── Side-by-side comparison if both modes ──
    if len(modes) == 2:
        print(f"  Side-by-side comparison:")
        print(f"  {'Length':>8} {'Standard':>12} {'Low-Mem':>12} {'Savings':>10}")
        print(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*10}")

        for length in sizes:
            std = next((r for r in results if r["length"] == length and r["mode"] == "standard"), None)
            lm = next((r for r in results if r["length"] == length and r["mode"] == "lowmem"), None)

            std_peak = std["peak_allocated_mb"] if std and std["peak_allocated_mb"] else None
            lm_peak = lm["peak_allocated_mb"] if lm and lm["peak_allocated_mb"] else None

            std_str = _format_mb(std_peak) if std_peak else std["status"] if std else "—"
            lm_str = _format_mb(lm_peak) if lm_peak else lm["status"] if lm else "—"

            if std_peak and lm_peak:
                savings = f"{(1 - lm_peak / std_peak) * 100:.0f}%"
            else:
                savings = "—"

            print(f"  {length:>8} {std_str:>12} {lm_str:>12} {savings:>10}")
        print()

    # ── Save JSON ──
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Results saved to {args.out}")

    # ── Plot ──
    if not args.no_plot:
        plot_results(results, sizes, modes, args.plot)


def plot_results(
    results: list[dict],
    sizes: list[int],
    modes: list[bool],
    plot_path: str,
) -> None:
    """Generate a peak memory scaling plot."""
    fig, ax = plt.subplots(figsize=(10, 6))

    style = {
        "standard": {"color": "#c44e52", "marker": "o", "label": "Standard"},
        "lowmem":   {"color": "#4c72b0", "marker": "s", "label": "Low-Memory"},
    }

    for low_mem in modes:
        mode_label = "lowmem" if low_mem else "standard"
        s = style[mode_label]

        lengths = []
        peaks = []
        for r in results:
            if r["mode"] == mode_label and r["peak_allocated_mb"] is not None:
                lengths.append(r["length"])
                peaks.append(r["peak_allocated_mb"] / 1024)  # Convert to GB

        if not lengths:
            continue

        ax.plot(
            lengths, peaks,
            color=s["color"], marker=s["marker"], markersize=8,
            linewidth=2, label=s["label"], zorder=3,
        )

        # Annotate each point
        for x, y in zip(lengths, peaks):
            ax.annotate(
                f"{y:.1f} GB",
                (x, y),
                textcoords="offset points",
                xytext=(0, 12),
                ha="center",
                fontsize=8,
                color=s["color"],
            )

        # Mark OOM points
        for r in results:
            if r["mode"] == mode_label and r["status"] == "OOM":
                ax.axvline(
                    r["length"], color=s["color"], linestyle=":",
                    alpha=0.5, linewidth=1,
                )
                ax.annotate(
                    f"OOM (L={r['length']})",
                    (r["length"], ax.get_ylim()[1] * 0.95),
                    ha="center", fontsize=8, color=s["color"], alpha=0.7,
                )

    ax.set_xlabel("Protein Length (residues)", fontsize=12)
    ax.set_ylabel("Peak GPU Memory (GB)", fontsize=12)
    ax.set_title("RFD3 Peak GPU Memory vs Protein Size", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    # Add savings annotation if both modes present
    if len(modes) == 2:
        for length in sizes:
            std = next((r for r in results if r["length"] == length and r["mode"] == "standard" and r["peak_allocated_mb"]), None)
            lm = next((r for r in results if r["length"] == length and r["mode"] == "lowmem" and r["peak_allocated_mb"]), None)
            if std and lm:
                savings = (1 - lm["peak_allocated_mb"] / std["peak_allocated_mb"]) * 100
                mid_y = (std["peak_allocated_mb"] + lm["peak_allocated_mb"]) / 2 / 1024
                ax.annotate(
                    f"{savings:.0f}%",
                    (length, mid_y),
                    ha="center", fontsize=8, fontweight="bold",
                    color="#555555",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#f0f0f0", edgecolor="#cccccc", alpha=0.8),
                )

    plt.tight_layout()
    out = Path(plot_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Plot saved to {plot_path}")


if __name__ == "__main__":
    main()
