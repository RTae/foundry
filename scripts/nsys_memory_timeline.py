#!/usr/bin/env python3
"""Visualize GPU memory allocation timeline from an Nsight Systems SQLite profile.

Produces a multi-panel figure correlating GPU memory waterline with NVTX trace
ranges so you can see *exactly* when each allocation happens and which code phase
is responsible.

Usage:
    python scripts/nsys_memory_timeline.py logs/nsys/rfd3_common_sim_single_model.sqlite \
        --out logs/nsys/memory_timeline.png

Environment:
    Requires matplotlib (``uv pip install matplotlib``).
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

MB = 1024.0 ** 2
NS_TO_MS = 1e-6
NS_TO_S = 1e-9

# ── Colours for top-level NVTX phases ──────────────────────────────────────────
PHASE_COLORS = {
    "rfd3.engine.initialize": "#4c72b0",
    "rfd3.engine.run_multi": "#55a868",
    "TokenInitializer": "#c44e52",
    "InputFeatureProcessing": "#dd8452",
    "Downcast": "#8172b3",
    "DiffusionTransformer": "#937860",
    "PairwiseEmbedding": "#da8bc3",
    "Pairformer": "#8c8c8c",
    "AtomTransformer": "#ccb974",
}


def _phase_color(text: str) -> str | None:
    for key, color in PHASE_COLORS.items():
        if key in text:
            return color
    return None


# ── Data loading ───────────────────────────────────────────────────────────────

def load_memory_events(con: sqlite3.Connection):
    """Return sorted list of (timestamp_ns, delta_bytes) tuples."""
    cur = con.cursor()
    rows = cur.execute("""
        SELECT start, bytes, memoryOperationType
        FROM CUDA_GPU_MEMORY_USAGE_EVENTS
        ORDER BY start
    """).fetchall()
    events = []
    for ts, size, op in rows:
        delta = size if op == 0 else -size  # 0=alloc, 1=dealloc
        events.append((ts, delta))
    return events


def build_waterline(events):
    """Compute cumulative memory waterline from events."""
    timestamps = np.array([e[0] for e in events], dtype=np.int64)
    deltas = np.array([e[1] for e in events], dtype=np.int64)
    waterline = np.cumsum(deltas)
    return timestamps, waterline


def load_nvtx_ranges(con: sqlite3.Connection, min_dur_ms: float = 0.1):
    """Load NVTX push/pop ranges with duration >= min_dur_ms."""
    cur = con.cursor()
    rows = cur.execute("""
        SELECT text, start, end
        FROM NVTX_EVENTS
        WHERE end IS NOT NULL AND start IS NOT NULL AND text IS NOT NULL
          AND (end - start) > ?
        ORDER BY start
    """, (int(min_dur_ms / NS_TO_MS),)).fetchall()
    return [(text, start, end) for text, start, end in rows]


def _load_nvtx_index(con: sqlite3.Connection):
    """Load NVTX ranges into numpy arrays for vectorized enclosure lookup.

    Returns (starts, ends, texts, durations) sorted by duration ascending.
    We filter to ranges > 10 ms with user-defined trace names (containing / or .).
    """
    cur = con.cursor()
    rows = cur.execute("""
        SELECT text, start, end FROM NVTX_EVENTS
        WHERE end IS NOT NULL AND start IS NOT NULL AND text IS NOT NULL
          AND (end - start) > 10000000
          AND (text LIKE '%/%' OR text LIKE '%.%')
        ORDER BY (end - start) ASC
    """).fetchall()
    texts = [r[0] for r in rows]
    starts = np.array([r[1] for r in rows], dtype=np.int64)
    ends = np.array([r[2] for r in rows], dtype=np.int64)
    durations = ends - starts
    return starts, ends, texts, durations


def _find_enclosing_batch(nvtx_index, timestamps):
    """For each timestamp, find the tightest enclosing NVTX range. Returns list of text."""
    starts, ends, texts, _ = nvtx_index
    n_ranges = len(texts)
    n_ts = len(timestamps)
    if n_ranges == 0 or n_ts == 0:
        return ["<no NVTX>"] * n_ts
    ts_arr = np.array(timestamps, dtype=np.int64)
    # Vectorized: (n_ts, n_ranges) bool matrix — ranges sorted by duration ASC
    enclosed = (ts_arr[:, None] >= starts[None, :]) & (ts_arr[:, None] <= ends[None, :])
    # For each timestamp, first True column is tightest enclosing range
    first_match = np.argmax(enclosed, axis=1)
    has_match = enclosed[np.arange(n_ts), first_match]
    results = []
    for i in range(n_ts):
        results.append(texts[first_match[i]] if has_match[i] else "<no NVTX>")
    return results


def load_top_allocations(con: sqlite3.Connection, nvtx_index, top_n: int = 20):
    """Return the top_n largest individual allocations with NVTX context."""
    cur = con.cursor()
    allocs = cur.execute("""
        SELECT start, bytes
        FROM CUDA_GPU_MEMORY_USAGE_EVENTS
        WHERE memoryOperationType = 0
        ORDER BY bytes DESC
        LIMIT ?
    """, (top_n,)).fetchall()
    timestamps = [a[0] for a in allocs]
    phases = _find_enclosing_batch(nvtx_index, timestamps)
    return [(ts, size, phase) for (ts, size), phase in zip(allocs, phases)]


def attribute_memory_to_phases(con: sqlite3.Connection, nvtx_index):
    """For each allocation, find the tightest enclosing NVTX range and sum by phase."""
    cur = con.cursor()
    allocs = cur.execute("""
        SELECT start, bytes
        FROM CUDA_GPU_MEMORY_USAGE_EVENTS
        WHERE memoryOperationType = 0
        ORDER BY start
    """).fetchall()

    timestamps = [a[0] for a in allocs]
    sizes = [a[1] for a in allocs]
    phases = _find_enclosing_batch(nvtx_index, timestamps)

    phase_bytes: dict[str, int] = {}
    phase_count: dict[str, int] = {}
    for phase, size in zip(phases, sizes):
        phase_bytes[phase] = phase_bytes.get(phase, 0) + size
        phase_count[phase] = phase_count.get(phase, 0) + 1

    ranked = sorted(phase_bytes.items(), key=lambda x: -x[1])
    return [(name, phase_bytes[name], phase_count[name]) for name, _ in ranked]


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_memory_timeline(
    sqlite_path: str | Path,
    out_path: str | Path,
    focus_range: tuple[float, float] | None = None,
):
    con = sqlite3.connect(str(sqlite_path))

    # Load data
    events = load_memory_events(con)
    timestamps, waterline = build_waterline(events)
    ranges = load_nvtx_ranges(con, min_dur_ms=5.0)
    nvtx_index = _load_nvtx_index(con)
    top_allocs = load_top_allocations(con, nvtx_index, top_n=15)
    phase_attribution = attribute_memory_to_phases(con, nvtx_index)
    con.close()

    t0 = timestamps[0]  # base timestamp
    ts_ms = (timestamps - t0) * NS_TO_MS
    wl_mb = waterline / MB
    peak_idx = np.argmax(waterline)
    peak_mb = wl_mb[peak_idx]
    peak_ms = ts_ms[peak_idx]

    # ── Figure layout ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 1.2, 1], hspace=0.35, wspace=0.3)

    # Panel 1: Full memory waterline with NVTX phase bands
    ax1 = fig.add_subplot(gs[0, :])
    ax1.fill_between(ts_ms, wl_mb, alpha=0.3, color="#4c72b0", step="post")
    ax1.step(ts_ms, wl_mb, where="post", linewidth=0.8, color="#4c72b0")
    ax1.axhline(peak_mb, color="red", linestyle="--", linewidth=0.8, alpha=0.7)
    ax1.annotate(
        f"Peak: {peak_mb:.0f} MB",
        xy=(peak_ms, peak_mb),
        xytext=(peak_ms + 500, peak_mb + 200),
        fontsize=9,
        color="red",
        arrowprops=dict(arrowstyle="->", color="red", lw=0.8),
    )

    # Overlay NVTX phase bands (only top-level ranges)
    top_level_keywords = [
        "rfd3.engine.initialize",
        "TokenInitializer/InputFeatureProcessing",
        "RFD3/Layers/Downcast",
        "DiffusionTransformer",
    ]
    legend_patches = []
    seen_colors = set()
    for text, rstart, rend in ranges:
        color = _phase_color(text)
        if color and color not in seen_colors:
            # Only draw if it's a substantial range
            r_start_ms = (rstart - t0) * NS_TO_MS
            r_end_ms = (rend - t0) * NS_TO_MS
            if r_end_ms - r_start_ms > 50:
                ax1.axvspan(r_start_ms, r_end_ms, alpha=0.10, color=color)
                short = text.split("/")[-1] if "/" in text else text.split(".")[-1]
                seen_colors.add(color)
                legend_patches.append(mpatches.Patch(color=color, alpha=0.3, label=short))

    ax1.set_xlabel("Time (ms from first alloc)")
    ax1.set_ylabel("GPU Memory Allocated (MB)")
    ax1.set_title("GPU Memory Waterline with NVTX Phase Overlay", fontsize=13, fontweight="bold")
    ax1.legend(handles=legend_patches, loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Panel 2A: Zoomed view around TokenInitializer (peak allocation zone)
    ax2 = fig.add_subplot(gs[1, 0])
    # Find TokenInitializer time range from NVTX
    ti_start, ti_end = None, None
    for text, rstart, rend in ranges:
        if "TokenInitializer" in text and "Input" not in text and "Atom" not in text:
            ti_start = (rstart - t0) * NS_TO_MS
            ti_end = (rend - t0) * NS_TO_MS
            break

    if ti_start is not None:
        margin = (ti_end - ti_start) * 0.3
        zoom_start = ti_start - margin
        zoom_end = ti_end + margin
        mask = (ts_ms >= zoom_start) & (ts_ms <= zoom_end)
        ax2.fill_between(ts_ms[mask], wl_mb[mask], alpha=0.3, color="#c44e52", step="post")
        ax2.step(ts_ms[mask], wl_mb[mask], where="post", linewidth=1.0, color="#c44e52")
        ax2.axvspan(ti_start, ti_end, alpha=0.15, color="#c44e52")
        ax2.set_xlim(zoom_start, zoom_end)

        # Mark individual large allocations in this zone
        for alloc_ts, alloc_size, phase in top_allocs:
            a_ms = (alloc_ts - t0) * NS_TO_MS
            if zoom_start <= a_ms <= zoom_end and alloc_size > 100 * MB:
                ax2.axvline(a_ms, color="orange", alpha=0.5, linewidth=0.7)
                ax2.annotate(
                    f"{alloc_size / MB:.0f} MB",
                    xy=(a_ms, ax2.get_ylim()[1] * 0.85),
                    fontsize=7,
                    rotation=45,
                    color="darkorange",
                )
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("GPU Memory (MB)")
    ax2.set_title("Zoom: TokenInitializer Phase", fontsize=11, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    # Panel 2B: Per-alloc size histogram
    ax3 = fig.add_subplot(gs[1, 1])
    alloc_sizes = [e[1] for e in events if e[1] > 0]
    alloc_mb = [s / MB for s in alloc_sizes]
    # Log-scale bins
    if alloc_mb:
        bins = np.logspace(np.log10(max(min(alloc_mb), 0.001)), np.log10(max(alloc_mb)), 50)
        ax3.hist(alloc_mb, bins=bins, color="#55a868", edgecolor="white", linewidth=0.5)
        ax3.set_xscale("log")
        ax3.set_xlabel("Allocation Size (MB, log scale)")
        ax3.set_ylabel("Count")
        ax3.set_title("Allocation Size Distribution", fontsize=11, fontweight="bold")
        ax3.grid(True, alpha=0.3)
        # Mark top allocation
        ax3.axvline(max(alloc_mb), color="red", linestyle="--", linewidth=1)
        ax3.annotate(
            f"Max: {max(alloc_mb):.0f} MB",
            xy=(max(alloc_mb), 0),
            xytext=(max(alloc_mb) * 0.3, ax3.get_ylim()[1] * 0.8),
            fontsize=9,
            color="red",
            arrowprops=dict(arrowstyle="->", color="red", lw=0.8),
        )

    # Panel 3A: Phase attribution bar chart
    ax4 = fig.add_subplot(gs[2, 0])
    top_phases = phase_attribution[:10]
    names = [p[0].split("/")[-1] if "/" in p[0] else p[0].split(".")[-1] for p in top_phases]
    totals = [p[1] / MB for p in top_phases]
    counts = [p[2] for p in top_phases]
    colors_bar = [_phase_color(p[0]) or "#aaaaaa" for p in top_phases]

    bars = ax4.barh(range(len(names)), totals, color=colors_bar, edgecolor="white")
    ax4.set_yticks(range(len(names)))
    ax4.set_yticklabels(names, fontsize=8)
    ax4.invert_yaxis()
    ax4.set_xlabel("Total Allocated (MB)")
    ax4.set_title("Memory Allocated by NVTX Phase (Top 10)", fontsize=11, fontweight="bold")
    for i, (total, count) in enumerate(zip(totals, counts)):
        ax4.text(total + 20, i, f"{total:.0f} MB ({count} allocs)", va="center", fontsize=7)

    # Panel 3B: Annotation text table
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis("off")
    table_data = [
        ["Metric", "Value"],
        ["Peak Active Memory", f"{peak_mb:.0f} MB ({peak_mb / 1024:.2f} GB)"],
        ["Total Allocations", f"{len([e for e in events if e[1] > 0]):,}"],
        ["Total Deallocations", f"{len([e for e in events if e[1] < 0]):,}"],
        ["Largest Single Alloc", f"{max(alloc_mb):.0f} MB" if alloc_mb else "N/A"],
        ["Time to Peak", f"{peak_ms:.0f} ms"],
    ]

    if top_allocs:
        table_data.append(["", ""])
        table_data.append(["Top Allocations", ""])
        for ts, size, phase in top_allocs[:7]:
            short = phase.split("/")[-1] if "/" in phase else phase
            table_data.append([f"  {short}", f"{size / MB:.0f} MB"])

    tb = ax5.table(
        cellText=table_data,
        cellLoc="left",
        loc="center",
        colWidths=[0.45, 0.5],
    )
    tb.auto_set_font_size(False)
    tb.set_fontsize(8)
    tb.scale(1, 1.3)
    # Style header
    for j in range(2):
        tb[0, j].set_facecolor("#4c72b0")
        tb[0, j].set_text_props(color="white", fontweight="bold")
    ax5.set_title("Memory Summary", fontsize=11, fontweight="bold")

    fig.suptitle(
        f"RFD3 GPU Memory Profile — Peak {peak_mb:.0f} MB ({peak_mb / 1024:.1f} GB)",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved → {out_path}  ({out_path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite", type=Path, help="Path to Nsight Systems .sqlite export")
    parser.add_argument("--out", type=Path, default=None, help="Output image path")
    args = parser.parse_args()
    out = args.out or args.sqlite.with_name("memory_timeline.png")
    plot_memory_timeline(args.sqlite, out)
