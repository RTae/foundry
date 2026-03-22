#!/usr/bin/env python3
"""Summarize a single Nsight Systems profile for attention-block performance.

This script extracts three metric groups from an Nsight Systems SQLite export
or .nsys-rep file:
1) DiffusionTransformer block timing and dominant kernel category
2) CUDA memory allocation behavior and estimated peak active device memory
3) Kernel bottlenecks by total GPU time

"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sqlite3
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


NS_PER_MS = 1_000_000.0
NS_PER_S = 1_000_000_000.0
BYTES_PER_MB = 1024.0 * 1024.0


def kernel_category(kernel_name: str) -> str:
    lower = kernel_name.lower()
    if "gemm" in lower or "cublas" in lower or "cutlass" in lower:
        return "gemm"
    if "gather" in lower or "index" in lower or "scatter" in lower:
        return "gather/index"
    return "other"


def run_cmd(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            + " ".join(cmd)
            + "\n\nstdout:\n"
            + proc.stdout
            + "\n\nstderr:\n"
            + proc.stderr
        )


def ensure_sqlite(input_path: Path, force_regen: bool = False) -> Path:
    if input_path.suffix == ".sqlite":
        if not input_path.exists():
            raise FileNotFoundError(f"Missing sqlite file: {input_path}")
        return input_path

    if input_path.suffix != ".nsys-rep":
        raise ValueError(f"Unsupported input type: {input_path}")

    if not input_path.exists():
        raise FileNotFoundError(f"Missing nsys report: {input_path}")

    sqlite_path = input_path.with_suffix(".sqlite")
    if sqlite_path.exists() and not force_regen:
        return sqlite_path

    cmd = [
        "nsys",
        "stats",
        "--force-export",
        "true" if force_regen else "false",
        "--report",
        "nvtx_sum",
        "--format",
        "csv",
        str(input_path),
    ]
    run_cmd(cmd)
    if not sqlite_path.exists():
        raise RuntimeError(f"Failed to export sqlite from {input_path}")
    return sqlite_path


@dataclass
class BlockStat:
    block_id: int
    instances: int
    avg_ms: float
    med_ms: float
    min_ms: float
    max_ms: float
    std_ms: float
    gpu_active_ratio: float
    dominant_kernel: str
    dominant_kernel_ms: float
    dominant_kernel_category: str
    dominant_category_share: float


@dataclass
class MemorySummary:
    alloc_count: int
    dealloc_count: int
    total_alloc_mb: float
    total_dealloc_mb: float
    largest_alloc_mb: float
    estimated_peak_active_mb: float
    top_alloc_sizes_mb: list[tuple[float, int]]
    avg_allocs_per_block: float
    avg_alloc_mb_per_block: float


@dataclass
class KernelSummaryRow:
    name: str
    calls: int
    total_ms: float
    avg_ms: float
    max_ms: float
    avg_registers: float
    avg_threads_per_launch: float
    category: str


@dataclass
class ProfileSummary:
    label: str
    source: str
    block_stats: list[BlockStat]
    memory: MemorySummary
    top_kernels: list[KernelSummaryRow]
    category_totals_ms: dict[str, float]


def connect_db(sqlite_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(sqlite_path))
    conn.row_factory = sqlite3.Row
    return conn


def build_block_events_table(conn: sqlite3.Connection) -> list[tuple[int, int, int]]:
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS _block_events")
    cur.execute("CREATE TEMP TABLE _block_events (block_id INTEGER, start INTEGER, end INTEGER)")

    rows = cur.execute(
        """
        SELECT start, end, text
        FROM NVTX_EVENTS
        WHERE text LIKE '%/DiffusionTransformer/Block_%'
          AND text NOT LIKE '%/Block_%/%'
          AND end IS NOT NULL
        ORDER BY start
        """
    ).fetchall()

    block_events: list[tuple[int, int, int]] = []
    for row in rows:
        text = row["text"] or ""
        try:
            block_id = int(text.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            continue
        start = int(row["start"])
        end = int(row["end"])
        if end <= start:
            continue
        block_events.append((block_id, start, end))

    cur.executemany("INSERT INTO _block_events(block_id, start, end) VALUES (?, ?, ?)", block_events)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_block_events_time ON _block_events(start, end)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_block_events_id ON _block_events(block_id)")
    conn.commit()
    return block_events


def block_time_window_ns(block_events: list[tuple[int, int, int]]) -> tuple[int, int] | None:
    if not block_events:
        return None
    starts = [s for _, s, _ in block_events]
    ends = [e for _, _, e in block_events]
    return (min(starts), max(ends))


def materialize_kernel_window(
    conn: sqlite3.Connection,
    window_ns: tuple[int, int] | None,
) -> str:
    """Create an indexed temp table of kernels in the relevant time window."""
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS _kernels_window")
    if window_ns is None:
        cur.execute(
            """
            CREATE TEMP TABLE _kernels_window AS
            SELECT start, end, shortName, registersPerThread, gridX, gridY, gridZ, blockX, blockY, blockZ
            FROM CUPTI_ACTIVITY_KIND_KERNEL
            """
        )
    else:
        start_ns, end_ns = window_ns
        cur.execute(
            """
            CREATE TEMP TABLE _kernels_window AS
            SELECT start, end, shortName, registersPerThread, gridX, gridY, gridZ, blockX, blockY, blockZ
            FROM CUPTI_ACTIVITY_KIND_KERNEL
            WHERE end > ? AND start < ?
            """,
            (start_ns, end_ns),
        )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_kernels_window_time ON _kernels_window(start, end)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_kernels_window_name ON _kernels_window(shortName)")
    conn.commit()
    return "_kernels_window"


def materialize_memory_window(
    conn: sqlite3.Connection,
    window_ns: tuple[int, int] | None,
) -> str:
    """Create an indexed temp table of device memory events in the relevant time window."""
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS _mem_window")
    if window_ns is None:
        cur.execute(
            """
            CREATE TEMP TABLE _mem_window AS
            SELECT start, address, bytes, memKind, memoryOperationType
            FROM CUDA_GPU_MEMORY_USAGE_EVENTS
            WHERE memKind = 2
            """
        )
    else:
        start_ns, end_ns = window_ns
        cur.execute(
            """
            CREATE TEMP TABLE _mem_window AS
            SELECT start, address, bytes, memKind, memoryOperationType
            FROM CUDA_GPU_MEMORY_USAGE_EVENTS
            WHERE memKind = 2
              AND start >= ?
              AND start <= ?
            """,
            (start_ns, end_ns),
        )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_mem_window_time ON _mem_window(start)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_mem_window_addr ON _mem_window(address)")
    conn.commit()
    return "_mem_window"


def summarize_blocks(conn: sqlite3.Connection, kernel_table: str = "CUPTI_ACTIVITY_KIND_KERNEL") -> list[BlockStat]:
    block_events = build_block_events_table(conn)
    if not block_events:
        return []

    durations_by_block: dict[int, list[int]] = defaultdict(list)
    for block_id, start, end in block_events:
        durations_by_block[block_id].append(end - start)

    cur = conn.cursor()

    # Aggregate overlapping kernel durations for each block and kernel name.
    overlap_rows = cur.execute(
        f"""
        SELECT
          b.block_id AS block_id,
          COALESCE(s.value, '<unknown>') AS kernel_name,
          SUM(
            CASE
              WHEN MIN(k.end, b.end) > MAX(k.start, b.start)
              THEN MIN(k.end, b.end) - MAX(k.start, b.start)
              ELSE 0
            END
          ) AS overlap_ns
                FROM _block_events b
                JOIN {kernel_table} k
          ON k.end > b.start
         AND k.start < b.end
        LEFT JOIN StringIds s
          ON s.id = k.shortName
        GROUP BY b.block_id, kernel_name
                """
    ).fetchall()

    by_block_kernel: dict[int, list[tuple[str, int]]] = defaultdict(list)
    for row in overlap_rows:
        overlap = int(row["overlap_ns"] or 0)
        if overlap <= 0:
            continue
        by_block_kernel[int(row["block_id"])].append((str(row["kernel_name"]), overlap))

    result: list[BlockStat] = []
    for block_id in sorted(durations_by_block):
        durations = durations_by_block[block_id]
        avg_ns = statistics.mean(durations)
        med_ns = statistics.median(durations)
        min_ns = min(durations)
        max_ns = max(durations)
        std_ns = statistics.pstdev(durations) if len(durations) > 1 else 0.0

        total_kernel_overlap = sum(v for _, v in by_block_kernel.get(block_id, []))
        dominant_kernel = "<none>"
        dominant_overlap = 0
        dominant_category = "other"
        dominant_category_share = 0.0

        if by_block_kernel.get(block_id):
            dominant_kernel, dominant_overlap = max(by_block_kernel[block_id], key=lambda x: x[1])
            cat_totals: dict[str, int] = defaultdict(int)
            for name, overlap in by_block_kernel[block_id]:
                cat_totals[kernel_category(name)] += overlap
            dominant_category = max(cat_totals.items(), key=lambda x: x[1])[0]
            dominant_category_share = (
                cat_totals[dominant_category] / total_kernel_overlap if total_kernel_overlap > 0 else 0.0
            )

        gpu_active_ratio = min(1.0, (total_kernel_overlap / (avg_ns * len(durations))) if avg_ns > 0 else 0.0)

        result.append(
            BlockStat(
                block_id=block_id,
                instances=len(durations),
                avg_ms=avg_ns / NS_PER_MS,
                med_ms=med_ns / NS_PER_MS,
                min_ms=min_ns / NS_PER_MS,
                max_ms=max_ns / NS_PER_MS,
                std_ms=std_ns / NS_PER_MS,
                gpu_active_ratio=gpu_active_ratio,
                dominant_kernel=dominant_kernel,
                dominant_kernel_ms=dominant_overlap / NS_PER_MS,
                dominant_kernel_category=dominant_category,
                dominant_category_share=dominant_category_share,
            )
        )

    return result


def summarize_memory(conn: sqlite3.Connection, memory_table: str = "CUDA_GPU_MEMORY_USAGE_EVENTS") -> MemorySummary:
    cur = conn.cursor()
    rows = cur.execute(
        f"""
        SELECT start, address, bytes, memKind, memoryOperationType
        FROM {memory_table}
        ORDER BY start
        """
    ).fetchall()

    alloc_count = 0
    dealloc_count = 0
    total_alloc_bytes = 0
    total_dealloc_bytes = 0
    largest_alloc_bytes = 0
    size_counter: Counter[int] = Counter()

    live_by_address: dict[int, int] = {}
    active_bytes = 0
    peak_active_bytes = 0

    for row in rows:
        addr = int(row["address"])
        size = int(row["bytes"])
        op = int(row["memoryOperationType"])

        if op == 0:  # Allocation
            alloc_count += 1
            total_alloc_bytes += size
            largest_alloc_bytes = max(largest_alloc_bytes, size)
            size_counter[size] += 1

            previous = live_by_address.get(addr, 0)
            if previous:
                active_bytes -= previous
            live_by_address[addr] = size
            active_bytes += size
        elif op == 1:  # Deallocation
            dealloc_count += 1
            total_dealloc_bytes += size
            known = live_by_address.pop(addr, size)
            active_bytes -= known

        if active_bytes > peak_active_bytes:
            peak_active_bytes = active_bytes

    if peak_active_bytes < 0:
        peak_active_bytes = 0

    per_block = cur.execute(
        f"""
        SELECT b.block_id, COUNT(*) AS allocs, SUM(m.bytes) AS alloc_bytes
                FROM _block_events b
                LEFT JOIN {memory_table} m
          ON m.start >= b.start
         AND m.start <= b.end
         AND m.memoryOperationType = 0
        GROUP BY b.block_id
        ORDER BY b.block_id
        """
    ).fetchall()

    allocs_per_block = [int(r["allocs"] or 0) for r in per_block]
    bytes_per_block = [int(r["alloc_bytes"] or 0) for r in per_block]

    top_sizes = sorted(size_counter.items(), key=lambda kv: (-kv[1], -kv[0]))[:8]
    top_sizes_mb = [(size / BYTES_PER_MB, count) for size, count in top_sizes]

    return MemorySummary(
        alloc_count=alloc_count,
        dealloc_count=dealloc_count,
        total_alloc_mb=total_alloc_bytes / BYTES_PER_MB,
        total_dealloc_mb=total_dealloc_bytes / BYTES_PER_MB,
        largest_alloc_mb=largest_alloc_bytes / BYTES_PER_MB,
        estimated_peak_active_mb=peak_active_bytes / BYTES_PER_MB,
        top_alloc_sizes_mb=top_sizes_mb,
        avg_allocs_per_block=statistics.mean(allocs_per_block) if allocs_per_block else 0.0,
        avg_alloc_mb_per_block=(statistics.mean(bytes_per_block) / BYTES_PER_MB) if bytes_per_block else 0.0,
    )


def summarize_kernels(
    conn: sqlite3.Connection,
    top_n: int = 15,
    kernel_table: str = "CUPTI_ACTIVITY_KIND_KERNEL",
) -> tuple[list[KernelSummaryRow], dict[str, float]]:
    cur = conn.cursor()
    rows = cur.execute(
        f"""
        SELECT
          COALESCE(s.value, '<unknown>') AS kernel_name,
          COUNT(*) AS calls,
          SUM(k.end - k.start) AS total_ns,
          AVG(k.end - k.start) AS avg_ns,
          MAX(k.end - k.start) AS max_ns,
          AVG(k.registersPerThread) AS avg_registers,
          AVG(
            CAST(k.gridX AS REAL) * k.gridY * k.gridZ * k.blockX * k.blockY * k.blockZ
          ) AS avg_threads
                FROM {kernel_table} k
        LEFT JOIN StringIds s
          ON s.id = k.shortName
        GROUP BY kernel_name
        ORDER BY total_ns DESC
                """
    ).fetchall()

    parsed: list[KernelSummaryRow] = []
    totals_by_category: dict[str, float] = defaultdict(float)
    for row in rows:
        name = str(row["kernel_name"])
        total_ms = float(row["total_ns"] or 0) / NS_PER_MS
        cat = kernel_category(name)
        totals_by_category[cat] += total_ms
        parsed.append(
            KernelSummaryRow(
                name=name,
                calls=int(row["calls"] or 0),
                total_ms=total_ms,
                avg_ms=float(row["avg_ns"] or 0) / NS_PER_MS,
                max_ms=float(row["max_ns"] or 0) / NS_PER_MS,
                avg_registers=float(row["avg_registers"] or 0),
                avg_threads_per_launch=float(row["avg_threads"] or 0),
                category=cat,
            )
        )

    return parsed[:top_n], dict(totals_by_category)


def analyze_profile(label: str, input_path: Path, force_regen: bool = False, top_kernels: int = 15) -> ProfileSummary:
    sqlite_path = ensure_sqlite(input_path, force_regen=force_regen)
    conn = connect_db(sqlite_path)
    try:
        block_events = build_block_events_table(conn)
        window_ns = block_time_window_ns(block_events)
        kernel_table = materialize_kernel_window(conn, window_ns)
        memory_table = materialize_memory_window(conn, window_ns)

        # Reuse the temp block table and scoped temp event tables for faster joins.
        block_stats = summarize_blocks(conn, kernel_table=kernel_table)
        memory = summarize_memory(conn, memory_table=memory_table)
        top_rows, category_totals = summarize_kernels(conn, top_n=top_kernels, kernel_table=kernel_table)
    finally:
        conn.close()

    return ProfileSummary(
        label=label,
        source=str(input_path),
        block_stats=block_stats,
        memory=memory,
        top_kernels=top_rows,
        category_totals_ms=category_totals,
    )


def safe_pct(numer: float, denom: float) -> float:
    return (100.0 * numer / denom) if denom else 0.0


def print_block_section(summary: ProfileSummary) -> None:
    print(f"\n[{summary.label}] Attention Block Timing")
    if not summary.block_stats:
        print("  No DiffusionTransformer block NVTX ranges found.")
        return

    avg_of_blocks = statistics.mean(b.avg_ms for b in summary.block_stats)
    print(f"  Blocks found: {len(summary.block_stats)}")
    print(f"  Avg block duration (mean of block means): {avg_of_blocks:.3f} ms")
    print("  Per-block summary:")
    print(
        "    "
        + "block avg_ms med_ms gpu_active% dominant_kernel dominant_kernel_ms dominant_category(category_share%)"
    )
    for b in summary.block_stats:
        print(
            "    "
            + f"{b.block_id:>2d} {b.avg_ms:>7.3f} {b.med_ms:>7.3f} {100*b.gpu_active_ratio:>9.1f} "
            + f"{b.dominant_kernel[:42]:<42} {b.dominant_kernel_ms:>8.3f} "
            + f"{b.dominant_kernel_category:<12}({100*b.dominant_category_share:>5.1f}%)"
        )


def print_memory_section(summary: ProfileSummary) -> None:
    m = summary.memory
    print(f"\n[{summary.label}] CUDA Memory Allocation")
    print(f"  Device allocations: {m.alloc_count}")
    print(f"  Device deallocations: {m.dealloc_count}")
    print(f"  Total allocated: {m.total_alloc_mb:.1f} MB")
    print(f"  Total deallocated: {m.total_dealloc_mb:.1f} MB")
    print(f"  Largest single allocation: {m.largest_alloc_mb:.2f} MB")
    print(f"  Estimated peak active device memory: {m.estimated_peak_active_mb:.1f} MB")
    print(f"  Avg allocations per block: {m.avg_allocs_per_block:.2f}")
    print(f"  Avg allocated MB per block: {m.avg_alloc_mb_per_block:.2f} MB")
    print("  Common allocation sizes:")
    for size_mb, count in m.top_alloc_sizes_mb:
        print(f"    {size_mb:8.3f} MB x {count}")


def print_kernel_section(summary: ProfileSummary) -> None:
    print(f"\n[{summary.label}] Kernel Bottlenecks")
    totals = summary.category_totals_ms
    total_ms = sum(totals.values())
    print(
        "  Category totals: "
        + ", ".join(
            f"{k}={v:.1f} ms ({safe_pct(v, total_ms):.1f}%)" for k, v in sorted(totals.items())
        )
    )
    print("  Top kernels by total GPU time:")
    print("    total_ms avg_ms max_ms calls avg_regs avg_threads category name")
    for r in summary.top_kernels:
        print(
            f"    {r.total_ms:8.2f} {r.avg_ms:7.4f} {r.max_ms:7.4f} {r.calls:6d}"
            f" {r.avg_registers:8.1f} {r.avg_threads_per_launch:11.1f} {r.category:12s} {r.name}"
        )


def profile_to_dict(summary: ProfileSummary) -> dict[str, Any]:
    return {
        "label": summary.label,
        "source": summary.source,
        "block_stats": [
            {
                "block_id": b.block_id,
                "instances": b.instances,
                "avg_ms": b.avg_ms,
                "med_ms": b.med_ms,
                "min_ms": b.min_ms,
                "max_ms": b.max_ms,
                "std_ms": b.std_ms,
                "gpu_active_ratio": b.gpu_active_ratio,
                "dominant_kernel": b.dominant_kernel,
                "dominant_kernel_ms": b.dominant_kernel_ms,
                "dominant_kernel_category": b.dominant_kernel_category,
                "dominant_category_share": b.dominant_category_share,
            }
            for b in summary.block_stats
        ],
        "memory": {
            "alloc_count": summary.memory.alloc_count,
            "dealloc_count": summary.memory.dealloc_count,
            "total_alloc_mb": summary.memory.total_alloc_mb,
            "total_dealloc_mb": summary.memory.total_dealloc_mb,
            "largest_alloc_mb": summary.memory.largest_alloc_mb,
            "estimated_peak_active_mb": summary.memory.estimated_peak_active_mb,
            "top_alloc_sizes_mb": summary.memory.top_alloc_sizes_mb,
            "avg_allocs_per_block": summary.memory.avg_allocs_per_block,
            "avg_alloc_mb_per_block": summary.memory.avg_alloc_mb_per_block,
        },
        "top_kernels": [
            {
                "name": r.name,
                "calls": r.calls,
                "total_ms": r.total_ms,
                "avg_ms": r.avg_ms,
                "max_ms": r.max_ms,
                "avg_registers": r.avg_registers,
                "avg_threads_per_launch": r.avg_threads_per_launch,
                "category": r.category,
            }
            for r in summary.top_kernels
        ],
        "category_totals_ms": summary.category_totals_ms,
    }


def write_block_csv(path: Path, summary: ProfileSummary) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "label",
                "source",
                "block_id",
                "instances",
                "avg_ms",
                "med_ms",
                "min_ms",
                "max_ms",
                "std_ms",
                "gpu_active_ratio",
                "dominant_kernel",
                "dominant_kernel_ms",
                "dominant_kernel_category",
                "dominant_category_share",
            ]
        )
        for b in summary.block_stats:
            writer.writerow(
                [
                    summary.label,
                    summary.source,
                    b.block_id,
                    b.instances,
                    b.avg_ms,
                    b.med_ms,
                    b.min_ms,
                    b.max_ms,
                    b.std_ms,
                    b.gpu_active_ratio,
                    b.dominant_kernel,
                    b.dominant_kernel_ms,
                    b.dominant_kernel_category,
                    b.dominant_category_share,
                ]
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Nsight Systems attention and memory metrics")
    parser.add_argument("--input", type=Path, required=True, help="Path to Nsight Systems profile (.nsys-rep or .sqlite)")
    parser.add_argument("--label", type=str, default="profile", help="Label for displayed output")
    parser.add_argument("--force-export", action="store_true", help="Force re-export of sqlite from .nsys-rep")
    parser.add_argument("--top-kernels", type=int, default=15, help="How many top kernels to show")
    parser.add_argument("--json-out", type=Path, default=None, help="Optional JSON output path")
    parser.add_argument("--csv-out", type=Path, default=None, help="Optional CSV output path for per-block metrics")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    summary = analyze_profile(
        label=args.label,
        input_path=args.input,
        force_regen=args.force_export,
        top_kernels=args.top_kernels,
    )
    print_block_section(summary)
    print_memory_section(summary)
    print_kernel_section(summary)

    if args.csv_out:
        write_block_csv(args.csv_out, summary)

    if args.json_out:
        payload: dict[str, Any] = {"profile": profile_to_dict(summary)}
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
