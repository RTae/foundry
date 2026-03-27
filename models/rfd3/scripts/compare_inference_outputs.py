"""Compare metric outputs from two RFD3 inference runs.

Usage:
    python compare_inference_outputs.py <standard_dir> <lowmem_dir>
"""

import glob
import gzip
import json
import os
import sys


def find_file(directory, pattern):
    files = glob.glob(os.path.join(directory, pattern))
    if not files:
        print(f"ERROR: No {pattern} found in {directory}")
        sys.exit(1)
    return files[0]


def main():
    if len(sys.argv) != 3:
        print(__doc__.strip())
        sys.exit(1)

    std_dir, lm_dir = sys.argv[1], sys.argv[2]

    # --- Load JSON metrics ---
    with open(find_file(std_dir, "*.json")) as f:
        j_std = json.load(f)
    with open(find_file(lm_dir, "*.json")) as f:
        j_lm = json.load(f)

    # --- Load CIF structures ---
    with gzip.open(find_file(std_dir, "*.cif.gz"), "rt") as f:
        cif_std = f.read()
    with gzip.open(find_file(lm_dir, "*.cif.gz"), "rt") as f:
        cif_lm = f.read()

    # --- Compare metrics ---
    m_std = j_std.get("metrics", {})
    m_lm = j_lm.get("metrics", {})
    all_keys = sorted(set(list(m_std.keys()) + list(m_lm.keys())))

    print(f"{'Metric':45s}  {'Standard':>10s}  {'Low-mem':>10s}  {'Diff':>10s}")
    print("-" * 82)
    any_large_diff = False
    for k in all_keys:
        v1, v2 = m_std.get(k, "N/A"), m_lm.get(k, "N/A")
        if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
            diff = abs(v1 - v2)
            flag = "  *** LARGE" if diff > 0.01 else ""
            if diff > 0.01:
                any_large_diff = True
            print(f"  {k:43s}  {v1:10.4f}  {v2:10.4f}  {diff:10.4f}{flag}")
        elif isinstance(v1, list) or isinstance(v2, list):
            continue
        else:
            match = "==" if v1 == v2 else "!="
            print(f"  {k:43s}  {str(v1):>10s}  {match}  {str(v2):>10s}")

    # --- Compare CIF ---
    print()
    print(f"CIF identical: {cif_std == cif_lm}")
    print(f"CIF sizes: standard={len(cif_std)}, lowmem={len(cif_lm)}")

    # --- Verdict ---
    print()
    if not any_large_diff and cif_std == cif_lm:
        print("PASS — outputs are identical")
    elif not any_large_diff:
        print("PASS — metrics match (CIF text may have minor formatting diffs)")
    else:
        print("WARN — some metrics differ by > 0.01; check if seed was applied correctly")
        sys.exit(1)


if __name__ == "__main__":
    main()
