# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jason Prater (JP Astrophotography)
"""GPU result vs the literal CPU port of Siril's rejection + mean, on real per-pixel stacks.

Stacks a few rows of a real sequence, keeps the per-pixel stacks of a small patch, recomputes
each pixel with gpustacker.engine.reference and reports the differences.

Usage: .venv\\Scripts\\python tools\\check_reference.py SEQ X Y [SIZE] [winsorized|sigma|none] [weighting] [feather]
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gpustacker.engine import reference as ref  # noqa: E402
from gpustacker.engine.job import Job, JobConfig  # noqa: E402


def main():
    seq, x, y = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    size = int(sys.argv[4]) if len(sys.argv) > 4 else 16
    mode = sys.argv[5] if len(sys.argv) > 5 else "winsorized"
    weighting = sys.argv[6] if len(sys.argv) > 6 else "none"
    feather = float(sys.argv[7]) if len(sys.argv) > 7 else 0.0
    r0 = (y // 64) * 64
    cfg = JobConfig(seq_path=seq, rows=[r0, r0 + 64], rejection=mode, weighting=weighting, feather=feather,
                    output_norm=False, debug_patch=[x, y, size], preview_max=256)
    job = Job(cfg, lambda ev: print(ev["msg"]) if ev["ev"] == "log" else None)
    t = time.perf_counter()
    job.run()
    print(f"GPU band in {time.perf_counter() - t:.1f} s")
    dbg = job.debug
    rels, depths, kept_rej = [], [], 0
    t = time.perf_counter()
    for c, V, W, imgw in dbg["stacks"]:
        for j in range(size):
            for i in range(size):
                v = V[:, j, i]
                ok = ~np.isnan(v)
                w = (W[:, j, i].astype(np.float64) * imgw)[ok]
                cpu = ref.pixel(v[ok], w, mode, 3.0, 3.0, weighted=not dbg["plain"])
                gpu = float(dbg["values"][c, j, i])
                rels.append(abs(gpu - cpu) / max(abs(cpu), 1e-12))
                depths.append(int(ok.sum()))
    rels = np.array(rels)
    print(f"CPU reference: {len(rels)} pixel-channels, depth {min(depths)}-{max(depths)} "
          f"({time.perf_counter() - t:.0f} s)")
    print(f"  relative difference: median {np.median(rels):.1e}, max {rels.max():.1e}, "
          f"above 1e-6: {(rels > 1e-6).sum()}, above 1e-4: {(rels > 1e-4).sum()}")


if __name__ == "__main__":
    main()
