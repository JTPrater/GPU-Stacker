# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jason Prater (JP Astrophotography)
"""End-to-end check of the Siril script (siril/GPU Stacker/GPU-Stacker.py) with the real Siril, from the command line.

Usage: .venv\\Scripts\\python tools\\check_siril_script.py [WORK_DIR] [--siril PATH_TO_siril-cli.exe]

Writes synthetic Seestar-style lights, then one siril-cli run: Siril converts, registers (Bayer drizzle)
and stacks them itself, and the script stacks the same r_ sequence and the raw lights through
`pyscript`. Our sequence stack must match Siril's (same math); the raw-lights one must exist.

Siril stacks twice. Its first stack normalizes with statistics measured just then, at full precision, and
saves them in r_light_.seq rounded to 6 significant digits; every later stack, Siril's own and ours, reads
the rounded values. So ours is compared with Siril's second stack (same inputs). The first stack differs
from both by a few float steps and on a few borderline rejected values.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "tests")]

from synth_raw import make_lights  # noqa: E402

from gpustacker.engine.fits import read_image  # noqa: E402

SCRIPT = os.path.join(ROOT, "siril", "GPU Stacker", "GPU-Stacker.py")
SSF = """requires 1.4.0
cd lights
convert light -out=../process
cd ../process
register light -2pass
seqapplyreg light -drizzle -scale=1.0 -pixfrac=1.0 -kernel=square -framing=max
stack r_light rej w 3 3 -norm=addscale -output_norm -rgb_equal -weight=noise -32b -out=../siril_stack_first
stack r_light rej w 3 3 -norm=addscale -output_norm -rgb_equal -weight=noise -32b -out=../siril_stack
pyscript "{script}" r_light --feather 0 --output ../gpu_seq.fit --no-load
cd ..
pyscript "{script}" --lights lights --output gpu_raw.fit
"""


def find_siril() -> str:
    for p in (shutil.which("siril-cli"), r"C:\Program Files\Siril\bin\siril-cli.exe"):
        if p and os.path.isfile(p):
            return p
    raise SystemExit("siril-cli not found: give --siril")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("work", nargs="?", default=os.path.join(tempfile.gettempdir(), "gpustacker_siril_check"))
    ap.add_argument("--siril", default="")
    args = ap.parse_args()
    work = os.path.abspath(args.work)
    shutil.rmtree(work, ignore_errors=True)
    make_lights(work, n=14, w=480, h=360, pattern="RGGB", seed=21)
    with open(os.path.join(work, "check.ssf"), "w", encoding="utf-8") as f:
        f.write(SSF.format(script=SCRIPT.replace("\\", "/")))
    siril = args.siril or find_siril()
    # siril-cli writes its whole settings file back when it exits (wd included): give it a copy of the user's,
    # through XDG_CONFIG_HOME (which Siril honours on Windows too), so the user's own Siril settings stay as they are
    cfg_home = os.path.join(work, "siril-config")
    os.makedirs(os.path.join(cfg_home, "siril"))
    local_siril = os.path.join(os.environ.get("LOCALAPPDATA", ""), "siril")
    for name in os.listdir(local_siril) if os.path.isdir(local_siril) else []:
        if name.startswith("config.") and name.endswith(".ini"):
            shutil.copyfile(os.path.join(local_siril, name), os.path.join(cfg_home, "siril", name))
    out = subprocess.run([siril, "-d", work, "-s", os.path.join(work, "check.ssf")], capture_output=True,
                         text=True, encoding="utf-8", errors="replace", env=dict(os.environ, XDG_CONFIG_HOME=cfg_home))
    log = out.stdout + out.stderr
    for line in log.splitlines():                     # what the script said, plus Siril's own errors
        if "GPU Stacker" in line or "error" in line.lower() or "pyscript" in line:
            print("  siril>", line.strip()[:200])
    ok = True
    for name in ("siril_stack_first.fit", "siril_stack.fit", "gpu_seq.fit", "gpu_raw.fit"):
        exists = os.path.isfile(os.path.join(work, name))
        print(f"{name}: {'written' if exists else 'MISSING'}")
        ok &= exists
    if not ok:
        return 1
    first, a, b = (read_image(os.path.join(work, n))[0] for n in ("siril_stack_first.fit", "siril_stack.fit",
                                                                     "gpu_seq.fit"))
    if a.shape != b.shape:
        print(f"shapes differ: Siril {a.shape}, GPU Stacker {b.shape}")
        return 1
    if not np.allclose(a, b, rtol=0, atol=1e-6) and np.allclose(a[:, ::-1], b, rtol=0, atol=1e-6):
        print("note: the two are stored in opposite row orders")
        first, a = first[:, ::-1], a[:, ::-1]
    report("Siril's first stack vs its second (statistics rounded in the .seq)", a, first)
    return 0 if report("sequence stack vs Siril's second", a, b) < 1e-5 else 1


def report(label: str, a: np.ndarray, b: np.ndarray) -> float:
    """Print how far b is from a, with the pixels that differ by more than 1e-6; -> the max difference."""
    d = np.abs(a.astype(np.float64) - b)
    same = float((a == b).mean()) * 100
    print(f"{label}: max difference {d.max():.2e}, mean {d.mean():.2e}, {same:.3f}% identical")
    big = np.argwhere(d > 1e-6)
    if len(big):
        print("  > 1e-6 at (x, y, channel), top-down:", " ".join(f"({x},{y},{c})" for c, y, x in big[:20]),
              "..." if len(big) > 20 else "")
    return float(d.max())


if __name__ == "__main__":
    sys.exit(main())
