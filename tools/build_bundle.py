# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jason Prater (JP Astrophotography)
"""Build a portable Windows download: dist\\GPU Stacker\\ (its own Python 3.12, the packages, the app) and
dist\\GPU-Stacker-<version>-win64.zip. Nothing to install: unzip, double-click "Start GPU Stacker.bat".

Usage: .venv\\Scripts\\python tools\\build_bundle.py [--no-zip]

The Python is copied from the interpreter this venv was made from (same version as the packages), the
packages from this venv. A python312._pth file makes the bundled Python ignore anything installed on the
target PC (PYTHONPATH, user site-packages, registry), so a friend's own Python setup can't interfere.
Left out: test-only packages, Python's test suite and Tk, Qt's designer / QML / translation tools and
files, Qt's software OpenGL fallback (the app draws with the raster engine), NVIDIA's alternate NVRTC
build and static libraries.
"""
import argparse
import fnmatch
import os
import shutil
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from gpustacker import __version__  # noqa: E402

DIST = os.path.join(ROOT, "dist")
NAME = "GPU Stacker"

PY_SKIP = ["Lib/site-packages", "Lib/test", "Lib/idlelib", "Lib/tkinter", "Lib/turtledemo", "Lib/ensurepip",
           "Lib/lib2to3", "Lib/__pycache__", "DLLs/_tkinter.pyd", "DLLs/tcl86t.dll", "DLLs/tk86t.dll",
           "DLLs/_test*.pyd", "DLLs/_ctypes_test.pyd", "DLLs/xxlimited*.pyd"]
PY_TOP = ["python.exe", "pythonw.exe", "python3.dll", "python312.dll", "vcruntime140.dll", "vcruntime140_1.dll",
          "LICENSE.txt"]
SITE_SKIP = ["pip", "pip-*", "pytest", "pytest-*", "_pytest", "pluggy", "pluggy-*", "iniconfig", "iniconfig-*",
             "pygments", "pygments-*", "packaging", "packaging-*", "colorama", "colorama-*", "py.py",
             "setuptools", "setuptools-*", "_distutils_hack", "distutils-precedence.pth", "pkg_resources",
             "nvidia/cuda_nvrtc/bin/*.alt.dll", "nvidia/cuda_runtime/lib",
             "PySide6/qml", "PySide6/translations", "PySide6/metatypes", "PySide6/opengl32sw.dll",
             "PySide6/*.exe", "PySide6/*.pyi", "PySide6/Qt6Designer*", "PySide6/QtDesigner.pyd",
             "PySide6/plugins/designer", "PySide6/Qt6Quick*", "PySide6/QtQuick*", "PySide6/Qt6Qml*",
             "PySide6/QtQml.pyd", "PySide6/plugins/qmltooling", "PySide6/plugins/qmllint",
             "PySide6/Qt6OpenGL*", "PySide6/QtOpenGL*", "PySide6/Qt6LabsSettings*", "PySide6/Qt6Help*",
             "PySide6/QtHelp.pyd", "PySide6/Qt6UiTools*", "PySide6/QtUiTools.pyd", "PySide6/include",
             "PySide6/typesystems", "PySide6/glue", "PySide6/examples"]
APP_SKIP = ["__pycache__", "*.pyc"]

# No "import site": the paths below are all the bundle needs (no .pth files are used), and without it the
# user's own site-packages (AppData\Roaming\Python\Python312) and its startup hooks stay out entirely.
PTH = "Lib\nDLLs\nLib\\site-packages\n..\\app\n"

LAUNCHER = r"""@echo off
rem GPU Stacker: starts the app. Optional argument: a folder of lights or a Siril .seq file to open.
start "" "%~dp0python\pythonw.exe" -m gpustacker %*
"""

README = """GPU Stacker {version}
====================

GPU Stacker stacks astrophotography subs on an NVIDIA graphics card.

Point it at a folder of raw lights (the Seestar's folder of subs works as it is) and it does the whole
job: it finds the stars in every sub, registers them (mosaics too), drizzles the colour data straight
from the camera's Bayer pattern, and combines everything with sigma or winsorized rejection,
normalization, weighting and feathering. The stacking math is the same as Siril's, so the results are
the same quality; the graphics card just gets there several times faster. No converted or registered
copies of your subs are written, only the result (and a cache folder that makes re-stacks fast).

It can also stack a sequence you registered in Siril, and it works from inside Siril as a script
(see "Using it from Siril" below).

What you need
-------------
- Windows 10 or 11, 64-bit
- An NVIDIA graphics card (GTX 10-series or newer) with a current driver.
  AMD, Intel and Mac graphics are not supported.
- Disk space for the result next to your data (about 100-150 MB per stack).

Getting started
---------------
1. Unzip this folder anywhere (not inside the zip viewer).
2. Double-click "Start GPU Stacker.bat". If Windows says it protected your PC because the file came
   from the internet, click "More info", then "Run anyway". The first start takes about a minute
   longer while the GPU code is prepared; later starts are quick.
3. Click "Open a folder of raw lights" and pick your lights folder (or the folder above it).
4. Optional: "Analyze" finds the stars and registers every sub, so you can check them in the Frames
   tab and untick bad ones first. "Test strip" stacks a narrow band to try settings quickly.
5. Click "Stack". The result is a 32-bit FITS file next to your data, ready for Siril, PixInsight or
   GraXpert. Colour-calibrate as usual: like any Bayer-drizzled stack, it is a little green before
   calibration.

Using it from Siril
-------------------
GPU Stacker can run from Siril's Scripts menu. In Siril, open Preferences > Scripts, add this
folder on a new line under "Scripts Storage Directories":
    app\\siril\\GPU Stacker      (inside this GPU Stacker folder; use the full path)
then click the refresh button under that list and Apply. (In GPU Stacker, File > "Add to Siril's
Scripts menu" copies the full path for you.) Siril then has Scripts > Python Scripts > GPU Stacker >
GPU-Stacker.py: it stacks the registered sequence you have loaded (the r_ sequence), or a folder of
raw lights, and opens the result in Siril. Siril scripts can call it as well, for example
    pyscript GPU-Stacker.py r_pp_light
("pyscript GPU-Stacker.py --help" lists the options). If you move the GPU Stacker folder, update the
path in Siril.

Good to know
------------
- Default settings: Bayer drizzle 1x, winsorized rejection 3/3, additive + scaling normalization,
  noise weighting, feather 20.
- The slow steps' results are cached in a "gpustacker" folder next to your data, so stacking the same
  subs again with other settings is fast. You can delete that folder any time.
- If something goes wrong, the Log tab shows what happened, and errors are also written to
  %LOCALAPPDATA%\\GPU Stacker\\gpu-stacker.log - please send that file along.
- To remove GPU Stacker, delete this folder (and the cache folders next to your data).

License
-------
GPU Stacker {version}
Copyright (C) 2026 Jason Prater (JP Astrophotography)

GPU Stacker is free software: you can redistribute it and/or modify it under the terms of the GNU
General Public License as published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version. It is distributed in the hope that it will be
useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See LICENSE.txt for the full license.

It contains code ported from Siril, RawTherapee, CFITSIO and OpenCV, and this download includes
Python, Qt for Python, NumPy, CuPy and NVIDIA's CUDA runtime: their notices and licenses are in
THIRD-PARTY-NOTICES.txt and the LICENSES folder. The complete source code is in the app folder.
"""


def _skip(rel: str, patterns) -> bool:
    rel = rel.replace("\\", "/")
    return any(fnmatch.fnmatch(rel, p) or rel.startswith(p.rstrip("*") + "/") and "*" not in p for p in patterns)


def copy_tree(src: str, dst: str, skip) -> tuple[int, int]:
    files = size = 0
    for d, dirs, names in os.walk(src):
        rel_d = os.path.relpath(d, src)
        rel_d = "" if rel_d == "." else rel_d
        dirs[:] = [x for x in dirs if not _skip(os.path.join(rel_d, x), skip)]
        for n in names:
            rel = os.path.join(rel_d, n)
            if _skip(rel, skip):
                continue
            out = os.path.join(dst, rel)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            shutil.copy2(os.path.join(d, n), out)
            files += 1
            size += os.path.getsize(out)
    return files, size


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-zip", action="store_true")
    ap.add_argument("--dist", default=DIST, help="output folder (default: dist); use another one while a copy "
                                                 "in dist is running")
    args = ap.parse_args()
    dist = os.path.abspath(args.dist)
    if sys.prefix == sys.base_prefix:
        raise SystemExit("run this with the project's venv Python (.venv\\Scripts\\python)")
    out = os.path.join(dist, NAME)
    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(out)

    base = sys.base_prefix
    py = os.path.join(out, "python")
    os.makedirs(py)
    for f in PY_TOP:
        shutil.copy2(os.path.join(base, f), os.path.join(py, f))
    n1, s1 = copy_tree(os.path.join(base, "Lib"), os.path.join(py, "Lib"),
                       [p[4:] for p in PY_SKIP if p.startswith("Lib/")])
    n2, s2 = copy_tree(os.path.join(base, "DLLs"), os.path.join(py, "DLLs"),
                       [p[5:] for p in PY_SKIP if p.startswith("DLLs/")])
    site = os.path.join(sys.prefix, "Lib", "site-packages")
    n3, s3 = copy_tree(site, os.path.join(py, "Lib", "site-packages"), SITE_SKIP)
    with open(os.path.join(py, "python312._pth"), "w", encoding="ascii") as f:
        f.write(PTH)
    n4, s4 = copy_tree(os.path.join(ROOT, "gpustacker"), os.path.join(out, "app", "gpustacker"), APP_SKIP)
    n5, s5 = copy_tree(os.path.join(ROOT, "siril"), os.path.join(out, "app", "siril"), APP_SKIP)    # Siril script
    for extra in ("tests", "tools"):                 # the complete source (GPL): tests and this build script too
        n, s = copy_tree(os.path.join(ROOT, extra), os.path.join(out, "app", extra), APP_SKIP)
        n5, s5 = n5 + n, s5 + s
    shutil.copy2(os.path.join(ROOT, "pyproject.toml"), os.path.join(out, "app", "pyproject.toml"))
    shutil.copy2(os.path.join(ROOT, "LICENSE"), os.path.join(out, "LICENSE.txt"))       # license files: verbatim
    shutil.copy2(os.path.join(ROOT, "THIRD-PARTY-NOTICES.txt"), os.path.join(out, "THIRD-PARTY-NOTICES.txt"))
    copy_tree(os.path.join(ROOT, "LICENSES"), os.path.join(out, "LICENSES"), [])
    with open(os.path.join(out, "Start GPU Stacker.bat"), "w", encoding="ascii", newline="\r\n") as f:
        f.write(LAUNCHER)
    with open(os.path.join(out, "README.txt"), "w", encoding="utf-8", newline="\r\n") as f:
        f.write(README.format(version=__version__))
    total = s1 + s2 + s3 + s4 + s5
    print(f"python: {n1 + n2} files {(s1 + s2) / 1e6:.0f} MB; packages: {n3} files {s3 / 1e6:.0f} MB; "
          f"app: {n4 + n5} files; total {total / 1e6:.0f} MB -> {out}")

    if not args.no_zip:
        zpath = os.path.join(dist, f"GPU-Stacker-{__version__}-win64.zip")
        if os.path.exists(zpath):
            os.remove(zpath)
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for d, _dirs, names in os.walk(out):
                for n in names:
                    p = os.path.join(d, n)
                    z.write(p, os.path.relpath(p, dist))
        print(f"zip: {os.path.getsize(zpath) / 1e6:.0f} MB -> {zpath}")


if __name__ == "__main__":
    main()
