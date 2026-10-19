# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jason Prater (JP Astrophotography)
"""GPU Stacker inside Siril: the Siril script runs this installation's engine.

The script lives in this installation, in siril/GPU Stacker/; the user adds that folder to Siril's script
folders (Preferences > Scripts) and Siril, which names each Python-script submenu after the script's folder,
shows it as Scripts > Python Scripts > GPU Stacker > GPU-Stacker.py. Nothing is copied into Siril's folders
and Siril's settings are never edited from here: every Siril process (siril-cli too) writes its whole settings
file back when it exits, so outside edits don't stick. At every start the app records where it is
(location.json, next to its log) for copies of the script kept elsewhere.
"""
from __future__ import annotations

import glob
import json
import os
import re

from .. import __version__
from .runner import ROOT, engine_python

SCRIPT_NAME = "GPU-Stacker.py"
SCRIPT_FOLDER = os.path.join(ROOT, "siril", "GPU Stacker")


def state_dir() -> str:
    d = os.path.join(os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"), "GPU Stacker")
    os.makedirs(d, exist_ok=True)
    return d


def script_source() -> str:
    return os.path.join(SCRIPT_FOLDER, SCRIPT_NAME)


def record_location() -> None:
    data = {"python": engine_python(), "package_dir": ROOT, "version": __version__}
    path = os.path.join(state_dir(), "location.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
    os.replace(tmp, path)


def _same(a: str, b: str) -> bool:
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def siril_config() -> str | None:
    """The newest config.<version>.ini Siril has written (it keeps its settings there)."""
    base = os.path.join(os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"), "siril")
    configs = glob.glob(os.path.join(base, "config.*.ini"))
    return max(configs, key=lambda p: tuple(int(x) for x in re.findall(r"\d+", os.path.basename(p))), default=None)


def _list_value(raw: str) -> list[str]:
    """A GLib key-file string list: items end with ';', backslashes are doubled."""
    return [p.replace("\\\\", "\\") for p in raw.strip().split(";") if p.strip()]


def siril_script_dirs() -> list[str]:
    """Siril's script folders as last saved in its settings (script_path); [] if it has none yet."""
    cfg = siril_config()
    if cfg:
        try:
            with open(cfg, encoding="utf-8", errors="replace") as f:
                for line in f:
                    if line.startswith("script_path="):
                        return _list_value(line.split("=", 1)[1])
        except OSError:
            pass
    return []


def siril_lists_folder() -> bool:
    """Whether Siril's saved script folders include this installation's script folder."""
    return any(_same(d, SCRIPT_FOLDER) for d in siril_script_dirs())


def on_start() -> None:
    """Record this installation for copies of the Siril script kept elsewhere; never fails."""
    try:
        record_location()
    except Exception:                                # noqa: BLE001 - nothing here may stop the app
        pass
