# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jason Prater (JP Astrophotography)
"""The Siril script (siril/GPU Stacker/GPU-Stacker.py) without Siril: a stand-in sirilpy, the real engine."""
import enum
import importlib.util
import json
import os
import sys
import types

import pytest

from gpustacker.engine.plan import auto_output_path, output_folder

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class LogColor(enum.IntEnum):
    DEFAULT = 0
    RED = 1
    SALMON = 2
    GREEN = 3
    BLUE = 4


class FakeSiril:
    """What the script uses of sirilpy.SirilInterface."""

    def __init__(self, wd, seq=None):
        self.wd, self.seq = wd, seq
        self.logs, self.cmds, self.progress = [], [], []

    def log(self, msg, color=LogColor.DEFAULT):
        self.logs.append((msg, int(color)))

    def update_progress(self, msg, p):
        self.progress.append((msg, p))

    def reset_progress(self):
        pass

    def cmd(self, *args):
        self.cmds.append(args)

    def get_siril_wd(self):
        return self.wd

    def is_sequence_loaded(self):
        return self.seq is not None

    def get_seq(self):
        return self.seq

    def is_cli(self):
        return True


@pytest.fixture(scope="module")
def script():
    fake = types.ModuleType("sirilpy")
    fake.SirilError = type("SirilError", (Exception,), {})
    fake.SirilConnectionError = fake.SirilError
    fake.LogColor = LogColor
    saved = sys.modules.get("sirilpy")
    sys.modules["sirilpy"] = fake
    try:
        spec = importlib.util.spec_from_file_location("gpu_stacker_siril", os.path.join(ROOT, "siril", "GPU Stacker", "GPU-Stacker.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        if saved is None:
            sys.modules.pop("sirilpy", None)
        else:
            sys.modules["sirilpy"] = saved
    return mod


def _touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").close()


def test_sequence_names_resolve_the_siril_way(script, tmp_path):
    _touch(tmp_path / "r_light_.seq")
    wd = str(tmp_path)
    for name in ("r_light", "r_light_", "r_light_.seq"):
        assert script.sequence_path(name, wd) == os.path.join(wd, "r_light_.seq")
    assert script.sequence_path("other", wd) == os.path.join(wd, "other_.seq")


def test_default_lights_skips_sirils_process_folder(script, tmp_path):
    _touch(tmp_path / "lights" / "Light_001.fit")
    _touch(tmp_path / "process" / "light_00001.fit")
    _touch(tmp_path / "process" / "light_.seq")
    assert script.default_lights(str(tmp_path / "process")) == str(tmp_path / "lights")
    assert script.default_lights(str(tmp_path)) == str(tmp_path / "lights")
    assert script.default_lights(str(tmp_path / "lights")) == str(tmp_path / "lights")
    assert not script.is_raw_lights(str(tmp_path / "process"))


def test_finds_gpu_stacker_in_the_download_and_in_a_checkout(script, tmp_path):
    zipped = tmp_path / "GPU Stacker"
    _touch(zipped / "python" / "python.exe")
    _touch(zipped / "app" / "gpustacker" / "__init__.py")
    want = {"python": str(zipped / "python" / "python.exe"), "package_dir": str(zipped / "app")}
    assert script.install_in(str(zipped)) == want
    assert script.install_in(str(zipped / "app")) == want
    checkout = tmp_path / "src"
    _touch(checkout / ".venv" / "Scripts" / "python.exe")
    _touch(checkout / "gpustacker" / "__init__.py")
    assert script.install_in(str(checkout))["package_dir"] == str(checkout)
    assert script.install_in(str(tmp_path / "nothing")) is None


def test_options_reach_the_job(script):
    a = script.parse_args(["r_x", "--rejection", "sigma", "--sigma", "2.5", "4", "--weighting", "wfwhm",
                           "--feather", "0", "--no-rgb-equal", "--scale", "2", "--pixfrac", "0.7"])
    assert a.sequence == "r_x" and a.sigma == [2.5, 4.0] and a.scale == 2.0
    st = dict(script.DEFAULTS, rejection=a.rejection, sigma_low=2.5, sigma_high=4.0, weighting=a.weighting,
              feather=a.feather, rgb_equal=False, drizzle_scale=a.scale, pixfrac=a.pixfrac)
    seq = script.job_config(st, seq_path="x.seq", include=[1, 3])
    assert seq["seq_path"] == "x.seq" and seq["include"] == [1, 3] and "lights_dir" not in seq
    assert seq["rejection"] == "sigma" and seq["rgb_equal"] is False and seq["feather"] == 0
    raw = script.job_config(st, lights_dir="L")
    assert raw["lights_dir"] == "L" and raw["drizzle_scale"] == 2.0 and raw["pixfrac"] == 0.7


def test_command_line_stacks_raw_lights_through_the_engine(script, tmp_path):
    sys.path.insert(0, os.path.join(ROOT, "tests"))
    from synth_raw import make_lights
    make_lights(str(tmp_path), n=6, seed=8)
    siril = FakeSiril(str(tmp_path))
    rc = script.run_cli(siril, ["--lights", "lights", "--output", "out.fit", "--gpu-stacker", ROOT])
    assert rc == 0, siril.logs[-5:]
    assert os.path.isfile(tmp_path / "out.fit")
    assert siril.cmds == [("load", '"' + str(tmp_path / "out.fit").replace("\\", "/") + '"')]
    assert any(c == LogColor.GREEN and "stacked" in m for m, c in siril.logs)
    assert any(m.startswith("GPU Stacker: Stacking") for m, _ in siril.progress)


def test_engine_errors_come_back_as_a_message(script, tmp_path):
    siril = FakeSiril(str(tmp_path))
    rc = script.run_cli(siril, ["missing_seq", "--gpu-stacker", ROOT])
    assert rc == 1
    assert any(c == LogColor.RED and "missing_seq" in m for m, c in siril.logs)


def test_output_names(tmp_path):
    p = auto_output_path(str(tmp_path), "M 45", 906, 15.1 * 3600)
    name = os.path.basename(p)
    assert name.startswith("M_45_906frames_15.1h_gpu_") and name.endswith(".fit")
    open(p, "w").close()
    assert auto_output_path(str(tmp_path), "M 45", 906, 15.1 * 3600).endswith("_2.fit")
    assert output_folder(str(tmp_path / "process")) == str(tmp_path)
    assert output_folder(str(tmp_path / "data")) == str(tmp_path / "data")


def test_siril_finds_the_script_in_the_installation(tmp_path, monkeypatch, script):
    """The script stays in siril/GPU Stacker (Siril names the submenu after that folder): it finds GPU Stacker
    above it, and the app tells whether Siril lists that folder, without ever editing Siril's settings."""
    from gpustacker.ui import siril_link
    assert os.path.basename(siril_link.SCRIPT_FOLDER) == "GPU Stacker"
    assert os.path.isfile(siril_link.script_source())
    monkeypatch.setattr(script, "LOCATION", str(tmp_path / "none.json"))     # no recorded location
    found = script.find_gpu_stacker()
    assert found and os.path.normcase(found["package_dir"]) == os.path.normcase(ROOT)

    local = tmp_path / "Local"
    cfg = local / "siril" / "config.1.4.ini"
    os.makedirs(cfg.parent)
    monkeypatch.setenv("LOCALAPPDATA", str(local))
    program_files = "C:\\\\Program Files\\\\Siril\\\\scripts;"           # as Siril saves it: backslashes doubled
    cfg.write_text("[gui]\nscript_path=" + program_files + "\n")
    assert siril_link.siril_script_dirs() == ["C:\\Program Files\\Siril\\scripts"]
    assert not siril_link.siril_lists_folder()
    folder = siril_link.SCRIPT_FOLDER.replace("\\", "\\\\")
    cfg.write_text("[gui]\nscript_path=" + program_files + folder + ";\n")
    before = cfg.read_text()
    assert siril_link.siril_lists_folder()
    siril_link.on_start()                              # records the location only
    assert cfg.read_text() == before
    with open(local / "GPU Stacker" / "location.json", encoding="utf-8") as f:
        loc = json.load(f)
    assert os.path.isfile(os.path.join(loc["package_dir"], "gpustacker", "__init__.py"))
