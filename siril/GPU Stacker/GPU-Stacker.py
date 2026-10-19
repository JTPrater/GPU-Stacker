# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jason Prater (JP Astrophotography)
r"""
GPU Stacker
===========
Stacks in Siril on an NVIDIA graphics card, with GPU Stacker doing the work:

- a registered sequence (the r_ sequence Siril writes when it applies the registration): the same
  result as Siril's own `stack ... rej` with additive + scaling normalization, weights and feather,
  usually several times faster;
- or a folder of raw lights straight from the telescope: star detection, registration, Bayer drizzle
  and stacking in one go, with nothing written but the result.

The result opens in Siril. The script runs from GPU Stacker's own folder, which Siril lists as a script
folder (Preferences > Scripts: <GPU Stacker>\app\siril\GPU Stacker); Siril then shows it as Scripts >
Python Scripts > GPU Stacker > GPU-Stacker.py. A copy kept elsewhere finds GPU Stacker through the location
GPU Stacker records each time it starts, or the folder you point it to once.

From the Scripts menu it opens a window. From a Siril script or the command line it runs without one:

    pyscript GPU-Stacker.py r_pp_light --feather 0
    pyscript GPU-Stacker.py --lights lights --scale 2 --pixfrac 0.8

(`pyscript GPU-Stacker.py --help` lists every option.)

SPDX-License-Identifier: GPL-3.0-or-later
"""
# Version history
# 1.0.0  First version

import argparse
import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time

import sirilpy as s

VERSION = "1.0.0"
TITLE = "GPU Stacker"
CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0
STATE_DIR = os.path.join(os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"), "GPU Stacker")
LOCATION = os.path.join(STATE_DIR, "location.json")        # written by GPU Stacker at every start
SETTINGS = os.path.join(STATE_DIR, "siril_script.json")    # this window's last settings
FITS_EXT = (".fit", ".fits", ".fts")

DEFAULTS = dict(rejection="winsorized", sigma_low=3.0, sigma_high=3.0, normalization="addscale",
                weighting="noise", feather=20.0, rgb_equal=True, output_norm=True, drizzle_scale=1.0,
                pixfrac=1.0, load_result=True)
REJECTIONS = [("winsorized", "Winsorized sigma clipping"), ("sigma", "Sigma clipping"), ("none", "None")]
NORMALIZATIONS = [("addscale", "Additive with scaling"), ("none", "None")]
WEIGHTINGS = [("noise", "Noise"), ("wfwhm", "Weighted FWHM"), ("nbstars", "Number of stars"), ("none", "None")]
SCALES = [(1.0, "1x"), (1.5, "1.5x"), (2.0, "2x"), (3.0, "3x")]


# ----------------------------------------------------------------------------- finding GPU Stacker

def _install_at(package_dir: str, python: str) -> dict | None:
    if python and os.path.isfile(python) and os.path.isfile(os.path.join(package_dir, "gpustacker", "__init__.py")):
        return {"python": python, "package_dir": package_dir}
    return None


def install_in(folder: str) -> dict | None:
    """GPU Stacker in a folder the user picked: the unzipped download (or its app folder), or a source
    checkout with its .venv."""
    folder = os.path.abspath(folder)
    for package_dir, python in ((os.path.join(folder, "app"), os.path.join(folder, "python", "python.exe")),
                                (folder, os.path.join(os.path.dirname(folder), "python", "python.exe")),
                                (folder, os.path.join(folder, ".venv", "Scripts", "python.exe"))):
        found = _install_at(package_dir, python)
        if found:
            return found
    return None


def find_gpu_stacker(folder: str = "") -> dict | None:
    """{'python', 'package_dir'} of GPU Stacker: the folder given, the installation this script sits in
    (<GPU Stacker>/app/siril/GPU Stacker, or a source checkout's siril/GPU Stacker), or the location GPU Stacker
    last recorded (for a copy of the script kept elsewhere)."""
    if folder:
        return install_in(folder)
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(3):
        d = os.path.dirname(d)
        found = install_in(d)
        if found:
            return found
    try:
        with open(LOCATION, encoding="utf-8") as f:
            loc = json.load(f)
        return _install_at(loc.get("package_dir", ""), loc.get("python", ""))
    except (OSError, ValueError, AttributeError):
        return None


def remember_install(install: dict) -> None:
    try:
        os.makedirs(STATE_DIR, exist_ok=True)
        with open(LOCATION, "w", encoding="utf-8") as f:
            json.dump(install, f, indent=1)
    except OSError:
        pass


# ----------------------------------------------------------------------------- what to stack

def fits_count(folder: str) -> int:
    try:
        return sum(1 for n in os.listdir(folder) if n.lower().endswith(FITS_EXT))
    except OSError:
        return 0


def lights_in(folder: str) -> str:
    """The folder of lights GPU Stacker will use for `folder`: itself, or its 'lights' subfolder."""
    sub = os.path.join(folder, "lights")
    return sub if fits_count(sub) else folder


def is_raw_lights(folder: str) -> bool:
    """FITS files and no Siril sequence (a folder with a .seq holds Siril's converted or registered frames)."""
    try:
        return fits_count(folder) > 0 and not any(n.lower().endswith(".seq") for n in os.listdir(folder))
    except OSError:
        return False


def default_lights(wd: str) -> str:
    """Raw lights near Siril's working directory: its 'lights' folder, the one next to it (when the working
    directory is Siril's 'process' folder), or the directory itself."""
    for d in (os.path.join(wd, "lights"), os.path.join(os.path.dirname(os.path.normpath(wd)), "lights"), wd):
        if is_raw_lights(d):
            return os.path.normpath(d)
    return ""


def loaded_sequence(siril) -> dict | None:
    """The sequence open in Siril: its .seq file and the frames selected right now (unsaved changes too)."""
    try:
        if not siril.is_sequence_loaded():
            return None
        seq = siril.get_seq()
    except s.SirilError:
        return None
    if seq is None:
        return None
    include = [im.filenum for im in seq.imgparam if im.incl]
    path = sequence_path(seq.seqname, siril.get_siril_wd())
    return {"name": seq.seqname, "path": path, "include": include, "total": seq.number}


def sequence_path(name: str, wd: str) -> str:
    """The .seq file of a sequence name, Siril's way: r_light, r_light_ and r_light_.seq all mean r_light_.seq."""
    path = name if os.path.isabs(name) else os.path.join(wd, name)
    if path.lower().endswith(".seq"):
        return path
    for cand in (path + ".seq", path + "_.seq"):
        if os.path.isfile(cand):
            return cand
    return path + (".seq" if path.endswith("_") else "_.seq")


def job_config(st: dict, *, seq_path: str = "", include=None, lights_dir: str = "", output_path: str = "") -> dict:
    cfg = {k: st[k] for k in ("rejection", "sigma_low", "sigma_high", "normalization", "weighting", "feather",
                              "rgb_equal", "output_norm")}
    if lights_dir:
        cfg.update(lights_dir=lights_dir, drizzle_scale=st["drizzle_scale"], pixfrac=st["pixfrac"])
    else:
        cfg.update(seq_path=seq_path, include=include)
    cfg["output_path"] = output_path
    return cfg


# ----------------------------------------------------------------------------- running the engine

class Engine:
    """One GPU Stacker job in its own process: `python -m gpustacker.engine run JOB`, which reports
    JSON lines (log, phase, progress, done, error, ...). Stopping it = creating the cancel file; it also
    stops by itself if this script goes away."""

    def __init__(self, install: dict, cfg: dict):
        work = os.path.join(tempfile.gettempdir(), "gpustacker")
        os.makedirs(work, exist_ok=True)
        self.job = os.path.join(work, f"siril_job_{time.strftime('%Y%m%d-%H%M%S')}_{os.getpid()}.json")
        with open(self.job, "w", encoding="utf-8") as f:
            json.dump(dict(cfg, preview_dir=work, preview_max=64), f, indent=1)
        self.cancel_file = self.job + ".cancel"
        # a clean environment: Siril's own Python settings must not leak into GPU Stacker's Python
        env = {k: v for k, v in os.environ.items()
               if not k.upper().startswith(("PYTHON", "VIRTUAL_ENV", "CONDA", "MY_PIPE", "MY_SOCKET"))}
        env.update(PYTHONPATH=install["package_dir"], PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1",
                   PYTHONWARNINGS="ignore::UserWarning")
        self.stderr: list[str] = []
        self.proc = subprocess.Popen(
            [install["python"], "-m", "gpustacker.engine", "run", self.job, "--cancel-file", self.cancel_file,
             "--parent-pid", str(os.getpid())],
            cwd=install["package_dir"], env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace", bufsize=1,
            creationflags=CREATE_NO_WINDOW)
        threading.Thread(target=self._read_stderr, daemon=True).start()

    def _read_stderr(self) -> None:
        for line in self.proc.stderr:
            if line.strip():
                self.stderr = (self.stderr + [line.rstrip()])[-40:]

    def events(self):
        for line in self.proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except ValueError:
                self.stderr = (self.stderr + [line])[-40:]
        self.proc.wait()

    def stop(self) -> None:
        try:
            with open(self.cancel_file, "w") as f:
                f.write("cancel")
        except OSError:
            self.proc.kill()

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
        for p in (self.job, self.cancel_file):
            try:
                os.remove(p)
            except OSError:
                pass


class Report:
    """Engine events -> Siril's log and progress bar (and an optional status callback); keeps the outcome."""

    def __init__(self, siril, status=None):
        self.siril = siril
        self.status = status or (lambda text, fraction: None)
        self.label = ""
        self.last_progress = 0.0
        self.summary = None
        self.error = None
        self.cancelled = False

    def log(self, msg: str, color=s.LogColor.DEFAULT) -> None:
        try:
            self.siril.log(msg, color)
        except s.SirilError:
            print(msg)

    def progress(self, fraction: float, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self.last_progress < 0.25:
            return
        self.last_progress = now
        try:
            self.siril.update_progress(f"{TITLE}: {self.label}", fraction)
        except (s.SirilError, ValueError):
            pass
        self.status(self.label, fraction)

    def __call__(self, ev: dict) -> None:
        kind = ev.get("ev")
        if kind == "hello":
            gpu = ev.get("gpu") or {}
            self.log(f"{TITLE} {ev.get('version', '')} on {gpu.get('name', 'the graphics card')}")
        elif kind == "log":
            level = ev.get("level")
            color = s.LogColor.SALMON if level == "warn" else s.LogColor.RED if level == "error" else s.LogColor.DEFAULT
            self.log(ev.get("msg", ""), color)
        elif kind == "phase":
            self.label = ev.get("label") or ev.get("name", "")
            self.log(f"{TITLE}: {self.label}", s.LogColor.BLUE)
            self.progress(0.0, force=True)
        elif kind == "progress":
            total = ev.get("total") or 0
            self.progress(min(1.0, max(0.0, ev.get("done", 0) / total)) if total else -1.0)
        elif kind == "frame_bad":
            self.log(f"frame {ev.get('number', '?')} not used: {ev.get('reason', '')}", s.LogColor.SALMON)
        elif kind == "done":
            self.summary = ev
        elif kind == "cancelled":
            self.cancelled = True
        elif kind == "error":
            self.error = ev.get("msg") or "unknown error"
            if not ev.get("user") and ev.get("trace"):
                print(ev["trace"])


def run_job(siril, install: dict, cfg: dict, load_result: bool, status=None, started=None) -> Report:
    """Run one stack to the end, reporting into Siril; opens the result in Siril if asked."""
    rep = Report(siril, status)
    eng = Engine(install, cfg)
    if started:
        started(eng)
    try:
        for ev in eng.events():
            rep(ev)
    finally:
        eng.close()
    if rep.summary is None and rep.error is None and not rep.cancelled:
        rep.error = ("GPU Stacker stopped unexpectedly: " + " | ".join(eng.stderr[-3:])) if eng.stderr else \
            f"GPU Stacker stopped unexpectedly (exit code {eng.proc.returncode})"
    try:
        siril.reset_progress()
    except s.SirilError:
        pass
    if rep.error:
        rep.log(f"{TITLE}: {rep.error}", s.LogColor.RED)
    elif rep.cancelled:
        rep.log(f"{TITLE}: stopped", s.LogColor.SALMON)
    elif rep.summary:
        out = rep.summary.get("output")
        sm = rep.summary
        live = sm.get("livetime") or 0
        total = f"{live / 3600:.1f} h" if live >= 3600 else f"{live / 60:.0f} min"
        rep.log(f"{TITLE}: {sm.get('frames', 0)} frames ({total}) stacked in {sm.get('seconds_total', 0):.0f} s "
                f"-> {out}", s.LogColor.GREEN)
        if out and load_result:
            try:
                siril.cmd("load", '"' + out.replace("\\", "/") + '"')
            except s.SirilError as e:
                rep.log(f"{TITLE}: couldn't open the result in Siril: {e}", s.LogColor.SALMON)
    return rep


# ----------------------------------------------------------------------------- settings

def load_settings() -> dict:
    st = dict(DEFAULTS)
    try:
        with open(SETTINGS, encoding="utf-8") as f:
            saved = json.load(f)
        st.update({k: type(DEFAULTS[k])(v) for k, v in saved.items() if k in DEFAULTS})
    except (OSError, ValueError, TypeError):
        pass
    return st


def save_settings(st: dict) -> None:
    try:
        os.makedirs(STATE_DIR, exist_ok=True)
        with open(SETTINGS, "w", encoding="utf-8") as f:
            json.dump(st, f, indent=1)
    except OSError:
        pass


# ----------------------------------------------------------------------------- command line (pyscript)

def parse_args(argv):
    p = argparse.ArgumentParser(prog="GPU-Stacker.py", description="Stack with GPU Stacker from Siril.")
    p.add_argument("sequence", nargs="?", help="registered sequence to stack, in Siril's working directory "
                                               "(default: the one loaded in Siril)")
    p.add_argument("--lights", metavar="FOLDER", help="stack a folder of raw lights instead (registration, "
                                                      "Bayer drizzle and stacking in one go)")
    p.add_argument("--rejection", choices=[k for k, _ in REJECTIONS], default=DEFAULTS["rejection"])
    p.add_argument("--sigma", nargs=2, type=float, metavar=("LOW", "HIGH"),
                   default=[DEFAULTS["sigma_low"], DEFAULTS["sigma_high"]])
    p.add_argument("--normalization", choices=[k for k, _ in NORMALIZATIONS], default=DEFAULTS["normalization"])
    p.add_argument("--weighting", choices=[k for k, _ in WEIGHTINGS], default=DEFAULTS["weighting"])
    p.add_argument("--feather", type=float, default=DEFAULTS["feather"],
                   help="Siril's feather distance (default %(default)g, 0 = off)")
    p.add_argument("--no-rgb-equal", action="store_true", help="don't equalize the RGB channels")
    p.add_argument("--no-output-norm", action="store_true", help="don't normalize the result to [0, 1]")
    p.add_argument("--scale", type=float, choices=[k for k, _ in SCALES], default=DEFAULTS["drizzle_scale"],
                   help="drizzle scale (raw lights)")
    p.add_argument("--pixfrac", type=float, default=DEFAULTS["pixfrac"], help="drizzle pixfrac (raw lights)")
    p.add_argument("--output", "-o", metavar="FILE", default="",
                   help="result file (default: named after the stack, next to the data)")
    p.add_argument("--no-load", action="store_true", help="don't open the result in Siril")
    p.add_argument("--gpu-stacker", metavar="FOLDER", default="", help="GPU Stacker's folder, if not found")
    return p.parse_args(argv)


def run_cli(siril, argv) -> int:
    a = parse_args(argv)
    install = find_gpu_stacker(a.gpu_stacker)
    if install is None:
        siril.log(f"{TITLE}: GPU Stacker was not found. Start GPU Stacker once (it records where it is), or "
                  "give its folder with --gpu-stacker.", s.LogColor.RED)
        return 1
    wd = siril.get_siril_wd()
    st = dict(DEFAULTS, rejection=a.rejection, sigma_low=a.sigma[0], sigma_high=a.sigma[1],
              normalization=a.normalization, weighting=a.weighting, feather=a.feather,
              rgb_equal=not a.no_rgb_equal, output_norm=not a.no_output_norm, drizzle_scale=a.scale,
              pixfrac=a.pixfrac)
    output = os.path.join(wd, a.output) if a.output else ""
    if a.lights:
        folder = a.lights if os.path.isabs(a.lights) else os.path.join(wd, a.lights)
        cfg = job_config(st, lights_dir=folder, output_path=output)
    elif a.sequence:
        cfg = job_config(st, seq_path=sequence_path(a.sequence, wd), output_path=output)
        loaded = loaded_sequence(siril)
        if loaded and os.path.normcase(loaded["path"]) == os.path.normcase(cfg["seq_path"]):
            cfg["include"] = loaded["include"]
    else:
        loaded = loaded_sequence(siril)
        if loaded:
            cfg = job_config(st, seq_path=loaded["path"], include=loaded["include"], output_path=output)
        elif default_lights(wd):
            cfg = job_config(st, lights_dir=default_lights(wd), output_path=output)
        else:
            siril.log(f"{TITLE}: nothing to stack: give a sequence or --lights FOLDER", s.LogColor.RED)
            return 1
    rep = run_job(siril, install, cfg, load_result=not a.no_load)
    return 0 if rep.summary else 1


# ----------------------------------------------------------------------------- window (Scripts menu)

class Window:
    """Classic Windows dialog (ttk's native theme): what to stack, the stacking options, progress."""

    def __init__(self, siril, install: dict | None):
        import tkinter as tk
        from tkinter import ttk
        self.tk, self.ttk = tk, ttk
        self.siril = siril
        self.install = install
        self.engine = None
        self.worker = None
        self.events = queue.Queue()
        st = load_settings()
        wd = siril.get_siril_wd()
        self.loaded = loaded_sequence(siril)

        if os.name == "nt":                          # crisp text on scaled displays (Tk is DPI-unaware otherwise)
            try:
                import ctypes
                ctypes.windll.shcore.SetProcessDpiAwareness(1)
            except (AttributeError, OSError):
                pass
        self.root = root = tk.Tk()
        root.title(f"{TITLE} for Siril")
        style = ttk.Style(root)
        for theme in ("vista", "winnative", "clam"):
            if theme in style.theme_names():
                style.theme_use(theme)
                break
        root.resizable(False, False)
        body = ttk.Frame(root, padding=(11, 11, 11, 7))
        body.grid(sticky="nsew")
        pad = dict(padx=6, pady=3)

        # ---- what to stack
        src = ttk.LabelFrame(body, text="Stack", padding=(9, 5))
        src.grid(row=0, column=0, sticky="ew", pady=(0, 7))
        self.source = tk.StringVar(value="sequence" if self.loaded else "lights")
        if self.loaded:
            seq_text = (f"The sequence loaded in Siril: {self.loaded['name']} ({len(self.loaded['include'])} of "
                        f"{self.loaded['total']} frames selected)")
        else:
            seq_text = "The sequence loaded in Siril (none is loaded)"
        self.rb_seq = ttk.Radiobutton(src, text=seq_text, value="sequence", variable=self.source,
                                      command=self._source_changed)
        self.rb_seq.grid(row=0, column=0, columnspan=3, sticky="w", pady=2)
        if not self.loaded:
            self.rb_seq.state(["disabled"])
        self.rb_lights = ttk.Radiobutton(src, text="A folder of raw lights:", value="lights", variable=self.source,
                                         command=self._source_changed)
        self.rb_lights.grid(row=1, column=0, sticky="w", pady=2)
        self.lights = tk.StringVar(value=default_lights(wd))
        self.lights_entry = ttk.Entry(src, textvariable=self.lights, width=46)
        self.lights_entry.grid(row=1, column=1, sticky="ew", padx=(4, 4))
        self.browse = ttk.Button(src, text="Browse...", command=self._browse)
        self.browse.grid(row=1, column=2)
        ttk.Label(src, text="Registration, Bayer drizzle and stacking in one go.",
                  foreground="#5a5a5a").grid(row=2, column=1, columnspan=2, sticky="w", padx=4)

        # ---- options
        opt = ttk.LabelFrame(body, text="Options", padding=(9, 5))
        opt.grid(row=1, column=0, sticky="ew", pady=(0, 7))
        self.rejection = self._combo(opt, 0, "Rejection:", REJECTIONS, st["rejection"], width=26)
        self.sigma_low = self._spin(opt, 0, 2, "Low:", st["sigma_low"], 0.1, 10, 0.1)
        self.sigma_high = self._spin(opt, 0, 4, "High:", st["sigma_high"], 0.1, 10, 0.1)
        self.rejection_box.bind("<<ComboboxSelected>>", lambda e: self._rejection_changed())
        self.normalization = self._combo(opt, 1, "Normalization:", NORMALIZATIONS, st["normalization"], width=26)
        self.weighting = self._combo(opt, 2, "Weighting:", WEIGHTINGS, st["weighting"], width=26)
        self.feather = self._spin(opt, 3, 0, "Feather:", st["feather"], 0, 1000, 1, wide=True)
        ttk.Label(opt, text="(0 = off)", foreground="#5a5a5a").grid(row=3, column=2, columnspan=3, sticky="w")
        self.rgb_equal = tk.BooleanVar(value=st["rgb_equal"])
        self.output_norm = tk.BooleanVar(value=st["output_norm"])
        checks = ttk.Frame(opt)
        checks.grid(row=4, column=1, columnspan=5, sticky="w", pady=(2, 0))
        self._checks = [ttk.Checkbutton(checks, text="Equalize RGB", variable=self.rgb_equal),
                        ttk.Checkbutton(checks, text="Normalize the result to [0, 1]", variable=self.output_norm)]
        self._checks[0].grid(row=0, column=0, padx=(0, 14))
        self._checks[1].grid(row=0, column=1)

        drz = ttk.LabelFrame(body, text="Drizzle (raw lights)", padding=(9, 5))
        drz.grid(row=2, column=0, sticky="ew", pady=(0, 7))
        self.drz_frame = drz
        self.scale = self._combo(drz, 0, "Scale:", SCALES, st["drizzle_scale"], width=6)
        self.pixfrac = self._spin(drz, 0, 2, "Pixfrac:", st["pixfrac"], 0.1, 1.0, 0.05)

        self.load_result = tk.BooleanVar(value=st["load_result"])
        self._checks.append(ttk.Checkbutton(body, text="Open the result in Siril", variable=self.load_result))
        self._checks[-1].grid(row=3, column=0, sticky="w")
        self.bar = ttk.Progressbar(body, mode="determinate", maximum=1.0)
        self.bar.grid(row=4, column=0, sticky="ew", pady=(8, 2))
        self.status = ttk.Label(body, text="" if install else "GPU Stacker was not found: use Locate...",
                                width=70)
        self.status.grid(row=5, column=0, sticky="w")

        btns = ttk.Frame(body)
        btns.grid(row=6, column=0, sticky="ew", pady=(8, 0))
        btns.columnconfigure(0, weight=1)
        self.locate = ttk.Button(btns, text="Locate...", command=self._locate)
        self.locate.grid(row=0, column=0, sticky="w")
        self.stack_btn = ttk.Button(btns, text="Stack", command=self._start, default="active")
        self.stack_btn.grid(row=0, column=1, padx=(0, 6))
        self.stop_btn = ttk.Button(btns, text="Stop", command=self._stop)
        self.stop_btn.grid(row=0, column=2, padx=(0, 6))
        ttk.Button(btns, text="Close", command=self._close).grid(row=0, column=3)
        root.bind("<Return>", lambda e: self._start())
        root.bind("<Escape>", lambda e: self._close())
        root.protocol("WM_DELETE_WINDOW", self._close)
        self._inputs = [self.rb_lights, self.lights_entry, self.browse, self.rejection_box, self.normalization_box,
                        self.weighting_box, self.scale_box, self.locate] + self._spins + self._checks
        if self.loaded:
            self._inputs.append(self.rb_seq)
        self._source_changed()
        self._rejection_changed()
        self._set_running(False)
        root.after(100, self._poll)

    # -- widgets
    def _combo(self, parent, row, label, choices, value, width):
        ttk = self.ttk
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 6), pady=3)
        names = [n for _, n in choices]
        box = ttk.Combobox(parent, values=names, state="readonly", width=width)
        box.set(next((n for k, n in choices if k == value), names[0]))
        box.grid(row=row, column=1, sticky="w", pady=3)
        key = {"Rejection:": "rejection", "Normalization:": "normalization", "Weighting:": "weighting",
               "Scale:": "scale"}[label]
        setattr(self, key + "_box", box)
        return lambda: next(k for k, n in choices if n == box.get())

    def _spin(self, parent, row, col, label, value, lo, hi, step, wide=False):
        ttk = self.ttk
        if not hasattr(self, "_spins"):
            self._spins = []
        ttk.Label(parent, text=label).grid(row=row, column=col, sticky="w", padx=(0 if col == 0 else 14, 6))
        var = self.tk.StringVar(value=f"{value:g}")
        sp = ttk.Spinbox(parent, from_=lo, to=hi, increment=step, textvariable=var, width=8 if wide else 5)
        sp.grid(row=row, column=col + 1, sticky="w", pady=3)
        self._spins.append(sp)
        return lambda: float(var.get().replace(",", "."))

    # -- state
    def _source_changed(self) -> None:
        lights = self.source.get() == "lights"
        for w in (self.lights_entry, self.browse):
            w.state(["!disabled"] if lights and not self._running else ["disabled"])
        for w in self.drz_frame.winfo_children():
            if not isinstance(w, self.ttk.Label):
                w.state(["!disabled"] if lights and not self._running else ["disabled"])

    def _rejection_changed(self) -> None:
        on = self.rejection() != "none"
        for sp in self._spins[:2]:
            sp.state(["!disabled"] if on and not self._running else ["disabled"])

    _running = False

    def _set_running(self, running: bool) -> None:
        self._running = running
        for w in self._inputs:
            w.state(["disabled"] if running else ["!disabled"])
        if not self.loaded:
            self.rb_seq.state(["disabled"])
        self.stack_btn.state(["disabled"] if running or not self.install else ["!disabled"])
        self.stop_btn.state(["!disabled"] if running else ["disabled"])
        if not running:
            self._source_changed()
            self._rejection_changed()

    def _set_status(self, text: str, error: bool = False) -> None:
        self.status.configure(text=text, foreground="#c00000" if error else "")

    # -- actions
    def _browse(self) -> None:
        from tkinter import filedialog
        d = filedialog.askdirectory(parent=self.root, title="Folder of raw lights",
                                    initialdir=self.lights.get() or self.siril.get_siril_wd())
        if d:
            self.lights.set(os.path.normpath(lights_in(d)))
            self.source.set("lights")
            self._source_changed()

    def _locate(self) -> None:
        from tkinter import filedialog, messagebox
        d = filedialog.askdirectory(parent=self.root, title="GPU Stacker's folder")
        if not d:
            return
        found = install_in(d)
        if found is None:
            messagebox.showerror(TITLE, "That folder doesn't contain GPU Stacker (the unzipped folder with "
                                        "'Start GPU Stacker.bat' in it).", parent=self.root)
            return
        self.install = found
        remember_install(found)
        self._set_status(f"Using GPU Stacker in {found['package_dir']}")
        self._set_running(False)

    def settings(self) -> dict:
        return dict(rejection=self.rejection(), sigma_low=self.sigma_low(), sigma_high=self.sigma_high(),
                    normalization=self.normalization(), weighting=self.weighting(), feather=self.feather(),
                    rgb_equal=bool(self.rgb_equal.get()), output_norm=bool(self.output_norm.get()),
                    drizzle_scale=self.scale(), pixfrac=self.pixfrac(), load_result=bool(self.load_result.get()))

    def _start(self) -> None:
        from tkinter import messagebox
        if self._running or not self.install:
            return
        try:
            st = self.settings()
        except (ValueError, StopIteration):
            messagebox.showerror(TITLE, "Please check the numbers in the options.", parent=self.root)
            return
        if self.source.get() == "sequence" and self.loaded:
            cfg = job_config(st, seq_path=self.loaded["path"], include=self.loaded["include"])
        else:
            folder = self.lights.get().strip()
            if not folder or not fits_count(lights_in(folder)):
                messagebox.showerror(TITLE, "Choose a folder that contains the raw light frames (FITS).",
                                     parent=self.root)
                return
            cfg = job_config(st, lights_dir=lights_in(folder))
        save_settings(st)
        self._set_running(True)
        self.bar.configure(value=0)
        self._set_status("Starting...")
        self.worker = threading.Thread(target=self._work, args=(cfg, st["load_result"]), daemon=True)
        self.worker.start()

    def _work(self, cfg: dict, load_result: bool) -> None:
        try:
            rep = run_job(self.siril, self.install, cfg, load_result,
                          status=lambda text, f: self.events.put(("progress", text, f)),
                          started=lambda eng: setattr(self, "engine", eng))
            self.events.put(("finished", rep))
        except Exception as e:                       # noqa: BLE001 - shown in the window
            self.events.put(("failed", f"{type(e).__name__}: {e}"))
        finally:
            self.engine = None

    def _poll(self) -> None:
        try:
            while True:
                kind, *rest = self.events.get_nowait()
                if kind == "progress":
                    text, f = rest
                    self.bar.configure(mode="determinate", value=f if f >= 0 else 0)
                    self._set_status(f"{text} {f * 100:.0f}%" if f >= 0 else text)
                elif kind == "finished":
                    rep = rest[0]
                    self._set_running(False)
                    if rep.error:
                        self._set_status(rep.error, error=True)
                        from tkinter import messagebox
                        messagebox.showerror(TITLE, rep.error, parent=self.root)
                    elif rep.cancelled:
                        self._set_status("Stopped.")
                        self.bar.configure(value=0)
                    else:
                        sm = rep.summary or {}
                        self.bar.configure(value=1.0)
                        self._set_status(f"Done: {os.path.basename(sm.get('output') or '')}  ({sm.get('frames', 0)} "
                                         f"frames, {sm.get('seconds_total', 0):.0f} s)")
                elif kind == "failed":
                    self._set_running(False)
                    self._set_status(rest[0], error=True)
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def _stop(self) -> None:
        if self.engine is not None:
            self._set_status("Stopping...")
            self.engine.stop()

    def _close(self) -> None:
        if self._running:
            self._stop()
            if self.worker is not None:
                self.worker.join(timeout=15)
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


# ----------------------------------------------------------------------------- entry

def main() -> int:
    siril = s.SirilInterface()
    try:
        siril.connect()
    except s.SirilConnectionError as e:
        print(f"{TITLE}: not started from Siril ({e})")
        return 1
    try:
        siril.cmd("requires", "1.4.0")
    except s.SirilError:
        return 1
    if siril.is_cli() or len(sys.argv) > 1:
        return run_cli(siril, sys.argv[1:])
    Window(siril, find_gpu_stacker()).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
