# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jason Prater (JP Astrophotography)
"""Runs the engine in a separate process and turns its JSON-lines events into Qt signals.

The engine runs out of process so the window stays responsive, a CUDA failure can't take the
UI down, and GPU memory is fully released when a job ends. Cancelling writes a flag file the
engine polls; the engine also stops by itself if this process disappears.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import time

from PySide6.QtCore import QObject, QTimer, Signal

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0


def engine_python() -> str:
    """python.exe next to the running interpreter (pythonw.exe has no usable console streams)."""
    exe = sys.executable
    d, name = os.path.split(exe)
    if name.lower() == "pythonw.exe":
        cand = os.path.join(d, "python.exe")
        if os.path.isfile(cand):
            return cand
    return exe


def _env() -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = ROOT + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["PYTHONWARNINGS"] = "ignore::UserWarning"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def work_dir() -> str:
    d = os.path.join(tempfile.gettempdir(), "gpustacker")
    os.makedirs(d, exist_ok=True)
    return d


def _kill_tree(pid: int) -> None:
    if os.name == "nt":
        subprocess.call(["taskkill", "/PID", str(pid), "/T", "/F"], stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL, creationflags=CREATE_NO_WINDOW)
    else:
        try:
            os.kill(pid, 9)
        except OSError:
            pass


class EngineRunner(QObject):
    event = Signal(dict)
    stderr_line = Signal(str)
    finished = Signal(int)

    KILL_AFTER_MS = 30_000

    def __init__(self, parent=None):
        super().__init__(parent)
        self.proc: subprocess.Popen | None = None
        self._cancel_file = ""
        self._kill_timer = QTimer(self)
        self._kill_timer.setSingleShot(True)
        self._kill_timer.timeout.connect(self.kill)
        self.cancelling = False

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, cfg: dict) -> None:
        if self.running():
            raise RuntimeError("a job is already running")
        stamp = time.strftime("%Y%m%d-%H%M%S")
        job = os.path.join(work_dir(), f"job_{stamp}_{os.getpid()}.json")
        cfg = dict(cfg, preview_dir=work_dir())
        with open(job, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=1)
        self._cancel_file = job + ".cancel"
        try:
            os.remove(self._cancel_file)
        except OSError:
            pass
        self.cancelling = False
        args = [engine_python(), "-m", "gpustacker.engine", "run", job, "--cancel-file", self._cancel_file,
                "--parent-pid", str(os.getpid())]
        self.proc = subprocess.Popen(args, cwd=ROOT, env=_env(), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
                                     bufsize=1, creationflags=CREATE_NO_WINDOW)
        proc = self.proc
        threading.Thread(target=self._read_stdout, args=(proc,), daemon=True).start()
        threading.Thread(target=self._read_stderr, args=(proc,), daemon=True).start()

    def _read_stdout(self, proc) -> None:
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except ValueError:
                self.stderr_line.emit(line)
                continue
            self.event.emit(ev)
        code = proc.wait()
        self.finished.emit(code)

    def _read_stderr(self, proc) -> None:
        for line in proc.stderr:
            line = line.rstrip()
            if line:
                self.stderr_line.emit(line)

    def cancel(self) -> None:
        if not self.running():
            return
        self.cancelling = True
        try:
            with open(self._cancel_file, "w") as f:
                f.write("cancel")
        except OSError:
            self.kill()
            return
        self._kill_timer.start(self.KILL_AFTER_MS)

    def kill(self) -> None:
        if self.running():
            _kill_tree(self.proc.pid)

    def cleanup(self) -> None:
        self._kill_timer.stop()
        try:
            os.remove(self._cancel_file)
        except OSError:
            pass


class Probe(QObject):
    """GPU info from `python -m gpustacker.engine probe` (in the background)."""
    done = Signal(dict)

    def start(self) -> None:
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        try:
            out = subprocess.run([engine_python(), "-m", "gpustacker.engine", "probe"], cwd=ROOT, env=_env(),
                                 capture_output=True, text=True, timeout=120, creationflags=CREATE_NO_WINDOW)
            line = [ln for ln in out.stdout.splitlines() if ln.strip().startswith("{")]
            info = json.loads(line[-1]) if line else {"error": (out.stderr or "no output").strip()[-500:]}
        except Exception as e:                       # noqa: BLE001
            info = {"error": f"{type(e).__name__}: {e}"}
        self.done.emit(info)
