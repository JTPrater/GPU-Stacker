# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jason Prater (JP Astrophotography)
"""Application entry point."""
from __future__ import annotations

import os
import sys
import traceback
from datetime import datetime


def log_dir() -> str:
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    d = os.path.join(base, "GPU Stacker")
    os.makedirs(d, exist_ok=True)
    return d


def _install_error_handling() -> None:
    """Started with pythonw there is no console: keep stray output and crashes in a log file and
    show unexpected errors instead of failing silently."""
    path = os.path.join(log_dir(), "gpu-stacker.log")
    if sys.stdout is None or sys.stderr is None:
        f = open(path, "a", encoding="utf-8", buffering=1)
        sys.stdout = sys.stdout or f
        sys.stderr = sys.stderr or f

    def hook(exc_type, exc, tb):
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(f"\n[{datetime.now():%Y-%m-%d %H:%M:%S}] unexpected error\n{text}")
        except OSError:
            pass
        try:
            from PySide6.QtWidgets import QApplication, QMessageBox
            if QApplication.instance() is not None:
                box = QMessageBox(QMessageBox.Icon.Critical, "GPU Stacker - unexpected error",
                                  f"{exc_type.__name__}: {exc}\n\nDetails were saved to {path}")
                box.setDetailedText(text)
                box.exec()
        except Exception:                   # noqa: BLE001
            pass
    sys.excepthook = hook


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    _install_error_handling()
    if os.name == "nt":                     # own taskbar identity (icon/grouping) instead of python.exe's
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("GPUStacker.App.1")
        except Exception:                   # noqa: BLE001
            pass
    from PySide6.QtWidgets import QApplication

    from .main_window import MainWindow
    from .siril_link import on_start
    from .theme import apply_theme
    from .widgets import app_icon

    on_start()                              # where this installation is, for the Siril script

    QApplication.setOrganizationName("GPU Stacker")
    QApplication.setApplicationName("GPU Stacker")
    app = QApplication(argv)
    apply_theme(app)
    app.setWindowIcon(app_icon())
    w = MainWindow()
    w.show()
    args = [a for a in argv[1:] if not a.startswith("-")]
    if args and (args[0].lower().endswith(".seq") or os.path.isdir(args[0])):
        w.open_path(args[0])
    return app.exec()
