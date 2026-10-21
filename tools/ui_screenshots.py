# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jason Prater (JP Astrophotography)
"""Render the app offscreen and save PNG screenshots (for reviewing the UI without opening a window).

Usage: .venv\\Scripts\\python tools\\ui_screenshots.py OUT_DIR [PATH] [--analyze] [--run] [--size=WxH]
  PATH      a .seq file or a folder of raw lights (default: a synthetic sequence)
  --analyze  (lights) run Analyze and capture the Frames tab afterwards
  --run      also run a test strip and capture the live preview and the result
  --size     window size, e.g. --size=1280x720 for a laptop screen (default 1600x1000)
"""
import os
import sys
import tempfile
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if os.name == "nt":
    os.environ.setdefault("QT_QPA_FONTDIR", os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts"))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    out = args[0]
    os.makedirs(out, exist_ok=True)
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication

    from gpustacker.ui.main_window import MainWindow
    from gpustacker.ui.session import open_any
    from gpustacker.ui.theme import apply_theme

    app = QApplication(sys.argv[:1])
    apply_theme(app)
    ini = os.path.join(tempfile.mkdtemp(prefix="gs_shots_"), "settings.ini")
    w = MainWindow(QSettings(ini, QSettings.Format.IniFormat))
    w.ask_before_quit = False
    w._warn = lambda t, m: print(f"WARN [{t}] {m}")
    w._error_box = lambda ev: print("ERROR", ev.get("msg"), ev.get("trace", "")[-2000:])
    size = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--size=")), "1600x1000")
    w.resize(*(int(v) for v in size.lower().split("x")))
    w.show()

    def pump(seconds=0.3):
        end = time.time() + seconds
        while time.time() < end:
            app.processEvents()
            time.sleep(0.01)

    def wait(cond, timeout):
        end = time.time() + timeout
        while time.time() < end and not cond():
            app.processEvents()
            time.sleep(0.02)

    def shot(name):
        pump(0.4)
        p = os.path.join(out, f"{name}.png")
        w.grab().save(p)
        print("saved", p, flush=True)

    wait(lambda: w.gpu is not None, 120)
    shot("01_welcome")

    if len(args) > 1:
        seq = args[1]
    else:
        from synth import make_sequence
        seq = make_sequence(tempfile.mkdtemp(prefix="gs_synth_"), n=24, canvas=(220, 150))
    t = time.time()
    w._loaded(open_any(seq))
    print(f"loaded in {time.time() - t:.1f} s")
    shot("02_mosaic")
    w.tabs.setCurrentWidget(w.frames)
    shot("03_frames")
    w.frames.table.selectRow(5)
    w.frames.metric_box.setCurrentIndex(2)
    shot("04_frames_stars")
    w.tabs.setCurrentWidget(w.mosaic)
    w.settings_panel.crop.setChecked(True)
    w.settings_panel.crop_frames.setValue(50 if len(args) > 1 else 6)
    shot("05_crop")
    w.settings_panel.crop.setChecked(False)

    if "--analyze" in sys.argv and w.session.kind == "lights":
        t = time.time()
        w.start_analyze()
        wait(lambda: not w.runner.running() and w.job_kind == "", 1800)
        pump(1.0)
        print(f"analyzed in {time.time() - t:.1f} s")
        shot("05b_analyzed")
        w.tabs.setCurrentWidget(w.mosaic)
        shot("05c_registered")

    if "--run" in sys.argv:
        w.settings_panel.weighting.setCurrentIndex(w.settings_panel.weighting.findData("none"))
        pump(0.3)
        w.start_test()
        wait(lambda: w.mosaic.view.live is not None and w.mosaic.view.live_front > w.mosaic.view.live_rect.top()
             + 0.45 * w.mosaic.view.live_rect.height(), 600)
        w.mosaic.view._render_live()
        shot("06_running")
        wait(lambda: not w.runner.running() and w.job_kind == "", 900)
        pump(1.0)
        shot("07_result")
        w.tabs.setCurrentWidget(w.log)
        shot("08_log")
    w.close()


if __name__ == "__main__":
    main()
