# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jason Prater (JP Astrophotography)
"""UI tests, offscreen, driving the real engine process on synthetic sequences."""
import json
import os
import time

import pytest

pytest.importorskip("cupy")

from synth import make_sequence  # noqa: E402

from gpustacker.engine.fits import read_image  # noqa: E402
from gpustacker.ui.frames_view import C_CHECK  # noqa: E402
from gpustacker.ui.session import load_session  # noqa: E402


@pytest.fixture
def window(qapp, tmp_path):
    from PySide6.QtCore import QSettings

    from gpustacker.ui.main_window import MainWindow
    w = MainWindow(QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat))
    w.ask_before_quit = False
    w.messages = []
    w._warn = lambda title, text: w.messages.append((title, text))
    w._error_box = lambda ev: w.messages.append(("error", ev.get("msg")))
    w.resize(1400, 900)
    w.show()
    yield w
    w.close()


def _wait(qapp, cond, timeout):
    end = time.time() + timeout
    while time.time() < end:
        qapp.processEvents()
        if cond():
            return True
        time.sleep(0.02)
    return False


def _open(w, path):
    w._loaded(load_session(path))


def test_open_sequence_fills_the_window(window, tmp_path):
    seq = make_sequence(str(tmp_path / "seq"), n=10, excluded=(3,))
    _open(window, seq)
    assert window.pages.currentWidget() is window.tabs
    assert "9</b> of 10 frames" in window.seq_card.lines.text()
    assert window.frames.model.rowCount() == 10
    assert window.depth is not None and window.depth[0].max() >= 1
    assert "≈" in window.estimate_label.text()
    assert window.settings_panel.output.text().endswith(".fit")


def test_unchecking_a_frame_updates_selection_and_is_remembered(window, tmp_path):
    seq = make_sequence(str(tmp_path / "seq"), n=8)
    _open(window, seq)
    m = window.frames.model
    m.setData(m.index(2, C_CHECK), False, role=__import__("PySide6.QtCore").QtCore.Qt.ItemDataRole.CheckStateRole)
    s = window.session
    assert s.seq.frames[2].filenum not in s.included
    assert "Custom selection" in window.seq_card.lines.text()
    saved = json.load(open(s.selection_path()))
    assert s.seq.frames[2].filenum not in saved["included"]
    again = load_session(seq)
    assert again.included == s.included


def test_settings_reach_the_job_config(window, tmp_path):
    _open(window, make_sequence(str(tmp_path / "seq"), n=6))
    sp = window.settings_panel
    sp.rejection.setCurrentIndex(sp.rejection.findData("sigma"))
    sp.sig_lo.setValue(2.5)
    sp.weighting.setCurrentIndex(sp.weighting.findData("wfwhm"))
    sp.feather.setValue(0)
    cfg = window._base_cfg()
    assert (cfg["rejection"], cfg["sigma_low"], cfg["weighting"], cfg["feather"]) == ("sigma", 2.5, "wfwhm", 0.0)
    assert sorted(cfg["include"]) == sorted(window.session.included)


def test_test_strip_runs_and_shows_a_result(qapp, window, tmp_path):
    _open(window, make_sequence(str(tmp_path / "seq"), n=10))
    window.start_test()
    assert window.runner.running()
    assert _wait(qapp, lambda: window.job_kind == "" and not window.runner.running(), 300), window.messages
    assert window.job_outcome == "done", window.messages
    assert window.result.view.data is not None
    assert "Test strip done" in window.runbar.phase.text()
    assert not any(f.endswith(".fit") for f in os.listdir(tmp_path))


def test_full_stack_writes_the_file(qapp, window, tmp_path):
    _open(window, make_sequence(str(tmp_path / "seq"), n=10))
    out = str(tmp_path / "result.fit")
    window.settings_panel.output.setText(out)
    window.start_stack()
    assert _wait(qapp, lambda: window.job_kind == "" and not window.runner.running(), 300), window.messages
    assert window.job_outcome == "done", window.messages
    data, hdr = read_image(out)
    assert data.shape[0] == 3 and hdr.get("STACKCNT") == 10
    assert window.result.open_btn.isEnabled()
    assert window.session.noise(window.session.frames[0]) is not None     # noise now cached and shown


def test_lights_folder_analyze_reference_and_test_strip(qapp, window, tmp_path):
    from PySide6.QtCore import Qt
    from synth_raw import make_lights

    from gpustacker.ui.session import open_any
    truth = make_lights(str(tmp_path / "M 45"), n=8, cloudy=(3,))
    window._loaded(open_any(str(tmp_path / "M 45")))          # the parent of 'lights' works too
    s = window.session
    assert s.kind == "lights" and len(s.frames) == 8 and s.lights_dir == truth.lights_dir
    assert window.analyze_btn.isVisible() and window.analyze_btn.isEnabled()
    assert not window.test_btn.isEnabled()                    # nothing registered yet
    assert "not analyzed yet" in window.seq_card.lines.text()
    assert "Bayer GRBG" in window.seq_card.lines.text()

    window.start_analyze()
    assert _wait(qapp, lambda: window.job_kind == "" and not window.runner.running(), 300), window.messages
    assert window.job_outcome == "done", window.messages
    assert s.registered and window.depth is not None and window.test_btn.isEnabled()
    assert s.bad_reason(s.frames[3]).startswith("only")        # the clouded-out sub
    auto = s.reference_name
    others = [i for i, fr in enumerate(s.frames) if i not in (3, 5) and s.frame_name(fr) != auto]
    assert s.status(s.frames[others[0]]).startswith("Registered")
    assert "Registered 7 frames" in window.seq_card.lines.text()

    # exclude a frame: remembered next time the folder is opened
    m = window.frames.model
    m.setData(m.index(5, C_CHECK), False, role=Qt.ItemDataRole.CheckStateRole)
    assert s.frames[5].filenum not in s.included
    assert open_any(truth.lights_dir).included == s.included

    # pick another reference; the test strip re-registers on it
    k = others[-1]
    window.frames.table.selectRow(k)
    window.frames.ref_btn.click()
    assert s.reference_file == truth.names[k] and window.settings_panel.ref_label.text() == f"#{k + 1}"
    assert _wait(qapp, lambda: "registers" in window.estimate_label.text(), 5)      # debounced refresh
    window.start_test()
    assert _wait(qapp, lambda: window.job_kind == "" and not window.runner.running(), 300), window.messages
    assert window.job_outcome == "done", window.messages
    assert window.result.view.data is not None
    assert s.registration.reference == truth.names[k] and s.status(s.frames[k]) == "Reference"
    assert window.job_cfg["include_files"] == [s.frame_name(fr) for fr in s.frames if fr.filenum in s.included]


def test_cancel_from_the_ui(qapp, window, tmp_path):
    _open(window, make_sequence(str(tmp_path / "seq"), n=30, w=160, h=120, canvas=(260, 200)))
    out = str(tmp_path / "cancelled.fit")
    window.settings_panel.output.setText(out)
    window.settings_panel.strip.setValue(8)
    window.start_stack()
    assert _wait(qapp, lambda: window.runbar.cancel.isVisible(), 60)
    window.cancel_job()
    assert _wait(qapp, lambda: window.job_kind == "" and not window.runner.running(), 120)
    assert window.job_outcome == "cancelled" and window.runbar.phase.text() == "Cancelled"
    assert not os.path.exists(out)
