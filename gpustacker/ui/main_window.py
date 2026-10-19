# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jason Prater (JP Astrophotography)
"""Main window: data + options sidebar, Canvas / Frames / Result / Log tabs, run bar."""
from __future__ import annotations

import base64
import math
import os
import shutil
import threading
import time
from datetime import datetime

import numpy as np
from PySide6.QtCore import QEventLoop, QObject, QSettings, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QFontMetrics, QKeySequence, QPixmap, QTextCursor
from PySide6.QtWidgets import (QApplication, QFileDialog, QFrame, QHBoxLayout, QLabel, QMainWindow, QMessageBox,
                               QPlainTextEdit, QProgressBar, QPushButton, QScrollArea, QStackedWidget, QStyle,
                               QTabWidget, QVBoxLayout, QWidget)

from .. import __version__
from ..engine.plan import auto_output_path, estimate, output_folder
from ..engine.sequence import largest_rectangle
from .frames_view import FramesPanel
from .mosaic_view import MosaicPanel
from .result_view import ResultPanel
from .runner import EngineRunner, Probe
from .session import (LightsSession, Session, binning_for_view, format_bytes, format_duration, format_hours,
                      open_any)
from .settings_panel import SettingsPanel
from .theme import COLORS, mono_font, set_role
from .widgets import Card, CommandLink, ElidedLabel, app_icon, hbox, hline

MAX_RECENT = 8
# rough per-frame costs for a 2160 x 3840 Seestar S50 Pro raw frame (scaled by pixel count)
RAW_COSTS = dict(analyze=0.045, register=0.03, measure=0.07, stack=0.10)


class _Loader(QObject):
    progress = Signal(int, int)
    loaded = Signal(object)
    failed = Signal(str)

    def start(self, path: str) -> None:
        threading.Thread(target=self._run, args=(path,), daemon=True).start()

    def _run(self, path: str) -> None:
        try:
            self.loaded.emit(open_any(path, progress=lambda n, t: self.progress.emit(n, t)))
        except Exception as e:                       # noqa: BLE001
            self.failed.emit(f"{e}")


class SequenceCard(Card):
    def __init__(self, parent=None):
        super().__init__("Data", parent)
        self.name = QLabel("Nothing open")
        set_role(self.name, "title")
        self.open_btn = QPushButton("Open…")
        self.open_btn.setToolTip("Open a folder of raw lights (Ctrl+O) or a Siril sequence (Ctrl+Shift+O)")
        self.body.addLayout(hbox(self.name, None, self.open_btn))
        self.folder = ElidedLabel("")
        set_role(self.folder, "muted")
        self.body.addWidget(self.folder)
        self.lines = QLabel("")
        self.lines.setWordWrap(True)
        self.lines.setTextFormat(Qt.TextFormat.RichText)
        self.body.addWidget(self.lines)
        self.notice = QLabel("")
        self.notice.setWordWrap(True)
        self.notice.hide()
        self.body.addWidget(self.notice)

    def show_notice(self, text: str, kind: str = "warning") -> None:
        if not text:
            self.notice.hide()
            return
        self.notice.setText(text)
        set_role(self.notice, "warning", kind == "warning")
        set_role(self.notice, "error", kind == "error")
        self.notice.show()


class WelcomePage(QWidget):
    open_clicked = Signal()
    open_seq_clicked = Signal()
    recent_clicked = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 24, 24, 24)
        outer.addStretch(1)
        self.card = QFrame()
        self.card.setObjectName("welcomePanel")
        self.card.setMaximumWidth(640)
        lay = QVBoxLayout(self.card)
        lay.setContentsMargins(22, 18, 22, 16)
        lay.setSpacing(6)
        head = QHBoxLayout()
        head.setSpacing(12)
        icon = QLabel()
        icon.setPixmap(app_icon().pixmap(48, 48))
        icon.setAlignment(Qt.AlignmentFlag.AlignTop)
        head.addWidget(icon)
        col = QVBoxLayout()
        col.setSpacing(4)
        t = QLabel("Open your subs to start stacking")
        set_role(t, "title")
        info = QLabel("GPU Stacker finds the stars, registers every frame, drizzles the Bayer data and does the "
                      "rejection stack on your graphics card. No Siril needed.")
        info.setWordWrap(True)
        col.addWidget(t)
        col.addWidget(info)
        head.addLayout(col, 1)
        lay.addLayout(head)
        lay.addSpacing(8)
        self.open_link = CommandLink("Open a folder of raw lights",
                                     "The Seestar's lights folder works as is (or pick the folder above it).")
        self.open_link.clicked.connect(self.open_clicked)
        self.seq_link = CommandLink("Open a Siril sequence",
                                    "Already registered in Siril? Open the r_….seq file from the process folder.")
        self.seq_link.clicked.connect(self.open_seq_clicked)
        lay.addWidget(self.open_link)
        lay.addWidget(self.seq_link)
        self.recent_box = QVBoxLayout()
        self.recent_box.setSpacing(3)
        self.recent_box.setContentsMargins(10, 8, 0, 0)
        lay.addLayout(self.recent_box)
        lay.addSpacing(10)
        drop = QLabel("Tip: you can also drag a folder or a .seq file onto this window.")
        set_role(drop, "muted")
        lay.addWidget(drop)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(self.card, 4)
        row.addStretch(1)
        outer.addLayout(row)
        outer.addStretch(2)

    def set_recent(self, paths: list[str]) -> None:
        while self.recent_box.count():
            item = self.recent_box.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        if not paths:
            return
        head = QLabel("Recent folders")
        set_role(head, "section")
        self.recent_box.addWidget(head)
        fm = QFontMetrics(self.font())
        for p in paths:
            b = QPushButton(fm.elidedText(p, Qt.TextElideMode.ElideMiddle, 520).replace("&", "&&"))
            set_role(b, "flat")
            b.setToolTip(p)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.clicked.connect(lambda _c=False, path=p: self.recent_clicked.emit(path))
            self.recent_box.addWidget(b, 0, Qt.AlignmentFlag.AlignLeft)

    def set_dragging(self, on: bool) -> None:
        set_role(self.card, "dragging", on)


class RunBar(QFrame):
    cancel_clicked = Signal()

    ICONS = {"ok": QStyle.StandardPixmap.SP_DialogApplyButton, "warn": QStyle.StandardPixmap.SP_MessageBoxWarning,
             "error": QStyle.StandardPixmap.SP_MessageBoxCritical}

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("runBar")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 4, 8, 4)
        lay.setSpacing(8)
        self.dot = QLabel()                      # status icon
        self.dot.setFixedSize(16, 16)
        self.phase = QLabel("Ready")
        self.detail = ElidedLabel("")
        set_role(self.detail, "muted")
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setFixedWidth(220)
        self.bar.setTextVisible(False)
        self.eta = QLabel("")
        self.eta.setMinimumWidth(100)
        self.eta.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.cancel = QPushButton("Cancel")
        self.cancel.setMinimumWidth(75)
        self.cancel.clicked.connect(self.cancel_clicked)
        for w in (self.dot, self.phase):
            lay.addWidget(w)
        lay.addWidget(self.detail, 1)
        lay.addWidget(self.bar)
        lay.addWidget(self.eta)
        lay.addWidget(self.cancel)
        self.set_idle("Ready")

    def _icon(self, kind: str) -> None:
        sp = self.ICONS.get(kind)
        self.dot.setPixmap(self.style().standardIcon(sp).pixmap(16, 16) if sp is not None else QPixmap())

    def set_idle(self, text: str, kind: str = "idle") -> None:
        self._icon(kind)
        self.phase.setText(text)
        self.detail.setText("")
        self.eta.setText("")
        self.bar.hide()
        self.cancel.hide()

    def set_running(self, phase: str) -> None:
        self._icon("")
        self.phase.setText(phase)
        self.bar.show()
        self.cancel.show()
        self.cancel.setEnabled(True)
        self.cancel.setText("Cancel")


class LogPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setFont(mono_font(9))
        self.text.setMaximumBlockCount(50000)
        lay.addWidget(self.text, 1)
        copy = QPushButton("Copy")
        save = QPushButton("Save…")
        clear = QPushButton("Clear")
        for b in (copy, save, clear):
            b.setMinimumWidth(75)
        lay.addLayout(hbox(None, clear, copy, save))
        copy.clicked.connect(lambda: QApplication.clipboard().setText(self.text.toPlainText()))
        clear.clicked.connect(self.text.clear)
        save.clicked.connect(self._save)

    def add(self, msg: str, level: str = "info") -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        col = {"warn": COLORS["warning"], "error": COLORS["error"], "ok": COLORS["success"]}.get(level)
        esc = msg.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
        body = f"<span style='color:{col}'>{esc}</span>" if col else esc
        self.text.appendHtml(f"<span style='color:{COLORS['text_disabled']}'>{ts}</span>&nbsp;&nbsp;{body}")
        self.text.moveCursor(QTextCursor.MoveOperation.End)

    def _save(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save log", "gpu-stacker-log.txt", "Text (*.txt)")
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.text.toPlainText())


class MainWindow(QMainWindow):
    def __init__(self, settings: QSettings | None = None):
        super().__init__()
        self.settings = settings or QSettings("GPU Stacker", "GPU Stacker")
        self.session: Session | None = None
        self.gpu: dict | None = None
        self.runner = EngineRunner(self)
        self.probe = Probe(self)
        self.loader = _Loader(self)
        self.job_kind = ""
        self.job_cfg: dict = {}
        self.job_t0 = 0.0
        self.job_outcome = ""
        self.canvas_ev: dict | None = None
        self.crop_rect: tuple | None = None
        self.depth = None
        self.output_custom = False
        self.ask_before_quit = True
        self._stderr_tail: list[str] = []
        self.setWindowTitle("GPU Stacker")
        self.setWindowIcon(app_icon())
        self.setAcceptDrops(True)
        self._build()
        self._menus()
        self.settings_panel.restore(self.settings)
        self._restore_geometry()
        self.welcome.set_recent(self._recent())
        self._wire()
        self._estimate_timer = QTimer(self)
        self._estimate_timer.setSingleShot(True)
        self._estimate_timer.setInterval(120)
        self._estimate_timer.timeout.connect(self._refresh_plan)
        self._update_actions()
        self.status_gpu.setText("Checking GPU…")
        self.probe.start()

    # ================================================================ layout
    def _build(self) -> None:
        central = QWidget()
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        root = QHBoxLayout()
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        outer.addLayout(root, 1)

        side = QWidget()
        side.setObjectName("sidebar")
        side.setFixedWidth(380)
        sl = QVBoxLayout(side)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner = QWidget()
        il = QVBoxLayout(inner)
        il.setContentsMargins(8, 6, 6, 4)
        il.setSpacing(8)
        self.seq_card = SequenceCard()
        il.addWidget(self.seq_card)
        self.settings_panel = SettingsPanel()
        il.addWidget(self.settings_panel)
        il.addStretch(1)
        scroll.setWidget(inner)
        sl.addWidget(scroll, 1)

        act = QWidget()
        al = QVBoxLayout(act)
        al.setContentsMargins(8, 4, 8, 8)
        al.setSpacing(6)
        al.addWidget(hline())
        self.estimate_label = QLabel("")
        self.estimate_label.setWordWrap(True)
        self.estimate_label.setTextFormat(Qt.TextFormat.RichText)
        set_role(self.estimate_label, "muted")
        al.addWidget(self.estimate_label)
        self.analyze_btn = QPushButton("Analyze")
        self.analyze_btn.setToolTip("Detect the stars in every frame and register them, without stacking. Fills in "
                                    "FWHM / roundness / star counts so you can weed out bad frames first.")
        self.test_btn = QPushButton("Test strip")
        self.test_btn.setToolTip("Stack a narrow band through the densest part of the image with these settings "
                                 "and preview it, without writing a file. Good for trying settings quickly.")
        self.stack_btn = QPushButton("Stack")
        self.stack_btn.setToolTip("Stack the selected frames and write the FITS file  (Ctrl+Enter)")
        set_role(self.stack_btn, "primary")                  # the default button
        for b in (self.analyze_btn, self.test_btn, self.stack_btn):
            b.setMinimumWidth(86)
        al.addLayout(hbox(self.analyze_btn, self.test_btn, None, self.stack_btn))
        sl.addWidget(act)
        root.addWidget(side)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(2, 6, 8, 8)
        rl.setSpacing(0)
        self.pages = QStackedWidget()
        self.welcome = WelcomePage()
        self.tabs = QTabWidget()
        self.mosaic = MosaicPanel()
        self.frames = FramesPanel()
        self.result = ResultPanel()
        self.log = LogPanel()
        self.tabs.addTab(self.mosaic, "Canvas")
        self.tabs.addTab(self.frames, "Frames")
        self.tabs.addTab(self.result, "Result")
        self.tabs.addTab(self.log, "Log")
        self.pages.addWidget(self.welcome)
        self.pages.addWidget(self.tabs)
        rl.addWidget(self.pages, 1)
        root.addWidget(right, 1)
        self.runbar = RunBar()
        outer.addWidget(self.runbar)
        self.setCentralWidget(central)

        sb = self.statusBar()
        self.status_gpu = QLabel("")
        ver = QLabel(f"v{__version__}")
        sb.addPermanentWidget(self.status_gpu)
        sb.addPermanentWidget(ver)

    def _menus(self) -> None:
        m = self.menuBar().addMenu("&File")
        self.open_act = QAction("&Open lights folder…", self)
        self.open_act.setShortcut(QKeySequence.StandardKey.Open)
        self.open_act.triggered.connect(self.open_dialog)
        m.addAction(self.open_act)
        self.open_seq_act = QAction("Open Siril &sequence…", self)
        self.open_seq_act.setShortcut(QKeySequence("Ctrl+Shift+O"))
        self.open_seq_act.triggered.connect(self.open_seq_dialog)
        m.addAction(self.open_seq_act)
        self.recent_menu = m.addMenu("Open &recent")
        self.reload_act = QAction("Re&load", self)
        self.reload_act.setShortcut(QKeySequence("F5"))
        self.reload_act.triggered.connect(lambda: self.session and self.open_path(self.session.path))
        m.addAction(self.reload_act)
        m.addSeparator()
        self.stack_act = QAction("&Stack", self)
        self.stack_act.setShortcut(QKeySequence("Ctrl+Return"))
        self.stack_act.triggered.connect(self.start_stack)
        m.addAction(self.stack_act)
        m.addSeparator()
        siril_act = QAction("Add to Siril's Scripts &menu…", self)
        siril_act.triggered.connect(self._install_siril_script)
        m.addAction(siril_act)
        m.addSeparator()
        quit_act = QAction("E&xit", self)
        quit_act.triggered.connect(self.close)
        m.addAction(quit_act)
        h = self.menuBar().addMenu("&Help")
        about = QAction("&About GPU Stacker", self)
        about.triggered.connect(self._about)
        h.addAction(about)
        about_qt = QAction("About &Qt", self)
        about_qt.triggered.connect(lambda: QMessageBox.aboutQt(self, "About Qt"))
        h.addAction(about_qt)
        self._fill_recent_menu()

    def _wire(self) -> None:
        self.seq_card.open_btn.clicked.connect(self.open_dialog)
        self.welcome.open_clicked.connect(self.open_dialog)
        self.welcome.open_seq_clicked.connect(self.open_seq_dialog)
        self.welcome.recent_clicked.connect(self.open_path)
        self.analyze_btn.clicked.connect(self.start_analyze)
        self.frames.reference_changed.connect(self._reference_changed)
        self.settings_panel.reference_reset.connect(self._reference_reset)
        self.loader.progress.connect(self._load_progress)
        self.loader.loaded.connect(self._loaded)
        self.loader.failed.connect(self._load_failed)
        self.settings_panel.changed.connect(self._options_changed)
        self.settings_panel.output_edited.connect(lambda: setattr(self, "output_custom", True))
        self.frames.selection_edited.connect(self._selection_changed)
        self.stack_btn.clicked.connect(self.start_stack)
        self.test_btn.clicked.connect(self.start_test)
        self.runbar.cancel_clicked.connect(self.cancel_job)
        self.runner.event.connect(self._on_event)
        self.runner.stderr_line.connect(self._on_stderr)
        self.runner.finished.connect(self._on_finished)
        self.probe.done.connect(self._probe_done)

    # ================================================================ sequence
    def open_dialog(self) -> None:
        if self.runner.running():
            return
        start = self.settings.value("last_dir", os.path.expanduser("~"))
        path = QFileDialog.getExistingDirectory(self, "Open a folder of raw lights (or its parent)", start)
        if path:
            self.open_path(path)

    def open_seq_dialog(self) -> None:
        if self.runner.running():
            return
        start = self.settings.value("last_dir", os.path.expanduser("~"))
        path, _ = QFileDialog.getOpenFileName(self, "Open registered Siril sequence", start,
                                              "Siril sequence (*.seq);;All files (*)")
        if path:
            self.open_path(path)

    def open_path(self, path: str) -> None:
        if self.runner.running():
            QMessageBox.information(self, "GPU Stacker", "Wait for the running job to finish (or cancel it) first.")
            return
        path = os.path.normpath(path)
        self.settings.setValue("last_dir", path if os.path.isdir(path) else os.path.dirname(path))
        self.runbar.set_running("Opening")
        self.runbar.cancel.hide()
        self.runbar.detail.setText(path)
        self.runbar.bar.setValue(0)
        self.loader.start(path)

    def _load_progress(self, n: int, total: int) -> None:
        self.runbar.bar.setValue(int(1000 * n / max(total, 1)))
        self.runbar.detail.setText(f"Reading frame headers {n:,} / {total:,}")

    def _load_failed(self, msg: str) -> None:
        self.runbar.set_idle("Couldn't open that", "error")
        self._warn("Couldn't open that", msg)

    def _loaded(self, session) -> None:
        self.session = session
        self._add_recent(session.path if session.kind == "lights" else session.seq.path)
        self.output_custom = False
        self.settings_panel.set_mode(session.kind)
        if session.kind == "lights":
            v = self.settings_panel.values()
            session.scale, session.pixfrac = v["drizzle_scale"], v["pixfrac"]
        self.frames.set_session(session)
        self.result.view.data = None
        self.mosaic.view.clear_live()
        self.mosaic.stack_available(False)
        self.pages.setCurrentWidget(self.tabs)
        self.tabs.setCurrentWidget(self.mosaic)
        self.settings_panel.output.setText(self._default_output())
        self._refresh_reference()
        self._refresh_coverage()
        self._refresh_sequence_card()
        self._refresh_plan()
        self.runbar.set_idle(f"Opened {session.name}  ·  {len(session.frames):,} frames", "ok")
        self.log.add(f"opened {session.path}: {len(session.frames)} frames, {len(session.included)} selected")
        for p in session.problems:
            self.log.add(p, "warn")
        self._update_actions()

    def _default_output(self) -> str:
        s = self.session
        parent = s.directory if s.kind == "lights" else output_folder(s.directory)
        frames = s.usable_selected()
        return auto_output_path(parent, s.object_name or s.name.strip("_"), len(frames),
                                s.integration_seconds(frames))

    def _refresh_reference(self) -> None:
        s = self.session
        if s is None or s.kind != "lights":
            return
        chosen = bool(s.reference_file)
        name = s.reference_name
        if name:
            fr = next((f for f in s.frames if s.frame_name(f) == name), None)
            text = (f"#{fr.number}" if fr else name) + ("" if chosen else " (automatic)")
        else:
            text = "Automatic (best frame)"
        self.settings_panel.set_reference(text, chosen)

    def _reference_changed(self) -> None:
        if self.session is not None:
            self.session.save_selection()
            self._refresh_reference()
            self._refresh_sequence_card()
            self._estimate_timer.start()

    def _reference_reset(self) -> None:
        s = self.session
        if s is not None and s.kind == "lights":
            s.reference_file = ""
            self._reference_changed()
            self.frames.refresh()

    def _refresh_sequence_card(self) -> None:
        s = self.session
        c = self.seq_card
        if s is None:
            return
        c.name.setText(s.name)
        c.folder.setText(s.path)
        frames = s.usable_selected()
        groups = s.exposure_groups(frames)
        exp = " + ".join(f"{n:,} × {e:g} s" for e, n in groups) or "exposure unknown"
        nights = s.nights(frames)
        if nights:
            a, b = datetime.fromisoformat(nights[0]), datetime.fromisoformat(nights[-1])
            span = a.strftime("%b %d") + ("" if a == b else " – " + b.strftime("%b %d"))
            night_txt = f"{len(nights)} night{'s' if len(nights) != 1 else ''}, {span}"
        else:
            night_txt = ""
        lines = [f"<b>{len(frames):,}</b> of {len(s.frames):,} frames  ·  <b>{format_hours(s.integration_seconds(frames))}</b>",
                 exp]
        if night_txt:
            lines.append(night_txt)
        bits = []
        if s.object_name:
            bits.append(s.object_name)
        if s.filter_name:
            bits.append(f"filter {s.filter_name}")
        if s.kind == "lights":
            bits.append(f"{s.width} × {s.height}" + (f" Bayer {s.bayer}" if s.bayer else ""))
        else:
            bits.append("RGB" if s.layers == 3 else "mono")
            if s.has_drizzle:
                bits.append("drizzle weights")
        lines.append("  ·  ".join(bits))
        if self.canvas_size():
            w, h = self.canvas_size()
            lines.append(f"Canvas {w:,} × {h:,} px")
        if s.kind == "lights":
            if not s.analyzed:
                n = sum(1 for fr in s.usable_selected() if s.frame_name(fr) not in s.metrics)
                lines.append(f"<span style='color:{COLORS['warning']}'>{n:,} frame(s) not analyzed yet</span>")
            elif s.registered:
                nreg = sum(1 for fr in s.usable_selected() if fr.has_reg)
                lines.append(f"Registered {nreg:,} frames")
        changed = s.selection_changes()
        if changed:
            what = "from Siril" if s.kind == "seq" else "excluded"
            lines.append(f"<span style='color:{COLORS['link']}'>Custom selection ({changed} "
                         f"{'change' + ('s' if changed != 1 else '') + ' ' if s.kind == 'seq' else ''}{what})</span>")
        c.lines.setText("<br>".join(lines))
        if s.problems:
            stoppers = [p for p in s.problems if "will be skipped" not in p]
            c.show_notice(" ".join(s.problems), "error" if stoppers else "warning")
        else:
            c.show_notice("")

    def canvas_size(self):
        return self._canvas_wh if getattr(self, "_canvas_wh", None) else None

    def _refresh_coverage(self) -> None:
        s = self.session
        self._canvas_wh = None
        self.depth = None
        if s is None:
            self.mosaic.view.set_coverage(None, 1, 0, 0)
            return
        res = s.canvas_and_coverage(binning_for_view) if s.usable_selected() else None
        if res is None:
            self.mosaic.view.set_coverage(None, 1, 0, 0)
            if s.kind == "lights" and s.usable_selected():
                self.mosaic.view.placeholder = ("Click Analyze (or Stack): GPU Stacker finds the stars in every frame "
                                                "and registers them, then this shows how they line up.")
                self.mosaic.info.setText("Frames not registered yet")
            else:
                self.mosaic.view.placeholder = "No frames selected"
                self.mosaic.info.setText("No frames selected")
            self.mosaic.view.update()
            return
        cw, ch, depth, b = res
        self._canvas_wh = (cw, ch)
        self.depth = (depth, b, (cw, ch))
        self.mosaic.view.set_coverage(depth, b, cw, ch)
        word = "canvas" if s.kind == "seq" else "stack"
        self.mosaic.info.setText(f"{cw:,} × {ch:,} px {word}  ·  up to {int(depth.max()):,} frames deep  ·  "
                                 "scroll to zoom, drag to pan")
        self._refresh_crop()

    def _refresh_crop(self) -> None:
        v = self.settings_panel.values()
        self.crop_rect = None
        info = self.settings_panel.crop_info
        if not v["crop_enabled"] or self.depth is None:
            self.mosaic.view.set_crop(None)
            info.setText("")
            return
        res = self.session.canvas_and_coverage(lambda w, h: max(1, math.ceil(max(w, h) / 450)))
        if res is None:
            return
        cw, ch, depth, b = res
        top, left, hh, ww = largest_rectangle(depth >= v["crop_min_frames"])
        if hh <= 2 or ww <= 2:
            self.mosaic.view.set_crop(None)
            info.setText("No area is covered by that many frames.")
            return
        x, y = (left + 1) * b, (top + 1) * b                 # one cell inside, to be safe
        w = min(cw - x, (ww - 2) * b)
        h = min(ch - y, (hh - 2) * b)
        self.crop_rect = (x, y, w, h)
        text = f"{w:,} × {h:,} px  ·  ≥ {v['crop_min_frames']:,} frames"
        self.mosaic.view.set_crop(self.crop_rect, text)
        info.setText(f"Crop {w:,} × {h:,} px at x {x:,}, y {y:,}")

    def _selection_changed(self) -> None:
        if self.session is None:
            return
        try:
            self.session.save_selection()
        except OSError as e:
            self.log.add(f"couldn't save the frame selection: {e}", "warn")
        if not self.output_custom:
            self.settings_panel.output.setText(self._default_output())
        self._refresh_coverage()
        self._refresh_sequence_card()
        self._estimate_timer.start()
        self._update_actions()
        self.output_path_tooltip()

    def output_path_tooltip(self) -> None:
        self.settings_panel.output.setToolTip(self.settings_panel.output.text())

    def _options_changed(self) -> None:
        self.settings_panel.save(self.settings)
        s = self.session
        if s is not None and s.kind == "lights":
            v = self.settings_panel.values()
            if (v["drizzle_scale"], v["pixfrac"]) != (s.scale, s.pixfrac):
                s.scale, s.pixfrac = v["drizzle_scale"], v["pixfrac"]
                self._refresh_coverage()
                self._refresh_sequence_card()
        self._refresh_crop()
        self._estimate_timer.start()
        self._update_actions()

    # ================================================================ estimate
    def _rate(self, key: str, default: float) -> float:
        try:
            return max(50e6, float(self.settings.value(f"rates/{key}", default)))
        except (TypeError, ValueError):
            return default

    def _refresh_plan(self) -> None:
        s = self.session
        if s is None:
            self.estimate_label.setText("")
            return
        v = self.settings_panel.values()
        frames = s.usable_selected()
        if len(frames) < 2:
            self.estimate_label.setText("Select at least 2 frames.")
            return
        if s.kind == "lights":
            self.estimate_label.setText(self._raw_estimate(s, frames, v))
            return
        est = estimate(s.seq, frames, s.info, v["normalization"], v["weighting"], v["feather"],
                       crop=list(self.crop_rect) if (v["crop_enabled"] and self.crop_rect) else None,
                       drizzle=s.has_drizzle)
        t_measure = est.measure_bytes / self._rate("measure", 400e6)
        t_stack = est.stack_bytes / self._rate("stack", 380e6)
        total = t_measure + t_stack + 10
        parts = [f"<b style='color:{COLORS['text']}'>≈ {format_duration(total)}</b>"]
        if est.to_measure:
            parts.append(f"measures {est.to_measure:,} frame{'s' if est.to_measure != 1 else ''} first "
                         f"({format_bytes(est.measure_bytes)}, cached afterwards)")
        parts.append(f"stacks {format_bytes(est.stack_bytes)}")
        if est.region and est.canvas:
            x0, y0, x1, y1 = est.region
            parts.append(f"output {x1 - x0:,} × {y1 - y0:,} px")
        self.estimate_label.setText(" · ".join(parts))

    def _raw_estimate(self, s, frames, v) -> str:
        names = [s.frame_name(fr) for fr in frames]
        todo_analyze = sum(1 for n in names if n not in s.metrics)
        ref = s.reference_file or s.registration.reference
        needs_reg = todo_analyze > 0 or not s.registered or (s.reference_file and ref != s.registration.reference)
        todo_measure = sum(1 for fr in frames if s.frame_name(fr) not in s.measures.frames)
        f = (s.width * s.height) / (2160 * 3840) if s.width else 1.0
        per_stack = float(self.settings.value("rates/raw_stack_per_frame", RAW_COSTS["stack"]))
        t = (todo_analyze * RAW_COSTS["analyze"] + (len(frames) * RAW_COSTS["register"] if needs_reg else 0)
             + todo_measure * RAW_COSTS["measure"] + len(frames) * per_stack * v["drizzle_scale"] ** 2) * f + 8
        parts = [f"<b style='color:{COLORS['text']}'>≈ {format_duration(t)}</b>"]
        first = []
        if todo_analyze:
            first.append(f"analyzes {todo_analyze:,} frames")
        if needs_reg:
            first.append("registers")
        if todo_measure:
            first.append(f"measures {todo_measure:,}")
        if first:
            parts.append(", ".join(first) + " first (cached afterwards)")
        parts.append(f"reads {format_bytes(s.raw_bytes(frames))} of raw files, no temporary files")
        if self.canvas_size():
            w, h = self.canvas_size()
            if v["crop_enabled"] and self.crop_rect:
                w, h = self.crop_rect[2], self.crop_rect[3]
            parts.append(f"output {w:,} × {h:,} px")
        return " · ".join(parts)

    # ================================================================ jobs
    def _base_cfg(self) -> dict:
        v = self.settings_panel.values()
        s = self.session
        cfg = dict(rejection=v["rejection"], sigma_low=v["sigma_low"], sigma_high=v["sigma_high"],
                   normalization=v["normalization"], rgb_equal=v["rgb_equal"], weighting=v["weighting"],
                   feather=v["feather"], output_norm=v["output_norm"], gpu_mem_gib=v["gpu_mem_gib"],
                   workers=v["workers"], strip=v["strip"], tile=v["tile"], save_coverage=v["save_coverage"],
                   preview_max=1600)
        if s.kind == "lights":
            cfg.update(lights_dir=s.lights_dir, reference_file=s.reference_file, drizzle_scale=v["drizzle_scale"],
                       pixfrac=v["pixfrac"], transform=v["transform"],
                       include_files=[s.frame_name(fr) for fr in s.frames if fr.filenum in s.included])
        else:
            cfg.update(seq_path=s.seq.path, include=sorted(s.included))
        if v["crop_enabled"] and self.crop_rect:
            cfg["crop"] = list(self.crop_rect)
        return cfg

    @staticmethod
    def _cfg_count(cfg: dict) -> int:
        return len(cfg.get("include_files") or cfg.get("include") or [])

    def start_analyze(self) -> None:
        s = self.session
        if not self._can_run() or s.kind != "lights":
            return
        cfg = self._base_cfg()
        cfg.update(analyze_only=True, output_path="")
        cfg.pop("crop", None)
        self._start(cfg, "analyze")

    def _can_run(self) -> bool:
        s = self.session
        if s is None or self.runner.running():
            return False
        stoppers = [p for p in s.problems if "will be skipped" not in p]
        return not stoppers and len(s.usable_selected()) >= 2 and not (self.gpu and self.gpu.get("error"))

    def start_stack(self) -> None:
        if not self._can_run():
            return
        out = self.settings_panel.output.text().strip()
        if not out:
            self._warn("No output file", "Choose where to save the stacked image.")
            return
        out = os.path.normpath(os.path.abspath(out))
        if not os.path.splitext(out)[1]:
            out += ".fit"
        folder = os.path.dirname(out)
        if not os.path.isdir(folder):
            try:
                os.makedirs(folder, exist_ok=True)
            except OSError as e:
                self._warn("Can't use that folder", str(e))
                return
        overwrite = False
        if os.path.exists(out):
            r = QMessageBox.question(self, "Overwrite?", f"{os.path.basename(out)} already exists.\n\nReplace it?")
            if r != QMessageBox.StandardButton.Yes:
                return
            overwrite = True
        cfg = self._base_cfg()
        s = self.session
        w, h = self.canvas_size() or (0, 0)
        if not w and s.kind == "lights":                     # not registered yet: frame size, a bit larger
            sc = cfg.get("drizzle_scale", 1.0)
            w, h = int(s.width * sc * 1.15), int(s.height * sc * 1.15)
        if cfg.get("crop"):
            w, h = cfg["crop"][2], cfg["crop"][3]
        need = s.layers * w * h * 4 * (1.25 if cfg["save_coverage"] else 1.0)
        free = shutil.disk_usage(folder).free
        if need > free * 0.98:
            self._warn("Not enough disk space", f"The output needs about {format_bytes(need)} but the drive has "
                       f"{format_bytes(free)} free.")
            return
        cfg.update(output_path=out, overwrite=overwrite)
        self._start(cfg, "stack")

    def start_test(self) -> None:
        if not self._can_run() or self.depth is None:
            return
        depth, b, (cw, ch) = self.depth
        cfg = self._base_cfg()
        lo_row, hi_row = 0, ch
        if cfg.get("crop"):
            lo_row, hi_row = cfg["crop"][1], cfg["crop"][1] + cfg["crop"][3]
        rows_depth = depth.mean(axis=1)
        gy0, gy1 = lo_row // b, max(lo_row // b + 1, hi_row // b)
        center = (gy0 + int(np.argmax(rows_depth[gy0:gy1]))) * b + b // 2
        half = 96
        r0 = max(lo_row, center - half)
        r1 = min(hi_row, r0 + 2 * half)
        cfg.update(rows=[r0, r1], output_path="", preview_max=max(cw, ch))      # full resolution
        self.mosaic.view.set_band((r0, r1))
        self._start(cfg, "test")

    def _start(self, cfg: dict, kind: str) -> None:
        self.job_kind = kind
        self.job_cfg = cfg
        self.job_t0 = time.time()
        self.job_outcome = ""
        self.canvas_ev = None
        self._stderr_tail = []
        if kind != "test":
            self.mosaic.view.set_band(None)
        label = {"test": "Test strip", "stack": "Stack", "analyze": "Analyze"}[kind]
        extra = (f", drizzle x{cfg['drizzle_scale']:g} pixfrac {cfg['pixfrac']:g}" if cfg.get("lights_dir") else "")
        self.log.add(f"{label}: {self._cfg_count(cfg)} frames selected, rejection {cfg['rejection']}, weighting "
                     f"{cfg['weighting']}, feather {cfg['feather']:g}{extra}"
                     + (f" -> {cfg['output_path']}" if cfg.get("output_path") else ""))
        try:
            self.runner.start(cfg)
        except Exception as e:                       # noqa: BLE001
            self._warn("Couldn't start the engine", str(e))
            return
        self.runbar.set_running("Starting the GPU engine")
        self.runbar.bar.setValue(0)
        self._set_running(True)
        self.tabs.setCurrentWidget(self.frames if kind == "analyze" else self.mosaic)

    def cancel_job(self) -> None:
        if self.runner.running():
            self.runner.cancel()
            self.runbar.cancel.setEnabled(False)
            self.runbar.cancel.setText("Cancelling…")
            self.log.add("cancel requested", "warn")

    def _set_running(self, running: bool) -> None:
        self.settings_panel.set_running(running)
        self.frames.set_locked(running)
        self.seq_card.open_btn.setEnabled(not running)
        self.open_act.setEnabled(not running)
        self.recent_menu.setEnabled(not running)
        self.reload_act.setEnabled(not running)
        self._update_actions()
        if not running:
            self.setWindowTitle("GPU Stacker")

    def _update_actions(self) -> None:
        ok = self._can_run()
        s = self.session
        lights = s is not None and s.kind == "lights"
        self.stack_btn.setEnabled(ok)
        self.test_btn.setEnabled(ok and self.depth is not None)
        self.test_btn.setToolTip("Stack a narrow band through the densest part of the image with these settings and "
                                 "preview it, without writing a file." if self.depth is not None or not lights else
                                 "Analyze the frames first (the strip is placed on the registered image).")
        self.analyze_btn.setVisible(lights)
        self.analyze_btn.setEnabled(ok and lights)
        self.stack_act.setEnabled(ok)
        running = self.runner.running()
        self.stack_btn.setText("Stacking…" if running and self.job_kind == "stack" else "Stack")
        self.test_btn.setText("Testing…" if running and self.job_kind == "test" else "Test strip")
        self.analyze_btn.setText("Analyzing…" if running and self.job_kind == "analyze" else "Analyze")

    # ---------------------------------------------------------------- engine events
    def _on_event(self, ev: dict) -> None:
        kind = ev.get("ev")
        if kind == "hello":
            g = ev.get("gpu", {})
            self.log.add(f"engine {ev.get('version')} on {g.get('name')} ({g.get('vram_total', 0) / 2**30:.1f} GB)")
        elif kind == "log":
            self.log.add(ev.get("msg", ""), ev.get("level", "info"))
        elif kind == "phase":
            label = {"headers": "Reading headers", "measure": "Measuring frames", "analyze": "Detecting stars",
                     "register": "Registering frames",
                     "stack": "Stacking" if self.job_kind == "stack" else "Stacking test strip",
                     "write": "Writing FITS"}.get(ev.get("name"), ev.get("label", ""))
            self.runbar.set_running(label)
            self.runbar.bar.setValue(0)
            if ev.get("name") == "measure":
                self.runbar.detail.setText(f"{ev.get('frames', 0):,} frames need statistics (first run only)")
            if ev.get("name") == "analyze" and self.session is not None:
                self.tabs.setCurrentWidget(self.frames)
        elif kind == "progress":
            self._progress(ev)
        elif kind == "analyzed":
            if self.session is not None and self.session.kind == "lights":
                self.session.apply_analyzed(ev.get("frames", []))
                self._live_refresh()
        elif kind == "registered":
            self.log.add(f"registered {ev['registered']} frames on {ev['reference']}"
                         + (f", {ev['failed']} failed" if ev.get("failed") else "")
                         + f"; median residual {ev.get('rms_median', 0):.2f} px, rotation "
                           f"{ev['rotation'][0]:.2f} to {ev['rotation'][1]:.2f} deg", "ok")
        elif kind == "frame_bad":
            name = f" ({ev['name']})" if ev.get("name") else ""
            self.log.add(f"frame #{ev['number']}{name} skipped: {ev['reason']}", "warn")
        elif kind == "canvas":
            self.canvas_ev = ev
            self.mosaic.view.begin_live(ev["region"], ev["bw"], ev["bh"], ev["channels"])
            self.mosaic.stack_available(True, select=True)
        elif kind == "rows":
            raw = base64.b64decode(ev["data"])
            block = np.frombuffer(raw, np.float16).astype(np.float32).reshape(ev["channels"], ev["rows"], ev["width"])
            self.mosaic.view.add_rows(ev["y"], block)
        elif kind == "done":
            self._done(ev)
        elif kind == "error":
            self.job_outcome = "error"
            self.log.add(ev.get("msg", "error"), "error")
            if ev.get("trace"):
                self.log.add(ev["trace"], "error")
            self._error_box(ev)
        elif kind == "cancelled":
            self.job_outcome = "cancelled"
            self.log.add("cancelled", "warn")

    def _progress(self, ev: dict) -> None:
        total = ev.get("total") or 1
        frac = min(1.0, ev.get("done", 0) / total)
        self.runbar.bar.setValue(int(frac * 1000))
        phase = ev.get("phase")
        rate = ev.get("rate")
        eta = ev.get("eta")
        bits = []
        if phase in ("measure", "analyze", "register"):
            bits.append(f"frame {ev.get('frames_done', 0):,} / {ev.get('frames_total', 0):,}")
        elif phase == "stack":
            if ev.get("rows_total"):
                bits.append(f"rows {ev.get('rows', 0):,} / {ev['rows_total']:,}")
            if ev.get("frames"):
                bits.append(f"{ev['frames']:,} frames in this strip")
            bits.append(f"{format_bytes(ev.get('done', 0))} of {format_bytes(total)} read")
        elif phase == "headers":
            bits.append(f"{int(ev.get('done', 0)):,} / {int(total):,}")
        if rate:
            bits.append(f"{rate / 1e6:.0f} MB/s")
        self.runbar.detail.setText("   ·   ".join(bits))
        self.runbar.eta.setText(f"{format_duration(eta)} left" if eta else "")
        if phase in ("measure", "stack", "analyze"):
            self.setWindowTitle(f"{frac * 100:.0f}% · {self.runbar.phase.text()} — GPU Stacker")
        lights = self.session is not None and self.session.kind == "lights"
        if rate and self.job_kind == "stack" and not lights and phase in ("measure", "stack") and frac > 0.15:
            self.settings.setValue(f"rates/{phase}", rate)     # learn throughput from full runs only

    def _live_refresh(self) -> None:
        """Throttled table/chart refresh while metrics stream in."""
        if getattr(self, "_live_timer", None) is None:
            self._live_timer = QTimer(self)
            self._live_timer.setSingleShot(True)
            self._live_timer.setInterval(600)
            self._live_timer.timeout.connect(self.frames.refresh)
        if not self._live_timer.isActive():
            self._live_timer.start()

    def _done(self, ev: dict) -> None:
        self.job_outcome = "done"
        if ev.get("analysis"):
            took = format_duration(ev.get("seconds_total", time.time() - self.job_t0))
            self.log.add(f"analysis done in {took}: {ev['frames']} frames registered", "ok")
            self.runbar.set_idle(f"Analyzed and registered {ev['frames']:,} frames in {took}", "ok")
            return
        lights = self.session is not None and self.session.kind == "lights"
        if lights and self.job_kind == "stack" and ev.get("frames") and ev.get("seconds_stack"):
            self.settings.setValue("rates/raw_stack_per_frame", ev["seconds_stack"] / ev["frames"]
                                   / max(self.job_cfg.get("drizzle_scale", 1.0) ** 2, 1.0))
        self.mosaic.view.end_live()
        preview = None
        path = ev.get("preview")
        if path and os.path.isfile(path):
            try:
                preview = np.load(path)
            finally:
                try:
                    os.remove(path)
                except OSError:
                    pass
        binf = self.canvas_ev["bin"] if self.canvas_ev else 1
        if preview is not None:
            self.mosaic.view.set_final(preview, ev["region"])
        self.result.show_result(ev, preview, binf, self.session.object_name if self.session else "")
        took = format_duration(ev.get("seconds_total", time.time() - self.job_t0))
        if ev.get("output"):
            self.log.add(f"wrote {ev['output']} in {took}", "ok")
            self.runbar.set_idle(f"Done in {took}  ·  {os.path.basename(ev['output'])}", "ok")
        else:
            self.log.add(f"test strip done in {took}", "ok")
            self.runbar.set_idle(f"Test strip done in {took}", "ok")
        self.tabs.setCurrentWidget(self.result)
        QApplication.alert(self, 0)

    def _on_stderr(self, line: str) -> None:
        self._stderr_tail = (self._stderr_tail + [line])[-40:]

    def _on_finished(self, code: int) -> None:
        self.runner.cleanup()
        self.mosaic.view.end_live()
        if self.job_outcome == "cancelled":
            self.runbar.set_idle("Cancelled", "warn")
        elif self.job_outcome == "error":
            self.runbar.set_idle("Stopped with an error (see the Log tab)", "error")
        elif self.job_outcome != "done":
            tail = "\n".join(self._stderr_tail[-15:])
            self.log.add(f"the engine exited unexpectedly (code {code})\n{tail}", "error")
            self.runbar.set_idle("The engine stopped unexpectedly (see the Log tab)", "error")
            if not self.runner.cancelling:
                self._error_box({"msg": f"The GPU engine stopped unexpectedly (exit code {code}).", "trace": tail})
        if self.job_kind == "test":
            self.mosaic.view.set_band(None)
        if self.session is not None:
            self.session.reload()
            self.frames.refresh()
            if self.session.kind == "lights":
                had_stack = self.mosaic.stack_btn.isChecked()
                self._refresh_coverage()
                self._refresh_reference()
                if had_stack and self.mosaic.view.live is not None:
                    self.mosaic.view.set_show_stack(True)
            self._refresh_sequence_card()
            self._refresh_plan()
        self.job_kind = ""
        self._set_running(False)

    # ================================================================ misc
    def _probe_done(self, info: dict) -> None:
        self.gpu = info
        if info.get("error"):
            self.status_gpu.setText("No usable GPU")
            self.log.add(f"GPU check failed: {info['error']}", "error")
            self.status_gpu.setToolTip(info["error"])
        else:
            rt = info.get("cuda_runtime", 0)
            self.status_gpu.setText(f"{info.get('name')}  ·  {info.get('vram_total', 0) / 2**30:.0f} GB  ·  "
                                    f"CUDA {rt // 1000}.{(rt % 1000) // 10}")
        self._update_actions()

    def _recent(self) -> list[str]:
        v = self.settings.value("recent", [])
        if isinstance(v, str):
            v = [v]
        return [p for p in (v or []) if isinstance(p, str)][:MAX_RECENT]

    def _add_recent(self, path: str) -> None:
        r = [path] + [p for p in self._recent() if os.path.normcase(p) != os.path.normcase(path)]
        self.settings.setValue("recent", r[:MAX_RECENT])
        self._fill_recent_menu()
        self.welcome.set_recent(self._recent())

    def _fill_recent_menu(self) -> None:
        self.recent_menu.clear()
        for p in self._recent():
            a = self.recent_menu.addAction(p)
            a.triggered.connect(lambda _c=False, path=p: self.open_path(path))
        self.recent_menu.setEnabled(bool(self._recent()))

    def _warn(self, title: str, text: str) -> None:
        QMessageBox.warning(self, title, text)

    def _error_box(self, ev: dict) -> None:
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning if ev.get("user") else QMessageBox.Icon.Critical)
        box.setWindowTitle("Stacking stopped")
        box.setText(ev.get("msg", "Unknown error"))
        if ev.get("trace"):
            box.setDetailedText(ev["trace"])
        box.open()

    def _about(self) -> None:
        from .runner import ROOT

        def where(*names: str) -> str:              # source checkout: LICENSE; download: LICENSE.txt above app
            for d in (ROOT, os.path.dirname(ROOT)):
                for n in names:
                    if os.path.isfile(os.path.join(d, n)):
                        return os.path.join(d, n)
            return names[0]
        QMessageBox.about(
            self, "About GPU Stacker",
            f"<b>GPU Stacker {__version__}</b><br>Copyright © 2026 Jason Prater (JP Astrophotography)<br><br>"
            "Stacks raw lights (or a Siril-registered sequence) on an NVIDIA GPU (CUDA via CuPy): star detection, "
            "registration (mosaics too), Bayer drizzle, then Siril 1.4's normalization, sigma / winsorized "
            "rejection, noise / wFWHM / star weighting, feathering and max framing. Writes a 32-bit FITS with the "
            "reference frame's header.<br><br>"
            "This program comes with ABSOLUTELY NO WARRANTY. It is free software: you can redistribute it and/or "
            "modify it under the terms of the GNU General Public License, version 3 or (at your option) any later "
            "version.<br><br>"
            "It contains code ported from Siril, RawTherapee, CFITSIO and OpenCV, and uses Qt for Python "
            "(Copyright © The Qt Company Ltd. and other contributors) under the GNU Lesser General Public "
            "License, version 3. The licenses and notices are in<br>"
            f"{where('LICENSE', 'LICENSE.txt')}<br>{where('THIRD-PARTY-NOTICES.txt')}")

    def _install_siril_script(self) -> None:
        from .siril_link import SCRIPT_FOLDER, on_start, siril_lists_folder
        on_start()
        what = ("It stacks the registered sequence loaded in Siril, or a folder of raw lights, on this graphics "
                "card and opens the result in Siril.")
        if siril_lists_folder():
            QMessageBox.information(self, "GPU Stacker", "Siril already has GPU Stacker: Scripts > Python Scripts > "
                                                         f"GPU Stacker > GPU-Stacker.py.\n\n{what}")
            return
        QApplication.clipboard().setText(SCRIPT_FOLDER)
        QMessageBox.information(
            self, "GPU Stacker",
            "To add GPU Stacker to Siril's Scripts menu:\n\n"
            "1. In Siril, open Preferences > Scripts.\n"
            "2. Under Scripts Storage Directories, add this folder on a new line (it is on the clipboard, so just "
            f"paste):\n    {SCRIPT_FOLDER}\n"
            "3. Click the refresh button under that list, then Apply.\n\n"
            f"Siril then shows Scripts > Python Scripts > GPU Stacker > GPU-Stacker.py. {what}")

    def _restore_geometry(self) -> None:
        g = self.settings.value("geometry")
        if g is not None:
            self.restoreGeometry(g)
        else:
            self.resize(1480, 920)

    # ---------------------------------------------------------------- drag & drop
    def dragEnterEvent(self, ev) -> None:
        urls = ev.mimeData().urls()
        if urls and not self.runner.running():
            p = urls[0].toLocalFile()
            if p.lower().endswith((".seq", ".fit", ".fits", ".fts")) or os.path.isdir(p):
                ev.acceptProposedAction()
                self.welcome.set_dragging(True)

    def dragLeaveEvent(self, ev) -> None:
        self.welcome.set_dragging(False)

    def dropEvent(self, ev) -> None:
        self.welcome.set_dragging(False)
        urls = ev.mimeData().urls()
        if urls:
            p = urls[0].toLocalFile()
            if p.lower().endswith((".fit", ".fits", ".fts")):      # a light frame: open its folder
                p = os.path.dirname(p)
            self.open_path(p)

    # ---------------------------------------------------------------- closing
    def closeEvent(self, ev) -> None:
        if self.runner.running():
            if self.ask_before_quit:
                r = QMessageBox.question(self, "Stop stacking?", "A stack is still running. Cancel it and quit?")
                if r != QMessageBox.StandardButton.Yes:
                    ev.ignore()
                    return
            self.runner.cancel()
            loop = QEventLoop()
            QTimer.singleShot(20000, loop.quit)
            self.runner.finished.connect(loop.quit)
            if self.runner.running():
                loop.exec()
            self.runner.kill()
        self.settings_panel.save(self.settings)
        self.settings.setValue("geometry", self.saveGeometry())
        super().closeEvent(ev)
