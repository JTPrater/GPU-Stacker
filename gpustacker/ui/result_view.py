# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jason Prater (JP Astrophotography)
"""Result tab: auto-stretched preview of the finished stack and what was written."""
from __future__ import annotations

import os
import subprocess

import numpy as np
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QDesktopServices, QGuiApplication
from PySide6.QtCore import QUrl
from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from .imageview import ImageView, qimage_rgb
from .session import format_duration, format_hours
from .stretch import apply_stf, auto_stf
from .theme import set_role


class ResultView(ImageView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.placeholder = "The stacked image will appear here"
        self.data: np.ndarray | None = None
        self.binning = 1
        self.hover_text = ""
        self.hovered.connect(self._on_hover)

    def _on_hover(self, x: float, y: float) -> None:
        if x < 0 or self.data is None:
            self.hover_text = ""
        else:
            gx, gy = int(x) // self.binning, int(y) // self.binning
            gy, gx = min(gy, self.data.shape[1] - 1), min(gx, self.data.shape[2] - 1)
            vals = self.data[:, gy, gx]
            v = "  ".join(f"{c} {val:.4f}" for c, val in zip("RGB" if len(vals) == 3 else "L", vals))
            self.hover_text = f"x {int(x):,}  y {int(y):,}   ·   {v}"
        self.update()

    def paint_overlay(self, p) -> None:
        if self.hover_text:
            self.draw_badge(p, self.hover_text, "br")


class ResultPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(8)
        head = QHBoxLayout()
        col = QVBoxLayout()
        col.setSpacing(2)
        self.title = QLabel("No result yet")
        set_role(self.title, "title")
        self.details = QLabel("")
        self.details.setWordWrap(True)                 # wraps instead of squeezing the buttons on small screens
        set_role(self.details, "muted")
        self.details.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        col.addWidget(self.title)
        col.addWidget(self.details)
        head.addLayout(col, 1)
        self.linked = QCheckBox("Linked stretch")
        self.linked.setToolTip("Same stretch for all channels (shows colour casts). Unlinked neutralizes the "
                               "background. Display only: the FITS file stays linear.")
        self.open_btn = QPushButton("Open")
        self.open_btn.setToolTip("Open the FITS file with its default program (Siril, PixInsight, ...)")
        self.folder_btn = QPushButton("Show in folder")
        self.copy_btn = QPushButton("Copy path")
        for w in (self.linked, self.open_btn, self.folder_btn, self.copy_btn):
            head.addWidget(w)
        lay.addLayout(head)
        self.view = ResultView()
        lay.addWidget(self.view, 1)
        self.path = ""
        self.region = (0, 0, 0, 0)
        self._set_buttons()
        self.linked.toggled.connect(lambda _on: self._render())
        self.open_btn.clicked.connect(self._open)
        self.folder_btn.clicked.connect(self._folder)
        self.copy_btn.clicked.connect(lambda: QGuiApplication.clipboard().setText(self.path))

    def _set_buttons(self) -> None:
        ok = bool(self.path) and os.path.isfile(self.path)
        for b in (self.open_btn, self.folder_btn, self.copy_btn):
            b.setEnabled(ok)

    def show_result(self, summary: dict, preview: np.ndarray | None, binning: int, object_name: str = "") -> None:
        self.path = summary.get("output") or ""
        self.region = tuple(summary.get("region", (0, 0, summary["width"], summary["height"])))
        name = os.path.basename(self.path) if self.path else "Test strip (not saved)"
        self.title.setText(name)
        lo = summary.get("rejected_low") or [0]
        hi = summary.get("rejected_high") or [0]
        bits = [f"{summary['width']:,} × {summary['height']:,}",
                f"{summary['frames']:,} frames", format_hours(summary.get("livetime", 0)) + " integration",
                f"rejected {np.mean(lo) * 100:.2f}% low / {np.mean(hi) * 100:.2f}% high",
                f"done in {format_duration(summary.get('seconds_total', 0))}"]
        if summary.get("test_strip"):
            bits.insert(0, f"rows {self.region[1]:,}–{self.region[3]:,}")
        if summary.get("skipped"):
            bits.append(f"{len(summary['skipped'])} unusable frame(s) skipped")
        self.details.setText("  ·  ".join(bits))
        self._set_buttons()
        self.view.data = preview
        self.view.binning = binning
        if preview is not None:
            self.view.set_scene(summary["width"], summary["height"])
            self._render()
            if summary.get("test_strip"):
                self._focus_strip(preview, binning, summary["height"])

    def _focus_strip(self, preview: np.ndarray, binning: int, height: int) -> None:
        """A test strip is a thin band: zoom so it fills most of the view, centred on its brightest part."""
        lum = preview.mean(axis=0).mean(axis=0)
        k = max(1, lum.size // 40)
        smooth = np.convolve(lum, np.ones(k) / k, mode="same")
        m = lum.size // 10                       # not the ends: few frames reach them, so they look noisy
        if lum.size - 2 * m > 0:
            smooth[:m] = smooth[lum.size - m:] = -np.inf
        cx = (int(np.argmax(smooth)) + 0.5) * binning
        self.view.focus_on(cx, height / 2, height)

    def _render(self) -> None:
        d = self.view.data
        if d is None:
            return
        rgb = apply_stf(d, auto_stf(d, linked=self.linked.isChecked()))
        img = qimage_rgb(rgb)
        w, h = self.view._scene.width(), self.view._scene.height()
        self.view.set_layers([(img, QRectF(0, 0, d.shape[2] * self.view.binning, d.shape[1] * self.view.binning), 1.0)])
        self.view.set_scene(w, h, keep_view=True)

    def _open(self) -> None:
        if self.path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.path))

    def _folder(self) -> None:
        if not self.path:
            return
        if os.name == "nt":
            subprocess.Popen(["explorer", "/select,", os.path.normpath(self.path)])
        else:
            QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.dirname(self.path)))
