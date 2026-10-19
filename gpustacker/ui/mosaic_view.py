# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jason Prater (JP Astrophotography)
"""Canvas tab: coverage heat map of the canvas, crop / test-strip overlays, live stack preview."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QLinearGradient, QPainter, QPen
from PySide6.QtWidgets import QButtonGroup, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from .imageview import ImageView, dashed_pen, qimage_rgba, tooltip_box
from .stretch import COVERAGE_STOPS, apply_stf, auto_stf, coverage_rgba
from .theme import color, set_role, ui_font
from .widgets import ElidedLabel


class MosaicView(ImageView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.placeholder = "Open a sequence to see its mosaic coverage"
        self.depth: np.ndarray | None = None
        self.depth_bin = 1
        self.vmax = 1
        self._cov_img: QImage | None = None
        self.crop: tuple | None = None           # (x, y, w, h) canvas pixels
        self.crop_text = ""
        self.band: tuple | None = None           # (y0, y1) test strip rows
        self.show_stack = False
        self.live: np.ndarray | None = None      # (C, bh, bw) linear, binned
        self.live_rect = QRectF()
        self.live_front = -1.0                   # canvas row reached so far (-1 = done / none)
        self._live_img: QImage | None = None
        self._dirty = False
        self._params = None
        self._param_fill = 0.0
        self._timer = QTimer(self)
        self._timer.setInterval(700)
        self._timer.timeout.connect(self._render_live)
        self.hover_text = ""
        self.hovered.connect(self._on_hover)

    # ---------------------------------------------------------------- coverage
    def set_coverage(self, depth: np.ndarray | None, binning: int, width: int, height: int) -> None:
        self.depth, self.depth_bin = depth, binning
        if depth is None:
            self._cov_img = None
            self.set_scene(0, 0)
            self._relayer()
            return
        self.vmax = int(depth.max()) if depth.size else 1
        self._cov_img = qimage_rgba(coverage_rgba(depth, self.vmax))
        self.set_scene(width, height, keep_view=True)
        self._relayer()

    def set_crop(self, crop: tuple | None, text: str = "") -> None:
        self.crop, self.crop_text = crop, text
        self.update()

    def set_band(self, band: tuple | None) -> None:
        self.band = band
        self.update()

    # ---------------------------------------------------------------- live preview
    def begin_live(self, region, bw: int, bh: int, channels: int) -> None:
        x_lo, y_lo, x_hi, y_hi = region
        self.live = np.zeros((channels, bh, bw), np.float32)
        self.live_rect = QRectF(x_lo, y_lo, x_hi - x_lo, y_hi - y_lo)
        self.live_front = float(y_lo)
        self._live_img = None
        self._params = None
        self._param_fill = 0.0
        self.show_stack = True
        self._timer.start()
        self._relayer()

    def add_rows(self, y: int, block: np.ndarray) -> None:
        if self.live is None:
            return
        k = min(block.shape[1], self.live.shape[1] - y)
        if k <= 0:
            return
        self.live[:, y:y + k, :block.shape[2]] = block[:, :k, :self.live.shape[2]]
        frac = (y + k) / self.live.shape[1]
        self.live_front = self.live_rect.top() + frac * self.live_rect.height()
        self._dirty = True

    def end_live(self) -> None:
        self._timer.stop()
        self.live_front = -1.0
        if self._dirty:
            self._render_live()
        self.update()

    def set_final(self, preview: np.ndarray, region) -> None:
        """Show a finished stack (binned linear data) as the stack layer."""
        x_lo, y_lo, x_hi, y_hi = region
        self.live = preview.astype(np.float32, copy=False)
        self.live_rect = QRectF(x_lo, y_lo, x_hi - x_lo, y_hi - y_lo)
        self.live_front = -1.0
        self._params = None
        self._dirty = True
        self._render_live()

    def clear_live(self) -> None:
        self._timer.stop()
        self.live = None
        self._live_img = None
        self.live_front = -1.0
        self.show_stack = False
        self._relayer()

    def _render_live(self) -> None:
        if not self._dirty or self.live is None:
            return
        self._dirty = False
        data = self.live
        fill = 1.0 if self.live_front < 0 else (self.live_front - self.live_rect.top()) / max(1.0, self.live_rect.height())
        if self._params is None or fill - self._param_fill > 0.08 or fill >= 1.0:
            self._params = auto_stf(data)            # re-derive the stretch as more of the image arrives
            self._param_fill = fill
        rgb = apply_stf(data, self._params)
        has = np.any(data != 0, axis=0)
        rgba = np.dstack([rgb, np.where(has, 255, 0).astype(np.uint8)])
        self._live_img = qimage_rgba(rgba)
        self._relayer()

    def set_show_stack(self, on: bool) -> None:
        self.show_stack = on
        self._relayer()

    def _relayer(self) -> None:
        layers = []
        if self._cov_img is not None:
            w, h = self.depth.shape[1] * self.depth_bin, self.depth.shape[0] * self.depth_bin
            layers.append((self._cov_img, QRectF(0, 0, w, h), 0.28 if (self.show_stack and self._live_img) else 1.0))
        if self.show_stack and self._live_img is not None:
            layers.append((self._live_img, self.live_rect, 1.0))
        self.set_layers(layers)

    # ---------------------------------------------------------------- overlays
    def _on_hover(self, x: float, y: float) -> None:
        if x < 0 or self.depth is None:
            self.hover_text = ""
        else:
            gy, gx = int(y) // self.depth_bin, int(x) // self.depth_bin
            d = int(self.depth[gy, gx]) if 0 <= gy < self.depth.shape[0] and 0 <= gx < self.depth.shape[1] else 0
            self.hover_text = f"x {int(x):,}  y {int(y):,}   ·   {d:,} frame{'s' if d != 1 else ''}"
        self.update()

    def paint_overlay(self, p: QPainter) -> None:
        if not self.has_scene():
            return
        if self.band:
            r = self.scene_rect_to_widget(QRectF(0, self.band[0], self._scene.width(), self.band[1] - self.band[0]))
            p.setPen(dashed_pen(color("band"), 1.4))
            p.setBrush(color("band", 30))
            p.drawRect(r)
            self._tag(p, r.topLeft() + QPointF(6, -22), "Test strip")
        if self.crop:
            x, y, w, h = self.crop
            r = self.scene_rect_to_widget(QRectF(x, y, w, h))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(QColor(0, 0, 0, 120), 3))
            p.drawRect(r)
            p.setPen(dashed_pen(QColor("#ffffff"), 1.4))
            p.drawRect(r)
            if self.crop_text:
                self._tag(p, r.topLeft() + QPointF(6, 6), self.crop_text)
        if self.live_front >= 0 and self.live is not None:
            y = self.to_widget(0, self.live_front).y()
            x0 = self.to_widget(self.live_rect.left(), 0).x()
            x1 = self.to_widget(self.live_rect.right(), 0).x()
            g = QLinearGradient(0, y - 18, 0, y)
            g.setColorAt(0, QColor(51, 153, 255, 0))
            g.setColorAt(1, QColor(51, 153, 255, 90))
            p.fillRect(QRectF(x0, y - 18, x1 - x0, 18), g)
            p.setPen(QPen(color("accent"), 1.5))
            p.drawLine(QPointF(x0, y), QPointF(x1, y))
        if self._cov_img is not None and not (self.show_stack and self._live_img is not None):
            self._legend(p)
        if self.hover_text:
            self.draw_badge(p, self.hover_text, "br")

    def _tag(self, p: QPainter, at: QPointF, text: str) -> None:
        p.setFont(ui_font(9))
        w = p.fontMetrics().horizontalAdvance(text) + 12
        rect = QRectF(at.x(), at.y(), w, p.fontMetrics().height() + 6)
        tooltip_box(p, rect)
        p.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)

    def _legend(self, p: QPainter) -> None:
        x, y, w, h = 18, self.height() - 42, 150, 8
        g = QLinearGradient(x, 0, x + w, 0)
        for pos, (r, gg, b) in COVERAGE_STOPS:
            g.setColorAt(pos, QColor(r, gg, b))
        tooltip_box(p, QRectF(x - 8, y - 23, w + 16, 47))
        p.setPen(QPen(QColor("#828790"), 1))
        p.setBrush(g)
        p.drawRect(QRectF(x, y, w, h))
        p.setFont(ui_font(8))
        p.setPen(QColor("#000000"))
        p.drawText(QRectF(x, y - 19, w, 14), Qt.AlignmentFlag.AlignLeft, "Frames per pixel")
        p.drawText(QRectF(x, y + 9, w, 14), Qt.AlignmentFlag.AlignLeft, "1")
        p.drawText(QRectF(x, y + 9, w, 14), Qt.AlignmentFlag.AlignRight, f"{self.vmax:,}")


class MosaicPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)
        bar = QHBoxLayout()
        bar.setSpacing(4)
        self.cov_btn = QPushButton("Coverage")
        self.stack_btn = QPushButton("Stack preview")
        group = QButtonGroup(self)
        for b in (self.cov_btn, self.stack_btn):
            b.setCheckable(True)
            group.addButton(b)
            bar.addWidget(b)
        self.cov_btn.setChecked(True)
        self.stack_btn.setEnabled(False)
        bar.addSpacing(12)
        self.info = ElidedLabel("", Qt.TextElideMode.ElideRight)    # shortens instead of squeezing the buttons
        set_role(self.info, "muted")
        bar.addWidget(self.info, 1)
        fit = QPushButton("Fit")
        fit.setMinimumWidth(75)
        fit.setToolTip("Fit to window (double-click the image: 100% / fit)")
        bar.addWidget(fit)
        lay.addLayout(bar)
        self.view = MosaicView()
        lay.addWidget(self.view, 1)
        fit.clicked.connect(self.view.fit)
        self.stack_btn.toggled.connect(self.view.set_show_stack)

    def stack_available(self, on: bool, select: bool = False) -> None:
        self.stack_btn.setEnabled(on)
        if select and on:
            self.stack_btn.setChecked(True)
        if not on:
            self.cov_btn.setChecked(True)
