# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jason Prater (JP Astrophotography)
"""Zoom / pan image view drawn with QPainter. Layers are QImages placed in 'scene' coordinates
(full-resolution canvas pixels), so binned previews and overlays share one coordinate system."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QLinearGradient, QMouseEvent, QPainter, QPen, QWheelEvent
from PySide6.QtWidgets import QWidget

from .theme import COLORS, ui_font


def qimage_rgb(rgb: np.ndarray) -> QImage:
    """(h, w, 3) uint8 -> QImage (owns a copy)."""
    rgb = np.ascontiguousarray(rgb)
    h, w, _ = rgb.shape
    return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


def qimage_rgba(rgba: np.ndarray) -> QImage:
    rgba = np.ascontiguousarray(rgba)
    h, w, _ = rgba.shape
    return QImage(rgba.data, w, h, 4 * w, QImage.Format.Format_RGBA8888).copy()


class ImageView(QWidget):
    hovered = Signal(float, float)          # scene coordinates, (-1, -1) when outside

    ZOOM_MIN, ZOOM_MAX = 1 / 64, 16.0

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setMinimumSize(200, 150)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)
        self._scene = QRectF()               # extent of the scene (canvas pixels)
        self._layers: list[tuple[QImage, QRectF, float]] = []
        self._scale = 1.0
        self._ox = self._oy = 0.0
        self._fit = True
        self._focus: tuple | None = None       # (x, y, height, frac, max_scale) kept centred across resizes
        self._drag: QPointF | None = None
        self._drag_origin = (0.0, 0.0)
        self.placeholder = ""

    # ---------------------------------------------------------------- content
    def set_scene(self, width: float, height: float, keep_view: bool = False) -> None:
        changed = (self._scene.width(), self._scene.height()) != (width, height)
        self._scene = QRectF(0, 0, width, height)
        if changed or not keep_view:
            self._fit = True
            self._focus = None
            self._apply_fit()
        self.update()

    def set_layers(self, layers: list[tuple[QImage, QRectF, float]]) -> None:
        self._layers = layers
        self.update()

    def has_scene(self) -> bool:
        return self._scene.width() > 0

    # ---------------------------------------------------------------- view
    def _apply_fit(self) -> None:
        if not self.has_scene():
            return
        m = 12.0
        sx = (self.width() - 2 * m) / self._scene.width()
        sy = (self.height() - 2 * m) / self._scene.height()
        self._scale = max(min(sx, sy), 1e-6)
        self._ox = (self.width() - self._scene.width() * self._scale) / 2
        self._oy = (self.height() - self._scene.height() * self._scale) / 2

    def fit(self) -> None:
        self._fit = True
        self._focus = None
        self._apply_fit()
        self.update()

    def focus_on(self, x: float, y: float, height: float, frac: float = 0.7, max_scale: float = 1.0) -> None:
        """Centre scene point (x, y), scaled so `height` scene pixels fill `frac` of the view height (at most
        max_scale). Re-applied on every resize until the user zooms, pans or fits, so it also works on a
        widget that isn't laid out yet (e.g. in a hidden tab)."""
        self._focus = (x, y, height, frac, max_scale)
        self._fit = False
        self._apply_focus()
        self.update()

    def _apply_focus(self) -> None:
        x, y, height, frac, max_scale = self._focus
        s = min(max_scale, frac * max(self.height(), 1) / max(height, 1e-6))
        self._scale = min(max(s, self.ZOOM_MIN), self.ZOOM_MAX)
        self._ox = self.width() / 2 - x * self._scale
        self._oy = self.height() / 2 - y * self._scale

    def zoom_at(self, factor: float, pos: QPointF) -> None:
        new = min(max(self._scale * factor, self.ZOOM_MIN), self.ZOOM_MAX)
        sx, sy = self.to_scene(pos)
        self._scale = new
        self._ox = pos.x() - sx * new
        self._oy = pos.y() - sy * new
        self._fit = False
        self._focus = None
        self.update()

    def to_scene(self, pos: QPointF) -> tuple[float, float]:
        return (pos.x() - self._ox) / self._scale, (pos.y() - self._oy) / self._scale

    def to_widget(self, x: float, y: float) -> QPointF:
        return QPointF(self._ox + x * self._scale, self._oy + y * self._scale)

    def scene_rect_to_widget(self, r: QRectF) -> QRectF:
        return QRectF(self.to_widget(r.x(), r.y()), self.to_widget(r.right(), r.bottom()))

    @property
    def zoom(self) -> float:
        return self._scale

    # ---------------------------------------------------------------- painting
    def paintEvent(self, _ev) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(COLORS["viewer_bg"]))
        if not self.has_scene():
            if self.placeholder:
                p.setPen(QColor(COLORS["viewer_text"]))
                p.setFont(ui_font(9))
                p.drawText(self.rect().adjusted(20, 0, -20, 0),
                           Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, self.placeholder)
        else:
            smooth = self._scale < 2.0
            p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, smooth)
            for img, rect, opacity in self._layers:
                if img is None or img.isNull():
                    continue
                p.setOpacity(opacity)
                p.drawImage(self.scene_rect_to_widget(rect), img)
            p.setOpacity(1.0)
            p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            self.paint_overlay(p)
            p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        p.setPen(QColor(COLORS["border_strong"]))           # classic picture box frame
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(self.rect().adjusted(0, 0, -1, -1))
        p.end()

    def paint_overlay(self, p: QPainter) -> None:
        """Subclasses draw annotations here (widget coordinates; use to_widget)."""

    def draw_badge(self, p: QPainter, text: str, corner: str = "br") -> None:
        p.setFont(ui_font(9))
        w, h = p.fontMetrics().horizontalAdvance(text) + 12, p.fontMetrics().height() + 6
        x = self.width() - w - 10 if corner.endswith("r") else 10
        y = self.height() - h - 10 if corner.startswith("b") else 10
        tooltip_box(p, QRectF(x, y, w, h))
        p.drawText(QRectF(x, y, w, h), Qt.AlignmentFlag.AlignCenter, text)


def tooltip_box(p: QPainter, r: QRectF) -> None:
    """A Windows 7 tooltip: white to light blue-grey, thin grey border. Leaves the pen set for the text."""
    g = QLinearGradient(0, r.top(), 0, r.bottom())
    g.setColorAt(0, QColor("#ffffff"))
    g.setColorAt(1, QColor("#e4e5f0"))
    p.save()
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    p.setPen(QPen(QColor(COLORS["tip_border"]), 1))
    p.setBrush(g)
    p.drawRoundedRect(r.adjusted(0.5, 0.5, -0.5, -0.5), 2, 2)
    p.restore()
    p.setPen(QColor(COLORS["tip_text"]))

    # ---------------------------------------------------------------- interaction
    def resizeEvent(self, ev) -> None:
        if self._fit:
            self._apply_fit()
        elif self._focus is not None:
            self._apply_focus()
        super().resizeEvent(ev)

    def wheelEvent(self, ev: QWheelEvent) -> None:
        if not self.has_scene():
            return
        steps = ev.angleDelta().y() / 120.0
        self.zoom_at(1.25 ** steps, ev.position())

    def mousePressEvent(self, ev: QMouseEvent) -> None:
        if ev.button() == Qt.MouseButton.LeftButton and self.has_scene():
            self._drag = ev.position()
            self._drag_origin = (self._ox, self._oy)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, ev: QMouseEvent) -> None:
        if self._drag is not None:
            d = ev.position() - self._drag
            self._ox, self._oy = self._drag_origin[0] + d.x(), self._drag_origin[1] + d.y()
            self._fit = False
            self._focus = None
            self.update()
        if self.has_scene():
            x, y = self.to_scene(ev.position())
            inside = 0 <= x < self._scene.width() and 0 <= y < self._scene.height()
            self.hovered.emit(x if inside else -1.0, y if inside else -1.0)

    def mouseReleaseEvent(self, ev: QMouseEvent) -> None:
        self._drag = None
        self.unsetCursor()

    def mouseDoubleClickEvent(self, ev: QMouseEvent) -> None:
        if self._fit and self.has_scene():
            self.zoom_at(1.0 / self._scale, ev.position())          # 100%
        else:
            self.fit()

    def leaveEvent(self, ev) -> None:
        self.hovered.emit(-1.0, -1.0)
        super().leaveEvent(ev)


def dashed_pen(color: QColor, width: float = 1.5) -> QPen:
    pen = QPen(color, width)
    pen.setStyle(Qt.PenStyle.DashLine)
    pen.setCosmetic(True)
    return pen
