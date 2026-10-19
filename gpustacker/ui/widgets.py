# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jason Prater (JP Astrophotography)
"""Small reusable widgets: group boxes, form rows, elided labels, a collapsible section, Windows 7 style
command links, the app icon."""
from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import (QColor, QFontMetrics, QIcon, QLinearGradient, QPainter, QPainterPath, QPen, QPixmap,
                           QPolygonF)
from PySide6.QtWidgets import (QAbstractButton, QComboBox, QFrame, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
                               QSizePolicy, QStyledItemDelegate, QToolButton, QVBoxLayout, QWidget)

from .theme import ui_font


class Card(QGroupBox):
    """Classic group box: a thin frame with its title on the line. Content goes in .body."""

    def __init__(self, title: str = "", parent=None):
        super().__init__(title.replace("&", "&&"), parent)
        self.outer = QVBoxLayout(self)
        self.outer.setContentsMargins(9, 14 if title else 9, 9, 9)
        self.outer.setSpacing(6)
        self.body = QVBoxLayout()
        self.body.setSpacing(6)
        self.outer.addLayout(self.body)


class Combo(QComboBox):
    """Drop-down whose list uses the standard item delegate. Fusion's default menu-style list takes its
    highlight colour from the combo's own background, a gradient in this theme, which renders solid black."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setItemDelegate(QStyledItemDelegate(self))


class FormGrid(QGridLayout):
    """Two-column label / control grid, labels ending in a colon like a Windows dialog."""

    def __init__(self):
        super().__init__()
        self.setHorizontalSpacing(8)
        self.setVerticalSpacing(6)
        self.setColumnStretch(1, 1)
        self._row = 0

    def add(self, text: str, widget, tip: str = "") -> QLabel:
        lab = QLabel(text + ":")
        if tip:
            lab.setToolTip(tip)
            if isinstance(widget, QWidget):
                widget.setToolTip(tip)
        self.addWidget(lab, self._row, 0, Qt.AlignmentFlag.AlignVCenter)
        if isinstance(widget, QWidget):
            self.addWidget(widget, self._row, 1)
        else:
            self.addLayout(widget, self._row, 1)
        self._row += 1
        return lab

    def add_full(self, widget) -> None:
        if isinstance(widget, QWidget):
            self.addWidget(widget, self._row, 0, 1, 2)
        else:
            self.addLayout(widget, self._row, 0, 1, 2)
        self._row += 1


class ElidedLabel(QLabel):
    """Label that elides its text in the middle to fit (full text in the tooltip)."""

    def __init__(self, text: str = "", mode=Qt.TextElideMode.ElideMiddle, parent=None):
        super().__init__(parent)
        self._full = ""
        self._mode = mode
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setText(text)

    def setText(self, text: str) -> None:
        self._full = text
        self.setToolTip(text)
        self._elide()

    def full_text(self) -> str:
        return self._full

    def resizeEvent(self, ev) -> None:
        self._elide()
        super().resizeEvent(ev)

    def _elide(self) -> None:
        fm = QFontMetrics(self.font())
        super().setText(fm.elidedText(self._full, self._mode, max(10, self.width())))


class Collapsible(QWidget):
    """Header button that shows / hides a content widget."""
    toggled = Signal(bool)

    def __init__(self, title: str, content: QWidget, expanded: bool = False, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        self.button = QToolButton()
        self.button.setObjectName("collapsibleHeader")
        self.button.setText(title)
        self.button.setCheckable(True)
        self.button.setChecked(expanded)
        self.button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.button.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self.button.toggled.connect(self._on)
        self.content = content
        content.setVisible(expanded)
        lay.addWidget(self.button)
        lay.addWidget(content)

    def _on(self, on: bool) -> None:
        self.button.setArrowType(Qt.ArrowType.DownArrow if on else Qt.ArrowType.RightArrow)
        self.content.setVisible(on)
        self.toggled.emit(on)


class CommandLink(QAbstractButton):
    """Windows 7 command link: a green arrow, a 12 pt dark blue title and a note underneath, with a light
    blue box while hovered, like the choices in a Windows 7 task dialog."""

    TEXT = QColor("#151c55")

    def __init__(self, title: str, note: str = "", parent=None):
        super().__init__(parent)
        self.setText(title)
        self.note = note
        self.setAttribute(Qt.WidgetAttribute.WA_Hover)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self._title_font = ui_font(12)
        self._note_font = ui_font(9)
        self._arrow = command_arrow(20)

    def _note_height(self, width: int) -> int:
        if not self.note:
            return 0
        fm = QFontMetrics(self._note_font)
        r = fm.boundingRect(0, 0, max(50, width - 52), 1000, int(Qt.TextFlag.TextWordWrap), self.note)
        return r.height() + 2

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return 16 + QFontMetrics(self._title_font).height() + self._note_height(width)

    def sizeHint(self) -> QSize:
        return QSize(420, self.heightForWidth(max(self.width(), 360)))

    def resizeEvent(self, ev) -> None:
        self.setFixedHeight(self.heightForWidth(self.width()))
        super().resizeEvent(ev)

    def paintEvent(self, _ev) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        hot, down = self.underMouse() and self.isEnabled(), self.isDown()
        if hot or down or self.hasFocus():
            g = QLinearGradient(0, 0, 0, self.height())
            if down:
                border, top, bottom = "#a8c9e8", "#e3f0fb", "#cfe5f8"
            elif hot:
                border, top, bottom = "#c6e3f7", "#f7fbfe", "#e8f3fc"
            else:                                          # keyboard focus only
                border, top, bottom = "#d6e7f5", "#ffffff", "#ffffff"
            g.setColorAt(0, QColor(top))
            g.setColorAt(1, QColor(bottom))
            p.setPen(QPen(QColor(border), 1))
            p.setBrush(g)
            p.drawRoundedRect(r, 3, 3)
        title_h = QFontMetrics(self._title_font).height()
        p.drawPixmap(QPointF(10, 8 + (title_h - 20) / 2), self._arrow)
        p.setPen(self.TEXT if self.isEnabled() else QColor("#8d8d8d"))
        p.setFont(self._title_font)
        p.drawText(QRectF(38, 7, self.width() - 46, title_h),
                   int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter), self.text())
        if self.note:
            p.setFont(self._note_font)
            p.drawText(QRectF(38, 8 + title_h, self.width() - 52, self.height() - title_h - 8),
                       int(Qt.TextFlag.TextWordWrap) | int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop),
                       self.note)
        p.end()


def command_arrow(size: int) -> QPixmap:
    """The green right arrow of Windows 7 command links (drawn at 2x for sharp edges)."""
    dpr = 2.0
    pm = QPixmap(int(size * dpr), int(size * dpr))
    pm.setDevicePixelRatio(dpr)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    s = size / 20.0
    poly = QPolygonF([QPointF(x * s, y * s) for x, y in
                      ((2, 7.5), (10, 7.5), (10, 3), (18, 10), (10, 17), (10, 12.5), (2, 12.5))])
    g = QLinearGradient(0, 3 * s, 0, 17 * s)
    g.setColorAt(0, QColor("#8fe07f"))
    g.setColorAt(0.5, QColor("#3cb83a"))
    g.setColorAt(1, QColor("#1f8f22"))
    p.setPen(QPen(QColor("#1d7a20"), 1.0 * s))
    p.setBrush(g)
    p.drawPolygon(poly)
    p.end()
    return pm


def hline() -> QFrame:
    f = QFrame()
    f.setProperty("separator", True)
    f.setFrameShape(QFrame.Shape.NoFrame)
    return f


def hbox(*widgets, spacing: int = 6, stretch_at: int | None = None) -> QHBoxLayout:
    lay = QHBoxLayout()
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(spacing)
    for i, w in enumerate(widgets):
        if stretch_at == i:
            lay.addStretch(1)
        if w is None:
            lay.addStretch(1)
        elif isinstance(w, QWidget):
            lay.addWidget(w)
        else:
            lay.addLayout(w)
    return lay


def app_icon() -> QIcon:
    """Stacked frames with a star, drawn in code (no image files needed)."""
    icon = QIcon()
    for size in (16, 24, 32, 48, 64, 128, 256):
        icon.addPixmap(_icon_pixmap(size))
    return icon


def _icon_pixmap(size: int) -> QPixmap:
    pm = QPixmap(QSize(size, size))
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    s = size / 64.0
    bg = QLinearGradient(0, 0, size, size)
    bg.setColorAt(0, QColor("#1d2a4d"))
    bg.setColorAt(1, QColor("#0d1222"))
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(bg)
    p.drawRoundedRect(QRectF(1 * s, 1 * s, 62 * s, 62 * s), 14 * s, 14 * s)
    for i, (alpha, dy) in enumerate(((90, 16), (150, 8), (255, 0))):     # three stacked frames
        c = QColor("#5b8cff")
        c.setAlpha(alpha)
        p.setPen(QPen(c, max(1.0, 2.2 * s)))
        p.setBrush(QColor(91, 140, 255, 28 + 18 * i))
        p.drawRoundedRect(QRectF((12 + dy * 0.5) * s, (14 + dy) * s, 30 * s, 24 * s), 4 * s, 4 * s)
    star = QPainterPath()
    cx, cy, r1, r2 = 44 * s, 22 * s, 9 * s, 2.4 * s
    for k in range(8):
        r = r1 if k % 2 == 0 else r2
        a = math.pi / 4 * k - math.pi / 2
        pt = QPointF(cx + r * math.cos(a), cy + r * math.sin(a))
        star.moveTo(pt) if k == 0 else star.lineTo(pt)
    star.closeSubpath()
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor("#fff4d6"))
    p.drawPath(star)
    p.end()
    return pm
