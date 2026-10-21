# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jason Prater (JP Astrophotography)
"""Frames tab: per-frame metrics chart + sortable table with include / exclude."""
from __future__ import annotations

import math
from datetime import timedelta

import numpy as np
from PySide6.QtCore import QAbstractTableModel, QItemSelectionModel, QModelIndex, QPointF, QRectF, \
    QSortFilterProxyModel, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QHBoxLayout,
                               QHeaderView, QLabel, QPushButton, QTableView, QVBoxLayout, QWidget)

from .session import Session, parse_date
from .imageview import tooltip_box
from .widgets import Combo
from .theme import COLORS, color, set_role, ui_font

COLUMNS = ["", "#", "Exposure", "Time (UTC)", "FWHM", "Roundness", "Stars", "Background", "Noise ×10⁻³", "Status",
           "File"]
C_CHECK, C_NUM, C_EXP, C_TIME, C_FWHM, C_ROUND, C_STARS, C_BKG, C_NOISE, C_STATUS, C_FILE = range(11)
SORT_ROLE = Qt.ItemDataRole.UserRole + 1

METRICS = [("FWHM (px)", "fwhm", "high"), ("Roundness", "roundness", "low"), ("Stars", "nstars", "low"),
           ("Background", "background", "high"), ("Noise", "noise", "high")]


def metric_value(session: Session, fr, key: str) -> float:
    if key == "noise":
        n = session.noise(fr)
        return float("nan") if n is None else n
    v = getattr(fr, key)
    return float(v) if v is not None else float("nan")


class FramesModel(QAbstractTableModel):
    selection_edited = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.session: Session | None = None
        self.locked = False

    def set_session(self, session: Session | None) -> None:
        self.beginResetModel()
        self.session = session
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() or self.session is None else len(self.session.frames)

    def columnCount(self, parent=QModelIndex()) -> int:
        return len(COLUMNS)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return COLUMNS[section]
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.ToolTipRole:
            return {C_FWHM: "Star FWHM in pixels, from Siril's registration", C_ROUND: "Star roundness (1 = round)",
                    C_STARS: "Stars detected at registration", C_BKG: "Background level from registration",
                    C_NOISE: "Background noise (green), measured when stacking with noise weighting",
                    C_CHECK: "Include in the stack"}.get(section)
        return None

    def frame(self, row: int):
        return self.session.frames[row]

    def status(self, fr) -> str:
        return self.session.status(fr)

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or self.session is None:
            return None
        fr = self.frame(index.row())
        col = index.column()
        meta = self.session.meta.get(fr.filenum)
        included = fr.filenum in self.session.included
        if role == Qt.ItemDataRole.CheckStateRole and col == C_CHECK:
            return Qt.CheckState.Checked if included else Qt.CheckState.Unchecked
        if role in (Qt.ItemDataRole.DisplayRole, SORT_ROLE):
            sort = role == SORT_ROLE
            if col == C_CHECK:
                return int(included) if sort else None
            if col == C_NUM:
                return fr.number
            if col == C_EXP:
                e = meta.exptime if meta else 0.0
                return e if sort else (f"{e:g} s" if e else "")
            if col == C_TIME:
                d = meta.date_obs if meta else ""
                if sort:
                    return d
                dt = parse_date(d)
                return dt.strftime("%b %d  %H:%M:%S") if dt else ""
            if col == C_FWHM:
                return fr.fwhm if sort else (f"{fr.fwhm:.2f}" if fr.fwhm == fr.fwhm else "")
            if col == C_ROUND:
                return fr.roundness if sort else (f"{fr.roundness:.3f}" if fr.roundness == fr.roundness else "")
            if col == C_STARS:
                return fr.nstars
            if col == C_BKG:
                return fr.background if sort else (f"{fr.background:.4f}" if fr.background == fr.background else "")
            if col == C_NOISE:
                n = self.session.noise(fr)
                return (n if n is not None else -1.0) if sort else (f"{n * 1e3:.3f}" if n is not None else "")
            if col == C_STATUS:
                return self.status(fr)
            if col == C_FILE:
                return self.session.frame_name(fr)
        if role == Qt.ItemDataRole.ForegroundRole:
            if self.session.bad_reason(fr):
                return color("error")
            if not included:
                return color("text_disabled")
        if role == Qt.ItemDataRole.TextAlignmentRole and col in (C_NUM, C_EXP, C_FWHM, C_ROUND, C_STARS, C_BKG, C_NOISE):
            return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        if role == Qt.ItemDataRole.ToolTipRole and col in (C_STATUS, C_FILE, C_NUM):
            return f"{self.session.frame_name(fr)}\n{self.status(fr)}".strip()
        return None

    def flags(self, index: QModelIndex):
        f = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
        if index.column() == C_CHECK and not self.locked and self.session and \
                not self.session.bad_reason(self.frame(index.row())):
            f |= Qt.ItemFlag.ItemIsUserCheckable
        return f

    def setData(self, index: QModelIndex, value, role=Qt.ItemDataRole.EditRole) -> bool:
        if role != Qt.ItemDataRole.CheckStateRole or index.column() != C_CHECK or self.locked:
            return False
        fr = self.frame(index.row())
        checked = Qt.CheckState(value) == Qt.CheckState.Checked if not isinstance(value, bool) else value
        self.set_included([fr.filenum], checked)
        return True

    def set_included(self, filenums, on: bool) -> None:
        if self.session is None or self.locked:
            return
        changed = False
        for n in filenums:
            if on and n not in self.session.included:
                self.session.included.add(n)
                changed = True
            elif not on and n in self.session.included:
                self.session.included.discard(n)
                changed = True
        if changed:
            self.refresh()
            self.selection_edited.emit()

    def refresh(self) -> None:
        if self.rowCount():
            self.dataChanged.emit(self.index(0, 0), self.index(self.rowCount() - 1, self.columnCount() - 1))


class FramesProxy(QSortFilterProxyModel):
    FILTERS = ["All frames", "Included", "Excluded", "Unusable"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.mode = 0
        self.setSortRole(SORT_ROLE)

    def set_mode(self, mode: int) -> None:
        self.mode = mode
        self.invalidateFilter()

    def filterAcceptsRow(self, row: int, parent) -> bool:
        m: FramesModel = self.sourceModel()
        if self.mode == 0 or m.session is None:
            return True
        fr = m.frame(row)
        bad = bool(m.session.bad_reason(fr))
        inc = fr.filenum in m.session.included
        return {1: inc and not bad, 2: not inc and not bad, 3: bad}[self.mode]


class MetricChart(QWidget):
    """Scatter of one metric against frame number, with night bands. Click or drag to pick frames."""
    picked = Signal(list, bool)          # file numbers, extend selection

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(170)
        self.setMouseTracking(True)
        self.session: Session | None = None
        self.key, self.worse = "fwhm", "high"
        self.highlight: set[int] = set()
        self._pts = None
        self._drag: QPointF | None = None
        self._drag_now: QPointF | None = None
        self._hover = -1

    def set_metric(self, key: str, worse: str) -> None:
        self.key, self.worse = key, worse
        self.update()

    def set_session(self, session) -> None:
        self.session = session
        self.highlight = set()
        self.update()

    def set_highlight(self, nums: set[int]) -> None:
        self.highlight = nums
        self.update()

    def _plot_rect(self) -> QRectF:
        return QRectF(52, 12, max(10, self.width() - 66), max(10, self.height() - 36))

    def _data(self):
        s = self.session
        frames = s.frames
        x = np.array([fr.number for fr in frames], float)
        y = np.array([metric_value(s, fr, self.key) for fr in frames], float)
        state = np.array([2 if s.bad_reason(fr) else (0 if fr.filenum in s.included else 1) for fr in frames])
        return frames, x, y, state

    COLS = {0: QColor(79, 129, 189, 215), 1: QColor(165, 165, 165, 190), 2: QColor(192, 80, 77, 235)}

    def _frame(self, p: QPainter) -> None:
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        p.setPen(QColor(COLORS["border_strong"]))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(self.rect().adjusted(0, 0, -1, -1))

    def paintEvent(self, _ev) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor("#ffffff"))
        r = self._plot_rect()
        p.setFont(ui_font(8))
        if self.session is None or not self.session.frames:
            self._frame(p)
            p.end()
            return
        frames, x, y, state = self._data()
        ok = np.isfinite(y)
        if not ok.any():
            p.setPen(color("text_muted"))
            msg = "No values yet" + (" - noise is measured during the first stack with noise weighting"
                                     if self.key == "noise" else "")
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, msg)
            self._frame(p)
            p.end()
            return
        lo, hi = np.percentile(y[ok], [0.5, 99.5])
        pad = (hi - lo) * 0.08 or abs(hi) * 0.05 or 1.0
        lo, hi = lo - pad, hi + pad
        x0, x1 = x.min() - 0.5, x.max() + 0.5

        def X(v):
            return r.left() + (v - x0) / (x1 - x0) * r.width()

        def Y(v):
            return r.bottom() - (min(max(v, lo), hi) - lo) / (hi - lo) * r.height()

        # night bands
        nights, start = [], None
        prev = None
        for fr in frames:
            dt = parse_date(self.session.meta.get(fr.filenum).date_obs if self.session.meta.get(fr.filenum) else "")
            night = (dt - timedelta(hours=12)).date() if dt else None
            if night != prev:
                if prev is not None:
                    nights.append((start, fr.number - 0.5, prev))
                start, prev = fr.number - 0.5, night
        if prev is not None:
            nights.append((start, x1, prev))
        for i, (a, b, night) in enumerate(nights):
            if i % 2:
                p.fillRect(QRectF(X(a), r.top(), X(b) - X(a), r.height()), QColor("#f2f2f2"))
            if night and X(b) - X(a) > 42:
                p.setPen(QColor("#7f7f7f"))
                p.drawText(QRectF(X(a) + 4, r.top() + 2, X(b) - X(a) - 8, 14), Qt.AlignmentFlag.AlignLeft,
                           night.strftime("%b %d"))
        # axes and gridlines
        for t in np.linspace(lo + pad, hi - pad, 3):
            yy = Y(t)
            p.setPen(QPen(QColor("#d9d9d9"), 1))
            p.drawLine(QPointF(r.left(), yy), QPointF(r.right(), yy))
            p.setPen(QColor("#000000"))
            p.drawText(QRectF(0, yy - 8, 46, 16), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                       _fmt(t, self.key))
        p.setPen(QPen(QColor("#868686"), 1))
        p.drawLine(r.bottomLeft(), r.bottomRight())
        p.drawLine(r.bottomLeft(), r.topLeft())
        p.setPen(QColor("#000000"))
        for t in _ticks(x.min(), x.max(), 6):
            p.drawText(QRectF(X(t) - 30, r.bottom() + 3, 60, 14), Qt.AlignmentFlag.AlignCenter, f"{int(t)}")
        # median of included
        inc = ok & (state == 0)
        if inc.any():
            med = float(np.median(y[inc]))
            p.setPen(QPen(QColor(31, 73, 125, 170), 1, Qt.PenStyle.DashLine))
            p.drawLine(QPointF(r.left(), Y(med)), QPointF(r.right(), Y(med)))
        # points
        n = len(frames)
        rad = 2.6 if n < 400 else (2.0 if n < 1500 else 1.6)
        pts = np.stack([np.array([X(v) for v in x]), np.array([Y(v) if np.isfinite(v) else np.nan for v in y])], 1)
        self._pts = pts
        p.setPen(Qt.PenStyle.NoPen)
        for st in (1, 0, 2):
            p.setBrush(self.COLS[st])
            for i in np.flatnonzero((state == st) & ok):
                p.drawEllipse(QPointF(pts[i, 0], pts[i, 1]), rad, rad)
        if self.highlight:
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(QColor("#000000"), 1.2))
            for i, fr in enumerate(frames):
                if fr.number in self.highlight and ok[i]:
                    p.drawEllipse(QPointF(pts[i, 0], pts[i, 1]), rad + 2.2, rad + 2.2)
        if 0 <= self._hover < n and ok[self._hover]:
            fr = frames[self._hover]
            txt = f"#{fr.number}  {_fmt(y[self._hover], self.key)}"
            px, py = pts[self._hover]
            p.setPen(QPen(QColor("#000000"), 1.4))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(QPointF(px, py), rad + 3, rad + 3)
            p.setFont(ui_font(9))
            w = p.fontMetrics().horizontalAdvance(txt) + 12
            h = p.fontMetrics().height() + 6
            bx = min(max(px - w / 2, r.left()), r.right() - w)
            by = py - h - 8 if py - h - 8 > r.top() else py + 10
            tooltip_box(p, QRectF(bx, by, w, h))
            p.drawText(QRectF(bx, by, w, h), Qt.AlignmentFlag.AlignCenter, txt)
        if self._drag is not None and self._drag_now is not None:     # Explorer-style marquee
            p.setPen(QPen(QColor("#3399ff"), 1))
            p.setBrush(QColor(51, 153, 255, 50))
            p.drawRect(QRectF(self._drag, self._drag_now).normalized())
        self._frame(p)
        p.end()

    def _nearest(self, pos: QPointF) -> int:
        if self._pts is None:
            return -1
        d = np.hypot(self._pts[:, 0] - pos.x(), self._pts[:, 1] - pos.y())
        d[~np.isfinite(d)] = np.inf
        i = int(np.argmin(d))
        return i if d[i] < 8 else -1

    def mousePressEvent(self, ev) -> None:
        if ev.button() == Qt.MouseButton.LeftButton:
            self._drag = ev.position()
            self._drag_now = None

    def mouseMoveEvent(self, ev) -> None:
        if self._drag is not None:
            self._drag_now = ev.position()
            self.update()
            return
        h = self._nearest(ev.position())
        if h != self._hover:
            self._hover = h
            self.update()

    def mouseReleaseEvent(self, ev) -> None:
        if self._drag is None or self.session is None:
            return
        extend = bool(ev.modifiers() & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier))
        if self._drag_now is not None and (self._drag_now - self._drag).manhattanLength() > 4:
            box = QRectF(self._drag, self._drag_now).normalized()
            frames = self.session.frames
            sel = [frames[i].filenum for i in range(len(frames))
                   if self._pts is not None and np.isfinite(self._pts[i, 1])
                   and box.contains(QPointF(self._pts[i, 0], self._pts[i, 1]))]
            self.picked.emit(sel, extend)
        else:
            i = self._nearest(ev.position())
            if i >= 0:
                self.picked.emit([self.session.frames[i].filenum], extend)
        self._drag = self._drag_now = None
        self.update()

    def leaveEvent(self, ev) -> None:
        self._hover = -1
        self.update()


def _fmt(v: float, key: str) -> str:
    if key == "noise":
        return f"{v * 1e3:.3f}e-3"
    if key == "nstars":
        return f"{v:.0f}"
    if key == "background":
        return f"{v:.4f}"
    return f"{v:.2f}"


def _ticks(a: float, b: float, n: int) -> list[float]:
    span = max(b - a, 1)
    step = 10 ** math.floor(math.log10(span / n))
    for m in (1, 2, 5, 10):
        if span / (step * m) <= n:
            step *= m
            break
    start = math.ceil(a / step) * step
    return [t for t in np.arange(start, b + 1e-9, step)]


class WorstDialog(QDialog):
    """Exclude the worst N% of the included frames by a metric."""

    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Exclude worst frames")
        self.session = session
        lay = QVBoxLayout(self)
        # Never name an attribute after a Qt method: "metric" made PySide treat the combo as an override of
        # QWidget.metric(), which Qt calls for the screen DPI, so the dialog failed as it opened (issue #1).
        self.metric_box = Combo()
        for text, key, worse in METRICS:
            self.metric_box.addItem(text, (key, worse))
        self.pct = QDoubleSpinBox()
        self.pct.setRange(0.5, 90)
        self.pct.setValue(5.0)
        self.pct.setSuffix(" %")
        self.info = QLabel()
        set_role(self.info, "muted")
        row = QHBoxLayout()
        row.addWidget(QLabel("Exclude the worst"))
        row.addWidget(self.pct)
        row.addWidget(QLabel("by"))
        row.addWidget(self.metric_box, 1)
        lay.addLayout(row)
        lay.addWidget(self.info)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self.metric_box.currentIndexChanged.connect(self._update)
        self.pct.valueChanged.connect(self._update)
        self.result_frames: list[int] = []
        self._update()

    def _update(self) -> None:
        key, worse = self.metric_box.currentData()
        frames = self.session.usable_selected()
        vals = [(metric_value(self.session, fr, key), fr.filenum) for fr in frames]
        vals = [(v, n) for v, n in vals if v == v]
        if not vals:
            self.info.setText("No values for this metric yet.")
            self.result_frames = []
            return
        vals.sort(reverse=(worse == "high"))
        k = int(round(len(vals) * self.pct.value() / 100.0))
        self.result_frames = [n for _, n in vals[:k]]
        if k:
            limit = vals[k - 1][0]
            self.info.setText(f"{k} of {len(vals)} included frames ({'≥' if worse == 'high' else '≤'} "
                              f"{_fmt(limit, key)})")
        else:
            self.info.setText("No frames.")


class FramesPanel(QWidget):
    selection_edited = Signal()
    reference_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(8)
        top = QHBoxLayout()
        self.metric_box = Combo()
        for text, key, worse in METRICS:
            self.metric_box.addItem(text, (key, worse))
        self.count = QLabel()
        set_role(self.count, "muted")
        legend = QLabel("<span style='color:#4f81bd'>●</span> included &nbsp; "
                        "<span style='color:#a5a5a5'>●</span> excluded &nbsp; "
                        "<span style='color:#c0504d'>●</span> unusable")
        top.addWidget(QLabel("Plot:"))
        top.addWidget(self.metric_box)
        top.addSpacing(10)
        top.addWidget(legend)
        top.addStretch(1)
        top.addWidget(self.count)
        lay.addLayout(top)
        self.chart = MetricChart()
        lay.addWidget(self.chart, 2)

        bar = QHBoxLayout()
        self.include_btn = QPushButton("Include")
        self.exclude_btn = QPushButton("Exclude")
        self.worst_btn = QPushButton("Exclude worst…")
        self.reset_btn = QPushButton("Reset to Siril selection")
        self.ref_btn = QPushButton("Set as reference")
        for b, tip in ((self.include_btn, "Include the selected rows (Space toggles)"),
                       (self.exclude_btn, "Exclude the selected rows"),
                       (self.worst_btn, "Exclude a percentage of the worst included frames by a metric"),
                       (self.reset_btn, "Go back to the default selection"),
                       (self.ref_btn, "Register every frame onto the selected frame (normally the best frame is "
                                      "picked automatically). Takes effect on the next Analyze / Stack.")):
            b.setToolTip(tip)
            bar.addWidget(b)
        self.ref_btn.hide()
        bar.addStretch(1)
        self.filter = Combo()
        self.filter.addItems(FramesProxy.FILTERS)
        bar.addWidget(QLabel("Show:"))
        bar.addWidget(self.filter)
        lay.addLayout(bar)

        self.model = FramesModel(self)
        self.proxy = FramesProxy(self)
        self.proxy.setSourceModel(self.model)
        self.table = QTableView()
        self.table.setModel(self.proxy)
        self.table.setSortingEnabled(True)
        self.table.sortByColumn(C_NUM, Qt.SortOrder.AscendingOrder)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(False)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(24)
        self.table.setWordWrap(False)                                 # one line per row, even for "Light_M 45_…"
        self.table.setTextElideMode(Qt.TextElideMode.ElideMiddle)     # file names differ at the end
        hh = self.table.horizontalHeader()
        hh.setStretchLastSection(True)
        hh.setHighlightSections(False)
        for col, w in ((C_CHECK, 34), (C_NUM, 64), (C_EXP, 82), (C_TIME, 140), (C_FWHM, 70), (C_ROUND, 86),
                       (C_STARS, 64), (C_BKG, 96), (C_NOISE, 104), (C_STATUS, 200)):
            hh.resizeSection(col, w)
        hh.setSectionResizeMode(C_CHECK, QHeaderView.ResizeMode.Fixed)
        lay.addWidget(self.table, 5)

        self.metric_box.currentIndexChanged.connect(lambda _i: self.chart.set_metric(*self.metric_box.currentData()))
        self.filter.currentIndexChanged.connect(self.proxy.set_mode)
        self.include_btn.clicked.connect(lambda: self._set_selected(True))
        self.exclude_btn.clicked.connect(lambda: self._set_selected(False))
        self.worst_btn.clicked.connect(self._worst)
        self.reset_btn.clicked.connect(self._reset)
        self.ref_btn.clicked.connect(self._set_reference)
        self.model.selection_edited.connect(self._edited)
        self.chart.picked.connect(self._picked)
        self.table.selectionModel().selectionChanged.connect(self._table_selection)

    # ---------------------------------------------------------------- state
    def set_session(self, session: Session | None) -> None:
        self.model.set_session(session)
        self.chart.set_session(session)
        if session is not None:
            self.reset_btn.setText(session.default_label)
            self.ref_btn.setVisible(session.kind == "lights")
        self._update_count()

    def set_locked(self, locked: bool) -> None:
        self.model.locked = locked
        for b in (self.include_btn, self.exclude_btn, self.worst_btn, self.reset_btn, self.ref_btn):
            b.setEnabled(not locked)

    def refresh(self) -> None:
        self.model.refresh()
        self.chart.update()
        self._update_count()

    def _update_count(self) -> None:
        s = self.model.session
        if s is None:
            self.count.setText("")
            return
        usable = len(s.usable_selected())
        bad = sum(1 for fr in s.frames if s.bad_reason(fr))
        txt = f"{usable} of {len(s.frames)} frames will be stacked"
        if bad:
            txt += f" · {bad} unusable"
        self.count.setText(txt)

    def _edited(self) -> None:
        self.chart.update()
        self._update_count()
        self.selection_edited.emit()

    # ---------------------------------------------------------------- actions
    def _selected_filenums(self) -> list[int]:
        rows = {self.proxy.mapToSource(i).row() for i in self.table.selectionModel().selectedRows()}
        return [self.model.frame(r).filenum for r in sorted(rows)]

    def _set_selected(self, on: bool) -> None:
        nums = self._selected_filenums()
        if nums:
            self.model.set_included(nums, on)

    def keyPressEvent(self, ev) -> None:
        if ev.key() == Qt.Key.Key_Space and not self.model.locked:
            nums = self._selected_filenums()
            s = self.model.session
            if nums and s:
                on = not all(n in s.included for n in nums)
                self.model.set_included(nums, on)
                return
        super().keyPressEvent(ev)

    def _worst(self) -> None:
        s = self.model.session
        if s is None:
            return
        dlg = WorstDialog(s, self)
        if dlg.exec() and dlg.result_frames:
            self.model.set_included(dlg.result_frames, False)

    def _reset(self) -> None:
        s = self.model.session
        if s is None:
            return
        s.included = set(s.default_included)
        self.model.refresh()
        self._edited()

    def _set_reference(self) -> None:
        s = self.model.session
        nums = self._selected_filenums()
        if s is None or s.kind != "lights" or len(nums) != 1:
            return
        fr = next(f for f in s.frames if f.filenum == nums[0])
        s.reference_file = s.frame_name(fr)
        self.model.refresh()
        self.reference_changed.emit()

    def _picked(self, nums: list, extend: bool) -> None:
        sm = self.table.selectionModel()
        flags = QItemSelectionModel.SelectionFlag.Rows
        if not extend:
            sm.clearSelection()
        rows = {fr.filenum: i for i, fr in enumerate(self.model.session.frames)}
        first = None
        for n in nums:
            idx = self.proxy.mapFromSource(self.model.index(rows[n], 0))
            if idx.isValid():
                sm.select(idx, QItemSelectionModel.SelectionFlag.Select | flags)
                first = first or idx
        if first is not None:
            self.table.scrollTo(first, QAbstractItemView.ScrollHint.PositionAtCenter)

    def _table_selection(self, *_a) -> None:
        s = self.model.session
        if s is None:
            return
        nums = {self.model.frame(self.proxy.mapToSource(i).row()).number
                for i in self.table.selectionModel().selectedRows()}
        self.chart.set_highlight(nums)
