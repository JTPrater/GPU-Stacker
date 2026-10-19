# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jason Prater (JP Astrophotography)
"""Classic light theme in the style of Windows 7 (Aero): Fusion style + light palette + a style sheet that
draws the glossy buttons, button-like drop-downs, white tab pages, blue-grey list headers and the green
progress bar. Arrows and check marks are small SVG files written to a temp folder at startup."""
from __future__ import annotations

import os
import sys
import tempfile

from PySide6.QtCore import QDir, Qt
from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import QApplication, QLabel, QWidget

COLORS: dict[str, str] = {
    "window": "#f0f0f0",          # dialog grey
    "panel": "#ffffff",
    "panel_alt": "#f7f9fc",
    "input": "#ffffff",
    "border": "#d5dfe5",          # group box line
    "border_strong": "#828790",   # list / picture box frame
    "hover": "#e5f3fb",
    "text": "#000000",
    "text_muted": "#5a5a5a",
    "text_disabled": "#8d8d8d",
    "heading": "#003399",         # Windows 7 "main instruction" blue
    "link": "#0066cc",
    "accent": "#3399ff",          # system highlight
    "accent_hover": "#3c7fb1",
    "accent_pressed": "#2c628b",
    "selection": "#cce4f7",
    "warning": "#b35a00",         # text on light backgrounds
    "band": "#ffa53d",            # overlays on the dark image area
    "error": "#c00000",
    "success": "#007a00",
    "viewer_bg": "#262626",
    "viewer_text": "#c8c8c8",
    "tip_text": "#575757",
    "tip_border": "#767676",
}

# Aero gradients (top half / bottom half split at 0.5, like the real theme)
GRAD_NORMAL = ("qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #f2f2f2, stop:0.49 #ebebeb, stop:0.5 #dddddd, "
               "stop:1 #cfcfcf)")
GRAD_HOVER = ("qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #eaf6fd, stop:0.49 #d9f0fc, stop:0.5 #bee6fd, "
              "stop:1 #a7d9f5)")
GRAD_PRESSED = ("qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #e5f4fc, stop:0.49 #c4e5f6, stop:0.5 #98d1ef, "
                "stop:1 #68b3db)")
GRAD_DEFAULT = ("qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #f1f8fd, stop:0.49 #e3f0fa, stop:0.5 #d3e7f6, "
                "stop:1 #c4dff3)")


def color(name: str, alpha: int | None = None) -> QColor:
    c = QColor(COLORS[name])
    if alpha is not None:
        c.setAlpha(alpha)
    return c


def ui_font(point_size: float = 9.0, bold: bool = False) -> QFont:
    f = QFont("Segoe UI" if sys.platform == "win32" else QFont().family())
    f.setPointSizeF(point_size)
    f.setBold(bold)
    return f


def mono_font(point_size: float = 9.0) -> QFont:
    f = QFont("Consolas" if sys.platform == "win32" else "monospace")
    f.setStyleHint(QFont.StyleHint.Monospace)
    f.setPointSizeF(point_size)
    return f


def light_palette() -> QPalette:
    p = QPalette()
    role = QPalette.ColorRole
    for r, c in ((role.Window, "#f0f0f0"), (role.WindowText, "#000000"), (role.Base, "#ffffff"),
                 (role.AlternateBase, COLORS["panel_alt"]), (role.ToolTipBase, "#ffffff"),
                 (role.ToolTipText, COLORS["tip_text"]), (role.PlaceholderText, COLORS["text_disabled"]),
                 (role.Text, "#000000"), (role.Button, "#f0f0f0"), (role.ButtonText, "#000000"),
                 (role.BrightText, "#ffffff"), (role.Highlight, COLORS["accent"]), (role.HighlightedText, "#ffffff"),
                 (role.Link, COLORS["link"]), (role.LinkVisited, "#800080"), (role.Light, "#ffffff"),
                 (role.Midlight, "#e3e3e3"), (role.Mid, "#a0a0a0"), (role.Dark, "#696969"), (role.Shadow, "#000000")):
        p.setColor(r, QColor(c))
    disabled = QPalette.ColorGroup.Disabled
    for r in (role.WindowText, role.Text, role.ButtonText):
        p.setColor(disabled, r, QColor(COLORS["text_disabled"]))
    p.setColor(disabled, role.Highlight, QColor("#c0c0c0"))
    return p


# ----------------------------------------------------------------------------- SVG glyphs

def _tri(direction: str, fill: str) -> str:
    pts = {"down": "0,0 7,0 3.5,4", "up": "0,4 7,4 3.5,0", "left": "4,0 4,7 0,3.5", "right": "0,0 0,7 4,3.5"}[direction]
    w, h = (7, 4) if direction in ("down", "up") else (4, 7)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
            f'<polygon points="{pts}" fill="{fill}"/></svg>')


def _chevron(direction: str, stroke: str) -> str:
    d = "M0.5,4 L4,0.8 L7.5,4" if direction == "up" else "M0.5,0.8 L4,4 L7.5,0.8"
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="8" height="5" viewBox="0 0 8 5">'
            f'<path d="{d}" fill="none" stroke="{stroke}" stroke-width="1.2"/></svg>')


def _check(stroke: str) -> str:
    return ('<svg xmlns="http://www.w3.org/2000/svg" width="11" height="11" viewBox="0 0 11 11">'
            f'<path d="M2,5.6 L4.4,8.2 L9,2.6" fill="none" stroke="{stroke}" stroke-width="1.9" '
            'stroke-linecap="square"/></svg>')


GLYPHS = {
    **{f"{d}.svg": _tri(d, "#3c3c3c") for d in ("down", "up", "left", "right")},
    **{f"{d}-disabled.svg": _tri(d, "#a3a3a3") for d in ("down", "up", "left", "right")},
    "sort-up.svg": _chevron("up", "#97a5b8"),
    "sort-down.svg": _chevron("down", "#97a5b8"),
    "check.svg": _check("#2b2b2b"),
    "check-disabled.svg": _check("#a3a3a3"),
}


def _glyph_dir() -> str:
    d = os.path.join(tempfile.gettempdir(), "gpu-stacker-theme")
    os.makedirs(d, exist_ok=True)
    for name, svg in GLYPHS.items():
        path = os.path.join(d, name)
        try:
            with open(path, encoding="utf-8") as f:
                if f.read() == svg:
                    continue
        except OSError:
            pass
        with open(path, "w", encoding="utf-8") as f:
            f.write(svg)
    return d


STYLE_SHEET = """
QMainWindow, QDialog { background: #f0f0f0; }
QToolTip { color: %(tip_text)s; border: 1px solid %(tip_border)s; border-radius: 2px; padding: 2px 3px;
           background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ffffff, stop:1 #e4e5f0); }

/* ---- menus and status bar */
QMenuBar { background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #fdfeff, stop:1 #e9eef5);
           border-bottom: 1px solid #d5dbe4; padding: 1px 2px; }
QMenuBar::item { padding: 3px 8px; background: transparent; border: 1px solid transparent; border-radius: 3px; }
QMenuBar::item:selected { border-color: #b6d1f0;
                          background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #f1f7fe, stop:1 #dae9fa); }
QMenuBar::item:pressed { border-color: #9cbde6;
                         background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #e3eefa, stop:1 #cfe1f7); }
QMenu { background: #f0f0f0; border: 1px solid #979797; padding: 2px; }
QMenu::item { padding: 4px 24px 4px 26px; border: 1px solid transparent; border-radius: 3px; background: transparent; }
QMenu::item:selected { border-color: #aecff7;
                       background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #f4f8fd, stop:1 #ddeaf9); }
QMenu::item:disabled { color: %(text_disabled)s; }
QMenu::separator { height: 1px; background: #e0e0e0; margin: 3px 2px 3px 28px; }
QStatusBar { background: #f0f0f0; border-top: 1px solid #d7d7d7; }
QStatusBar::item { border: none; }
QStatusBar QLabel { padding: 0 6px; }

/* ---- tabs: white pages, glossy tabs */
QTabWidget::pane { border: 1px solid #898c95; background: #ffffff; top: -1px; }
QTabWidget::tab-bar { left: 4px; }
QTabBar::tab { color: #000000; border: 1px solid #898c95; border-bottom: none; padding: 3px 14px 4px 14px;
               margin-top: 2px; margin-right: -1px; min-width: 46px;
               border-top-left-radius: 2px; border-top-right-radius: 2px;
               background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #fdfdfd, stop:0.49 #f1f1f1,
                                           stop:0.5 #e7e7e7, stop:1 #dddddd); }
QTabBar::tab:hover:!selected { border-color: #3c7fb1; background: %(grad_hover)s; }
QTabBar::tab:selected { background: #ffffff; margin-top: 0; padding-bottom: 6px; }
QTabBar::tab:disabled { color: %(text_disabled)s; }

/* ---- buttons */
QPushButton { color: #000000; border: 1px solid #707070; border-radius: 3px; padding: 2px 12px;
              min-height: 17px; min-width: 49px; background: %(grad_normal)s; }
QPushButton:hover { border-color: #3c7fb1; background: %(grad_hover)s; }
QPushButton:pressed, QPushButton:checked { border-color: #2c628b; background: %(grad_pressed)s; }
QPushButton:disabled { color: #838383; border-color: #adb2b5; background: #f4f4f4; }
QPushButton:focus { outline: none; }
QPushButton[primary="true"] { border-color: #3c7fb1; background: %(grad_default)s; }
QPushButton[primary="true"]:hover { background: %(grad_hover)s; }
QPushButton[primary="true"]:pressed { border-color: #2c628b; background: %(grad_pressed)s; }
QPushButton[primary="true"]:disabled { color: #838383; border-color: #adb2b5; background: #f4f4f4; }
QPushButton[small="true"] { min-width: 0; padding: 2px 4px; }
QPushButton[flat="true"] { color: %(link)s; border: none; background: transparent; padding: 0 1px; min-width: 0;
                           min-height: 0; text-align: left; }
QPushButton[flat="true"]:hover { color: #3399ff; text-decoration: underline; }
QPushButton[flat="true"]:pressed { background: transparent; }
QToolButton { color: #000000; background: transparent; border: 1px solid transparent; border-radius: 3px;
              padding: 2px 6px; }
QToolButton:hover { border-color: #3c7fb1; background: %(grad_hover)s; }
QToolButton:pressed, QToolButton:checked { border-color: #2c628b; background: %(grad_pressed)s; }
QToolButton:disabled { color: %(text_disabled)s; }
QToolButton#collapsibleHeader { border: none; background: transparent; padding: 1px 0; }
QToolButton#collapsibleHeader:hover { color: %(link)s; }

/* ---- text boxes and spin boxes (Windows 7: darker top edge, blue when hovered / focused) */
QLineEdit, QSpinBox, QDoubleSpinBox { color: #000000; background: #ffffff; border: 1px solid;
              border-color: #abadb3 #dbdfe6 #e3e9ef #e2e3ea; padding: 2px 3px; min-height: 17px;
              selection-background-color: #3399ff; selection-color: #ffffff; }
QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover { border-color: #5794bf #b7d5ea #c7e2f1 #b7d5ea; }
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus { border-color: #3d7bad #a4c9e3 #b7d9ed #a4c9e3; }
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled { color: #6d6d6d; background: #f4f4f4;
              border-color: #afafaf #d9d9d9 #dedede #d9d9d9; }
QSpinBox, QDoubleSpinBox { padding-right: 18px; }
QSpinBox::up-button, QDoubleSpinBox::up-button { subcontrol-origin: border; subcontrol-position: top right;
              width: 15px; margin: 1px 1px 0 0; border: 1px solid #a7a7a7; border-top-right-radius: 2px;
              background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #f8f8f8, stop:1 #dfdfdf); }
QSpinBox::down-button, QDoubleSpinBox::down-button { subcontrol-origin: border; subcontrol-position: bottom right;
              width: 15px; margin: 0 1px 1px 0; border: 1px solid #a7a7a7; border-top: none;
              border-bottom-right-radius: 2px;
              background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #f8f8f8, stop:1 #dfdfdf); }
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover, QSpinBox::down-button:hover,
QDoubleSpinBox::down-button:hover { border-color: #3c7fb1; background: %(grad_hover)s; }
QSpinBox::up-button:pressed, QDoubleSpinBox::up-button:pressed, QSpinBox::down-button:pressed,
QDoubleSpinBox::down-button:pressed { background: %(grad_pressed)s; }
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow { image: url(theme:up.svg); width: 7px; height: 4px; }
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow { image: url(theme:down.svg); width: 7px; height: 4px; }
QSpinBox::up-arrow:disabled, QDoubleSpinBox::up-arrow:disabled,
QSpinBox::up-arrow:off, QDoubleSpinBox::up-arrow:off { image: url(theme:up-disabled.svg); }
QSpinBox::down-arrow:disabled, QDoubleSpinBox::down-arrow:disabled,
QSpinBox::down-arrow:off, QDoubleSpinBox::down-arrow:off { image: url(theme:down-disabled.svg); }

/* ---- drop-down lists look like buttons, as in Windows 7 */
QComboBox { color: #000000; border: 1px solid #707070; border-radius: 3px; padding: 2px 22px 2px 6px;
            min-height: 17px; background: %(grad_normal)s; }
QComboBox:hover { border-color: #3c7fb1; background: %(grad_hover)s; }
QComboBox:on { border-color: #2c628b; background: %(grad_pressed)s; }
QComboBox:disabled { color: #838383; border-color: #adb2b5; background: #f4f4f4; }
QComboBox::drop-down { subcontrol-origin: padding; subcontrol-position: center right; width: 18px; border: none; }
QComboBox::down-arrow { image: url(theme:down.svg); width: 7px; height: 4px; }
QComboBox::down-arrow:disabled { image: url(theme:down-disabled.svg); }
QComboBox QAbstractItemView { color: #000000; background: #ffffff; border: 1px solid #646464; outline: none;
            selection-background-color: #3399ff; selection-color: #ffffff; }
QComboBox QAbstractItemView::item { min-height: 19px; padding: 0 4px; }

/* ---- check boxes */
QCheckBox { spacing: 5px; }
QCheckBox:disabled { color: %(text_disabled)s; }
QCheckBox::indicator, QTableView::indicator { width: 11px; height: 11px; border: 1px solid #8e8f8f;
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #dadbdc, stop:1 #fbfbfb); }
QCheckBox::indicator:hover, QTableView::indicator:hover { border-color: #3c7fb1;
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #b9def8, stop:1 #eef8fe); }
QCheckBox::indicator:pressed { border-color: #2c628b;
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #9bd0ef, stop:1 #def0fb); }
QCheckBox::indicator:checked, QTableView::indicator:checked { image: url(theme:check.svg); }
QCheckBox::indicator:disabled { border-color: #bcbcbc; background: #f4f4f4; }
QCheckBox::indicator:checked:disabled { image: url(theme:check-disabled.svg); }

/* ---- group boxes */
QGroupBox { border: 1px solid %(border)s; border-radius: 3px; margin-top: 8px; padding: 0; }
QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; left: 8px; padding: 0 3px;
                   color: #000000; background: #f0f0f0; }

/* ---- scroll areas and bars */
QScrollArea { background: transparent; border: none; }
QScrollArea > QWidget > QWidget { background: transparent; }
QScrollBar:vertical { background: #f0f0f0; width: 17px; margin: 17px 0 17px 0; border: none; }
QScrollBar:horizontal { background: #f0f0f0; height: 17px; margin: 0 17px 0 17px; border: none; }
QScrollBar::handle:vertical { min-height: 24px; margin: 0 1px; border: 1px solid #959595; border-radius: 2px;
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #f4f4f4, stop:0.49 #e9e9e9,
                                        stop:0.5 #dadada, stop:1 #d0d0d0); }
QScrollBar::handle:horizontal { min-width: 24px; margin: 1px 0; border: 1px solid #959595; border-radius: 2px;
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #f4f4f4, stop:0.49 #e9e9e9,
                                        stop:0.5 #dadada, stop:1 #d0d0d0); }
QScrollBar::handle:vertical:hover { border-color: #3c7fb1;
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #eaf6fd, stop:0.49 #d9f0fc,
                                        stop:0.5 #bee6fd, stop:1 #a7d9f5); }
QScrollBar::handle:horizontal:hover { border-color: #3c7fb1; background: %(grad_hover)s; }
QScrollBar::add-line:vertical { subcontrol-origin: margin; subcontrol-position: bottom; height: 17px; }
QScrollBar::sub-line:vertical { subcontrol-origin: margin; subcontrol-position: top; height: 17px; }
QScrollBar::add-line:horizontal { subcontrol-origin: margin; subcontrol-position: right; width: 17px; }
QScrollBar::sub-line:horizontal { subcontrol-origin: margin; subcontrol-position: left; width: 17px; }
QScrollBar::add-line, QScrollBar::sub-line { border: 1px solid transparent; border-radius: 2px; background: transparent; }
QScrollBar::add-line:hover, QScrollBar::sub-line:hover { border-color: #3c7fb1; background: %(grad_hover)s; }
QScrollBar::up-arrow:vertical { image: url(theme:up.svg); width: 7px; height: 4px; }
QScrollBar::down-arrow:vertical { image: url(theme:down.svg); width: 7px; height: 4px; }
QScrollBar::left-arrow:horizontal { image: url(theme:left.svg); width: 4px; height: 7px; }
QScrollBar::right-arrow:horizontal { image: url(theme:right.svg); width: 4px; height: 7px; }
QScrollBar::add-page, QScrollBar::sub-page { background: none; }

/* ---- lists: white, Explorer-style headers */
QTableView { color: #000000; background: #ffffff; alternate-background-color: %(panel_alt)s;
             border: 1px solid %(border_strong)s; gridline-color: #e4e4e4; outline: none;
             selection-background-color: %(selection)s; selection-color: #000000; }
QTableView::item { padding: 0 4px; border: none; }
QHeaderView { background: #ffffff; }
QHeaderView::section { color: #4c607a; border: none; border-right: 1px solid #e5e5e5; border-bottom: 1px solid #d5d5d5;
             padding: 3px 6px; background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ffffff, stop:1 #f1f2f4); }
QHeaderView::section:hover { border-right-color: #88cbeb;
             background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #e3f7ff, stop:1 #bdedff); }
QHeaderView::section:pressed { background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #bce4f9, stop:1 #8dd6f7); }
QHeaderView::up-arrow { subcontrol-origin: padding; subcontrol-position: top center; image: url(theme:sort-down.svg);
             width: 8px; height: 5px; }
QHeaderView::down-arrow { subcontrol-origin: padding; subcontrol-position: top center; image: url(theme:sort-up.svg);
             width: 8px; height: 5px; }
QTableCornerButton::section { background: #ffffff; border: none; border-bottom: 1px solid #d5d5d5; }
QPlainTextEdit { color: #000000; background: #ffffff; border: 1px solid %(border_strong)s;
             selection-background-color: #3399ff; selection-color: #ffffff; }

/* ---- the green Windows 7 progress bar */
QProgressBar { border: 1px solid #bcbcbc; border-radius: 2px; text-align: center; color: transparent;
               min-height: 13px; max-height: 13px;
               background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #e6e6e6, stop:0.5 #f7f7f7, stop:1 #e3e3e3); }
QProgressBar::chunk { border-radius: 1px;
               background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #8ee683, stop:0.47 #2ecb2e,
                                           stop:0.5 #11b014, stop:1 #33c833); }

/* ---- label roles */
QLabel[muted="true"] { color: %(text_muted)s; }
QLabel[section="true"] { color: #1e3287; }
QLabel[title="true"] { color: %(heading)s; font-size: 12pt; }
QLabel[warning="true"] { color: #000000; background: #ffffe1; border: 1px solid #d9d2a5; padding: 5px 7px; }
QLabel[error="true"] { color: #8b0000; background: #fdecea; border: 1px solid #e8afa9; padding: 5px 7px; }

QWidget#sidebar { background: #f0f0f0; }
QFrame#runBar { background: #f0f0f0; border-top: 1px solid #d7d7d7; }
QFrame#runBar QLabel { background: transparent; }
QFrame#welcomePanel { background: #ffffff; border: 1px solid #d9d9d9; }
QFrame#welcomePanel[dragging="true"] { background: #eef6fd; border: 1px solid #70c0e7; }
QFrame#welcomePanel QLabel { background: transparent; }
QFrame[separator="true"] { background: #d7d7d7; max-height: 1px; min-height: 1px; border: none; }
"""


def apply_theme(app: QApplication) -> None:
    hints = app.styleHints()
    if hasattr(hints, "setColorScheme"):  # Qt >= 6.8: light native title bar to match
        hints.setColorScheme(Qt.ColorScheme.Light)
    QDir.addSearchPath("theme", _glyph_dir())
    app.setStyle("Fusion")
    app.setPalette(light_palette())
    app.setFont(ui_font(9.0))
    app.setStyleSheet(STYLE_SHEET % {**COLORS, "grad_normal": GRAD_NORMAL, "grad_hover": GRAD_HOVER,
                                     "grad_pressed": GRAD_PRESSED, "grad_default": GRAD_DEFAULT})


def set_role(widget: QWidget, role: str, on: bool = True) -> QWidget:
    """Boolean style-sheet property such as 'primary', 'muted', 'section', 'warning', 'error'."""
    widget.setProperty(role, on)
    widget.style().unpolish(widget)
    widget.style().polish(widget)
    return widget


def label(text: str = "", role: str | None = None, wrap: bool = False) -> QLabel:
    lab = QLabel(text)
    lab.setWordWrap(wrap)
    if role:
        set_role(lab, role)
    return lab
