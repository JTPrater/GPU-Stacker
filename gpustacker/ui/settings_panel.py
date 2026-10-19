# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jason Prater (JP Astrophotography)
"""Stacking options (left sidebar)."""
from __future__ import annotations

import os

from PySide6.QtCore import QSettings, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QLabel, QLineEdit, QPushButton,
                               QSpinBox, QVBoxLayout, QWidget)

from .theme import set_role
from .widgets import Card, Collapsible, Combo, FormGrid, hbox

REJECTIONS = [("Winsorized sigma clipping", "winsorized"), ("Sigma clipping", "sigma"), ("No rejection", "none")]
WEIGHTINGS = [("Noise (best for mixed exposures)", "noise"), ("Weighted FWHM", "wfwhm"),
              ("Number of stars", "nbstars"), ("None", "none")]
NORMALIZATIONS = [("Additive + scaling", "addscale"), ("None", "none")]

DEFAULTS = dict(rejection="winsorized", sigma_low=3.0, sigma_high=3.0, weighting="noise", normalization="addscale",
                rgb_equal=True, feather=20.0, output_norm=True, crop_enabled=False, crop_min_frames=50,
                gpu_mem_gib=0.0, workers=6, strip=64, tile=1024, save_coverage=False,
                drizzle_scale=1.0, pixfrac=1.0, transform="affine")
SCALES = [("1× (native resolution)", 1.0), ("1.5×", 1.5), ("2×", 2.0)]
TRANSFORMS = [("Affine (recommended)", "affine"), ("Similarity (rotation + shift)", "similarity")]


def _combo(items) -> QComboBox:
    c = Combo()
    for text, key in items:
        c.addItem(text, key)
    return c


def _set_combo(c: QComboBox, key) -> None:
    i = c.findData(key)
    if i >= 0:
        c.setCurrentIndex(i)


class SettingsPanel(QWidget):
    changed = Signal()
    output_edited = Signal()
    reference_reset = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        # ---- registration (raw lights only)
        self.reg_card = Card("Registration & drizzle")
        rg = FormGrid()
        self.scale = _combo(SCALES)
        rg.add("Drizzle", self.scale, "Output resolution. 1x keeps the camera's pixels; 2x needs well-dithered "
               "data and makes a 4x larger file.")
        self.pixfrac = QDoubleSpinBox()
        self.pixfrac.setRange(0.1, 1.0)
        self.pixfrac.setSingleStep(0.05)
        self.pixfrac.setDecimals(2)
        rg.add("Pixfrac", self.pixfrac, "Drizzle drop size. 1.0 is the smoothest; smaller values sharpen but need "
               "more frames (Siril -pixfrac).")
        self.transform = _combo(TRANSFORMS)
        rg.add("Alignment", self.transform, "How frames are mapped onto the reference. Affine also follows the slight "
               "scale and shear changes between nights.")
        self.ref_label = QLabel("Automatic (best frame)")
        self.ref_reset = QPushButton("Auto")
        set_role(self.ref_reset, "flat")
        self.ref_reset.setToolTip("Let the stacker pick the reference frame again")
        self.ref_reset.clicked.connect(self.reference_reset)
        rg.add("Reference", hbox(self.ref_label, None, self.ref_reset),
               "Frame everything is registered onto. Pick another in the Frames tab with 'Set as reference'.")
        self.reg_card.body.addLayout(rg)
        lay.addWidget(self.reg_card)
        self.reg_card.hide()

        # ---- stacking
        card = Card("Stacking")
        g = FormGrid()
        self.rejection = _combo(REJECTIONS)
        g.add("Rejection", self.rejection, "How outlier pixels (satellites, planes, hot pixels, cosmic rays) are "
              "thrown out before averaging. The same math as Siril's 'rej'.")
        self.sig_lo, self.sig_hi = QDoubleSpinBox(), QDoubleSpinBox()
        for s in (self.sig_lo, self.sig_hi):
            s.setRange(0.5, 10.0)
            s.setSingleStep(0.1)
            s.setDecimals(1)
        lo_lab, hi_lab = QLabel("low"), QLabel("high")
        for lab in (lo_lab, hi_lab):
            set_role(lab, "muted")
        self.sig_label = g.add("Sigma", hbox(lo_lab, self.sig_lo, hi_lab, self.sig_hi),
                               "Pixels more than this many sigmas below (low) or above (high) the median are rejected.")
        self.weighting = _combo(WEIGHTINGS)
        g.add("Weighting", self.weighting, "How much each frame counts. Noise: frames with less background noise "
              "count more, which is what you want when mixing 30 s and 60 s subs (Siril -weight=noise).")
        self.normalization = _combo(NORMALIZATIONS)
        g.add("Normalize", self.normalization, "Matches every frame's background level and spread to the reference "
              "frame before stacking (Siril -norm=addscale). Needed for multi-night data.")
        self.rgb_equal = QCheckBox("Equalize RGB channels")
        self.rgb_equal.setToolTip("Normalize all channels to the reference frame's green channel (Siril -rgb_equal).")
        g.add_full(self.rgb_equal)
        self.feather = QSpinBox()
        self.feather.setRange(0, 500)
        self.feather.setSpecialValueText("Off")
        self.feather_hint = QLabel()
        set_role(self.feather_hint, "muted")
        g.add("Feather", hbox(self.feather, self.feather_hint, None),
              "Blends frame edges in a mosaic so seams disappear. Same number as Siril's -feather: Siril measures it "
              "on 10%-size masks, so 20 blends over roughly 200 px.")
        self.output_norm = QCheckBox("Rescale output to 0–1")
        self.output_norm.setToolTip("Siril -output_norm. When off, values outside 0–1 are clipped instead.")
        g.add_full(self.output_norm)
        card.body.addLayout(g)
        lay.addWidget(card)

        # ---- output
        out = Card("Output")
        self.output = QLineEdit()
        self.output.setPlaceholderText("Output FITS file")
        browse = QPushButton("…")
        set_role(browse, "small")
        browse.setFixedWidth(30)
        browse.setToolTip("Choose where to save the stacked image")
        browse.clicked.connect(self._browse)
        out.body.addLayout(hbox(self.output, browse))
        self.crop = QCheckBox("Crop to area covered by at least")
        self.crop_frames = QSpinBox()
        self.crop_frames.setRange(1, 100000)
        self.crop_frames.setSuffix(" frames")
        self.crop.setToolTip("Write only the largest rectangle where at least this many frames overlap. Trims the "
                             "ragged mosaic edges, and frames outside it aren't read at all, so it's faster.")
        self.crop_frames.setToolTip(self.crop.toolTip())
        out.body.addLayout(hbox(self.crop, self.crop_frames, None))
        self.crop_info = QLabel()
        set_role(self.crop_info, "muted")
        out.body.addWidget(self.crop_info)
        lay.addWidget(out)

        # ---- advanced
        adv_w = QWidget()
        ag = FormGrid()
        adv_w.setLayout(ag)
        ag.setContentsMargins(0, 0, 0, 0)
        self.gpu_mem = QDoubleSpinBox()
        self.gpu_mem.setRange(0, 64)
        self.gpu_mem.setDecimals(1)
        self.gpu_mem.setSuffix(" GiB")
        self.gpu_mem.setSpecialValueText("Auto (75% of VRAM)")
        ag.add("GPU memory", self.gpu_mem, "Cap on the GPU memory the stacker uses. Going past the card's memory "
               "makes Windows page VRAM and runs 10-40x slower.")
        self.workers = QSpinBox()
        self.workers.setRange(1, 32)
        ag.add("Reader threads", self.workers, "Parallel file readers (disk throughput).")
        self.strip = QSpinBox()
        self.strip.setRange(8, 512)
        self.strip.setSingleStep(8)
        self.strip.setSuffix(" rows")
        ag.add("Strip height", self.strip, "Canvas rows per GPU pass (reduced automatically when memory is short).")
        self.tile = QSpinBox()
        self.tile.setRange(128, 8192)
        self.tile.setSingleStep(128)
        self.tile.setSuffix(" px")
        ag.add("Tile width", self.tile, "Columns per GPU tile.")
        self.save_coverage = QCheckBox("Also save a coverage map (frames per pixel)")
        ag.add_full(self.save_coverage)
        self.advanced = Collapsible("Advanced options", adv_w)
        adv_card = Card()
        adv_card.body.addWidget(self.advanced)
        lay.addWidget(adv_card)

        for sb, width in ((self.pixfrac, 80), (self.sig_lo, 64), (self.sig_hi, 64), (self.feather, 72),
                          (self.workers, 80), (self.strip, 100), (self.tile, 100)):
            sb.setMaximumWidth(width)
        self.set_values(DEFAULTS)
        for w in (self.rejection, self.weighting, self.normalization, self.scale, self.transform):
            w.currentIndexChanged.connect(self._changed)
        for w in (self.sig_lo, self.sig_hi, self.feather, self.crop_frames, self.gpu_mem, self.workers, self.strip,
                  self.tile, self.pixfrac):
            w.valueChanged.connect(self._changed)
        for w in (self.rgb_equal, self.output_norm, self.crop, self.save_coverage):
            w.toggled.connect(self._changed)
        self.output.textEdited.connect(lambda _t: self.output_edited.emit())
        self.output.editingFinished.connect(self._changed)
        self._sync()

    # ---------------------------------------------------------------- values
    def values(self) -> dict:
        return dict(rejection=self.rejection.currentData(), sigma_low=self.sig_lo.value(),
                    sigma_high=self.sig_hi.value(), weighting=self.weighting.currentData(),
                    normalization=self.normalization.currentData(), rgb_equal=self.rgb_equal.isChecked(),
                    feather=float(self.feather.value()), output_norm=self.output_norm.isChecked(),
                    crop_enabled=self.crop.isChecked(), crop_min_frames=self.crop_frames.value(),
                    gpu_mem_gib=self.gpu_mem.value(), workers=self.workers.value(), strip=self.strip.value(),
                    tile=self.tile.value(), save_coverage=self.save_coverage.isChecked(),
                    drizzle_scale=self.scale.currentData(), pixfrac=self.pixfrac.value(),
                    transform=self.transform.currentData())

    def set_values(self, v: dict) -> None:
        v = {**DEFAULTS, **v}
        _set_combo(self.rejection, v["rejection"])
        self.sig_lo.setValue(float(v["sigma_low"]))
        self.sig_hi.setValue(float(v["sigma_high"]))
        _set_combo(self.weighting, v["weighting"])
        _set_combo(self.normalization, v["normalization"])
        self.rgb_equal.setChecked(bool(v["rgb_equal"]))
        self.feather.setValue(int(round(float(v["feather"]))))
        self.output_norm.setChecked(bool(v["output_norm"]))
        self.crop.setChecked(bool(v["crop_enabled"]))
        self.crop_frames.setValue(int(v["crop_min_frames"]))
        self.gpu_mem.setValue(float(v["gpu_mem_gib"]))
        self.workers.setValue(int(v["workers"]))
        self.strip.setValue(int(v["strip"]))
        self.tile.setValue(int(v["tile"]))
        self.save_coverage.setChecked(bool(v["save_coverage"]))
        _set_combo(self.scale, float(v["drizzle_scale"]))
        self.pixfrac.setValue(float(v["pixfrac"]))
        _set_combo(self.transform, v["transform"])
        self._sync()

    def set_mode(self, kind: str) -> None:
        """'lights' shows the registration card; 'seq' (already registered by Siril) hides it."""
        self.reg_card.setVisible(kind == "lights")

    def set_reference(self, text: str, chosen: bool) -> None:
        self.ref_label.setText(text)
        self.ref_reset.setVisible(chosen)

    def save(self, s: QSettings) -> None:
        for k, val in self.values().items():
            s.setValue(f"options/{k}", val)

    def restore(self, s: QSettings) -> None:
        v = {}
        for k, default in DEFAULTS.items():
            raw = s.value(f"options/{k}", default)
            if isinstance(default, bool):
                v[k] = raw if isinstance(raw, bool) else str(raw).lower() in ("true", "1")
            elif isinstance(default, int):
                v[k] = int(float(raw))
            elif isinstance(default, float):
                v[k] = float(raw)
            else:
                v[k] = str(raw)
        self.set_values(v)

    # ---------------------------------------------------------------- behaviour
    def _changed(self, *_a) -> None:
        self._sync()
        self.changed.emit()

    def _sync(self) -> None:
        rej = self.rejection.currentData() != "none"
        self.sig_lo.setEnabled(rej)
        self.sig_hi.setEnabled(rej)
        f = self.feather.value()
        self.feather_hint.setText(f"≈ {f * 10} px blend" if f else "")
        self.rgb_equal.setEnabled(self.normalization.currentData() == "addscale")
        self.crop_frames.setEnabled(self.crop.isChecked())

    def _browse(self) -> None:
        start = self.output.text() or os.path.expanduser("~")
        path, _ = QFileDialog.getSaveFileName(self, "Save stacked image as", start, "FITS image (*.fit *.fits)")
        if path:
            if not os.path.splitext(path)[1]:
                path += ".fit"
            self.output.setText(os.path.normpath(path))
            self.output_edited.emit()
            self._changed()

    def set_running(self, running: bool) -> None:
        for w in self.findChildren(QWidget):
            if isinstance(w, (QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, QLineEdit, QPushButton)):
                w.setEnabled(not running)
        if not running:
            self._sync()
