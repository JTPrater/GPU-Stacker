# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jason Prater (JP Astrophotography)
"""What the UI has open: a Siril-registered sequence (Session) or a folder of raw lights
(LightsSession). Both expose the same small interface to the window, tables and views."""
from __future__ import annotations

import json
import math
import os
from collections import Counter
from datetime import datetime, timedelta

from ..engine.fits import read_header
from ..engine.lights import common_geometry, dataset_dir, find_lights_dir, list_fits, scan_lights
from ..engine.plan import load_stats
from ..engine.rawcache import Analysis, Measures, Registration, cache_dir, footprint, light_sig, quad_coverage, \
    raw_canvas
from ..engine.sequence import Frame, FrameMeta, check_stackable, coverage_map, max_framing, parse_seq, scan_headers


def parse_date(s: str) -> datetime | None:
    if not s:
        return None
    s = s.strip().replace("Z", "")
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s[:26], fmt)
        except ValueError:
            continue
    return None


class _Common:
    """Summary helpers shared by both session kinds."""
    frames: list
    meta: dict
    included: set
    default_included: set

    def usable_selected(self) -> list:
        raise NotImplementedError

    def exposure_groups(self, frames=None) -> list[tuple[float, int]]:
        frames = self.usable_selected() if frames is None else frames
        cnt = Counter(round(self.meta[fr.filenum].exptime, 2) for fr in frames if fr.filenum in self.meta)
        return sorted(cnt.items(), key=lambda kv: (-kv[1], -kv[0]))

    def integration_seconds(self, frames=None) -> float:
        frames = self.usable_selected() if frames is None else frames
        return sum(self.meta[fr.filenum].livetime for fr in frames if fr.filenum in self.meta)

    def nights(self, frames=None) -> list[str]:
        """Distinct observing nights (date of DATE-OBS minus 12 h), sorted."""
        frames = self.usable_selected() if frames is None else frames
        out = set()
        for fr in frames:
            dt = parse_date(self.meta.get(fr.filenum, FrameMeta()).date_obs)
            if dt:
                out.add((dt - timedelta(hours=12)).date().isoformat())
        return sorted(out)

    def selection_changes(self) -> int:
        return len(self.included ^ self.default_included)


# ============================================================================= Siril sequence

class Session(_Common):
    kind = "seq"
    default_label = "Reset to Siril selection"

    def __init__(self, seq, meta, info, included, siril_included, problems=None, object_name="", filter_name=""):
        self.seq = seq
        self.meta = meta
        self.info = info
        self.included = included
        self.default_included = siril_included
        self.problems = problems or []
        self.object_name = object_name
        self.filter_name = filter_name

    # identity
    @property
    def name(self) -> str:
        return self.seq.name

    @property
    def path(self) -> str:
        return self.seq.path

    @property
    def directory(self) -> str:
        return self.seq.directory

    @property
    def layers(self) -> int:
        return self.seq.layers

    @property
    def has_drizzle(self) -> bool:
        return self.seq.drizzle

    @property
    def frames(self) -> list[Frame]:
        return self.seq.frames

    # frames
    def bad_reason(self, fr: Frame) -> str:
        st = self.info.stats.get(fr.filenum)
        return st.bad if st else ""

    def usable_selected(self) -> list[Frame]:
        return [fr for fr in self.seq.frames if fr.filenum in self.included and fr.has_reg and not self.bad_reason(fr)]

    def noise(self, fr: Frame) -> float | None:
        st = self.info.stats.get(fr.filenum)
        if not st or not st.noise:
            return None
        c = 1 if self.seq.layers == 3 and len(st.noise) > 1 else 0
        return st.noise[c]

    def frame_name(self, fr: Frame) -> str:
        return os.path.basename(self.seq.frame_path(fr))

    def status(self, fr: Frame) -> str:
        bad = self.bad_reason(fr)
        if bad:
            return f"Unusable: {bad}"
        if not fr.has_reg:
            return "Not registered"
        if fr.filenum not in self.included:
            return "Excluded" + ("" if fr.filenum not in self.default_included else " here")
        if fr.filenum not in self.default_included:
            return "Included here (excluded in Siril)"
        if fr.index == self.seq.reference:
            return "Reference"
        return ""

    # geometry
    def canvas_and_coverage(self, binning_for):
        """(canvas_w, canvas_h, depth map, binning) for the usable selected frames, or None."""
        frames = self.usable_selected()
        if not frames:
            return None
        cv = max_framing(frames)
        b = binning_for(cv.width, cv.height)
        return cv.width, cv.height, coverage_map(frames, cv, b), b

    # persistence
    def selection_path(self) -> str:
        return os.path.join(self.seq.cache_dir(), f"{self.seq.name}selection.json")

    def save_selection(self) -> None:
        path = self.selection_path()
        if self.included == self.default_included:
            try:
                os.remove(path)
            except OSError:
                pass
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"included": sorted(self.included)}, f)

    def reload(self) -> None:
        self.info = load_stats(self.seq, self.seq.frames, need_mask=True)


def load_session(path: str, progress=None) -> Session:
    seq = parse_seq(path)
    problems = check_stackable(seq)
    meta = scan_headers(seq, progress=progress) if seq.frames else {}
    info = load_stats(seq, seq.frames, need_mask=True)
    siril = {fr.filenum for fr in seq.frames if fr.included}
    included = set(siril)
    sel_path = os.path.join(seq.cache_dir(), f"{seq.name}selection.json")
    try:
        with open(sel_path, encoding="utf-8") as f:
            saved = set(json.load(f).get("included", []))
        known = {fr.filenum for fr in seq.frames}
        if saved and saved <= known:
            included = saved
    except (OSError, ValueError):
        pass
    obj = filt = ""
    if seq.frames:
        try:
            hdr = read_header(seq.frame_path(seq.reference_frame()))
            obj = str(hdr.get("OBJECT", "") or "").strip()
            filt = str(hdr.get("FILTER", "") or "").strip()
        except (OSError, ValueError):
            pass
    missing = [fr for fr in seq.frames if meta.get(fr.filenum) and meta[fr.filenum].error]
    if missing and len(missing) == len(seq.frames):
        problems.append(f"None of the frame files could be read (e.g. {seq.frame_path(missing[0])}).")
    elif missing:
        problems.append(f"{len(missing)} frame file(s) are missing or unreadable; they will be skipped.")
        included -= {fr.filenum for fr in missing}
    return Session(seq, meta, info, included, siril, problems, obj, filt)


# ============================================================================= raw lights

class LightsSession(_Common):
    """A folder of raw lights; metrics, registration and measurements come from the engine's caches."""
    kind = "lights"
    default_label = "Include all frames"

    def __init__(self, folder: str, progress=None):
        self.lights_dir = find_lights_dir(folder)
        self.dataset = dataset_dir(self.lights_dir)
        self.cdir = cache_dir(self.lights_dir)
        self.lights = scan_lights(self.lights_dir, progress=progress)
        self.problems: list[str] = []
        if not self.lights:
            self.problems.append(f"No FITS files in {self.lights_dir}.")
        self.width, self.height, self.channels, self.bayer = common_geometry(self.lights)
        self.odd = {lt.name: (lt.error or f"different size or type ({lt.width} x {lt.height}, {lt.channels} ch)")
                    for lt in self.lights
                    if lt.error or (lt.width, lt.height, lt.channels, lt.bayer) != (self.width, self.height,
                                                                                    self.channels, self.bayer)}
        if self.odd:
            self.problems.append(f"{len(self.odd)} file(s) can't be used (unreadable or a different size); "
                                 "they will be skipped.")
        self.frames: list[Frame] = []
        self.meta: dict[int, FrameMeta] = {}
        self.by_num: dict[int, object] = {}
        for lt in self.lights:
            fr = Frame(index=lt.index, filenum=lt.index + 1, included=lt.name not in self.odd, width=lt.width,
                       height=lt.height, has_reg=False)
            self.frames.append(fr)
            self.meta[fr.filenum] = FrameMeta(exptime=lt.exptime, livetime=lt.exptime, date_obs=lt.date_obs,
                                              filter=lt.filter, shape=(lt.channels, lt.height, lt.width),
                                              bitpix=lt.bitpix, error=lt.error)
            self.by_num[fr.filenum] = lt
        self.default_included = {fr.filenum for fr in self.frames if fr.included}
        self.included = set(self.default_included)
        self.reference_file = ""                  # user override; "" = automatic
        good = [lt for lt in self.lights if lt.name not in self.odd]
        self.object_name = next((lt.object for lt in good if lt.object), "")
        self.filter_name = next((lt.filter for lt in good if lt.filter), "")
        self.scale, self.pixfrac = 1.0, 1.0
        self._load_selection()
        self.reload()

    # identity
    @property
    def name(self) -> str:
        return os.path.basename(os.path.normpath(self.dataset)) or self.dataset

    @property
    def path(self) -> str:
        return self.lights_dir

    @property
    def directory(self) -> str:
        return self.dataset

    @property
    def layers(self) -> int:
        return 3 if (self.bayer or self.channels == 3) else 1

    @property
    def has_drizzle(self) -> bool:
        return True

    def light(self, fr: Frame):
        return self.by_num[fr.filenum]

    def frame_name(self, fr: Frame) -> str:
        return self.light(fr).name

    # caches
    def reload(self) -> None:
        self.analysis = Analysis(self.cdir)
        self.registration = Registration(self.cdir)
        self.measures = Measures(self.cdir)
        self.metrics: dict[str, dict] = {}
        for fr in self.frames:
            lt = self.light(fr)
            e = self.analysis.entries.get(lt.name)
            if e and e.get("sig") == light_sig(lt):
                self.metrics[lt.name] = e
                fr.fwhm = e.get("fwhm") or float("nan")
                fr.roundness = e.get("roundness") or float("nan")
                fr.background = e.get("background") or float("nan")
                fr.nstars = int(e.get("nstars") or 0)
            r = self.registration.frames.get(lt.name)
            fr.has_reg = bool(r and r.get("sig") == light_sig(lt))
        ref = self.registration.reference
        nref = self.metrics.get(ref, {}).get("nstars") or max([m.get("nstars", 0) for m in self.metrics.values()] or [1])
        for fr in self.frames:
            m = self.metrics.get(self.light(fr).name)
            if m and m.get("fwhm"):
                fr.wfwhm = m["fwhm"] * (1.0 + 2.0 * (nref - m["nstars"]) / max(nref, 1))

    def apply_analyzed(self, batch: list[dict]) -> None:
        """Live metrics from the engine's 'analyzed' events."""
        by_name = {self.light(fr).name: fr for fr in self.frames}
        for e in batch:
            fr = by_name.get(e.get("name"))
            if fr is None:
                continue
            self.metrics[e["name"]] = e
            fr.fwhm = e.get("fwhm") or float("nan")
            fr.roundness = e.get("roundness") or float("nan")
            fr.background = e.get("background") or float("nan")
            fr.nstars = int(e.get("nstars") or 0)

    @property
    def analyzed(self) -> bool:
        return all(self.light(fr).name in self.metrics for fr in self.frames if fr.filenum in self.included
                   and self.light(fr).name not in self.odd)

    @property
    def registered(self) -> bool:
        sel = [fr for fr in self.usable_selected()]
        return bool(sel) and all(fr.has_reg for fr in sel)

    @property
    def reference_name(self) -> str:
        return self.reference_file or self.registration.reference

    # frames
    def bad_reason(self, fr: Frame) -> str:
        lt = self.light(fr)
        if lt.name in self.odd:
            return self.odd[lt.name]
        m = self.metrics.get(lt.name)
        if m is not None and m.get("nstars", 0) < 15:
            return f"only {m.get('nstars', 0)} stars (clouds, dawn or a bad frame?)"
        f = self.registration.failed.get(lt.name)
        if f and f.get("sig") == light_sig(lt):
            return f.get("reason", "couldn't be registered")
        e = self.measures.frames.get(lt.name)
        if e and e.get("bad") and e.get("sig") == light_sig(lt):
            return e["bad"]
        return ""

    def usable_selected(self) -> list[Frame]:
        return [fr for fr in self.frames if fr.filenum in self.included and not self.bad_reason(fr)]

    def noise(self, fr: Frame) -> float | None:
        e = self.measures.frames.get(self.light(fr).name)
        if not e or not e.get("noise"):
            return None
        n = e["noise"]
        v = n[1] if len(n) > 1 else n[0]
        return v if v else None

    def status(self, fr: Frame) -> str:
        bad = self.bad_reason(fr)
        if bad:
            return f"Unusable: {bad}"
        lt = self.light(fr)
        if fr.filenum not in self.included:
            return "Excluded"
        if lt.name not in self.metrics:
            return "Not analyzed yet"
        if lt.name == self.reference_name:
            return "Reference" + ("" if fr.has_reg or not self.reference_file else " (chosen)")
        if not fr.has_reg:
            return "Not registered yet"
        r = self.registration.frames.get(lt.name, {})
        via = r.get("via", "")
        return f"Registered, {r.get('rms', 0):.2f} px" + (f" ({via})" if via.startswith("via") else "")

    # geometry
    def transforms(self) -> dict:
        out = {}
        for fr in self.usable_selected():
            if fr.has_reg:
                t = self.registration.transform(self.light(fr).name)
                if t is not None:
                    out[fr.filenum] = t
        return out

    def canvas_and_coverage(self, binning_for):
        ts = self.transforms()
        if len(ts) < 1:
            return None
        cw, ch, t_canvas, boxes = raw_canvas(ts, self.width, self.height, self.scale, self.pixfrac)
        b = binning_for(cw, ch)
        quads = [footprint(m, self.width, self.height, self.pixfrac) for m in t_canvas.values()]
        return cw, ch, quad_coverage(quads, cw, ch, b), b

    # persistence
    def selection_path(self) -> str:
        return os.path.join(self.cdir, "selection.json")

    def _load_selection(self) -> None:
        try:
            with open(self.selection_path(), encoding="utf-8") as f:
                d = json.load(f)
            names = set(d.get("excluded", []))
            self.included = {fr.filenum for fr in self.frames if fr.included and self.light(fr).name not in names}
            self.reference_file = d.get("reference", "")
        except (OSError, ValueError):
            pass

    def save_selection(self) -> None:
        excluded = sorted(self.light(fr).name for fr in self.frames if fr.filenum not in self.included)
        if not excluded and not self.reference_file:
            try:
                os.remove(self.selection_path())
            except OSError:
                pass
            return
        os.makedirs(self.cdir, exist_ok=True)
        with open(self.selection_path(), "w", encoding="utf-8") as f:
            json.dump({"excluded": excluded, "reference": self.reference_file}, f)

    def raw_bytes(self, frames) -> int:
        return sum(self.light(fr).width * self.light(fr).height * self.light(fr).channels
                   * abs(self.light(fr).bitpix) // 8 for fr in frames)


def is_lights_folder(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    return bool(list_fits(find_lights_dir(path)))


def open_any(path: str, progress=None):
    """A Session for a .seq file, a LightsSession for a folder of FITS lights (or its parent)."""
    if os.path.isfile(path) and path.lower().endswith(".seq"):
        return load_session(path, progress)
    if is_lights_folder(path):
        return LightsSession(path, progress)
    if os.path.isdir(path):
        seqs = sorted(f for f in os.listdir(path) if f.lower().endswith(".seq"))
        pick = [f for f in seqs if f.startswith("r_")] or seqs
        if pick:
            return load_session(os.path.join(path, pick[0]), progress)
    raise ValueError(f"{path} has no FITS lights and no Siril sequence (.seq).")


def format_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds} s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{m} min {s:02d} s" if m < 10 else f"{m} min"
    h, m = divmod(m, 60)
    return f"{h} h {m:02d} min"


def format_hours(seconds: float) -> str:
    return f"{seconds / 3600:.1f} h" if seconds >= 3600 else f"{seconds / 60:.0f} min"


def format_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1000 or unit == "TB":
            return f"{n:.0f} {unit}" if unit in ("B", "KB", "MB") else f"{n:.1f} {unit}"
        n /= 1000
    return f"{n:.1f} TB"


def binning_for_view(w: int, h: int) -> int:
    return max(1, math.ceil(max(w, h) / 1400))


__all__ = ["Session", "LightsSession", "open_any", "load_session", "parse_date", "format_duration", "format_hours",
           "format_bytes", "binning_for_view"]
