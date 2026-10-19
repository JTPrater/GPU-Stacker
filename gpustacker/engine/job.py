# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jason Prater (JP Astrophotography)
# Contains code ported from Siril; see THIRD-PARTY-NOTICES.txt
"""One stacking job: validate, measure frames, normalize, stack, write. Emits progress events.

Events are plain dicts (the CLI prints them as JSON lines, the UI reads them):
  hello, log, phase, progress, frame_bad, canvas, rows, done, cancelled, error
"""
from __future__ import annotations

import base64
import concurrent.futures as cf
import math
import os
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from .. import __version__
from . import fits as F
from .plan import (auto_output_path, canvas_origin, load_stats, needs_measuring, output_folder, overlaps,
                   region_for, shift_region)
from .sequence import Sequence, SequenceError, check_stackable, max_framing, parse_seq, scan_headers
from .stats import FrameStats

REJECTIONS = ("winsorized", "sigma", "none")
WEIGHTINGS = ("none", "noise", "wfwhm", "nbstars")
NORMALIZATIONS = ("addscale", "none")


@dataclass
class JobConfig:
    seq_path: str = ""                   # a Siril-registered sequence, or ...
    lights_dir: str = ""                 # ... a folder of raw lights (calibrate-free Seestar subs)
    output_path: str = ""
    include: list | None = None          # file numbers to stack; None = the .seq selection
    rejection: str = "winsorized"
    sigma_low: float = 3.0
    sigma_high: float = 3.0
    normalization: str = "addscale"
    rgb_equal: bool = True
    weighting: str = "noise"
    feather: float = 0.0                 # Siril -feather distance (0 = off)
    output_norm: bool = True
    crop: list | None = None             # [x, y, w, h] canvas rectangle (top-down) to stack and write
    rows: list | None = None             # [r0, r1] canvas rows only (test strip)
    save_coverage: bool = False
    preview_max: int = 1600              # long side of preview images (pixels)
    gpu_mem_gib: float = 0.0             # 0 = 75% of the card
    workers: int = 6
    strip: int = 64
    tile: int = 1024
    overwrite: bool = False
    preview_dir: str = ""
    debug_patch: list | None = None      # [x, y, size]: keep per-pixel stacks for verification (tests)
    # raw lights only
    include_files: list | None = None    # light file names to use; None = all
    reference_file: str = ""             # registration reference (file name); "" = best frame
    drizzle_scale: float = 1.0
    pixfrac: float = 1.0
    transform: str = "affine"            # affine | similarity
    analyze_only: bool = False           # stop after star detection + registration

    @classmethod
    def from_dict(cls, d: dict) -> "JobConfig":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> None:
        if self.rejection not in REJECTIONS:
            raise ValueError(f"rejection must be one of {REJECTIONS}")
        if self.weighting not in WEIGHTINGS:
            raise ValueError(f"weighting must be one of {WEIGHTINGS}")
        if self.normalization not in NORMALIZATIONS:
            raise ValueError(f"normalization must be one of {NORMALIZATIONS}")
        if self.rejection != "none" and not (self.sigma_low > 0 and self.sigma_high > 0):
            raise ValueError("sigma values must be positive")
        if self.feather < 0:
            raise ValueError("feather must be >= 0")
        if not (self.seq_path or self.lights_dir):
            raise ValueError("give a sequence (seq_path) or a lights folder (lights_dir)")
        if not (0.05 <= self.pixfrac <= 1.0):
            raise ValueError("pixfrac must be between 0.05 and 1")
        if self.drizzle_scale not in (1.0, 1.5, 2.0, 3.0):
            raise ValueError("drizzle scale must be 1, 1.5, 2 or 3")
        if self.transform not in ("affine", "similarity"):
            raise ValueError("transform must be affine or similarity")


class JobError(Exception):
    """A problem the user can fix (shown without a traceback)."""


# ----------------------------------------------------------------------------- GPU info

def probe_gpu() -> dict:
    import cupy as cp
    dev = cp.cuda.Device(0)
    props = cp.cuda.runtime.getDeviceProperties(0)
    free, total = cp.cuda.runtime.memGetInfo()
    name = props["name"]
    return {"name": name.decode() if isinstance(name, bytes) else str(name), "vram_total": int(total),
            "vram_free": int(free), "compute": f"{dev.compute_capability[0]}.{dev.compute_capability[1]}",
            "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(), "cupy": cp.__version__}


# ----------------------------------------------------------------------------- the job

class Job:
    def __init__(self, cfg: JobConfig, emit, cancel: threading.Event | None = None):
        cfg.validate()
        self.cfg = cfg
        self._emit = emit
        self.cancel = cancel or threading.Event()
        self._last_progress = 0.0
        self.t0 = time.perf_counter()

    # -- events
    def emit(self, ev: str, **kw) -> None:
        self._emit({"ev": ev, **kw})

    def log(self, msg: str, level: str = "info") -> None:
        self.emit("log", level=level, msg=msg)

    def progress(self, phase: str, done: float, total: float, force: bool = False, **kw) -> None:
        now = time.perf_counter()
        if not force and now - self._last_progress < 0.2:
            return
        self._last_progress = now
        self.emit("progress", phase=phase, done=done, total=total, **kw)

    def check_cancel(self) -> None:
        from .stack import Cancelled
        if self.cancel.is_set():
            raise Cancelled()

    # -- main entry
    def start_gpu(self) -> float:
        """Probe the GPU, announce the job, cap the memory pool; returns the cap in bytes."""
        import cupy as cp
        gpu = probe_gpu()
        self.emit("hello", version=__version__, gpu=gpu)
        self.check_cancel()
        limit = self.cfg.gpu_mem_gib * 2**30 if self.cfg.gpu_mem_gib > 0 else 0.75 * gpu["vram_total"]
        cp.get_default_memory_pool().set_limit(size=int(limit))
        return limit

    def run(self) -> dict:
        if self.cfg.lights_dir:
            from .rawjob import run_raw
            return run_raw(self)
        return self.run_sequence()

    def run_sequence(self) -> dict:
        from .stack import region_bytes

        cfg = self.cfg
        limit = self.start_gpu()

        # ---- sequence and selection
        try:
            seq = parse_seq(cfg.seq_path)
        except (SequenceError, OSError) as e:
            raise JobError(str(e)) from None
        problems = check_stackable(seq)
        if problems:
            raise JobError(" ".join(problems))
        if cfg.include is not None:
            wanted = set(int(n) for n in cfg.include)
            frames = [fr for fr in seq.frames if fr.filenum in wanted]
        else:
            frames = seq.included
        frames = [fr for fr in frames if fr.has_reg]
        if len(frames) < 2:
            raise JobError(f"only {len(frames)} frame(s) selected; at least 2 are needed")
        out_path = cfg.output_path                   # "" = named after the stack, once that is known
        test_strip = cfg.rows is not None
        if out_path and not test_strip and os.path.exists(out_path) and not cfg.overwrite:
            raise JobError(f"{out_path} already exists (choose another name or allow overwriting)")
        layers = seq.layers
        self.log(f"{seq.name}: {len(frames)} of {len(seq.frames)} frames, {layers} layer(s)")

        # ---- files and headers
        self.emit("phase", name="headers", label="Reading frame headers", total=len(frames))
        missing = [seq.frame_path(fr) for fr in frames if not os.path.isfile(seq.frame_path(fr))]
        if missing:
            raise JobError(f"{len(missing)} frame file(s) missing, e.g. {missing[0]}")
        use_drizzle = seq.drizzle and os.path.isdir(os.path.join(seq.directory, "drizztmp"))
        if use_drizzle:
            nodz = [fr for fr in frames if not os.path.isfile(seq.drizzle_path(fr))]
            if nodz:
                self.log(f"{len(nodz)} drizzle weight file(s) missing in drizztmp; stacking without "
                         "drizzle weights", "warn")
                use_drizzle = False
        meta = scan_headers(seq, frames, progress=lambda n, t: self.progress("headers", n, t))
        self.check_cancel()
        bad_hdr = [fr for fr in frames if meta[fr.filenum].error]
        if bad_hdr:
            raise JobError(f"unreadable frame: {seq.frame_path(bad_hdr[0])}: {meta[bad_hdr[0].filenum].error}")
        for fr in frames:
            m = meta[fr.filenum]
            if m.shape is None or m.shape[0] != layers or m.shape[1:] != (fr.height, fr.width):
                raise JobError(f"{os.path.basename(seq.frame_path(fr))}: image is {m.shape}, the .seq says "
                               f"{layers} x {fr.height} x {fr.width}")
            if m.bitpix not in (-32, 16):
                raise JobError(f"{os.path.basename(seq.frame_path(fr))}: BITPIX {m.bitpix} is not supported "
                               "(32-bit float or 16-bit frames only)")

        # ---- statistics (normalization, noise) and feather masks
        need_norm = cfg.normalization == "addscale"
        need_noise = cfg.weighting == "noise"
        need_mask = cfg.feather > 0
        info = load_stats(seq, frames, need_mask)
        fstats, mask_path = info.stats, info.mask_paths
        ok = [fr for fr in frames if not fstats[fr.filenum].bad]
        if len(ok) < 2:
            raise JobError("fewer than 2 usable frames")
        # The crop / test-strip region is given on the canvas of the selection minus frames already
        # known to be unusable (what the UI shows); only frames that touch it need measuring.
        canvas0 = max_framing(ok)
        region0 = region_for(canvas0, cfg.crop, cfg.rows)
        origin0 = canvas_origin(ok)
        ref0 = seq.reference_frame(ok)
        todo = [(fr, info.sigs[fr.filenum]) for fr in ok
                if (fr is ref0 or overlaps(fr, canvas0, region0))
                and needs_measuring(fstats[fr.filenum], layers, need_norm, need_noise, need_mask,
                                    mask_path.get(fr.filenum))]
        if todo:
            self._measure(seq, todo, layers, need_norm, need_noise, need_mask, fstats, mask_path, info.cache, meta)

        bad = [fr for fr in frames if fstats[fr.filenum].bad]
        for fr in bad:
            self.emit("frame_bad", filenum=fr.filenum, number=fr.number, reason=fstats[fr.filenum].bad)
        if bad:
            self.log(f"skipping {len(bad)} unusable frame(s): " + ", ".join(
                f"#{fr.number} ({fstats[fr.filenum].bad})" for fr in bad[:5]) + (" ..." if len(bad) > 5 else ""),
                "warn")
            frames = [fr for fr in frames if not fstats[fr.filenum].bad]
            if len(frames) < 2:
                raise JobError("fewer than 2 usable frames left")

        # ---- canvas, region, normalization, weights
        canvas = max_framing(frames)
        if cfg.crop or cfg.rows:
            x_lo, y_lo, x_hi, y_hi = shift_region(region0, origin0, canvas_origin(frames), canvas)
        else:
            x_lo, y_lo, x_hi, y_hi = 0, 0, canvas.width, canvas.height
        if x_hi <= x_lo or y_hi <= y_lo:
            raise JobError("the crop / row range is outside the canvas")
        in_region = [fr for fr in frames if overlaps(fr, canvas, (x_lo, y_lo, x_hi, y_hi))]
        if not in_region:
            raise JobError("no frame covers the chosen area")
        ref = seq.reference_frame(frames)
        if need_norm and not fstats[ref.filenum].has_norm(layers):
            ref = in_region[0]                       # the reference turned out unusable
        if ref.index != seq.reference:
            self.log(f"the sequence's reference image is not usable here; normalizing to #{ref.number}", "warn")
        work = in_region + ([] if ref in in_region else [ref])
        a, off, imgw = self._coefficients(work, ref, layers, fstats, seq.reg_layer, frames)
        stack_frames = self._stack_frames(seq, work, canvas, a, off, imgw, use_drizzle, mask_path)
        frames = in_region                           # what ends up in the output (counts, dates)
        bpp = 4 if all(meta[fr.filenum].bitpix == -32 for fr in frames) else 2
        region = (x_lo, y_lo, x_hi, y_hi)
        livetime = sum(meta[fr.filenum].livetime for fr in frames)
        if not out_path:
            obj = next((meta[fr.filenum].object for fr in frames if meta[fr.filenum].object), "")
            out_path = auto_output_path(output_folder(seq.directory), obj or seq.name.strip("_"), len(frames),
                                        livetime)
        return self._stack_and_write(
            stack_frames=stack_frames, canvas_wh=(canvas.width, canvas.height), region=region, layers=layers,
            n_frames=len(frames), livetime=livetime,
            stackcnt=sum(meta[fr.filenum].stackcnt for fr in frames),
            header_fn=lambda lt, sc: self._header(seq, ref, canvas, frames, meta, region, lt, sc),
            out_path=out_path, test_strip=test_strip, skipped=[fr.filenum for fr in bad], limit=limit,
            total_bytes_fn=lambda strip: region_bytes(stack_frames, y_lo, y_hi, layers, bpp, use_drizzle),
            plain=not (use_drizzle or cfg.feather > 0 or cfg.weighting != "none"))

    # ------------------------------------------------------------------------- stack + output
    def _stack_and_write(self, *, stack_frames, canvas_wh, region, layers, n_frames, livetime, stackcnt,
                         header_fn, out_path, test_strip, skipped, limit, total_bytes_fn, plain) -> dict:
        from . import kernels as K
        from .stack import StackOptions, raw_group_budget, stack_region, uses_batched_raw

        cfg = self.cfg
        x_lo, y_lo, x_hi, y_hi = region
        opts = StackOptions(channels=layers, rejection={"none": K.REJECT_NONE, "sigma": K.REJECT_SIGMA,
                                                       "winsorized": K.REJECT_WINSORIZED}[cfg.rejection],
                            sigma_low=cfg.sigma_low, sigma_high=cfg.sigma_high, feather=cfg.feather,
                            plain=plain, workers=cfg.workers, want_coverage=cfg.save_coverage,
                            debug_patch=tuple(cfg.debug_patch) if cfg.debug_patch else None)
        opts.raw_budget = raw_group_budget(limit)
        reserve = opts.raw_budget if uses_batched_raw(stack_frames) else 0      # batched raw rows on the GPU
        opts.strip, opts.tile = self._tune(stack_frames, limit, cfg.strip, cfg.tile, cfg.feather > 0, reserve)
        total_bytes = total_bytes_fn(opts.strip)
        rw, rh = x_hi - x_lo, y_hi - y_lo
        binf = max(1, math.ceil(max(rw, rh) / max(64, cfg.preview_max)))
        self.emit("canvas", width=canvas_wh[0], height=canvas_wh[1], region=[x_lo, y_lo, x_hi, y_hi],
                  bin=binf, bw=-(-rw // binf), bh=-(-rh // binf), channels=layers,
                  frames=n_frames, test_strip=test_strip)
        self.log(f"canvas {canvas_wh[0]} x {canvas_wh[1]}; stacking x {x_lo}-{x_hi}, y {y_lo}-{y_hi} "
                 f"({total_bytes / 1e9:.1f} GB to read, strips of {opts.strip} rows, tiles of {opts.tile})")

        # ---- stack
        self.emit("phase", name="stack", label="Stacking", total=total_bytes, rows=rh)
        t_stack = time.perf_counter()
        binned_rows = [0]

        def on_progress(nbytes, rows_done, nframes):
            el = time.perf_counter() - t_stack
            rate = nbytes / el if el > 0 else 0.0
            eta = (total_bytes - nbytes) / rate if rate > 0 else None
            self.progress("stack", nbytes, total_bytes, rows=rows_done, rows_total=rh, frames=nframes,
                          rate=rate, eta=eta, elapsed=el)
            self.check_cancel()

        def on_strip(out, r_lo, r_hi):
            self._emit_rows(out, binned_rows, r_hi, binf, rh, final=False)

        res = stack_region(stack_frames, (x_lo, y_lo, x_hi, y_hi), opts, self.cancel, on_progress, on_strip)
        self.check_cancel()
        if cfg.debug_patch:
            px, py, ps = (int(v) for v in cfg.debug_patch)
            self.debug = dict(stacks=res.debug, plain=opts.plain,
                              values=res.data[:, py - y_lo:py - y_lo + ps, px - x_lo:px - x_lo + ps].copy())
        self._emit_rows(res.data, binned_rows, rh, binf, rh, final=True)
        t_stack = time.perf_counter() - t_stack
        for c in range(layers):
            self.log(f"pixel rejection, channel {c}: {res.rejected_low[c] * 100:.3f}% low, "
                     f"{res.rejected_high[c] * 100:.3f}% high")
        self.log(f"stacked in {t_stack:.1f} s: {res.bytes_read / 1e9:.1f} GB read "
                 f"({res.bytes_read / max(t_stack, 1e-9) / 1e6:.0f} MB/s), waiting on disk {res.t_wait:.0f} s")
        for k, v in sorted(res.timings.items(), key=lambda kv: -kv[1]):     # GPUSTACKER_PROFILE=1
            self.log(f"profile: {k:<36} {v:7.2f} s")

        # ---- output
        data = res.data
        if cfg.output_norm:
            lo, hi = _nonzero_minmax(data)
            if hi > lo:
                _normalize_nonzero(data, lo, hi)
        else:
            np.clip(data, 0.0, 1.0, out=data)              # Siril clamps when not renormalizing
        preview_path = self._write_preview(data, binf)
        written = None
        if not test_strip or cfg.output_path:
            cards = header_fn(livetime, stackcnt)
            self.emit("phase", name="write", label="Writing output", total=1)
            os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
            F.write_fits(out_path, data, cards, progress=lambda d, t: self.progress("write", d, t))
            written = out_path
            if cfg.save_coverage and res.coverage is not None:
                cov_path = os.path.splitext(out_path)[0] + "_coverage.fit"
                F.write_fits(cov_path, res.coverage[None].astype(np.float32),
                             cards + F.history("coverage map: frames per pixel (valid samples)"))
                self.log(f"wrote coverage map {cov_path}")
        summary = dict(output=written, preview=preview_path, width=rw, height=rh, channels=layers,
                       frames=n_frames, skipped=skipped, livetime=livetime,
                       stackcnt=stackcnt, rejected_low=res.rejected_low, rejected_high=res.rejected_high,
                       bytes_read=res.bytes_read, seconds_stack=t_stack,
                       seconds_total=time.perf_counter() - self.t0, region=[x_lo, y_lo, x_hi, y_hi],
                       canvas=list(canvas_wh), test_strip=test_strip)
        self.emit("done", **summary)
        return summary

    # ------------------------------------------------------------------------- measure pass
    def _measure(self, seq, todo, layers, need_norm, need_noise, need_mask, fstats, mask_path, cache, meta):
        import cupy as cp

        from . import kernels as K
        from .masks import make_mask, write_mask

        planes_needed = layers if (need_norm or need_noise) else 1
        mask_layer = 1 if layers == 3 else 0
        total = sum(meta[fr.filenum].shape[1] * meta[fr.filenum].shape[2] * planes_needed
                    * (4 if meta[fr.filenum].bitpix == -32 else 2) for fr, _ in todo)
        what = [w for w, on in (("normalization", need_norm), ("noise", need_noise), ("feather masks", need_mask)) if on]
        self.emit("phase", name="measure", label="Measuring frames", total=total, frames=len(todo))
        self.log(f"measuring {len(todo)} frame(s) ({', '.join(what)}); results are cached for next time")

        def read(item):
            fr, sig = item
            path = seq.frame_path(fr)
            hdr = F.read_header(path)
            _, h, w = hdr.shape
            plane_bytes = h * w * hdr.dtype.itemsize
            first = 0 if planes_needed == layers else mask_layer
            buf = np.empty(plane_bytes * planes_needed, np.uint8)
            with open(path, "rb", buffering=0) as f:
                f.seek(hdr.data_offset + first * plane_bytes)
                if f.readinto(buf) != buf.nbytes:
                    raise JobError(f"{path}: file is shorter than its header says")
            return fr, sig, hdr, buf, first

        t0 = time.perf_counter()
        nbytes = 0
        last_save = time.perf_counter()
        with cf.ThreadPoolExecutor(max_workers=4) as pool:
            it = _bounded(pool, read, todo, 8)
            for n, (fr, sig, hdr, buf, first) in enumerate(it, 1):
                if self.cancel.is_set():
                    cache.save()
                    self.check_cancel()
                nbytes += buf.nbytes
                c_, h, w = hdr.shape
                npl = buf.nbytes // (h * w * hdr.dtype.itemsize)
                if hdr.dtype.itemsize == 4:
                    planes = K.bswap_f32(cp.asarray(buf.view(np.uint32))).reshape(npl, h, w)
                else:
                    planes = K.bswap_i16(cp.asarray(buf.view(np.uint16)), np.float32(hdr.get("BSCALE", 1.0)),
                                         np.float32(hdr.get("BZERO", 0.0))).reshape(npl, h, w)
                st = fstats[fr.filenum]
                want_norm = need_norm and not st.has_norm(layers)
                want_noise = need_noise and not st.has_noise(layers)
                new = _measure_planes(planes, want_norm, want_noise)
                if want_norm:
                    st.loc, st.scale = new.loc, new.scale
                if want_noise:
                    st.noise = new.noise
                if npl == layers:
                    st.ngood = new.ngood
                st.bad = _why_bad(new.ngood, h * w, st, layers, need_norm, need_noise)
                if need_mask and mask_path.get(fr.filenum) is None and not st.bad:
                    m = make_mask(planes[mask_layer - first])          # FITS row order
                    mp = seq.own_mask_path(fr)
                    write_mask(mp, m[::-1])
                    mask_path[fr.filenum] = mp
                cache.put(fr.filenum, sig, st)
                del planes
                el = time.perf_counter() - t0
                rate = nbytes / el if el > 0 else 0
                self.progress("measure", nbytes, total, frames_done=n, frames_total=len(todo), rate=rate,
                              eta=(total - nbytes) / rate if rate > 0 else None, elapsed=el, force=n == len(todo))
                if time.perf_counter() - last_save > 20:
                    cache.save()
                    last_save = time.perf_counter()
        cache.save()
        cp.get_default_memory_pool().free_all_blocks()
        self.log(f"measured {len(todo)} frame(s) in {time.perf_counter() - t0:.0f} s "
                 f"({nbytes / max(time.perf_counter() - t0, 1e-9) / 1e6:.0f} MB/s)")

    # ------------------------------------------------------------------------- normalization, weights
    def _coefficients(self, frames, ref, layers, fstats, reg_layer, all_frames):
        """addscale coefficients and image weights for `frames`; wFWHM / star-count weights take their
        min and max over all usable frames (as for a full Siril stack)."""
        cfg = self.cfg
        n = len(frames)
        a = np.ones((n, layers))
        off = np.zeros((n, layers))
        if cfg.normalization == "addscale":
            rs = fstats[ref.filenum]
            refl = reg_layer if (reg_layer is not None and reg_layer < layers) else (1 if layers == 3 else 0)
            for i, fr in enumerate(frames):
                st = fstats[fr.filenum]
                for c in range(layers):
                    rl = refl if cfg.rgb_equal else c
                    s = st.scale[c]
                    a[i, c] = 1.0 if not s else rs.scale[rl] / s
                    off[i, c] = a[i, c] * st.loc[c] - rs.loc[rl]
        w = np.ones((n, layers))
        if cfg.weighting == "noise":
            for i, fr in enumerate(frames):
                st = fstats[fr.filenum]
                for c in range(layers):
                    w[i, c] = 1.0 / (a[i, c] * a[i, c] * st.noise[c] * st.noise[c])
        elif cfg.weighting == "wfwhm":
            fa = np.array([fr.wfwhm for fr in all_frames], dtype=np.float64)
            if not (fa > 0).any():
                raise JobError("wFWHM weighting needs registration FWHM values (none in the .seq)")
            fmin, fmax = fa[fa > 0].min(), fa.max()
            f = np.array([fr.wfwhm for fr in frames], dtype=np.float64)
            pos = f > 0
            if fmax == fmin:
                w[:] = 1.0
            else:
                inv = 1.0 / (1.0 / fmin**2 - 1.0 / fmax**2)
                ww = np.where(pos, (1.0 / np.where(pos, f, 1.0) ** 2 - 1.0 / fmax**2) * inv, 0.0)
                w = np.repeat(ww[:, None], layers, axis=1)
        elif cfg.weighting == "nbstars":
            sa = np.array([fr.nstars for fr in all_frames], dtype=np.float64)
            smin, smax = sa.min(), sa.max()
            s = np.array([fr.nstars for fr in frames], dtype=np.float64)
            ww = np.ones(n) if smax == smin else ((s - smin) / (smax - smin)) ** 2
            w = np.repeat(ww[:, None], layers, axis=1)
        if cfg.weighting != "none":
            norm = w.sum(axis=0) / n
            if np.any(norm == 0):
                raise JobError(f"{cfg.weighting} weighting gives every frame weight 0")
            w = w / norm
        return a, off, w

    def _stack_frames(self, seq, frames, canvas, a, off, imgw, use_drizzle, mask_path):
        from .stack import StackFrame
        out = []
        for i, fr in enumerate(frames):
            out.append(StackFrame(filenum=fr.filenum, path=seq.frame_path(fr),
                                  drz_path=seq.drizzle_path(fr) if use_drizzle else None,
                                  mask_path=mask_path.get(fr.filenum), x0=canvas.x0[fr.filenum],
                                  y0=canvas.y0[fr.filenum], w=fr.width, h=fr.height,
                                  a=list(a[i]), off=list(off[i]), imgw=list(imgw[i])))
        return out

    @staticmethod
    def _tune(frames, limit, strip, tile, feather, reserve=0):
        """Shrink strip rows / tile width until the per-strip GPU buffers fit the memory budget
        (minus `reserve` bytes kept for something else, e.g. the batched raw rows)."""
        if not frames:
            return strip, tile
        edges = sorted([(fr.y0, 1) for fr in frames] + [(fr.y0 + fr.h, -1) for fr in frames])
        kmax = cur = 0
        for _, d in edges:
            cur += d
            kmax = max(kmax, cur)
        wmax = max(fr.w for fr in frames)
        mask_bytes = 0.0101 * max(fr.w * fr.h for fr in frames) * 4 if feather else 0
        budget = 0.7 * limit - kmax * mask_bytes - 256 * 2**20 - reserve   # headroom: temporaries, fragmentation
        while True:
            need = kmax * strip * (wmax * 8 + tile * 32)
            if need <= budget or (strip <= 8 and tile <= 128):
                return strip, tile
            if tile > 256:
                tile //= 2
            elif strip > 8:
                strip //= 2
            else:
                tile //= 2

    # ------------------------------------------------------------------------- previews, header
    def _emit_rows(self, out, state, r_done, binf, rh, final):
        """Emit binned preview rows that are complete (all channels) up to region row r_done."""
        start = state[0]
        stop = (r_done // binf) if not final else -(-rh // binf)
        if stop <= start:
            return
        block = _bin(out[:, start * binf:min(rh, stop * binf), :], binf)
        state[0] = stop
        self.emit("rows", y=start, rows=int(block.shape[1]), width=int(block.shape[2]),
                  channels=int(block.shape[0]),
                  data=base64.b64encode(block.astype(np.float16).tobytes()).decode("ascii"))

    def _write_preview(self, data, binf) -> str:
        d = self.cfg.preview_dir or os.path.join(tempfile.gettempdir(), "gpustacker")
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"preview_{os.getpid()}_{int(time.time() * 1000)}.npy")
        np.save(path, _bin(data, binf))
        return path

    def _header(self, seq, ref, canvas, frames, meta, region, livetime, stackcnt):
        x_lo, y_lo, x_hi, y_hi = region
        hdr = F.read_header(seq.frame_path(ref))
        drop = {"FILENAME", "DATE", "MIPS-FLO", "MIPS-FHI", "MIPS-HI", "MIPS-LO", "COMMENT", "HISTORY", "CRPIX1",
                "CRPIX2", "DATE-OBS", "PROGRAM", "STACKCNT", "LIVETIME", "DATAMIN", "DATAMAX", "CHECKSUM", "DATASUM"}
        cards = [c for c in hdr.user_cards() if c[:8].strip() not in drop]
        if "CRPIX1" in hdr.values and "CRPIX2" in hdr.values:
            rx0, ry0 = canvas.x0[ref.filenum], canvas.y0[ref.filenum]
            cards += [F.card("CRPIX1", float(hdr.values["CRPIX1"]) + rx0 - x_lo, "Axis1 reference pixel"),
                      F.card("CRPIX2", float(hdr.values["CRPIX2"]) + y_hi - ry0 - ref.height, "Axis2 reference pixel")]
        dates = sorted(meta[fr.filenum].date_obs for fr in frames if meta[fr.filenum].date_obs)
        if dates:
            cards.append(F.card("DATE-OBS", dates[0], "first frame observation start"))
        cfg = self.cfg
        cards += [F.card("STACKCNT", int(stackcnt), "Stack frames"),
                  F.card("LIVETIME", float(livetime), "[s] Exposure time after deadtime correction"),
                  F.card("PROGRAM", f"GPU Stacker {__version__}", "Software that created this HDU")]
        rej = cfg.rejection if cfg.rejection == "none" else f"{cfg.rejection} {cfg.sigma_low:g} {cfg.sigma_high:g}"
        cards += F.history(f"GPU Stacker {__version__}: {len(frames)} frames of {seq.name}, rejection {rej}")
        cards += F.history(f"normalization {cfg.normalization}{', rgb_equal' if cfg.rgb_equal else ''}, weighting "
                           f"{cfg.weighting}, feather {cfg.feather:g}, maximize framing, "
                           f"{'output normalized' if cfg.output_norm else 'not renormalized'}")
        if (x_lo, y_lo, x_hi, y_hi) != (0, 0, canvas.width, canvas.height):
            cards += F.history(f"region x {x_lo}-{x_hi}, y {y_lo}-{y_hi} of a {canvas.width} x {canvas.height} canvas")
        return cards


# ----------------------------------------------------------------------------- helpers

def _bounded(pool, fn, items, window):
    import itertools
    items = iter(items)
    pending = [pool.submit(fn, it) for it in itertools.islice(items, window)]
    while pending:
        res = pending.pop(0).result()
        nxt = next(items, None)
        if nxt is not None:
            pending.append(pool.submit(fn, nxt))
        yield res


def _measure_planes(planes, need_norm, need_noise):
    from .stats import measure_planes
    return measure_planes(planes, need_norm, need_noise)


def _why_bad(ngood: list, npix: int, st: FrameStats, layers: int, need_norm: bool, need_noise: bool) -> str:
    good = [g for g in ngood if g is not None]
    if good and min(good) < 0.10 * npix:
        return f"blank: only {min(good) / npix * 100:.1f}% of the pixels have data"
    if need_norm and not st.has_norm(layers):
        return "no usable statistics (flat or empty data)"
    if need_noise and not st.has_noise(layers):
        return "background noise could not be measured"
    return ""


def _nonzero_minmax(data: np.ndarray) -> tuple[float, float]:
    """Min and max of the non-zero pixels as Siril's norm_to_0_1_range finds them: its search starts at the
    second pixel in FITS order, so the first one (channel 0, bottom-left: data[0, -1, 0]) is left out."""
    lo, hi = np.inf, -np.inf
    h = data.shape[1]
    for c in range(data.shape[0]):
        for r in range(0, h, 512):
            blk = data[c, r:r + 512]
            keep = blk != 0
            if c == 0 and r + 512 >= h:
                keep[-1, 0] = False
            nz = blk[keep]
            if nz.size:
                lo, hi = min(lo, float(nz.min())), max(hi, float(nz.max()))
    return (lo, hi) if lo <= hi else (0.0, 0.0)


def _normalize_nonzero(data: np.ndarray, lo: float, hi: float) -> None:
    """Siril -output_norm (norm_to_0_1_range): (v - min) / (max - min) on non-zero pixels, in place, in
    float like Siril."""
    lo32 = np.float32(lo)
    rng = np.float32(hi) - lo32
    for c in range(data.shape[0]):
        for r in range(0, data.shape[1], 512):
            blk = data[c, r:r + 512]
            nz = blk != 0
            blk[nz] = (blk[nz] - lo32) / rng


def _bin(block: np.ndarray, f: int) -> np.ndarray:
    """(C, h, w) -> (C, ceil(h/f), ceil(w/f)) mean of the non-zero values of each f x f cell."""
    c, h, w = block.shape
    if f == 1:
        return block.astype(np.float32, copy=True)
    bh, bw = -(-h // f), -(-w // f)
    pad = np.zeros((c, bh * f, bw * f), np.float32)
    pad[:, :h, :w] = block
    cells = pad.reshape(c, bh, f, bw, f)
    s = cells.sum(axis=(2, 4), dtype=np.float64)
    n = (cells != 0).sum(axis=(2, 4))
    return np.where(n > 0, s / np.maximum(n, 1), 0.0).astype(np.float32)
