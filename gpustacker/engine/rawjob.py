# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jason Prater (JP Astrophotography)
"""Raw lights -> stacked image, with no intermediate files.

  1. analyze    read each light once: stars, FWHM, roundness, background (cached)
  2. register   triangle-match every frame onto the reference (mosaics: chained) (cached)
  3. measure    drizzle each frame once on the GPU for its normalization statistics, background
                noise and feather mask, exactly what Siril measures on its r_ frames (cached)
  4. stack      per canvas strip, drizzle only the raw rows each frame contributes, then the same
                Siril-equivalent rejection + weighted mean as for Siril sequences
Siril would write and re-read ~120 MB of registered frame + drizzle weights per light; here the
raw files (~17 MB each for the Seestar S50 Pro) are read three times instead.
"""
from __future__ import annotations

import concurrent.futures as cf
import itertools
import os
import time

import numpy as np

from .. import __version__
from . import fits as F
from .lights import common_geometry, decode_gpu, find_lights_dir, read_frame_raw, scan_lights
from .plan import auto_output_path, region_for
from .rawcache import Analysis, Measures, Registration, cache_dir, frame_key, light_sig, raw_canvas
from .stats import FrameStats

MIN_STARS = 15
RAW_STRIP = 256


class _Adapter:
    """What Job._coefficients needs from a frame."""

    def __init__(self, lt, nstars: int, wfwhm: float):
        self.filenum = lt.index + 1
        self.index = lt.index
        self.nstars = nstars
        self.wfwhm = wfwhm


def _bounded(pool, fn, items, window):
    items = iter(items)
    pending = [pool.submit(fn, it) for it in itertools.islice(items, window)]
    while pending:
        res = pending.pop(0).result()
        nxt = next(items, None)
        if nxt is not None:
            pending.append(pool.submit(fn, nxt))
        yield res


def run_raw(job) -> dict:
    from .job import JobError

    cfg = job.cfg
    limit = job.start_gpu()
    if not os.path.isdir(cfg.lights_dir):
        raise JobError(f"folder not found: {cfg.lights_dir}")
    lights_dir = find_lights_dir(cfg.lights_dir)
    cdir = cache_dir(lights_dir)

    # ---- headers
    job.emit("phase", name="headers", label="Reading frame headers", total=1)
    lights = scan_lights(lights_dir, progress=lambda n, t: job.progress("headers", n, t))
    job.check_cancel()
    if not lights:
        raise JobError(f"no FITS files in {lights_dir}")
    w, h, nch, bayer = common_geometry(lights)
    odd = [lt for lt in lights if lt.error or (lt.width, lt.height, lt.channels, lt.bayer) != (w, h, nch, bayer)]
    for lt in odd:
        job.emit("frame_bad", name=lt.name, number=lt.number,
                 reason=lt.error or f"different size or type ({lt.width} x {lt.height}, {lt.channels} ch)")
    frames = [lt for lt in lights if lt not in odd]
    if cfg.include_files is not None:
        wanted = set(cfg.include_files)
        frames = [lt for lt in frames if lt.name in wanted]
    if len(frames) < 2:
        raise JobError(f"only {len(frames)} usable frame(s) selected; at least 2 are needed")
    lut = frames[0].cfa_lut()
    layers = 3 if (lut is not None or nch == 3) else 1
    job.log(f"{len(frames)} lights, {w} x {h}, " + (f"Bayer {bayer}" if lut is not None else f"{nch} channel(s)"))

    # ---- 1. analyze
    ana = Analysis(cdir)
    todo = [lt for lt in frames if ana.get(lt) is None]
    if todo:
        _analyze(job, todo, ana)
    stars, metrics = {}, {}
    from .stars import StarList
    for lt in frames:
        arr, m = ana.get(lt)
        stars[lt.index] = StarList.from_array(arr)
        metrics[lt.index] = m
    poor = [lt for lt in frames if metrics[lt.index]["nstars"] < MIN_STARS]
    for lt in poor:
        job.emit("frame_bad", name=lt.name, number=lt.number,
                 reason=f"only {metrics[lt.index]['nstars']} stars detected (clouds, dawn or a bad frame?)")
    usable = [lt for lt in frames if lt not in poor]
    if len(usable) < 2:
        raise JobError("fewer than 2 frames with enough stars")

    # ---- 2. register
    from .register import Transform, pick_reference, register_all
    from .stars import FrameMetrics
    by_name = {lt.name: lt for lt in usable}
    reg = Registration(cdir)
    if cfg.reference_file and cfg.reference_file in by_name:
        ref = by_name[cfg.reference_file]
    elif reg.reference in by_name and reg.model == cfg.transform:
        ref = by_name[reg.reference]          # keep the cached choice when frames are excluded
    else:
        fm = {lt.index: FrameMetrics(metrics[lt.index]["nstars"], metrics[lt.index]["fwhm"],
                                     metrics[lt.index]["roundness"], metrics[lt.index]["background"],
                                     metrics[lt.index]["noise"]) for lt in usable}
        ref = next(lt for lt in usable if lt.index == pick_reference(fm))
    why = reg.why_invalid(ref.name, cfg.transform, usable)
    if why:
        job.log(f"registering: {why}")
        job.emit("phase", name="register", label="Registering frames", total=len(usable))
        t0 = time.perf_counter()
        res = register_all({lt.index: stars[lt.index] for lt in usable}, ref.index, cfg.transform,
                           progress=lambda n, t: job.progress("register", n, t, frames_done=n, frames_total=t),
                           cancel=job.cancel.is_set)
        job.check_cancel()
        idx = {lt.index: lt for lt in usable}
        reg.reference, reg.model = ref.name, cfg.transform
        reg.frames = {idx[i].name: {"sig": light_sig(idx[i]), "t": tr.to_list(), "inliers": res.info[i][0],
                                    "rms": res.info[i][1], "via": res.info[i][2]} for i, tr in res.transforms.items()}
        reg.failed = {idx[i].name: {"sig": light_sig(idx[i]), "reason": why} for i, why in res.failed.items()}
        reg.save()
        chained = sum(1 for v in res.info.values() if v[2].startswith("via"))
        job.log(f"registered {len(res.transforms)} of {len(usable)} frames on #{ref.number} ({ref.name}) in "
                f"{time.perf_counter() - t0:.0f} s" + (f", {chained} through neighbouring frames (mosaic)" if chained else ""))
    for lt in usable:
        if lt.name in reg.failed:
            job.emit("frame_bad", name=lt.name, number=lt.number, reason=reg.failed[lt.name]["reason"])
    registered = [lt for lt in usable if lt.name in reg.frames]
    if len(registered) < 2:
        raise JobError("fewer than 2 frames could be registered")
    rms = [reg.frames[lt.name]["rms"] for lt in registered if lt is not ref]
    rot = [Transform.from_list(reg.frames[lt.name]["t"]).rotation_deg for lt in registered]
    job.emit("registered", reference=ref.name, registered=len(registered), failed=len(usable) - len(registered),
             rms_median=float(np.median(rms)) if rms else 0.0, rotation=[float(min(rot)), float(max(rot))])
    if cfg.analyze_only:
        summary = dict(analysis=True, frames=len(registered), reference=ref.name,
                       seconds_total=time.perf_counter() - job.t0)
        job.emit("done", **summary)
        return summary

    # ---- canvas and region
    transforms = {lt.name: reg.transform(lt.name) for lt in registered}
    cw, ch, t_canvas, boxes = raw_canvas(transforms, w, h, cfg.drizzle_scale, cfg.pixfrac)
    region = region_for(_Canvas(cw, ch), cfg.crop, cfg.rows)
    x_lo, y_lo, x_hi, y_hi = region
    if x_hi <= x_lo or y_hi <= y_lo:
        raise JobError("the crop / row range is outside the canvas")

    def touches(lt):
        bx, by, bw, bh = boxes[lt.name]
        return bx < x_hi and bx + bw > x_lo and by < y_hi and by + bh > y_lo

    in_region = [lt for lt in registered if touches(lt)]
    if not in_region:
        raise JobError("no frame covers the chosen area")

    # ---- 3. measure
    need_norm = cfg.normalization == "addscale"
    need_noise = cfg.weighting == "noise"
    need_mask = cfg.feather > 0
    ms = Measures(cdir)

    def box_t(lt):
        m = t_canvas[lt.name].copy()
        bx, by, bw, bh = boxes[lt.name]
        m[0, 2] -= bx
        m[1, 2] -= by
        return m

    keys = {lt.name: frame_key(box_t(lt), boxes[lt.name][2], boxes[lt.name][3], cfg.pixfrac) for lt in registered}

    def complete(lt):
        e = ms.get(lt, keys[lt.name])
        if e is None:
            return False
        st = _fstats(e)
        if e.get("bad"):
            return True
        return ((not need_norm or st.has_norm(layers)) and (not need_noise or st.has_noise(layers))
                and (not need_mask or (e.get("mask") and os.path.isfile(ms.mask_path(lt)))))

    work = in_region + ([] if ref in in_region else [ref])
    todo = [lt for lt in work if not complete(lt)]
    if todo:            # always measure everything, so the cache serves any weighting / normalization / feather
        _measure(job, todo, ms, keys, box_t, boxes, lut, cfg.pixfrac, layers, True, True, True,
                 w * h * cfg.drizzle_scale ** 2)
    bad = [lt for lt in work if ms.get(lt, keys[lt.name]).get("bad")]
    for lt in bad:
        job.emit("frame_bad", name=lt.name, number=lt.number, reason=ms.get(lt, keys[lt.name])["bad"])
    work = [lt for lt in work if lt not in bad]
    in_region = [lt for lt in in_region if lt not in bad]
    if not in_region or len(work) < 2:
        raise JobError("fewer than 2 usable frames in the chosen area")
    if ref in bad:
        ref = in_region[0]
        job.log(f"the reference frame is unusable here; normalizing to #{ref.number}", "warn")

    # ---- normalization and weights (same rules as for Siril sequences)
    fstats = {lt.index + 1: _fstats(ms.get(lt, keys[lt.name])) for lt in work}
    n_ref = max(1, metrics[ref.index]["nstars"])

    def adapt(lt):
        m = metrics[lt.index]
        return _Adapter(lt, m["nstars"], m["fwhm"] * (1.0 + 2.0 * (n_ref - m["nstars"]) / n_ref))

    adapters = {lt.name: adapt(lt) for lt in registered}
    a, off, imgw = job._coefficients([adapters[lt.name] for lt in work], adapters[ref.name], layers, fstats, 1,
                                     [adapters[lt.name] for lt in registered if lt not in bad])
    from .stack import RawStackFrame, raw_group_budget, raw_region_bytes, uses_batched_raw
    stack_frames = []
    for i, lt in enumerate(work):
        bx, by, bw, bh = boxes[lt.name]
        stack_frames.append(RawStackFrame(
            filenum=lt.index + 1, light=lt, t=t_canvas[lt.name], lut=lut, pixfrac=cfg.pixfrac,
            mask_path=ms.mask_path(lt) if need_mask else None, x0=bx, y0=by, w=bw, h=bh,
            a=list(a[i]), off=list(off[i]), imgw=list(imgw[i])))
    if cfg.strip == 64:                       # the default: raw strips can be taller (fewer GPU launches)
        cfg.strip = RAW_STRIP
    test_strip = cfg.rows is not None
    if cfg.output_path and not test_strip and os.path.exists(cfg.output_path) and not cfg.overwrite:
        raise JobError(f"{cfg.output_path} already exists (choose another name or allow overwriting)")
    livetime = sum(lt.exptime for lt in in_region)
    out_path = cfg.output_path or auto_output_path(os.path.dirname(cdir), next(
        (lt.object for lt in in_region if lt.object), ""), len(in_region), livetime)
    dates = sorted(lt.date_obs for lt in in_region if lt.date_obs)

    def header(lt_sum, count):
        return _header(ref, cfg, len(in_region), lt_sum, count, dates[0] if dates else "", region, (cw, ch))

    return job._stack_and_write(
        stack_frames=stack_frames, canvas_wh=(cw, ch), region=region, layers=layers, n_frames=len(in_region),
        livetime=livetime, stackcnt=len(in_region), header_fn=header, out_path=out_path, test_strip=test_strip,
        skipped=[lt.number for lt in poor + bad] + [lt.number for lt in usable if lt.name in reg.failed],
        limit=limit, total_bytes_fn=lambda strip: raw_region_bytes(
            stack_frames, y_lo, y_hi, strip, layers,
            group_budget=raw_group_budget(limit) if uses_batched_raw(stack_frames) else None),
        plain=False)


class _Canvas:
    def __init__(self, w, h):
        self.width, self.height = w, h


def _fstats(e: dict) -> FrameStats:
    return FrameStats(loc=e.get("loc", []), scale=e.get("scale", []), noise=e.get("noise", []),
                      ngood=e.get("ngood", []), bad=e.get("bad", ""))


# ----------------------------------------------------------------------------- passes

def _analyze(job, todo, ana: Analysis) -> None:
    import cupy as cp

    from .stars import detect, luminance
    total = sum(lt.width * lt.height * lt.channels * abs(lt.bitpix) // 8 for lt in todo)
    job.emit("phase", name="analyze", label="Detecting stars", total=total, frames=len(todo))
    job.log(f"analyzing {len(todo)} frame(s): star detection and quality metrics (cached for next time)")
    t0 = time.perf_counter()
    nbytes, batch, last_save, last_emit = 0, [], time.perf_counter(), time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=4) as pool:
        for n, (lt, buf) in enumerate(_bounded(pool, lambda lt: (lt, read_frame_raw(lt)), todo, 8), 1):
            if job.cancel.is_set():
                ana.save()
                job.check_cancel()
            nbytes += buf.nbytes
            planes = decode_gpu(lt, buf, lt.height, lt.channels)
            s, m = detect(luminance(planes, cfa=lt.cfa_lut() is not None))
            md = dict(nstars=m.nstars, fwhm=_num(m.fwhm), roundness=_num(m.roundness), background=_num(m.background),
                      noise=_num(m.noise))
            ana.put(lt, s.to_array(), md)
            batch.append({"name": lt.name, **md})
            el = time.perf_counter() - t0
            rate = nbytes / el if el > 0 else 0
            job.progress("analyze", nbytes, total, frames_done=n, frames_total=len(todo), rate=rate,
                         eta=(total - nbytes) / rate if rate > 0 else None, elapsed=el, force=n == len(todo))
            if time.perf_counter() - last_emit > 0.5 or n == len(todo):
                job.emit("analyzed", frames=batch)
                batch, last_emit = [], time.perf_counter()
            if time.perf_counter() - last_save > 30:
                ana.save()
                last_save = time.perf_counter()
            del planes
    ana.save()
    cp.get_default_memory_pool().free_all_blocks()
    job.log(f"analyzed {len(todo)} frame(s) in {time.perf_counter() - t0:.0f} s")


def _measure(job, todo, ms: Measures, keys, box_t, boxes, lut, pixfrac, layers, need_norm, need_noise, need_mask,
             npix) -> None:
    import cupy as cp

    from .drizzle import warp_frame
    from .masks import downscaled_mask, finish_mask, write_mask
    from .stats import measure_planes
    total = sum(lt.width * lt.height * lt.channels * abs(lt.bitpix) // 8 for lt in todo)
    job.emit("phase", name="measure", label="Measuring frames", total=total, frames=len(todo))
    what = [x for x, on in (("normalization", need_norm), ("noise", need_noise), ("feather masks", need_mask)) if on]
    job.log(f"measuring {len(todo)} frame(s) ({', '.join(what) or 'coverage'}); results are cached for next time")
    t0 = time.perf_counter()
    nbytes, last_save = 0, time.perf_counter()

    def mask_job(padded, path):              # CPU half of the mask, overlapped with the next frame's GPU work
        write_mask(path, finish_mask(padded))

    with cf.ThreadPoolExecutor(max_workers=4) as pool, cf.ThreadPoolExecutor(max_workers=2) as cpu:
        pending = []
        for n, (lt, buf) in enumerate(_bounded(pool, lambda lt: (lt, read_frame_raw(lt)), todo, 8), 1):
            if job.cancel.is_set():
                ms.save()
                job.check_cancel()
            nbytes += buf.nbytes
            planes = decode_gpu(lt, buf, lt.height, lt.channels)
            bx, by, bw, bh = boxes[lt.name]
            val, dz = warp_frame(planes, box_t(lt), lut, pixfrac, bw, bh)
            data = cp.where(dz > 0, val, 0.0).astype(cp.float32)
            st = measure_planes(data, need_norm, need_noise)
            e = {"sig": light_sig(lt), "key": keys[lt.name], "loc": st.loc, "scale": st.scale, "noise": st.noise,
                 "ngood": st.ngood, "bad": "", "mask": ""}
            good = max(g for g in st.ngood if g is not None) if st.ngood else 0
            if good < 0.02 * npix:
                e["bad"] = f"blank: only {good / max(npix, 1) * 100:.1f}% of the pixels have data"
            elif need_norm and not st.has_norm(layers):
                e["bad"] = "no usable statistics (flat or empty data)"
            elif need_noise and not st.has_noise(layers):
                e["bad"] = "background noise could not be measured"
            if need_mask and not e["bad"]:
                padded = downscaled_mask(data[1 if layers == 3 else 0])
                pending.append(cpu.submit(mask_job, padded, ms.mask_path(lt)))
                e["mask"] = os.path.basename(ms.mask_path(lt))
            ms.frames[lt.name] = e
            del planes, val, dz, data
            el = time.perf_counter() - t0
            rate = nbytes / el if el > 0 else 0
            job.progress("measure", nbytes, total, frames_done=n, frames_total=len(todo), rate=rate,
                         eta=(total - nbytes) / rate if rate > 0 else None, elapsed=el, force=n == len(todo))
            if time.perf_counter() - last_save > 30:
                for f in pending:
                    f.result()
                pending = []
                ms.save()
                last_save = time.perf_counter()
        for f in pending:
            f.result()
    ms.save()
    cp.get_default_memory_pool().free_all_blocks()
    job.log(f"measured {len(todo)} frame(s) in {time.perf_counter() - t0:.0f} s")


def _num(v) -> float | None:
    v = float(v)
    return v if np.isfinite(v) else None


def _header(ref, cfg, n, livetime, count, date_obs, region, canvas) -> list:
    hdr = F.read_header(ref.path)
    drop = {"BZERO", "BSCALE", "BAYERPAT", "XBAYROFF", "YBAYROFF", "ROWORDER", "DATE-OBS", "DATE-EXP", "STACKCNT",
            "LIVETIME", "TOTALEXP", "PROGRAM", "COMMENT", "HISTORY", "DATAMIN", "DATAMAX", "CHECKSUM", "DATASUM",
            "XPIXSZ", "YPIXSZ", "PIXSIZE1", "PIXSIZE2", "XBINNING", "YBINNING", "CCDXBIN", "CCDYBIN"}
    cards = [c for c in hdr.user_cards() if c[:8].strip() not in drop]
    s = cfg.drizzle_scale
    for k in ("XPIXSZ", "YPIXSZ"):
        v = hdr.get(k)
        if isinstance(v, (int, float)):
            cards.append(F.card(k, float(v) / s, "pixel size in microns (after drizzle)"))
    if date_obs:
        cards.append(F.card("DATE-OBS", date_obs, "first frame observation start"))
    cards += [F.card("STACKCNT", int(count), "Stack frames"),
              F.card("LIVETIME", float(livetime), "[s] Exposure time after deadtime correction"),
              F.card("PROGRAM", f"GPU Stacker {__version__}", "Software that created this HDU")]
    rej = cfg.rejection if cfg.rejection == "none" else f"{cfg.rejection} {cfg.sigma_low:g} {cfg.sigma_high:g}"
    cards += F.history(f"GPU Stacker {__version__}: {n} raw lights, {cfg.transform} star registration on "
                       f"{ref.name}, drizzle x{s:g} pixfrac {cfg.pixfrac:g} (square kernel)")
    cards += F.history(f"rejection {rej}, normalization {cfg.normalization}{', rgb_equal' if cfg.rgb_equal else ''}, "
                       f"weighting {cfg.weighting}, feather {cfg.feather:g}, "
                       f"{'output normalized' if cfg.output_norm else 'not renormalized'}")
    x_lo, y_lo, x_hi, y_hi = region
    if (x_lo, y_lo, x_hi, y_hi) != (0, 0, canvas[0], canvas[1]):
        cards += F.history(f"region x {x_lo}-{x_hi}, y {y_lo}-{y_hi} of a {canvas[0]} x {canvas[1]} canvas")
    return cards
