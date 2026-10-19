# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jason Prater (JP Astrophotography)
"""GPU strip stacker over a max-framing canvas.

Per output pixel, following Siril 1.4.4 (stacking/median_and_mean.c, rejection_float.c):
  valid   raw value != 0, normalized value != 0, drizzle weight != 0
  norm    v' = (float)(v * a - off) in double, a = s_ref / s_i, off = a * o_i - o_ref
  reject  sigma or winsorized sigma clipping (or none)
  mean    weighted mean (double sums) of valid values within [min kept, max kept],
          weight = drizzle * feather * image weight; plain mean of kept values if no weights

The canvas is processed in strips of rows; for each strip and channel every overlapping frame
delivers its rows (reader threads), normalized on the GPU into one flat buffer, then gathered
into dense (frames x pixels) tiles, radix-sorted per pixel, rejected and averaged.

Two kinds of frames: StackFrame reads a registered frame Siril already wrote (r_ files, with
drizztmp weights); RawStackFrame reads only the band of raw rows a strip needs and drizzles it
(Bayer drizzle for CFA data) on the fly, so no registered intermediates are ever written.
Rows are top-down (row 0 = top).
"""
from __future__ import annotations

import concurrent.futures as cf
import itertools
import math
import os
import queue
import threading
import time
from dataclasses import dataclass, field

import cupy as cp
import numpy as np

from . import kernels as K
from .fits import RowReader
from .masks import read_mask


class Cancelled(Exception):
    pass


PROFILE = os.environ.get("GPUSTACKER_PROFILE") == "1"   # time every stage (synchronizes the GPU, a bit slower)


def _tick(res, key: str, t0: float) -> float:
    """Profiling: add the time since t0 to res.timings[key] (after the GPU catches up)."""
    if not PROFILE:
        return t0
    cp.cuda.Stream.null.synchronize()
    now = time.perf_counter()
    res.timings[key] = res.timings.get(key, 0.0) + now - t0
    return now


@dataclass
class StackFrame:
    """A registered frame on disk (Siril r_ file); x0, y0, w, h: its box on the canvas."""
    filenum: int
    path: str
    drz_path: str | None
    mask_path: str | None
    x0: int
    y0: int
    w: int
    h: int
    a: list
    off: list
    imgw: list
    reader: RowReader | None = None
    dreader: RowReader | None = None
    dz_scale: float = 1.0
    dz_zero: float = 0.0
    mask_sent: bool = False

    def open(self) -> None:
        if self.reader is None:
            self.reader = RowReader(self.path)
            if self.drz_path:
                self.dreader = RowReader(self.drz_path)
                if self.dreader.dtype.itemsize != 1:
                    raise ValueError(f"{self.drz_path}: expected 8-bit drizzle weights")
                self.dz_scale = float(self.dreader.header.get("BSCALE", 1.0))
                self.dz_zero = float(self.dreader.header.get("BZERO", 0.0))

    def close(self) -> None:
        for r in (self.reader, self.dreader):
            if r is not None:
                r.close()
        self.reader = self.dreader = None

    def read(self, c: int, ra: int, rb: int):
        """Reader thread: bytes for local rows [ra, rb) of channel c."""
        self.open()
        d = self.reader.raw_rows(c, ra, rb)
        z = self.dreader.raw_rows(c if self.dreader.channels > 1 else 0, ra, rb) if self.dreader else None
        return d, z

    @staticmethod
    def payload_bytes(p) -> int:
        return p[0].nbytes + (p[1].nbytes if p[1] is not None else 0)

    def prepare(self, p, c: int, ra: int, rb: int, fw, dst_v, dst_w) -> None:
        """GPU: decode, normalize and weight local rows [ra, rb) into dst_v / dst_w."""
        d, z = p
        nr = rb - ra
        if self.reader.dtype.itemsize == 4:
            v = K.bswap_f32(cp.asarray(d.view(np.uint32))).reshape(nr, self.w)[::-1]
        else:
            hd = self.reader.header
            v = K.bswap_i16(cp.asarray(d.view(np.uint16)), np.float32(hd.get("BSCALE", 1.0)),
                            np.float32(hd.get("BZERO", 0.0))).reshape(nr, self.w)[::-1]
        a, off = np.float64(self.a[c]), np.float64(self.off[c])
        if z is not None:
            dz = cp.asarray(z).reshape(nr, self.w)[::-1]
            K.prep_px(v, dz, fw, a, off, np.float32(self.dz_scale), np.float32(self.dz_zero), dst_v, dst_w)
        else:
            K.prep_px_nodz(v, fw, a, off, dst_v, dst_w)


@dataclass
class RawStackFrame:
    """A raw light drizzled on the fly. t: 2x3 affine from image pixels to canvas pixels;
    x0, y0, w, h: its integer box on the canvas."""
    filenum: int
    light: object
    t: np.ndarray
    lut: np.ndarray | None
    pixfrac: float
    mask_path: str | None
    x0: int
    y0: int
    w: int
    h: int
    a: list
    off: list
    imgw: list
    f: object = None
    mask_sent: bool = False
    _tinv: np.ndarray | None = None

    def open(self) -> None:
        if self.f is None:
            self.f = open(self.light.path, "rb", buffering=0)

    def close(self) -> None:
        if self.f is not None:
            self.f.close()
            self.f = None

    def band(self, ra: int, rb: int) -> tuple[int, int]:
        """Image rows whose footprints can reach canvas box rows [ra, rb)."""
        if self._tinv is None:
            m = np.vstack([np.asarray(self.t, dtype=np.float64).reshape(2, 3), [0, 0, 1]])
            self._tinv = np.linalg.inv(m)[:2]
        mi = self._tinv
        xs = np.array([self.x0, self.x0 + self.w, self.x0 + self.w, self.x0], dtype=np.float64)
        ys = np.array([self.y0 + ra, self.y0 + ra, self.y0 + rb, self.y0 + rb], dtype=np.float64)
        iy = mi[1, 0] * xs + mi[1, 1] * ys + mi[1, 2]
        pad = 1.0 + self.pixfrac
        lo = max(0, int(math.floor(iy.min() - pad)))
        hi = min(self.light.height, int(math.ceil(iy.max() + pad)))
        return lo, max(lo, hi)

    def read(self, c: int, ra: int, rb: int):
        from .lights import read_rows_raw
        self.open()
        ia, ib = self.band(ra, rb)
        plane = 0 if (self.lut is not None or self.light.channels == 1) else c
        if ib <= ia:
            return ia, ib, None
        return ia, ib, read_rows_raw(self.light, self.f, plane, ia, ib)

    def read_into(self, ia: int, ib: int, out: memoryview) -> None:
        """Reader thread (batched path): undecoded bytes of image rows [ia, ib) of plane 0, in file order,
        straight into `out` (same bytes as lights.read_rows_raw)."""
        from .lights import raw_dtype
        self.open()
        lt = self.light
        first = (lt.height - ib) if lt.bottom_up else ia
        self.f.seek(lt.data_offset + first * lt.width * raw_dtype(lt).itemsize)
        if self.f.readinto(out) != len(out):
            raise IOError(f"{lt.path}: short read")

    @staticmethod
    def payload_bytes(p) -> int:
        return 0 if p[2] is None else p[2].nbytes

    def prepare(self, p, c: int, ra: int, rb: int, fw, dst_v, dst_w) -> None:
        from .drizzle import drizzle_into, finish
        from .lights import decode_gpu
        ia, ib, buf = p
        nr = rb - ra
        num = cp.zeros((1, nr, self.w), cp.float32)
        den = cp.zeros((1, nr, self.w), cp.float32)
        if buf is not None:
            img = cp.ascontiguousarray(decode_gpu(self.light, buf, ib - ia)[0])
            t = np.asarray(self.t, dtype=np.float64).reshape(2, 3).copy()
            t[0, 2] -= self.x0
            t[1, 2] -= self.y0 + ra
            drizzle_into(img, ia, t, self.lut, self.pixfrac, num, den, only_ch=c if self.lut is not None else -1)
        val, dz = finish(num, den)
        K.prep_px(val[0], dz[0], fw, np.float64(self.a[c]), np.float64(self.off[c]), np.float32(1.0),
                  np.float32(0.0), dst_v, dst_w)


_ISZ = {8: 1, 16: 2, -32: 4}


def raw_batchable(frames) -> bool:
    """True when the batched raw path can stack these frames: raw lights with one plane each (CFA or mono)
    and the same row width, sample type, Bayer layout and pixfrac."""
    from .drizzle import site_parities
    if not frames or not all(isinstance(fr, RawStackFrame) for fr in frames):
        return False

    def key(fr):
        lut = None if fr.lut is None else tuple(int(v) for v in np.asarray(fr.lut).ravel())
        return fr.light.width, fr.light.bitpix, lut, fr.pixfrac

    k0 = key(frames[0])
    if k0[2] is not None and any(site_parities(k0[2], ch) is None for ch in range(3)):
        return False
    return all((fr.lut is not None or fr.light.channels == 1) and key(fr) == k0 for fr in frames)


def uses_batched_raw(frames) -> bool:
    """Whether stack_region will take the batched raw path for these frames (GPUSTACKER_PER_FRAME=1 forces
    the per-frame path, for comparisons)."""
    return os.environ.get("GPUSTACKER_PER_FRAME") != "1" and raw_batchable(frames)


def raw_group_budget(limit: float) -> int:
    """Raw-row bytes per batched group, from the GPU memory limit (also the size of each host buffer)."""
    return int(min(1.5e9, max(128 * 2**20, 0.15 * limit)))


@dataclass
class _Group:
    """Consecutive frames of one strip whose raw row bands go to the GPU together."""
    frames: list
    j0: int                    # index of frames[0] in the strip's frame list
    bands: list                # per frame: (ra, rb, ia, ib, nbytes) box rows, image rows, bytes
    roff: np.ndarray           # byte offset of each band in the group buffer
    nbytes: int                # buffer size (16-byte aligned bands)


def _plan_groups(sfr: list, r0: int, r1: int, budget: int) -> list:
    groups, cur, bands, offs, total, j0 = [], [], [], [], 0, 0
    for j, fr in enumerate(sfr):
        ra, rb = max(r0, fr.y0) - fr.y0, min(r1, fr.y0 + fr.h) - fr.y0
        ia, ib = fr.band(ra, rb)
        nb = max(0, ib - ia) * fr.light.width * _ISZ[fr.light.bitpix]
        if cur and (total + nb > budget or len(cur) >= 65535):   # 65535: CUDA grid y limit
            groups.append(_Group(cur, j0, bands, np.array(offs, np.int64), total))
            cur, bands, offs, total, j0 = [], [], [], 0, j
        cur.append(fr)
        bands.append((ra, rb, ia, ib, nb))
        offs.append(total)
        total += -(-nb // 16) * 16
    if cur:
        groups.append(_Group(cur, j0, bands, np.array(offs, np.int64), total))
    return groups


def raw_region_bytes(frames: list[RawStackFrame], y_lo: int, y_hi: int, strip: int, channels: int,
                     group_budget: int | None = None) -> int:
    """Bytes the stack pass will read for raw frames. Per-frame path: one band per strip and channel.
    Batched path (group_budget given): once per strip, or once per channel when a strip needs several groups."""
    total = 0
    for r0 in range(y_lo, y_hi, strip):
        r1 = min(r0 + strip, y_hi)
        sfr = [fr for fr in frames if fr.y0 < r1 and fr.y0 + fr.h > r0]
        nb = 0
        for fr in sfr:
            ra, rb = max(r0, fr.y0) - fr.y0, min(r1, fr.y0 + fr.h) - fr.y0
            ia, ib = fr.band(ra, rb)
            nb += max(0, ib - ia) * fr.light.width * _ISZ[fr.light.bitpix]
        if group_budget is None:
            total += nb * channels
        else:
            total += nb * (1 if len(_plan_groups(sfr, r0, r1, group_budget)) <= 1 else channels)
    return total


@dataclass
class StackOptions:
    channels: int
    rejection: int = K.REJECT_WINSORIZED
    sigma_low: float = 3.0
    sigma_high: float = 3.0
    feather: float = 0.0
    plain: bool = False                  # no drizzle, feather or weighting: plain mean of kept values
    strip: int = 64
    tile: int = 1024
    workers: int = 6
    queue_items: int = 384
    want_coverage: bool = False
    debug_patch: tuple | None = None     # (x, y, size) canvas patch: keep its per-pixel stacks (tests)
    batch_raw: bool = True               # raw lights: one drizzle launch per strip (GPUSTACKER_PER_FRAME=1: off)
    raw_budget: int = 512 * 2**20        # bytes of raw rows per batched group (GPU and host buffers)


@dataclass
class StackResult:
    data: np.ndarray                     # (C, H, W) float32, top-down, region only
    coverage: np.ndarray | None          # (H, W) uint16 valid-frame count (reference channel)
    rejected_low: list = field(default_factory=list)    # per channel, fraction of valid samples
    rejected_high: list = field(default_factory=list)
    bytes_read: int = 0
    t_wait: float = 0.0
    t_prep: float = 0.0
    t_tiles: float = 0.0
    timings: dict = field(default_factory=dict)  # per stage, with GPUSTACKER_PROFILE=1
    debug: list = field(default_factory=list)   # (channel, V, W, imgw) for StackOptions.debug_patch


def region_bytes(frames: list[StackFrame], y_lo: int, y_hi: int, channels: int, bpp: int, dz: bool) -> int:
    total = 0
    for fr in frames:
        rows = max(0, min(y_hi, fr.y0 + fr.h) - max(y_lo, fr.y0))
        total += rows * fr.w * channels * (bpp + (1 if dz else 0))
    return total


def _bounded_map(pool, fn, items, window, stop):
    """pool.map that keeps at most `window` results in flight (host RAM is scarce)."""
    items = iter(items)
    pending = [pool.submit(fn, it) for it in itertools.islice(items, window)]
    while pending:
        res = pending.pop(0).result()
        if stop.is_set():
            for f in pending:
                f.cancel()
            return
        nxt = next(items, None)
        if nxt is not None:
            pending.append(pool.submit(fn, nxt))
        yield res


def _put(q, item, stop) -> bool:
    while not stop.is_set():
        try:
            q.put(item, timeout=0.2)
            return True
        except queue.Full:
            continue
    return False


def _produce(jobs, q, pool, window, feather, stop) -> None:
    """Reader thread: for each (strip, channel) job, queue every frame's rows, then None."""
    try:
        for r0, r1, c, sfr in jobs:
            def job(fr, r0=r0, r1=r1, c=c):
                ra, rb = max(r0, fr.y0) - fr.y0, min(r1, fr.y0 + fr.h) - fr.y0
                payload = fr.read(c, ra, rb)
                m = None
                if feather > 0 and not fr.mask_sent:   # each frame's mask travels once
                    fr.mask_sent = True
                    m = read_mask(fr.mask_path)
                return fr, ra, rb, payload, m

            for item in _bounded_map(pool, job, sfr, window, stop):
                if not _put(q, item, stop):
                    return
            if not _put(q, None, stop):
                return
    except BaseException as e:               # surface reader errors in the main thread
        _put(q, e, stop)


def _produce_raw(units: list, q, pool, feather: float, stop) -> None:
    """Reader thread (batched raw path): for each group, every frame's raw rows read in parallel straight
    into one host buffer; feather masks travel with the first group that has the frame."""
    loaded: set[int] = set()
    try:
        for g in units:
            buf = np.empty(max(16, g.nbytes), np.uint8)
            mv = memoryview(buf)
            need = []
            for fr in g.frames:
                need.append(feather > 0 and fr.filenum not in loaded)
                loaded.add(fr.filenum)

            def read(k, g=g, mv=mv, need=need):
                fr = g.frames[k]
                nb = g.bands[k][4]
                if nb:
                    o = int(g.roff[k])
                    fr.read_into(g.bands[k][2], g.bands[k][3], mv[o:o + nb])
                return fr.filenum, (read_mask(fr.mask_path) if need[k] else None)

            mlist = list(pool.map(read, range(len(g.frames))))
            if stop.is_set() or not _put(q, (g, buf, mlist), stop):
                return
    except BaseException as e:               # surface reader errors in the main thread
        _put(q, e, stop)


def stack_region(frames: list[StackFrame], region: tuple[int, int, int, int], opts: StackOptions,
                 cancel: threading.Event, on_progress=None, on_strip=None) -> StackResult:
    """Stack canvas region (x_lo, y_lo, x_hi, y_hi), top-down.

    on_progress(bytes_done, rows_done, frames_in_strip) after every strip channel;
    on_strip(out, row_lo, row_hi) after every finished strip (region-relative rows)."""
    x_lo, y_lo, x_hi, y_hi = region
    C = opts.channels
    W, H = x_hi - x_lo, y_hi - y_lo
    frames = [fr for fr in frames if fr.y0 < y_hi and fr.y0 + fr.h > y_lo and fr.x0 < x_hi and fr.x0 + fr.w > x_lo]
    out = np.zeros((C, H, W), dtype=np.float32)
    cov = np.zeros((H, W), dtype=np.uint16) if opts.want_coverage else None
    cov_ch = 1 if C == 3 else 0
    res = StackResult(out, cov)
    if not frames:
        res.rejected_low = res.rejected_high = [0.0] * C
        return res

    ramp = cp.asarray(K.ramp_table())
    feather = float(opts.feather)
    invdist = np.float32(np.float32(1.0) / np.float32(feather)) if feather > 0 else np.float32(0)
    rej_lo = [cp.zeros((), cp.int64) for _ in range(C)]
    rej_hi = [cp.zeros((), cp.int64) for _ in range(C)]
    nval = [cp.zeros((), cp.int64) for _ in range(C)]

    def layout(sfr, r0, r1):
        """Each frame's block of the strip in the flat buffers: rows, width, canvas x0 and first row, offsets."""
        nrs = np.array([min(r1, fr.y0 + fr.h) - max(r0, fr.y0) for fr in sfr], dtype=np.int64)
        wid = np.array([fr.w for fr in sfr], dtype=np.int64)
        x0s = np.array([fr.x0 for fr in sfr], dtype=np.int64)
        cys = np.array([max(r0, fr.y0) for fr in sfr], dtype=np.int64)  # canvas row of first stored row
        offs = np.concatenate(([0], np.cumsum(nrs * wid)))
        return nrs, wid, x0s, cys, offs

    def tiles(c, r0, r1, sfr, vals, wts, nrs, wid, x0s, cys, offs):
        """Per x-tile: gather dense (K, R*C), sort columns, reject, weighted mean, into out[c]."""
        R = r1 - r0
        imgw_all = np.array([fr.imgw[c] for fr in sfr], dtype=np.float64)
        t = time.perf_counter()
        for c0 in range(x_lo, x_hi, opts.tile):
            c1 = min(c0 + opts.tile, x_hi)
            Cw, P = c1 - c0, R * (c1 - c0)
            sel = np.nonzero((x0s < c1) & (x0s + wid > c0))[0]
            Kn = len(sel)
            if Kn == 0:
                continue
            V2 = cp.empty((Kn, P), dtype=cp.float32)
            W2 = cp.empty((Kn, P), dtype=cp.float32)
            meta = [cp.asarray(offs[sel]), cp.asarray(nrs[sel].astype(np.int32)),
                    cp.asarray(wid[sel].astype(np.int32)), cp.asarray(x0s[sel].astype(np.int32)),
                    cp.asarray(cys[sel].astype(np.int32))]
            tt = _tick(res, "tiles: setup", time.perf_counter())
            K.launch(K.gather_tile, Kn * P, vals, wts, *meta, np.int32(Kn), np.int32(R), np.int32(Cw),
                     np.int32(r0), np.int32(c0), V2, W2)
            tt = _tick(res, "tiles: gather", tt)
            S = K.sort_columns(V2)
            tt = _tick(res, "tiles: sort", tt)
            pmin = cp.empty(P, cp.float32)
            pmax = cp.empty(P, cp.float32)
            kept = cp.empty(P, cp.int32)
            nvalid = cp.empty(P, cp.int32)
            nlo = cp.empty(P, cp.int32)
            nhi = cp.empty(P, cp.int32)
            kmean = cp.empty(P, cp.float32)
            K.reject_columns(S, V2, Kn, P, opts.rejection, opts.sigma_low, opts.sigma_high,
                             pmin, pmax, kept, nvalid, nlo, nhi, kmean)
            tt = _tick(res, "tiles: reject", tt)
            del S
            outp = cp.empty(P, cp.float32)
            K.launch(K.wmean, P, V2, W2, cp.asarray(imgw_all[sel]), np.int32(Kn), np.int64(P),
                     pmin, pmax, kept, kmean, np.int32(1 if opts.plain else 0), outp)
            tt = _tick(res, "tiles: weighted mean", tt)
            rej_lo[c] += nlo.sum(dtype=cp.int64)
            rej_hi[c] += nhi.sum(dtype=cp.int64)
            nval[c] += nvalid.sum(dtype=cp.int64)
            out[c, r0 - y_lo:r1 - y_lo, c0 - x_lo:c1 - x_lo] = cp.asnumpy(outp.reshape(R, Cw))
            if cov is not None and c == cov_ch:
                cov[r0 - y_lo:r1 - y_lo, c0 - x_lo:c1 - x_lo] = cp.asnumpy(
                    cp.minimum(nvalid, 65535).astype(cp.uint16).reshape(R, Cw))
            if opts.debug_patch:
                px, py, ps = opts.debug_patch
                if r0 <= py and py + ps <= r1 and c0 <= px and px + ps <= c1:
                    sl = (slice(None), slice(py - r0, py - r0 + ps), slice(px - c0, px - c0 + ps))
                    res.debug.append((c, cp.asnumpy(V2.reshape(Kn, R, Cw)[sl]),
                                      cp.asnumpy(W2.reshape(Kn, R, Cw)[sl]), imgw_all[sel].copy()))
            _tick(res, "tiles: results to host", tt)
            del V2, W2
        cp.cuda.Stream.null.synchronize()
        res.t_tiles += time.perf_counter() - t

    mempool = cp.get_default_memory_pool()
    masks: dict[int, cp.ndarray] = {}
    batched = opts.batch_raw and uses_batched_raw(frames)
    stop = threading.Event()
    pool = cf.ThreadPoolExecutor(max_workers=opts.workers)
    q: queue.Queue
    vals = wts = None
    if batched:
        strips, units = [], []
        for r0 in range(y_lo, y_hi, opts.strip):
            r1 = min(r0 + opts.strip, y_hi)
            sfr = [fr for fr in frames if fr.y0 < r1 and fr.y0 + fr.h > r0]
            groups = _plan_groups(sfr, r0, r1, opts.raw_budget)
            strips.append((r0, r1, sfr, groups))
            units += groups if len(groups) <= 1 else groups * C      # several groups: re-read per channel
        q = queue.Queue(maxsize=1)
        producer = threading.Thread(target=_produce_raw, args=(units, q, pool, feather, stop), daemon=True)
    else:
        jobs = []
        for r0 in range(y_lo, y_hi, opts.strip):
            r1 = min(r0 + opts.strip, y_hi)
            sfr = [fr for fr in frames if fr.y0 < r1 and fr.y0 + fr.h > r0]
            jobs += [(r0, r1, c, sfr) for c in range(C)]
        q = queue.Queue(maxsize=opts.queue_items)
        producer = threading.Thread(target=_produce, args=(jobs, q, pool, 4 * opts.workers, feather, stop),
                                    daemon=True)
    producer.start()

    def get_item():
        while True:
            if cancel.is_set():
                raise Cancelled()
            t = time.perf_counter()
            try:
                item = q.get(timeout=0.25)
            except queue.Empty:
                res.t_wait += time.perf_counter() - t
                continue
            res.t_wait += time.perf_counter() - t
            if isinstance(item, BaseException):
                raise item
            return item

    def finish_strip(sfr, r0, r1):
        for fr in sfr:                          # strips go top-down: frames ending here are done
            if fr.y0 + fr.h <= r1:
                fr.close()
                masks.pop(fr.filenum, None)
        if on_strip:
            on_strip(out, r0 - y_lo, r1 - y_lo)

    try:
        if batched:
            _stack_strips_batched(strips, C, frames[0], feather, invdist, ramp, masks, res, get_item, layout,
                                  tiles, finish_strip, on_progress, y_lo, mempool, cancel)
        else:
            strip_of_buffers = None
            for r0, r1, c, sfr in jobs:
                nrs, wid, x0s, cys, offs = layout(sfr, r0, r1)
                if strip_of_buffers != (r0, r1):     # one pair of strip buffers, reused by every channel
                    vals = wts = None
                    mempool.free_all_blocks()
                    vals = cp.empty(int(offs[-1]), dtype=cp.float32)
                    wts = cp.empty(int(offs[-1]), dtype=cp.float32)
                    strip_of_buffers = (r0, r1)
                pos = {fr.filenum: j for j, fr in enumerate(sfr)}
                while True:
                    item = get_item()
                    if item is None:
                        break
                    t = time.perf_counter()
                    tp = _tick(res, "wait for GPU before frame", t)
                    fr, ra, rb, payload, m = item
                    nr, j = rb - ra, pos[fr.filenum]
                    res.bytes_read += fr.payload_bytes(payload)
                    if m is not None:
                        masks[fr.filenum] = cp.asarray(m)
                    if feather > 0:
                        mk = masks[fr.filenum]
                        mh, mw = mk.shape
                        fw = cp.empty((nr, fr.w), dtype=cp.float32)
                        K.feather_w(mk, ramp, np.int32(mh), np.int32(mw), np.int32(fr.w), np.int32(ra),
                                    np.float32(mh / fr.h), np.float32(mw / fr.w), np.float32(feather), invdist, fw)
                    else:
                        fw = cp.ones((nr, fr.w), dtype=cp.float32)
                    tp = _tick(res, "prep: feather weights", tp)
                    o0, o1 = int(offs[j]), int(offs[j + 1])
                    dst_v, dst_w = vals[o0:o1].reshape(nr, fr.w), wts[o0:o1].reshape(nr, fr.w)
                    fr.prepare(payload, c, ra, rb, fw, dst_v, dst_w)
                    _tick(res, "prep: decode + drizzle + normalize", tp)
                    res.t_prep += time.perf_counter() - t
                tiles(c, r0, r1, sfr, vals, wts, nrs, wid, x0s, cys, offs)
                if on_progress:
                    on_progress(res.bytes_read, r1 - y_lo if c == C - 1 else r0 - y_lo, len(sfr))
                if c == C - 1:
                    finish_strip(sfr, r0, r1)
        producer.join()
    finally:
        stop.set()
        while True:                              # unblock the producer if it is waiting to put
            try:
                q.get_nowait()
            except queue.Empty:
                break
        producer.join(timeout=10)
        pool.shutdown(wait=True, cancel_futures=True)
        for fr in frames:
            fr.close()
        masks.clear()
        vals = wts = None
        mempool.free_all_blocks()

    for c in range(C):
        n = max(1, int(nval[c].get()))
        res.rejected_low.append(int(rej_lo[c].get()) / n)
        res.rejected_high.append(int(rej_hi[c].get()) / n)
    return res


def _stack_strips_batched(strips, C, f0, feather, invdist, ramp, masks, res, get_item, layout, tiles,
                          finish_strip, on_progress, y_lo, mempool, cancel) -> None:
    """The batched raw path: per strip, the frames' raw rows go to the GPU together, and each colour is
    drizzled, finished, feathered and normalized for all frames in two launches before the tiles."""
    from .drizzle import _fn as drizzle_fn, batch_geometry, site_parities
    k_drizzle, k_finish = drizzle_fn("drizzle_batch"), drizzle_fn("finish_prep_batch")
    width, bitpix = f0.light.width, f0.light.bitpix
    sites = [site_parities(f0.lut, c) for c in range(C)] if f0.lut is not None else [(-2, -2)] * C
    none_u64, none_i32, none_f32 = cp.zeros(1, cp.uint64), cp.zeros(1, cp.int32), cp.zeros(1, cp.float32)
    raw = cp.empty(16, cp.uint8)
    state = {"vals": None, "wts": None}

    def upload(g, offs):
        """Next group from the reader: its raw rows on the GPU, masks stored, per-frame tables built."""
        nonlocal raw
        unit, buf, mlist = get_item()
        if unit is not g:
            raise RuntimeError("stack reader out of order")
        t = time.perf_counter()
        for fnum, m in mlist:
            if m is not None:
                masks[fnum] = cp.asarray(m)
        if raw.size < buf.size:
            raw = None
            raw = cp.empty(buf.size, cp.uint8)
        raw[:buf.size].set(buf)
        res.bytes_read += sum(b[4] for b in g.bands)
        n = len(g.frames)
        ra = np.array([b[0] for b in g.bands], np.int32)
        bnr = np.array([b[3] - b[2] for b in g.bands], np.int32)
        T = np.empty((n, 6), np.float64)
        for k, fr in enumerate(g.frames):
            tk = np.asarray(fr.t, dtype=np.float64).reshape(2, 3).copy()
            tk[0, 2] -= fr.x0
            tk[1, 2] -= fr.y0 + int(ra[k])
            T[k] = tk.ravel()
        tab = dict(n=n, base=int(offs[g.j0]), end=int(offs[g.j0 + n]), maxrows=int(bnr.max()) if n else 0,
                   roff=cp.asarray(g.roff), by0=cp.asarray(np.array([b[2] for b in g.bands], np.int32)),
                   bnr=cp.asarray(bnr), bflip=cp.asarray(np.array([int(fr.light.bottom_up) for fr in g.frames],
                                                                  np.int32)),
                   bscale=cp.asarray(np.array([fr.light.bscale for fr in g.frames], np.float32)),
                   bzero=cp.asarray(np.array([fr.light.bzero for fr in g.frames], np.float32)),
                   G=cp.asarray(batch_geometry(T, f0.pixfrac)),
                   soff=cp.asarray(offs[g.j0:g.j0 + n].astype(np.int64)),
                   sw=cp.asarray(np.array([fr.w for fr in g.frames], np.int32)),
                   snr=cp.asarray(np.array([b[1] - b[0] for b in g.bands], np.int32)), sra=cp.asarray(ra))
        if feather > 0:
            mk = [masks[fr.filenum] for fr in g.frames]
            tab.update(mptr=cp.asarray(np.array([m.data.ptr for m in mk], np.uint64)),
                       mh=cp.asarray(np.array([m.shape[0] for m in mk], np.int32)),
                       mw=cp.asarray(np.array([m.shape[1] for m in mk], np.int32)),
                       fy=cp.asarray(np.array([np.float32(m.shape[0] / fr.h) for m, fr in zip(mk, g.frames)],
                                              np.float32)),
                       fx=cp.asarray(np.array([np.float32(m.shape[1] / fr.w) for m, fr in zip(mk, g.frames)],
                                              np.float32)))
        _tick(res, "prep: upload raw rows", t)
        return tab

    def run(g, tab, c):
        """Drizzle colour c of this group's frames into their (zeroed) slots, then finish, feather and
        normalize them in place."""
        n, total = tab["n"], tab["end"] - tab["base"]
        vals, wts = state["vals"], state["wts"]
        t = time.perf_counter()
        cp0, cp1 = sites[c]
        if cp0 == -2:                                   # no CFA: every pixel
            nthreads = tab["maxrows"] * width
        elif cp0 < 0 and cp1 < 0:                       # colour absent from the pattern
            nthreads = 0
        else:                                           # that colour's sites: half of each row it occupies
            rows = tab["maxrows"] if (cp0 >= 0 and cp1 >= 0) else (tab["maxrows"] + 1) // 2
            nthreads = rows * ((width + 1) // 2)
        if nthreads > 0:
            k_drizzle(((nthreads + 255) // 256, n), (256,), (
                raw, tab["roff"], tab["by0"], tab["bnr"], tab["bflip"], tab["bscale"], tab["bzero"],
                np.int32(width), np.int32(bitpix), np.int32(cp0), np.int32(cp1), tab["G"],
                tab["soff"], tab["sw"], tab["snr"], vals, wts))
        t = _tick(res, f"prep: drizzle (batched) ch{c}", t)
        if total > 0:
            A = cp.asarray(np.array([fr.a[c] for fr in g.frames], np.float64))
            OFF = cp.asarray(np.array([fr.off[c] for fr in g.frames], np.float64))
            k_finish(((total + 255) // 256,), (256,), (
                vals, wts, np.int64(tab["base"]), np.int64(total), tab["soff"], np.int32(n), tab["sw"], tab["sra"],
                A, OFF, np.int32(1 if feather > 0 else 0), tab.get("mptr", none_u64), tab.get("mh", none_i32),
                tab.get("mw", none_i32), tab.get("fy", none_f32), tab.get("fx", none_f32), ramp,
                np.float32(feather), invdist))
        _tick(res, "prep: finish + feather + normalize (batched)", t)

    try:
        for r0, r1, sfr, groups in strips:
            nrs, wid, x0s, cys, offs = layout(sfr, r0, r1)
            state["vals"] = state["wts"] = None
            mempool.free_all_blocks()
            state["vals"] = cp.empty(max(1, int(offs[-1])), dtype=cp.float32)
            state["wts"] = cp.empty(max(1, int(offs[-1])), dtype=cp.float32)
            tab0 = upload(groups[0], offs) if len(groups) == 1 else None
            for c in range(C):
                if cancel.is_set():
                    raise Cancelled()
                t = time.perf_counter()
                vals, wts = state["vals"], state["wts"]
                vals.data.memset_async(0, vals.nbytes)          # drizzle sums accumulate in place
                wts.data.memset_async(0, wts.nbytes)
                if tab0 is not None:
                    run(groups[0], tab0, c)
                else:
                    for g in groups:
                        run(g, upload(g, offs), c)
                res.t_prep += time.perf_counter() - t
                tiles(c, r0, r1, sfr, vals, wts, nrs, wid, x0s, cys, offs)
                if on_progress:
                    on_progress(res.bytes_read, r1 - y_lo if c == C - 1 else r0 - y_lo, len(sfr))
            finish_strip(sfr, r0, r1)
    finally:
        state["vals"] = state["wts"] = None
        raw = None
