# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jason Prater (JP Astrophotography)
"""Engine tests on synthetic data (needs an NVIDIA GPU + CuPy)."""
import json
import os
import subprocess
import sys
import threading

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from synth import make_sequence  # noqa: E402

from gpustacker.engine import fits as F  # noqa: E402
from gpustacker.engine import kernels as K  # noqa: E402
from gpustacker.engine import reference as ref  # noqa: E402
from gpustacker.engine.job import Job, JobConfig, _nonzero_minmax, _normalize_nonzero  # noqa: E402
from gpustacker.engine.masks import chamfer_distance  # noqa: E402
from gpustacker.engine.sequence import check_stackable, max_framing, parse_seq, round_to_int  # noqa: E402
from gpustacker.engine.stack import Cancelled  # noqa: E402
from gpustacker.engine.stats import bgnoise  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_fits_roundtrip(tmp_path):
    data = np.random.default_rng(1).random((3, 5, 7)).astype(np.float32)
    p = str(tmp_path / "a.fit")
    F.write_fits(p, data, [F.card("OBJECT", "M 31 'core'"), F.card("EXPTIME", 60.0), F.card("PLTSOLVD", True)])
    back, hdr = F.read_image(p)
    assert np.array_equal(back, data)
    assert hdr.get("OBJECT") == "M 31 'core'" and hdr.get("EXPTIME") == 60.0 and hdr.get("PLTSOLVD") is True
    assert os.path.getsize(p) % F.BLOCK == 0 and not os.path.exists(p + ".partial")


def test_round_to_int_matches_siril():
    assert [round_to_int(v) for v in (0.5, 1.49, -0.5, -1.5, -1.49, 2.0)] == [1, 1, -1, -2, -1, 2]


def test_output_norm_matches_siril_float_rescale():
    """-output_norm as Siril's norm_to_0_1_range: (v - min) / (max - min) in float, and the min / max search
    starts at the second pixel in FITS order, so data[0, -1, 0] is not looked at."""
    rng = np.random.default_rng(5)
    data = rng.uniform(0.02, 0.6, (3, 40, 50)).astype(np.float32)
    data[1, :3] = 0
    data[0, -1, 0] = 0.9
    lo, hi = _nonzero_minmax(data)
    assert hi == float(np.delete(data.ravel(), (data.shape[1] - 1) * data.shape[2]).max()) and hi < 0.6
    out = data.copy()
    _normalize_nonzero(out, lo, hi)
    f32 = np.float32
    assert np.array_equal(out, np.where(data != 0, (data - f32(lo)) / (f32(hi) - f32(lo)), f32(0)))
    dbl = np.where(data != 0, (data.astype(np.float64) - lo) / (hi - lo), 0).astype(np.float32)
    assert (out != dbl).any()               # float and double rounding differ: the test tells them apart


def test_parse_seq_and_framing(tmp_path):
    seq_path = make_sequence(str(tmp_path), n=6, excluded=(2,))
    seq = parse_seq(seq_path)
    assert len(seq.frames) == 6 and len(seq.included) == 5 and seq.reg_layer == 1 and seq.layers == 3
    assert not check_stackable(seq)
    cv = max_framing(seq.frames)
    xs, ys = [fr.tx for fr in seq.frames], [fr.ty for fr in seq.frames]
    assert cv.width == int(max(x + 96 for x in xs)) - int(min(xs)) + 1
    assert cv.height == int(max(y + 72 for y in ys)) - int(min(ys)) + 1
    for fr in seq.frames:
        assert cv.x0[fr.filenum] == fr.tx - min(xs) and cv.y0[fr.filenum] == fr.ty - min(ys)


def _random_stacks(rng, P, kmax):
    V = np.full((kmax, P), np.nan, np.float32)
    for p in range(P):
        n = int(rng.integers(0, kmax + 1))
        v = rng.normal(0.1, 0.01, n).astype(np.float32)
        nout = int(rng.integers(0, max(1, n // 2) + 1))
        v[rng.choice(n, size=min(n, nout), replace=False)] += rng.choice([-1, 1], min(n, nout)) * \
            rng.uniform(0.03, 0.5, min(n, nout)).astype(np.float32)
        idx = np.sort(rng.choice(kmax, size=n, replace=False))
        V[idx, p] = np.clip(v, 1e-4, None)
    return V


def _reject(S, V, kmax, P, m, siglow, sighigh, warp):
    """-> (pmin, pmax, kept, nvalid, nlo, nhi, kmean) from the warp-per-pixel or the thread-per-pixel kernel."""
    outs = [cp.empty(P, t) for t in (cp.float32, cp.float32, cp.int32, cp.int32, cp.int32, cp.int32, cp.float32)]
    K.reject_columns(S, V, kmax, P, m, siglow, sighigh, *outs, warp=warp)
    return outs


@pytest.mark.parametrize("mode", ["winsorized", "sigma", "none"])
@pytest.mark.parametrize("plain", [False, True])
@pytest.mark.parametrize("warp", [True, False])
def test_reject_kernel_matches_siril_port(mode, plain, warp):
    rng = np.random.default_rng(7)
    P, kmax = 1500, 24
    V = _random_stacks(rng, P, kmax)
    W = np.where(np.isnan(V), 0, rng.uniform(0.2, 1.0, V.shape)).astype(np.float32)
    imgw = rng.uniform(0.5, 1.5, kmax)
    Vd, Wd = cp.asarray(V), cp.asarray(W)
    S = K.sort_columns(Vd)
    m = {"none": K.REJECT_NONE, "sigma": K.REJECT_SIGMA, "winsorized": K.REJECT_WINSORIZED}[mode]
    pmin, pmax, kept, nval, nlo, nhi, kmean = _reject(S, Vd, kmax, P, m, 2.5, 2.0, warp)
    out = cp.empty(P, cp.float32)
    K.launch(K.wmean, P, Vd, Wd, cp.asarray(imgw), np.int32(kmax), np.int64(P), pmin, pmax, kept, kmean,
             np.int32(1 if plain else 0), out)
    out = out.get()
    for p in range(P):
        ok = ~np.isnan(V[:, p])
        cpu = ref.pixel(V[ok, p], (W[ok, p].astype(np.float64) * imgw[ok]), mode, 2.5, 2.0, weighted=not plain)
        assert abs(out[p] - cpu) <= 2e-7 * max(abs(cpu), 1e-6), (p, int(ok.sum()), out[p], cpu)


def _sky_stacks(rng, P, kmax):
    """Stacks like a real sub's pixel: sky + noise, a few hot / satellite / dark outliers, part coverage,
    some negative values (normalization offsets), sorted per column by the GPU."""
    V = rng.normal(0.026, 0.0015, (kmax, P)).astype(np.float32)
    V[:, : P // 3] += rng.uniform(0, 0.8, P // 3).astype(np.float32)          # stars: brighter columns
    V[:, -50:] -= 0.03                                                          # around zero
    out = rng.random(V.shape) < 0.01
    V[out] += (rng.choice([-1, 1], out.sum()) * rng.uniform(0.005, 0.5, out.sum())).astype(np.float32)
    V[rng.random(V.shape) < np.linspace(0, 0.9, P)] = np.nan                   # coverage 100% -> 10%
    V = cp.asarray(V)
    return V, K.sort_columns(V)


@pytest.mark.parametrize("kmax", [5, 37, 300, 906])
@pytest.mark.parametrize("mode", ["winsorized", "sigma", "none"])
def test_warp_reject_is_bit_identical_to_thread_reject(kmax, mode):
    """reject_w sums exactly in fixed point where reject adds doubles: the same bits on real-like data."""
    P = 4099
    V, S = _sky_stacks(np.random.default_rng(kmax), P, kmax)
    m = {"none": K.REJECT_NONE, "sigma": K.REJECT_SIGMA, "winsorized": K.REJECT_WINSORIZED}[mode]
    a = _reject(S, V, kmax, P, m, 3.0, 3.0, False)
    b = _reject(S, V, kmax, P, m, 3.0, 3.0, True)
    for name, x, y in zip(("pmin", "pmax", "kept", "nvalid", "nlo", "nhi", "kmean"), a, b):
        x, y = x.get(), y.get()
        assert np.array_equal(x.view(np.int32), y.view(np.int32)), (name, int((x != y).sum()))


def _guard_stacks(rng, P, kmax):
    """Stacks on which Siril's keep-4 guard cuts a pass short, in frame order (NaN = no data). From 9 values: a
    tight sky level, a far layer the first pass removes, leaving the second pass a budget of a few values, and a
    near layer of more outliers than that. Below 9: more outliers than the first pass may take."""
    V = np.full((kmax, P), np.nan, np.float32)
    for p in range(P):
        n = int(rng.integers(5, kmax + 1))
        v = rng.normal(0.1, 0.001, n)
        side = rng.choice([-1.0, 1.0], n)
        far = max(1, (n - 4) // 2 - int(rng.integers(1, 3))) if n >= 9 else 0
        near = min(n - far - 2, max(0, n - 2 * far - 4) + int(rng.integers(1, 4)))
        v[:far] += side[:far] * rng.uniform(0.3, 1.0, far)
        v[far:far + near] += side[far:far + near] * rng.uniform(0.008, 0.03, near)
        V[np.sort(rng.choice(kmax, n, replace=False)), p] = rng.permutation(np.clip(v, 1e-4, None))
    return V


@pytest.mark.parametrize("mode, sig", [("winsorized", 3.0), ("sigma", 1.5)])
def test_keep4_guard_follows_siril_frame_order(mode, sig):
    """Where the keep-4 guard cuts a pass short at 9+ values, the outliers Siril drops depend on the order its
    quickselect leaves the stack in, i.e. on the frame order. Two frame orders of the same values give the
    same sorted columns, and each must match the CPU port of Siril in its own order, with both kernels."""
    rng = np.random.default_rng(11)
    P, kmax = 200, 40
    VA = _guard_stacks(rng, P, kmax)
    WA = np.where(np.isnan(VA), 0, rng.uniform(0.2, 1.0, VA.shape)).astype(np.float32)
    VB, WB = VA.copy(), WA.copy()                       # the same samples, frames in another order
    for p in range(P):
        rows = np.flatnonzero(~np.isnan(VA[:, p]))
        perm = rng.permutation(rows)
        VB[rows, p], WB[rows, p] = VA[perm, p], WA[perm, p]
    imgw = np.ones(kmax)
    m = {"sigma": K.REJECT_SIGMA, "winsorized": K.REJECT_WINSORIZED}[mode]
    S = K.sort_columns(cp.asarray(VA))
    assert np.array_equal(K.sort_columns(cp.asarray(VB)).get().view(np.uint32), S.get().view(np.uint32))
    plain_means = []
    for V, W in ((VA, WA), (VB, WB)):
        Vd, Wd = cp.asarray(V), cp.asarray(W)
        res = [_reject(S, Vd, kmax, P, m, sig, sig, warp) for warp in (False, True)]
        for x, y in zip(*res):
            assert np.array_equal(x.get().view(np.int32), y.get().view(np.int32))
        pmin, pmax, kept, nval, nlo, nhi, kmean = res[0]
        gpu = []
        for plain in (0, 1):
            out = cp.empty(P, cp.float32)
            K.launch(K.wmean, P, Vd, Wd, cp.asarray(imgw), np.int32(kmax), np.int64(P), pmin, pmax, kept, kmean,
                     np.int32(plain), out)
            gpu.append(out.get())
        kept, nval, nlo, nhi = kept.get(), nval.get(), nlo.get(), nhi.get()
        cpu_plain = []
        for p in range(P):
            ok = ~np.isnan(V[:, p])
            v, w, crej = V[ok, p], W[ok, p].astype(np.float64), [0, 0]
            cpu = (ref.pixel(v, w, mode, sig, sig, weighted=True, crej=crej),
                   ref.pixel(v, w, mode, sig, sig, weighted=False))
            assert (nval[p], nlo[p], nhi[p], kept[p]) == (ok.sum(), *crej, ok.sum() - sum(crej)), p
            for out, c in zip(gpu, cpu):
                assert abs(out[p] - c) <= 2e-7 * max(abs(c), 1e-6), (p, out[p], c)
            cpu_plain.append(cpu[1])
        plain_means.append(cpu_plain)
    assert sum(a != b for a, b in zip(*plain_means)) >= 10   # the frame order decided which outliers went


@pytest.mark.parametrize("kmax", [1, 2, 3, 24, 300, 906, 2500])
def test_bitonic_sort_matches_radix_sort(kmax):
    rng = np.random.default_rng(kmax)
    V = rng.normal(0, 1, (kmax, 3001)).astype(np.float32)
    V[rng.random(V.shape) < 0.2] = np.nan
    V[rng.random(V.shape) < 0.01] = 0.0
    V[rng.random(V.shape) < 0.01] = -0.0
    a = K.sort_columns(cp.asarray(V)).get()
    b = K.sort_columns_radix(cp.asarray(V)).get()
    assert np.array_equal(a.view(np.uint32), b.view(np.uint32))


def _noise1_cpu(img):
    """Straight port of Siril/CFITSIO FnNoise1_float for checking the GPU kernel."""
    rows = []
    for row in img:
        good = row[(row != 0) & ~np.isnan(row)]
        if good.size < 3:
            continue
        d = (good[:-1] - good[1:]).astype(np.float32)
        if d.size < 2:
            continue

        def ms(x):
            x = x[x != 0].astype(np.float64)
            if x.size > 1:
                m = x.sum() / x.size
                return m, np.sqrt(np.sum(x * x) / x.size - m * m)
            return (float(x.sum()), 0.0) if x.size == 1 else (0.0, 0.0)
        mean, sd = ms(d)
        if sd > 0:
            for _ in range(3):
                keep = np.abs(d - np.float32(mean)).astype(np.float64) < 5.0 * sd
                if keep.all():
                    break
                d = d[keep]
                mean, sd = ms(d)
        rows.append(sd)
    rows = np.sort(np.array(rows))
    n = rows.size
    return 0.70710678 * ((rows[(n - 1) // 2] + rows[n // 2]) / 2.0)


def test_bgnoise_matches_cfitsio_port():
    rng = np.random.default_rng(3)
    img = rng.normal(0.1, 0.01, (60, 80)).astype(np.float32)
    img[:10, :15] = 0                                   # zero (no-data) corner
    img[30, ::7] = 0.9                                  # outliers
    assert abs(bgnoise(cp.asarray(img)) - _noise1_cpu(img)) < 1e-12


def test_chamfer_is_exact_fixed_point():
    rng = np.random.default_rng(5)
    fg = (rng.random((40, 50)) > 0.03).astype(np.uint8)
    fg[0, :] = fg[-1, :] = fg[:, 0] = fg[:, -1] = 0
    got = chamfer_distance(fg)
    # relaxation to the fixed point of the same 3x3 chamfer metric (fixed point, 16-bit)
    HV, DG = 62587, 89738
    d = np.where(fg != 0, 2**40, 0).astype(np.int64)
    while True:
        p = np.pad(d, 1, constant_values=2**40)
        n = np.minimum.reduce([p[1:-1, 1:-1], p[:-2, 1:-1] + HV, p[2:, 1:-1] + HV, p[1:-1, :-2] + HV,
                               p[1:-1, 2:] + HV, p[:-2, :-2] + DG, p[:-2, 2:] + DG, p[2:, :-2] + DG, p[2:, 2:] + DG])
        n = np.where(fg != 0, n, 0)
        if np.array_equal(n, d):
            break
        d = n
    assert np.array_equal(got, (d / 65536.0).astype(np.float32))


def _events():
    evs = []
    return evs, evs.append


def test_job_end_to_end(tmp_path):
    gains = np.linspace(0.85, 1.2, 14)
    offsets = np.linspace(-0.008, 0.012, 14)
    seq_path = make_sequence(str(tmp_path), n=14, blank=(5,), excluded=(7,), gains=gains, offsets=offsets)
    out = str(tmp_path / "stack.fit")
    evs, emit = _events()
    cfg = JobConfig(seq_path=seq_path, output_path=out, weighting="noise", feather=3, output_norm=False,
                    preview_max=64, preview_dir=str(tmp_path))
    summary = Job(cfg, emit).run()
    kinds = [e["ev"] for e in evs]
    assert "hello" in kinds and "canvas" in kinds and "rows" in kinds and kinds[-1] == "done"
    bad = [e for e in evs if e["ev"] == "frame_bad"]
    assert [e["number"] for e in bad] == [6] and "blank" in bad[0]["reason"]
    assert summary["frames"] == 12                      # 14 - excluded #8 - blank #6
    data, hdr = F.read_image(out)
    seq = parse_seq(seq_path)
    used = [fr for fr in seq.frames if fr.included and fr.number != 6]
    cv = max_framing(used)
    assert data.shape == (3, cv.height, cv.width)
    assert hdr.get("STACKCNT") == 12
    exps = [60.0 if i % 3 else 30.0 for i in range(14)]
    assert hdr.get("LIVETIME") == pytest.approx(sum(exps[fr.index] for fr in used))
    ref_fr = seq.frames[0]
    assert hdr.get("CRPIX1") == pytest.approx(48.5 + cv.x0[ref_fr.filenum])
    assert hdr.get("CRPIX2") == pytest.approx(36.5 + cv.height - cv.y0[ref_fr.filenum] - ref_fr.height)
    assert np.isfinite(data).all() and data.min() >= 0 and data.max() <= 1
    # masks were generated (no Siril cache in the synthetic folder) and the stats were cached
    assert len(os.listdir(os.path.join(seq.directory, "gpustacker", "masks"))) >= 12
    assert os.path.isfile(seq.stats_cache_path())
    # normalization brings every frame to the reference frame's level: background ~ 0.05 * g0 + o0
    bg = np.median(data[1][data[1] > 0])
    assert bg == pytest.approx(0.05 * 1.1 * gains[0] + offsets[0], abs=0.004)
    assert os.path.isfile(summary["preview"])


def test_job_cancel_leaves_no_output(tmp_path):
    seq_path = make_sequence(str(tmp_path), n=10)
    out = str(tmp_path / "stack.fit")
    cancel = threading.Event()

    def emit(ev):
        if ev["ev"] == "phase" and ev["name"] == "stack":
            cancel.set()
    cfg = JobConfig(seq_path=seq_path, output_path=out, weighting="none", strip=8, preview_dir=str(tmp_path))
    with pytest.raises(Cancelled):
        Job(cfg, emit, cancel).run()
    assert not os.path.exists(out) and not os.path.exists(out + ".partial")


def _engine(args, **kw):
    env = dict(os.environ, PYTHONPATH=ROOT, PYTHONWARNINGS="ignore")
    return subprocess.Popen([sys.executable, "-m", "gpustacker.engine", *args], cwd=ROOT, env=env,
                            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, **kw)


def test_cli_run_and_cancel(tmp_path):
    seq_path = make_sequence(str(tmp_path), n=8)
    job = tmp_path / "job.json"
    job.write_text(json.dumps({"seq_path": seq_path, "output_path": str(tmp_path / "cli.fit"), "weighting": "none",
                               "preview_dir": str(tmp_path)}))
    p = _engine(["run", str(job), "--parent-pid", str(os.getpid())])
    out, err = p.communicate(timeout=300)
    evs = [json.loads(line) for line in out.splitlines() if line.strip()]
    assert p.returncode == 0, err
    assert evs[0]["ev"] == "hello" and evs[-1]["ev"] == "done" and os.path.isfile(tmp_path / "cli.fit")

    job.write_text(json.dumps({"seq_path": seq_path, "output_path": str(tmp_path / "cli2.fit"), "weighting": "none",
                               "strip": 4, "preview_dir": str(tmp_path)}))
    flag = tmp_path / "cancel.flag"
    flag.write_text("cancel")                           # already requested: stops at the first checkpoint
    p = _engine(["run", str(job), "--cancel-file", str(flag)])
    seen = [json.loads(line)["ev"] for line in p.stdout]
    p.wait(timeout=120)
    assert p.returncode == 3 and seen[-1] == "cancelled", (seen, p.stderr.read())
    assert not os.path.exists(tmp_path / "cli2.fit")


def test_cli_stops_when_parent_exits(tmp_path):
    seq_path = make_sequence(str(tmp_path), n=8)
    job = tmp_path / "job.json"
    job.write_text(json.dumps({"seq_path": seq_path, "output_path": str(tmp_path / "o.fit"), "weighting": "none",
                               "preview_dir": str(tmp_path)}))
    dummy = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    p = _engine(["run", str(job), "--parent-pid", str(dummy.pid)])
    dummy.kill()
    dummy.wait()
    out, _ = p.communicate(timeout=120)
    assert p.returncode == 3 and json.loads(out.splitlines()[-1])["ev"] == "cancelled"
