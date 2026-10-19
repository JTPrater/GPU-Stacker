# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jason Prater (JP Astrophotography)
"""Raw-lights pipeline tests on synthetic Seestar-style subs with known geometry (needs an NVIDIA GPU)."""
import os

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from synth_raw import COLOR, apply, make_lights  # noqa: E402

from gpustacker.engine import fits as F  # noqa: E402
from gpustacker.engine.drizzle import warp_frame  # noqa: E402
from gpustacker.engine.job import Job, JobConfig, JobError  # noqa: E402
from gpustacker.engine.lights import cfa_lut, decode_gpu, read_frame_raw, read_light, scan_lights  # noqa: E402
from gpustacker.engine.rawcache import Registration, cache_dir, raw_canvas  # noqa: E402
from gpustacker.engine.register import register_all  # noqa: E402
from gpustacker.engine.stars import detect, luminance  # noqa: E402

CORNERS = (np.array([0.0, 320.0, 320.0, 0.0]), np.array([0.0, 0.0, 240.0, 240.0]))


@pytest.fixture(scope="module")
def lights(tmp_path_factory):
    """12 dithered, slightly rotated subs; #5 is clouded out, #8 has a satellite trail."""
    return make_lights(str(tmp_path_factory.mktemp("raw")), n=12, cloudy=(4,), trail=7)


def _analyze(truth):
    stars, metrics = {}, {}
    for lt in scan_lights(truth.lights_dir):
        planes = decode_gpu(lt, read_frame_raw(lt), lt.height, lt.channels)
        stars[lt.index], metrics[lt.index] = detect(luminance(planes, cfa=True))
    return stars, metrics


def _corner_error(m, truth_m) -> float:
    a = np.array(apply(m, *CORNERS))
    b = np.array(apply(truth_m, *CORNERS))
    return float(np.abs(a - b).max())


def _isolated(truth, min_peak, margin=25):
    """Indices of bright stars well inside frame 0 with no comparable neighbour within 12 px."""
    t = truth.stars
    out = []
    for k, (x, y, p) in enumerate(t):
        if p < min_peak or not (margin < x < truth.width - margin and margin < y < truth.height - margin):
            continue
        d2 = (t[:, 0] - x) ** 2 + (t[:, 1] - y) ** 2
        d2[k] = np.inf
        if not (d2[t[:, 2] > 0.1 * p] < 144).any():
            out.append(k)
    return out


def _canvas_of_reference(truth):
    reg = Registration(cache_dir(truth.lights_dir))
    cw, ch, tc, _ = raw_canvas({n: reg.transform(n) for n in reg.frames}, truth.width, truth.height, 1.0, 1.0)
    return cw, ch, tc[truth.names[0]]


def _photometry(data, X, Y, r=4):
    ix, iy = int(X), int(Y)
    side = np.concatenate([data[:, iy - 12:iy + 13, ix - 12:ix - 8].reshape(3, -1),
                           data[:, iy - 12:iy + 13, ix + 9:ix + 13].reshape(3, -1)], 1)
    bg = np.median(side, axis=1)
    return (data[:, iy - r:iy + r + 1, ix - r:ix + r + 1] - bg[:, None, None]).sum(axis=(1, 2))


def _trail_excess(data, truth, m0):
    """Median (stack - stack 7 px above/below) along where frame #8's trail lands, per channel."""
    k, a, b = truth.trail
    out = []
    for s in np.linspace(0.05, 0.95, 200):
        x, y = apply(m0, *apply(truth.transforms[k], a[0] + s * (b[0] - a[0]), a[1] + s * (b[1] - a[1])))
        ix, iy = int(x), int(y)
        if 7 <= iy < data.shape[1] - 7 and 0 <= ix < data.shape[2]:
            out.append(data[:, iy, ix] - 0.5 * (data[:, iy - 7, ix] + data[:, iy + 7, ix]))
    return np.median(np.array(out), axis=0)


def _run(truth, tmp_path, **kw):
    evs = []
    cfg = dict(lights_dir=truth.lights_dir, reference_file=truth.names[0], output_norm=False, preview_max=64,
               preview_dir=str(tmp_path))
    cfg.update(kw)
    summary = Job(JobConfig(**cfg), evs.append).run()
    return summary, evs


# ----------------------------------------------------------------------------- files and CFA

def test_cfa_lut_follows_bayerpat_and_offsets():
    assert cfa_lut("GRBG").tolist() == [[1, 0], [2, 1]]
    assert cfa_lut("RGGB").tolist() == [[0, 1], [1, 2]]
    assert cfa_lut("RGGB", xoff=1).tolist() == cfa_lut("GRBG").tolist()
    assert cfa_lut("RGGB", yoff=1).tolist() == cfa_lut("GBRG").tolist()
    assert cfa_lut("") is None and cfa_lut("XYZW") is None


def test_top_down_and_bottom_up_files_decode_to_the_same_image(tmp_path):
    a = make_lights(str(tmp_path / "a"), n=1, seed=3)
    b = make_lights(str(tmp_path / "b"), n=1, seed=3, bottom_up=True)
    la, lb = read_light(os.path.join(a.lights_dir, a.names[0])), read_light(os.path.join(b.lights_dir, b.names[0]))
    assert (la.bottom_up, lb.bottom_up, la.bayer, la.bzero) == (False, True, "GRBG", 32768.0)
    pa = decode_gpu(la, read_frame_raw(la), la.height).get()
    pb = decode_gpu(lb, read_frame_raw(lb), lb.height).get()
    assert np.array_equal(pa, pb)
    assert pa.shape == (1, 240, 320) and abs(float(np.median(pa)) - a.sky) < 0.002


def test_scan_sorts_by_time_and_flags_odd_files(tmp_path):
    truth = make_lights(str(tmp_path), n=4)
    with open(os.path.join(truth.lights_dir, "Light_broken.fit"), "wb") as f:
        f.write(b"not a fits file")
    ls = scan_lights(truth.lights_dir)
    good = [lt.name for lt in ls if not lt.error]
    assert good == truth.names and [lt.name for lt in ls if lt.error] == ["Light_broken.fit"]


# ----------------------------------------------------------------------------- stars and registration

def test_star_detection_finds_the_stars(lights):
    stars, metrics = _analyze(lights)
    s = stars[0]
    found = 0
    errs = []
    iso = _isolated(lights, 0.08)
    for k in iso:
        x, y, _ = lights.stars[k]
        d = np.hypot(s.x - x, s.y - y)
        if d.min() < 1.0:
            found += 1
            errs.append(d.min())
    assert found >= 0.9 * len(iso) and len(iso) >= 15
    assert np.median(errs) < 0.35
    assert metrics[0].fwhm == pytest.approx(3.2, rel=0.2)      # the synthetic PSF, seen through the 2x2 binning
    assert metrics[0].background == pytest.approx(lights.sky, abs=0.002)
    assert metrics[4].nstars < 15                              # the clouded-out frame


def test_registration_recovers_the_known_transforms(lights):
    stars, metrics = _analyze(lights)
    good = {i: s for i, s in stars.items() if metrics[i].nstars >= 15}
    res = register_all(good, 0, "affine")
    assert sorted(res.transforms) == sorted(good) and not res.failed
    for i, t in res.transforms.items():
        assert _corner_error(t.m, lights.transforms[i]) < 0.35, i
        assert t.rotation_deg == pytest.approx(np.degrees(np.arctan2(lights.transforms[i][1, 0],
                                                                     lights.transforms[i][0, 0])), abs=0.05)


def test_mosaic_panels_register_through_their_neighbours(tmp_path):
    # three panels in a row: the last one barely overlaps the reference panel
    offs = [(0, 0), (3, 2), (-2, 1), (154, -3), (150, 2), (157, 0), (304, 1), (300, -2), (306, 3)]
    rots = [0.0, 0.2, -0.2, 0.1, -0.1, 0.3, 0.0, -0.3, 0.2]
    truth = make_lights(str(tmp_path), n=9, offsets=offs, rotations=rots, seed=5)
    stars, _ = _analyze(truth)
    res = register_all(stars, 0, "affine")
    assert sorted(res.transforms) == list(range(9)) and not res.failed
    assert any(res.info[i][2].startswith("via") for i in (6, 7, 8))
    for i, t in res.transforms.items():
        assert _corner_error(t.m, truth.transforms[i]) < 0.6, i


# ----------------------------------------------------------------------------- drizzle

def test_bayer_drizzle_identity_puts_each_pixel_in_its_colour_plane():
    rng = np.random.default_rng(0)
    img = rng.uniform(0.1, 0.9, (8, 10)).astype(np.float32)
    lut = cfa_lut("GRBG")
    val, dz = warp_frame(cp.asarray(img[None]), np.array([[1.0, 0, 0], [0, 1.0, 0]]), lut, 1.0, 10, 8)
    val, dz = val.get(), dz.get()
    yy, xx = np.mgrid[0:8, 0:10]
    own = lut[yy % 2, xx % 2]
    for c in range(3):
        assert np.allclose(val[c][own == c], img[own == c], atol=1e-6)
        assert (dz[c][own == c] == 255).all() and (dz[c][own != c] == 0).all()


def test_drizzle_keeps_surface_brightness_and_drops_area_with_pixfrac():
    h, w = 40, 50
    img = np.full((1, h, w), 0.37, np.float32)
    th = np.radians(7.0)
    t = np.array([[np.cos(th), -np.sin(th), 9.3], [np.sin(th), np.cos(th), 4.6]])
    for pixfrac in (1.0, 0.6):
        val, dz = warp_frame(cp.asarray(img), t, cfa_lut("RGGB"), pixfrac, 70, 60)
        val, dz = val.get(), dz.get()
        assert np.allclose(val[dz > 0], 0.37, atol=2e-6)
    # weights: each colour's input area lands in its own plane, shrunk by pixfrac^2 (frame fully inside)
    from gpustacker.engine.drizzle import drizzle_into
    for pixfrac in (1.0, 0.6):
        num = cp.zeros((3, 60, 70), cp.float32)
        den = cp.zeros((3, 60, 70), cp.float32)
        drizzle_into(cp.asarray(img[0]), 0, t, cfa_lut("RGGB"), pixfrac, num, den)
        area = den.sum(axis=(1, 2)).get()
        assert area == pytest.approx(np.array([0.25, 0.5, 0.25]) * h * w * pixfrac ** 2, rel=1e-4)


@pytest.mark.parametrize("pattern", ["GRBG", "RGGB"])
def test_batched_double_float_drizzle_matches_the_fp64_drizzle(pattern):
    """drizzle_batch (double-float corners, threads on one colour's sites, raw big-endian input) against
    drizzle_into (FP64) on the same band: same coverage and 8-bit weights, sums equal but for rare
    last-bit roundings. R and B are compared exactly (one drop per output pixel, no atomic-order noise)."""
    from gpustacker.engine.drizzle import _fn, batch_geometry, drizzle_into, site_parities
    rng = np.random.default_rng(5)
    W, H, y0, ow, oh = 600, 41, 1201, 680, 90               # odd band start and height on purpose
    lut = cfa_lut(pattern)
    img = rng.uniform(0.01, 0.2, (H, W)).astype(np.float32)
    raw = cp.asarray(np.frombuffer(img.astype(">f4").tobytes(), np.uint8))
    one = lambda v, t: cp.asarray(np.array([v], t))  # noqa: E731
    q = lambda d: np.floor(np.minimum(d, 1) * 255 + 0.5)  # noqa: E731
    covered = differing = 0
    for th, pixfrac, sc in ((0.05, 1.0, 1.0), (-0.13, 0.8, 1.0), (21.0, 0.7, 1.0), (0.02, 0.8, 2.0)):
        c_, s_ = sc * np.cos(np.radians(th)), sc * np.sin(np.radians(th))
        m = np.array([[c_, -s_, 0.0], [s_, c_, 0.0]])
        m[:, 2] = [ow / 2, oh / 2] - m[:, :2] @ [W / 2, y0 + H / 2] + rng.uniform(0, 1, 2)
        for ch in range(3):
            n_a, d_a = cp.zeros((1, oh, ow), cp.float32), cp.zeros((1, oh, ow), cp.float32)
            drizzle_into(cp.asarray(img), y0, m, lut, pixfrac, n_a, d_a, only_ch=ch)
            n_b, d_b = cp.zeros(oh * ow, cp.float32), cp.zeros(oh * ow, cp.float32)
            cp0, cp1 = site_parities(lut, ch)
            nthr = (H if min(cp0, cp1) >= 0 else (H + 1) // 2) * ((W + 1) // 2)
            _fn("drizzle_batch")(((nthr + 255) // 256, 1), (256,), (
                raw, one(0, np.int64), one(y0, np.int32), one(H, np.int32), one(0, np.int32), one(1, np.float32),
                one(0, np.float32), np.int32(W), np.int32(-32), np.int32(cp0), np.int32(cp1),
                cp.asarray(batch_geometry(m.ravel()[None], pixfrac)), one(0, np.int64), one(ow, np.int32),
                one(oh, np.int32), n_b, d_b))
            a_n, a_d, b_n, b_d = n_a.get().ravel(), d_a.get().ravel(), n_b.get(), d_b.get()
            assert np.array_equal(a_d > 0, b_d > 0) and np.array_equal(q(a_d), q(b_d))
            use = q(a_d) > 0
            assert np.allclose(b_n[use] / b_d[use], a_n[use] / a_d[use], rtol=1e-6)
            if ch != 1:
                covered += int((a_d > 0).sum())
                differing += int(((a_d != b_d) | (a_n != b_n)).sum())
    assert covered > 50000 and differing / covered < 0.002


def test_batched_stack_matches_the_per_frame_path(tmp_path, monkeypatch):
    """The batched raw path (one read per strip, all frames per launch) stacks what the per-frame path
    does, to float rounding."""
    truth = make_lights(str(tmp_path / "subs"), n=10, seed=11, pattern="RGGB",
                        rotations=[0.0] + [2.5 * k for k in range(1, 10)])
    kw = dict(rejection="winsorized", weighting="noise", feather=2, pixfrac=0.8)
    out_a, out_b = str(tmp_path / "per_frame.fit"), str(tmp_path / "batched.fit")
    monkeypatch.setenv("GPUSTACKER_PER_FRAME", "1")
    _run(truth, tmp_path, output_path=out_a, **kw)
    monkeypatch.delenv("GPUSTACKER_PER_FRAME")
    _run(truth, tmp_path, output_path=out_b, **kw)
    a, b = F.read_image(out_a)[0], F.read_image(out_b)[0]
    assert np.array_equal(np.isnan(a), np.isnan(b))
    ok = np.isfinite(a)
    assert np.allclose(b[ok], a[ok], rtol=1e-6, atol=1e-9)
    assert (a[ok] == b[ok]).mean() > 0.995


# ----------------------------------------------------------------------------- the whole job

def test_raw_job_end_to_end(lights, tmp_path):
    out = str(tmp_path / "stack.fit")
    summary, evs = _run(lights, tmp_path, output_path=out, rejection="winsorized", weighting="noise", feather=2)
    phases = [e["name"] for e in evs if e["ev"] == "phase"]
    assert phases == ["headers", "analyze", "register", "measure", "stack", "write"]
    bad = [e for e in evs if e["ev"] == "frame_bad"]
    assert [e["number"] for e in bad] == [5] and "stars" in bad[0]["reason"]
    assert summary["frames"] == 11 and summary["skipped"] == [5]
    reg = [e for e in evs if e["ev"] == "registered"][0]
    assert reg["registered"] == 11 and reg["reference"] == lights.names[0]

    data, hdr = F.read_image(out)
    cw, ch, m0 = _canvas_of_reference(lights)
    assert data.shape == (3, ch, cw)
    assert hdr.get("STACKCNT") == 11 and hdr.get("LIVETIME") == pytest.approx(11 * lights.exptime)
    assert hdr.get("OBJECT") == "Synthetic" and hdr.get("BAYERPAT") is None and hdr.get("BZERO") is None
    assert np.isfinite(data).all()

    # geometry: stars land where frame 0 says, to a small fraction of a pixel
    errs = []
    for k in _isolated(lights, 0.1):
        x, y = apply(m0, *lights.stars[k][:2])
        ix, iy = int(x), int(y)
        win = data[1, iy - 7:iy + 8, ix - 7:ix + 8].astype(np.float64)
        g = np.clip(win - np.median(data[1, iy - 12:iy + 13, ix - 12:ix - 8]), 0, None)
        yy, xx = np.mgrid[iy - 7:iy + 8, ix - 7:ix + 8] + 0.5
        errs.append(np.hypot((g * xx).sum() / g.sum() - x, (g * yy).sum() / g.sum() - y))
    assert len(errs) >= 10 and np.median(errs) < 0.15 and max(errs) < 0.4

    # colour: red stays red and blue stays blue (normalization rescales green, see the next test)
    ratios = np.array([_photometry(data, *apply(m0, *lights.stars[k][:2])) for k in _isolated(lights, 0.15)])
    rb = np.median(ratios[:, 0] / ratios[:, 2])
    assert rb == pytest.approx(COLOR[0] / COLOR[2], rel=0.06)

    # level and noise: background at the reference's level, noise well below one sub's
    assert float(np.median(data[1][data[1] > 0])) == pytest.approx(lights.sky, abs=0.002)
    c = data[1, 40:200, 40:280]
    d = c[:, 2:] - c[:, :-2]
    assert 1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2) < 0.5 * lights.noise

    # the satellite trail is rejected
    assert np.abs(_trail_excess(data, lights, m0)).max() < 0.002

    # caches are reused: no analysis, registration or measuring the second time; same image
    out2 = str(tmp_path / "again.fit")
    _, evs2 = _run(lights, tmp_path, output_path=out2, rejection="winsorized", weighting="noise", feather=2)
    assert [e["name"] for e in evs2 if e["ev"] == "phase"] == ["headers", "stack", "write"]
    again, _ = F.read_image(out2)
    assert np.allclose(again, data, atol=1e-6)


def test_without_rejection_the_trail_shows(lights, tmp_path):
    out = str(tmp_path / "norej.fit")
    _run(lights, tmp_path, output_path=out, rejection="none", weighting="noise", feather=2)
    data, _ = F.read_image(out)
    _, _, m0 = _canvas_of_reference(lights)
    assert (_trail_excess(data, lights, m0) > 0.01).all()      # so the rejection check above means something


def test_colours_are_exact_without_normalization(lights, tmp_path):
    """Bayer drizzle keeps each colour's flux. With additive+scaling normalization the green planes,
    which average neighbouring green sites after sub-pixel shifts, measure less noisy than the
    reference's, so green gets scaled up (Siril does the same on its drizzled frames); a white
    balance factor that colour calibration removes."""
    out = str(tmp_path / "nonorm.fit")
    _run(lights, tmp_path, output_path=out, normalization="none", weighting="none")
    data, _ = F.read_image(out)
    _, _, m0 = _canvas_of_reference(lights)
    f = np.array([_photometry(data, *apply(m0, *lights.stars[k][:2])) for k in _isolated(lights, 0.15)])
    assert np.median(f[:, 0] / f[:, 1]) == pytest.approx(COLOR[0] / COLOR[1], rel=0.05)
    assert np.median(f[:, 2] / f[:, 1]) == pytest.approx(COLOR[2] / COLOR[1], rel=0.05)


def test_test_strip_selection_and_drizzle_scale(lights, tmp_path):
    keep = [n for i, n in enumerate(lights.names) if i not in (2, 9)]
    summary, evs = _run(lights, tmp_path, rows=[100, 140], include_files=keep, weighting="none")
    assert summary["test_strip"] and summary["output"] is None and summary["frames"] == 9
    assert summary["height"] == 40
    out = str(tmp_path / "x2.fit")
    summary, _ = _run(lights, tmp_path, output_path=out, drizzle_scale=2.0, pixfrac=0.8, weighting="none",
                      crop=[200, 150, 300, 200])
    data, hdr = F.read_image(out)
    assert data.shape == (3, 200, 300) and hdr.get("XPIXSZ") == pytest.approx(1.45)
    assert np.isfinite(data).all() and float(np.median(data[1])) > 0


def test_analyze_only_registers_without_stacking(tmp_path):
    truth = make_lights(str(tmp_path), n=5, seed=9)
    summary, evs = _run(truth, tmp_path, analyze_only=True, reference_file="")
    assert summary["analysis"] and summary["frames"] == 5
    assert not [e for e in evs if e["ev"] == "phase" and e["name"] in ("measure", "stack")]
    assert sum(len(e["frames"]) for e in evs if e["ev"] == "analyzed") == 5
    reg = Registration(cache_dir(truth.lights_dir))
    assert len(reg.frames) == 5 and reg.reference in truth.names


def test_too_few_usable_frames_is_a_clear_error(tmp_path):
    truth = make_lights(str(tmp_path), n=3, cloudy=(1, 2))
    with pytest.raises(JobError, match="fewer than 2 frames"):
        _run(truth, tmp_path, output_path=str(tmp_path / "x.fit"))
