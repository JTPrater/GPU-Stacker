# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jason Prater (JP Astrophotography)
"""Side-by-side comparison of two stacks of the same data, with the same stretch applied to both.

Usage: .venv\\Scripts\\python tools\\side_by_side.py A.fit B.fit OUT.png [--labels "A" "B"] [--notes "..." "..."]
                                                    [--crop X Y W H]... [--zoom Z]
  B is aligned onto A by matching stars (a whole-pixel shift is used when the two are only shifted, so neither
  image is resampled), its brightness is matched to A per channel from the fluxes of the same stars, and both
  are stretched with one set of screen-transfer parameters computed from A. Crops are in A's pixel coordinates.
  Prints background noise and star FWHM for both.
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if os.name == "nt":
    os.environ.setdefault("QT_QPA_FONTDIR", os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts"))

from gpustacker.engine.fits import read_image  # noqa: E402
from gpustacker.ui.stretch import apply_stf, auto_stf  # noqa: E402


def stars_of(data):
    import cupy as cp

    from gpustacker.engine.stars import bin2, detect
    L = cp.asarray(np.nan_to_num(data.mean(axis=0)), dtype=cp.float32)
    return detect(bin2(L))


def align(a, b):
    """Transform taking B's pixel coordinates to A's (2x3), from matched stars."""
    from gpustacker.engine.register import match
    sa, ma = stars_of(a)
    sb, mb = stars_of(b)
    m = match(sb, sa, "affine")
    if m is None:
        raise SystemExit("couldn't match the stars of the two stacks")
    return m, ma, mb


def oriented(a, b):
    """FITS files without ROWORDER are ambiguous. Siril shows Seestar subs (and its own stacks of them) upside
    down, which is why its scripts end with `mirrorx`; a Siril stack saved without that flip has its rows in
    the other order from ours. Return A flipped vertically if that makes its stars match B's."""
    from gpustacker.engine.register import match
    sb, _ = stars_of(b)
    for flip in (False, True):
        aa = a[:, ::-1, :] if flip else a
        sa, _ = stars_of(aa)
        m = match(sb, sa, "similarity")
        if m is not None and m.inliers >= 50:
            return np.ascontiguousarray(aa), flip
    raise SystemExit("couldn't match the stars of the two stacks in either row order")


def star_fluxes(img, xs, ys, r_ap=6, r_in=7, r_out=10):
    """(channels, stars) aperture fluxes minus the local background; NaN where a star's box leaves the data."""
    yy, xx = np.mgrid[-r_out:r_out + 1, -r_out:r_out + 1]
    out = np.full((img.shape[0], len(xs)), np.nan)
    for k, (xi, yi) in enumerate(zip(xs, ys)):
        cx, cy = int(np.floor(xi)), int(np.floor(yi))
        cut = img[:, cy - r_out:cy + r_out + 1, cx - r_out:cx + r_out + 1]
        if cut.shape[1:] != yy.shape or not np.all(cut > 0):
            continue
        r = np.hypot(xx + cx + 0.5 - xi, yy + cy + 0.5 - yi)
        ann = (r > r_in) & (r < r_out)
        for c in range(img.shape[0]):
            out[c, k] = (cut[c] - np.median(cut[c][ann]))[r <= r_ap].sum()
    return out


def photometric_match(a, bb, both):
    """Per channel (gain, offset, stars used) putting B on A's brightness scale. The gain comes from the
    fluxes of the same isolated, unsaturated stars in both; the offset matches the median level of the area
    both cover. (A least-squares fit of A against B over all pixels is biased low by B's noise: on the M45
    comparison it scaled B down 4-6%, making it look darker and quieter than it is.)"""
    st, _ = stars_of(a)
    ok = ~st.saturated
    xs, ys, fl = st.x[ok], st.y[ok], st.flux[ok]
    iso = np.array([(np.hypot(xs - x, ys - y) < 15).sum() == 1 for x, y in zip(xs, ys)], bool)
    order = np.argsort(-fl[iso])[50:850]           # mid-bright: skip the ones close to saturation
    xs, ys = xs[iso][order], ys[iso][order]
    fa, fb = star_fluxes(a, xs, ys), star_fluxes(bb, xs, ys)
    out = []
    for c in range(a.shape[0]):
        k = np.isfinite(fa[c]) & np.isfinite(fb[c]) & (fa[c] > 0) & (fb[c] > 0)
        g = float(np.median(fa[c][k] / fb[c][k]))
        out.append((g, float(np.median(a[c][both]) - g * np.median(bb[c][both])), int(k.sum())))
    return out


def background_noise(img, box):
    """Per channel: robust sigma of 2-pixel differences / sqrt(2) in a star-poor box."""
    x, y, w, h = box
    out = []
    for c in range(img.shape[0]):
        p = img[c, y:y + h, x:x + w].astype(np.float64)
        d = (p[:, 2:] - p[:, :-2]).ravel()
        d = d[np.isfinite(d)]
        out.append(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2))
    return np.array(out)


def quiet_box(img, size=400):
    """The size x size box with the lowest median luminance gradient energy (few stars, no nebula)."""
    L = img.mean(axis=0)
    h, w = L.shape
    best, best_v = None, np.inf
    for y in range(size // 2, h - size - size // 2, size // 2):
        for x in range(size // 2, w - size - size // 2, size // 2):
            p = L[y:y + size, x:x + size]
            if not np.all(p > 0):
                continue
            v = np.percentile(p, 99) - np.percentile(p, 50)
            if v < best_v:
                best, best_v = (x, y, size, size), v
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("out")
    ap.add_argument("--labels", nargs=2, default=["A", "B"])
    ap.add_argument("--notes", nargs=2, default=["", ""])
    ap.add_argument("--crop", nargs=4, type=int, action="append", default=[])
    ap.add_argument("--zoom", type=float, default=1.0, help="magnification of the crops in the picture")
    args = ap.parse_args()

    a, _ = read_image(args.a)
    b, _ = read_image(args.b)
    print(f"A {a.shape}  B {b.shape}")
    a, flipped = oriented(a, b)
    if flipped:
        print("A's rows are stored in the opposite order (no ROWORDER keyword): flipped it to match B")
    m, ma, mb = align(a, b)
    t = m.t.m
    rot = np.degrees(np.arctan2(t[1, 0], t[0, 0]))
    print(f"B -> A: shift ({t[0, 2]:.2f}, {t[1, 2]:.2f}) px, rotation {rot:.4f} deg, scale {m.t.scale:.6f}, "
          f"{m.inliers} stars, rms {m.rms:.2f} px")
    if abs(rot) < 0.02 and abs(m.t.scale - 1) < 2e-4:
        dx, dy = int(round(t[0, 2])), int(round(t[1, 2]))
        bb = np.zeros_like(a)                      # B placed on A's grid by a whole-pixel shift
        ys, xs = max(0, dy), max(0, dx)
        ye, xe = min(a.shape[1], b.shape[1] + dy), min(a.shape[2], b.shape[2] + dx)
        bb[:, ys:ye, xs:xe] = b[:, ys - dy:ye - dy, xs - dx:xe - dx]
        print(f"using a whole-pixel shift ({dx}, {dy}); no resampling")
    else:
        raise SystemExit("the stacks differ by more than a shift; resampling is not implemented")

    both = np.all(a > 0, axis=0) & np.all(bb > 0, axis=0)
    for c, (g, o, n) in enumerate(photometric_match(a, bb, both)):     # brightness: B -> A
        bb[c] = np.where(bb[c] > 0, bb[c] * g + o, 0)
        print(f"channel {c}: B scaled x{g:.5f} {o:+.6f} (from {n} stars)")
    a = np.where(both[None], a, 0).astype(np.float32)
    bb = np.where(both[None], bb, 0).astype(np.float32)

    qb = quiet_box(a)
    na, nb = background_noise(a, qb), background_noise(bb, qb)
    print(f"background noise in box {qb}: A {np.round(na * 1e4, 3)}  B {np.round(nb * 1e4, 3)} (x1e-4, A's units)")
    print(f"star FWHM (px, median): A {ma.fwhm:.3f}  B {mb.fwhm:.3f};  stars found: A {ma.nstars}  B {mb.nstars}")

    params = auto_stf(a)                           # one stretch for both
    ra, rb = apply_stf(a, params), apply_stf(bb, params)

    from PySide6.QtCore import QRect, Qt
    from PySide6.QtGui import QColor, QFont, QGuiApplication, QImage, QPainter

    app = QGuiApplication.instance() or QGuiApplication(sys.argv[:1])   # noqa: F841 (needed for fonts)

    def qimg(rgb):
        rgb = np.ascontiguousarray(rgb)
        h, w, _ = rgb.shape
        return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()

    rows, cols = np.flatnonzero(both.any(axis=1)), np.flatnonzero(both.any(axis=0))
    ov = (int(cols[0]), int(rows[0]), int(cols[-1] - cols[0] + 1), int(rows[-1] - rows[0] + 1))
    crops = [tuple(c) for c in args.crop]
    pad, head, label_h = 16, 74, 26
    tiles = []
    scale_full = 560 / ov[3]                        # the area both cover, scaled down
    tiles.append((f"Whole stack, area both cover ({ov[2]}x{ov[3]} px), scaled down", ov, scale_full))
    for c in crops:
        tiles.append((f"Crop {c[2]}x{c[3]} px at ({c[0]}, {c[1]}), {args.zoom:g}x", c, args.zoom))
    tiles.append((f"Quiet background {qb[2]}x{qb[3]} px, 1x (noise)", qb, 1.0))

    widths = [int(t[1][2] * t[2]) for t in tiles]
    heights = [int(t[1][3] * t[2]) for t in tiles]
    W = pad * 3 + 2 * max(widths)
    H = head + sum(hh + label_h + pad for hh in heights) + pad
    canvas = QImage(W, H, QImage.Format.Format_RGB888)
    canvas.fill(QColor("#f0f0f0"))
    p = QPainter(canvas)
    p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
    big = QFont("Segoe UI", 12)
    small = QFont("Segoe UI", 9)
    colw = max(widths)
    for i, (lab, note) in enumerate(zip(args.labels, args.notes)):
        x = pad + i * (colw + pad)
        p.setFont(big)
        p.setPen(QColor("#003399"))
        p.drawText(QRect(x, 10, colw, 26), Qt.AlignmentFlag.AlignLeft, lab)
        p.setFont(small)
        p.setPen(QColor("#000000"))
        n = (na, nb)[i]
        f = (ma.fwhm, mb.fwhm)[i]
        p.drawText(QRect(x, 38, colw, 34), Qt.AlignmentFlag.AlignLeft | Qt.TextFlag.TextWordWrap,
                   f"{note}   ·   star FWHM {f:.2f} px   ·   background noise {np.mean(n) * 1e4:.2f}e-4")
    y = head
    for (title, (cx, cy, cw, ch), z), tw, th in zip(tiles, widths, heights):
        p.setFont(small)
        p.setPen(QColor("#5a5a5a"))
        p.drawText(QRect(pad, y, W - 2 * pad, label_h - 4), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom,
                   title)
        y += label_h
        for i, img in enumerate((ra, rb)):
            sub = img[cy:cy + ch, cx:cx + cw]
            q = qimg(sub).scaled(tw, th, Qt.AspectRatioMode.IgnoreAspectRatio,
                                 Qt.TransformationMode.SmoothTransformation if z < 1 else
                                 Qt.TransformationMode.FastTransformation)
            x = pad + i * (colw + pad)
            p.drawImage(x, y, q)
            p.setPen(QColor("#828790"))
            p.drawRect(x - 1, y - 1, tw + 1, th + 1)
        y += th + pad
    p.end()
    canvas.save(args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
