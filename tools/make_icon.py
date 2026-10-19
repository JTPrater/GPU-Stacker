# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jason Prater (JP Astrophotography)
"""Write gpustacker/assets/gpu-stacker.ico (PNG-compressed entries) from the icon drawn in code."""
import os
import struct
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from PySide6.QtCore import QBuffer, QByteArray, QIODevice  # noqa: E402
from PySide6.QtGui import QGuiApplication  # noqa: E402

from gpustacker.ui.widgets import _icon_pixmap  # noqa: E402


def main():
    QGuiApplication(sys.argv[:1])
    sizes = (16, 24, 32, 48, 64, 128, 256)
    pngs = []
    for s in sizes:
        ba = QByteArray()
        buf = QBuffer(ba)
        buf.open(QIODevice.OpenModeFlag.WriteOnly)
        _icon_pixmap(s).save(buf, "PNG")
        pngs.append(bytes(ba))
    out = os.path.join(ROOT, "gpustacker", "assets", "gpu-stacker.ico")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    offset = 6 + 16 * len(sizes)
    with open(out, "wb") as f:
        f.write(struct.pack("<HHH", 0, 1, len(sizes)))
        for s, png in zip(sizes, pngs):
            f.write(struct.pack("<BBBBHHII", s % 256, s % 256, 0, 0, 1, 32, len(png), offset))
            offset += len(png)
        for png in pngs:
            f.write(png)
    print("wrote", out)


if __name__ == "__main__":
    main()
