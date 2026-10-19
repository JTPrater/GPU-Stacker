# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jason Prater (JP Astrophotography)
"""python -m gpustacker [sequence.seq] - start the desktop app."""
import sys

from .ui.app import main

sys.exit(main())
