"""Shared units.

Its own module so plugin interfaces can use it without importing the document
machinery, which would import plugins right back.
"""

from __future__ import annotations

#: PDF user space is defined at 72 units per inch; render matrices scale from it.
PDF_UNITS_PER_INCH = 72.0
