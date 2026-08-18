"""Gerber parser -- STUB.

A worked skeleton for the multi-file case, and the example to copy when adding
a format. It registers, appears in the UI and reports the layer set it found,
but rendering raises with instructions rather than guessing: a half-right board
render is worse than an honest error, because a diff of two wrong renders looks
like a real difference.

Why a board is one document
---------------------------
A fabrication package is a directory of layers -- copper, mask, silkscreen,
drill -- that only mean anything together. Comparing them layer-by-layer as
separate documents would bury the change you care about. So this parser takes
the whole set via ``open(paths)`` and presents it as a single document, which is
what the rest of Redliner already expects.

Finishing it
------------
1. Add a renderer. `pcb-tools`, `gerbonara` or `pygerber` all parse RS-274X and
   can rasterize; `gerbonara` also converts to SVG, which would let you render
   through PyMuPDF and keep the vector-exact alignment path.
2. Fill in `page_sizes` from the board outline so pages measure correctly --
   everything downstream works in points, and magic align offsets are points.
3. Decide the page model. One page with layers composited is the usual answer;
   one page per layer is also defensible if your reviewers work layer by layer.
4. Leave `supports_subpixel` False unless your renderer can honour a fractional
   offset without resampling. Being honest here costs a little alignment
   precision; claiming it wrongly produces silent anti-alias fringing.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from ..plugins.parsers import PARSERS, DocumentSource, SourceParser

#: Conventional layer suffixes, most permissive first. Vendors disagree wildly,
#: which is exactly the kind of thing a fork will want to adjust.
LAYER_PATTERNS = {
    "top copper": r"(\.gtl|_f_cu|-f_cu|\.top)$",
    "bottom copper": r"(\.gbl|_b_cu|-b_cu|\.bot)$",
    "top mask": r"(\.gts|_f_mask|-f_mask)$",
    "bottom mask": r"(\.gbs|_b_mask|-b_mask)$",
    "top silk": r"(\.gto|_f_silks|-f_silks)$",
    "bottom silk": r"(\.gbo|_b_silks|-b_silks)$",
    "outline": r"(\.gko|\.gm1|_edge_cuts|-edge_cuts)$",
    "drill": r"(\.drl|\.txt|\.xln)$",
}

GERBER_SUFFIXES = frozenset({
    ".gbr", ".ger", ".gtl", ".gbl", ".gts", ".gbs", ".gto", ".gbo",
    ".gko", ".gm1", ".drl", ".xln",
})


def classify(path: Path) -> str:
    """Best guess at which layer a file is, for display and ordering."""
    name = path.name.lower()
    for layer, pattern in LAYER_PATTERNS.items():
        if re.search(pattern, name):
            return layer
    return "unknown"


class GerberSource(DocumentSource):
    """One board, made of many layer files."""

    # A raster renderer cannot place content at a fraction of a pixel without
    # resampling, and resampling introduces its own edge artefacts. Saying so
    # makes magic align fall back to whole-pixel shifting.
    supports_subpixel = False

    def __init__(self, paths: list[Path]) -> None:
        self.paths = list(paths)
        self.layers = {classify(p): p for p in self.paths}

    @property
    def page_count(self) -> int:
        return 1

    @property
    def page_sizes(self) -> list[tuple[float, float]]:
        # Should come from the outline layer's extents. US Letter is a
        # placeholder so the rest of the pipeline has something coherent.
        return [(612.0, 792.0)]

    def render(self, page_index: int, dpi: float,
               clip: tuple[float, float, float, float] | None = None,
               offset: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
        raise NotImplementedError(
            "Gerber rendering is not implemented. This stub found "
            f"{len(self.paths)} file(s): "
            + ", ".join(f"{name} ({path.name})"
                        for name, path in sorted(self.layers.items()))
            + ". See redliner/extensions/gerber_parser.py for how to finish it."
        )


@PARSERS.register
class GerberParser(SourceParser):
    name = "gerber"
    label = "Gerber (stub)"
    priority = 50
    extensions = GERBER_SUFFIXES

    @classmethod
    def matches(cls, paths: list[Path]) -> bool:
        # Requires a recognised Gerber extension, not merely a plausible one.
        # `.txt` appears in the drill patterns above and would otherwise let
        # this parser claim any stray text file dropped alongside a PDF.
        return any(p.suffix.lower() in cls.extensions for p in paths)

    @classmethod
    def open(cls, paths: list[Path]) -> DocumentSource:
        return GerberSource(paths)
