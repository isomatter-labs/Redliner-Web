"""The project model: which source pages line up with which, and rendering.

A project holds N source documents and, for each, a *sequence*: a list whose
length is the number of output pages and whose entries are either a source page
index or ``None`` for "this document has nothing here". All sequences are kept
the same length, so output page `i` is composed from `sequences[doc][i]` across
every document. Inserting a blank into one document's sequence is what lets a
user re-align a revision that gained or lost a sheet.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from .align import AlignPatch, apply_patches, normalize_rect, window
from .compose import DiffSettings, Layer, align, composite
from .documents import (PDF_UNITS_PER_INCH, SourceDoc, blank_like,
                        rasterize, rasterize_clip)
from .markup import Shape

#: Fallback output size (US Letter, points) when a slot has no source pages.
DEFAULT_PAGE_POINTS = (612.0, 792.0)

DEFAULT_COLORS: list[tuple[float, float, float]] = [
    (1.00, 0.45, 0.00),  # orange -- conventionally the old revision
    (0.00, 0.40, 1.00),  # blue   -- conventionally the new revision
    (0.00, 0.65, 0.25),  # green
    (0.80, 0.00, 0.75),  # magenta
    (0.85, 0.65, 0.00),  # amber
    (0.00, 0.65, 0.70),  # teal
]


@dataclass(slots=True)
class OutputPage:
    """One composed page of the result."""

    export: bool = True
    label: str = ""
    markups: list[Shape] = field(default_factory=list)
    patches: list[AlignPatch] = field(default_factory=list)


@dataclass(slots=True)
class Project:
    docs: list[SourceDoc] = field(default_factory=list)
    sequences: dict[str, list[int | None]] = field(default_factory=dict)
    pages: list[OutputPage] = field(default_factory=list)
    settings: DiffSettings = field(default_factory=DiffSettings)
    dpi: float = 150.0
    preview_dpi: float = 72.0

    # -- structure -------------------------------------------------------

    @property
    def page_count(self) -> int:
        return len(self.pages)

    def add_document(self, doc: SourceDoc) -> None:
        """Add a document, seeding its sequence as pages 0..n-1 in order.

        A document with no color yet is assigned one here rather than by the
        caller. Uploads are handled concurrently, so a caller that picks the
        color before awaiting its parse would hand the same color to every file
        in a multi-file drop; choosing at insert time is atomic because this
        runs to completion on the event loop.
        """
        if doc.color is None:
            doc.color = next_color([d.color for d in self.docs])
        self.docs.append(doc)
        self.sequences[doc.doc_id] = list(range(doc.page_count))
        self._normalize()

    def remove_document(self, doc_id: str) -> None:
        self.docs = [d for d in self.docs if d.doc_id != doc_id]
        self.sequences.pop(doc_id, None)
        self._normalize()

    def _normalize(self) -> None:
        """Pad every sequence to a common length and sync the page list."""
        length = max((len(s) for s in self.sequences.values()), default=0)
        for doc_id, seq in self.sequences.items():
            if len(seq) < length:
                seq.extend([None] * (length - len(seq)))

        while len(self.pages) < length:
            self.pages.append(OutputPage())
        del self.pages[length:]

    def insert_blank(self, doc_id: str, at: int) -> None:
        """Push one document's pages down by one slot, opening a gap at `at`."""
        self.sequences[doc_id].insert(at, None)
        self._normalize()

    def remove_slot(self, doc_id: str, at: int) -> None:
        """Remove one entry from a single document's sequence, closing the gap."""
        seq = self.sequences[doc_id]
        if 0 <= at < len(seq):
            seq.pop(at)
            seq.append(None)
        self._trim_trailing_blanks()

    def move_slot(self, doc_id: str, frm: int, to: int) -> None:
        seq = self.sequences[doc_id]
        if 0 <= frm < len(seq) and 0 <= to < len(seq):
            seq.insert(to, seq.pop(frm))

    def delete_output_page(self, at: int) -> None:
        """Drop output page `at` entirely, from every document at once."""
        for seq in self.sequences.values():
            if 0 <= at < len(seq):
                seq.pop(at)
        if 0 <= at < len(self.pages):
            self.pages.pop(at)
        self._normalize()

    def _trim_trailing_blanks(self) -> None:
        """Drop trailing slots where every document is blank."""
        while self.sequences and all(
            seq and seq[-1] is None for seq in self.sequences.values()
        ):
            for seq in self.sequences.values():
                seq.pop()
            if self.pages:
                self.pages.pop()
        self._normalize()

    def auto_align(self) -> None:
        """Reset every sequence to natural page order."""
        for doc in self.docs:
            self.sequences[doc.doc_id] = list(range(doc.page_count))
        self._normalize()

    # -- rendering -------------------------------------------------------

    def slot_sources(self, index: int) -> list[tuple[SourceDoc, int | None]]:
        """The (document, page index or None) pairs feeding one output page."""
        return [(doc, self.sequences[doc.doc_id][index]) for doc in self.docs]

    def page_size_points(self, index: int) -> tuple[float, float]:
        """Physical size of an output page: the bounding box of its sources."""
        sizes = [
            doc.page_sizes[page]
            for doc, page in self.slot_sources(index)
            if page is not None and page < len(doc.page_sizes)
        ]
        if not sizes:
            return DEFAULT_PAGE_POINTS
        return (max(w for w, _ in sizes), max(h for _, h in sizes))

    def render(self, index: int, dpi: float | None = None) -> np.ndarray:
        """Composite output page `index` into an (H, W, 3) uint8 RGB array."""
        if not 0 <= index < self.page_count:
            raise IndexError(f"no output page {index}")
        dpi = self.dpi if dpi is None else dpi

        sources = self.slot_sources(index)
        rasters: list[np.ndarray | None] = [
            rasterize(doc, page, dpi) if page is not None else None
            for doc, page in sources
        ]

        present = [r for r in rasters if r is not None]
        if not present:
            width_pt, height_pt = self.page_size_points(index)
            scale = dpi / PDF_UNITS_PER_INCH
            shape = (max(1, round(height_pt * scale)), max(1, round(width_pt * scale)))
            return np.full((*shape, 3), 255, dtype=np.uint8)

        shape = (max(r.shape[0] for r in present), max(r.shape[1] for r in present))
        filled = [r if r is not None else blank_like(shape) for r in rasters]

        corrected = apply_patches(
            align(filled), [doc.doc_id for doc, _ in sources],
            self.pages[index].patches, dpi,
            sample=self._sampler(sources, dpi),
        )
        layers = [
            Layer(ink=raster, color=doc.color)
            for raster, (doc, _) in zip(corrected, sources)
        ]
        return composite(layers, self.settings)

    # -- alignment -------------------------------------------------------

    def _sampler(self, sources: list[tuple[SourceDoc, int | None]], dpi: float):
        """A clip-window sampler over the documents feeding one output page.

        Handed to :func:`apply_patches` so alignment offsets are realised by
        re-rendering from the PDF at a shifted clip rectangle -- exact at any
        fraction of a pixel -- instead of nudging finished pixels around.
        """
        def sample(index: int, clip: tuple[float, float, float, float],
                   offset: tuple[float, float]) -> np.ndarray:
            doc, page_index = sources[index]
            if page_index is None:
                return np.zeros((0, 0), dtype=np.uint8)
            return rasterize_clip(doc, page_index, dpi, clip, offset)

        return sample

    def region_rasters(self, index: int, rect: tuple[float, float, float, float],
                       dpi: float,
                       offsets: dict[str, tuple[float, float]] | None = None
                       ) -> list[np.ndarray]:
        """Grayscale crops of `rect` from each document, with offsets applied.

        Renders each crop straight from the source at the offset clip window, so
        the align dialog shows the same sub-pixel-accurate result the final
        composite will produce.
        """
        offsets = offsets or {}
        x0, y0, x1, y1 = normalize_rect(rect)
        scale = dpi / PDF_UNITS_PER_INCH
        width = max(1, round((x1 - x0) * scale))
        height = max(1, round((y1 - y0) * scale))

        crops = []
        for doc, page_index in self.slot_sources(index):
            if page_index is None:
                crops.append(np.full((height, width), 255, dtype=np.uint8))
                continue
            crops.append(rasterize_clip(
                doc, page_index, dpi, (x0, y0, x1, y1),
                offsets.get(doc.doc_id, (0.0, 0.0)),
            ))

        # Clip windows round independently; pad to a common shape so the
        # compositor's equal-shape requirement holds.
        target = (max(c.shape[0] for c in crops), max(c.shape[1] for c in crops))
        return [c if c.shape == target
                else window(c, 0, 0, target[1], target[0]) for c in crops]

    def render_region(self, index: int, rect: tuple[float, float, float, float],
                      dpi: float,
                      offsets: dict[str, tuple[float, float]] | None = None
                      ) -> np.ndarray:
        """Difference-only composite of one region, for the align dialog."""
        crops = self.region_rasters(index, rect, dpi, offsets)
        settings = DiffSettings(highlight=False, show_unchanged=False,
                                ink_floor=self.settings.ink_floor)
        layers = [Layer(ink=crop, color=doc.color)
                  for crop, doc in zip(crops, self.docs)]
        return composite(layers, settings)

    def render_preview(self, index: int) -> np.ndarray:
        return self.render(index, dpi=self.preview_dpi)

    def export_indices(self) -> list[int]:
        return [i for i, page in enumerate(self.pages) if page.export]


def next_color(used: list[tuple[float, float, float]]) -> tuple[float, float, float]:
    """Pick the next unused default color, cycling once they run out."""
    for color in DEFAULT_COLORS:
        if color not in used:
            return color
    return DEFAULT_COLORS[len(used) % len(DEFAULT_COLORS)]


def rgb_to_hex(color: tuple[float, float, float]) -> str:
    return "#" + "".join(f"{round(c * 255):02x}" for c in color)


def hex_to_rgb(value: str) -> tuple[float, float, float]:
    value = value.lstrip("#")
    if len(value) == 3:
        value = "".join(c * 2 for c in value)
    return tuple(int(value[i : i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]
