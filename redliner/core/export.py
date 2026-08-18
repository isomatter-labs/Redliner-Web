"""Compose selected output pages into a downloadable PDF.

The composite itself is a raster, but the source PDFs already carry positioned
text, so the exported pages get an invisible text layer lifted straight from
them. That makes the result searchable and selectable without ever running OCR
-- which is both faster and far more accurate than re-recognizing text we
already have in vector form. Scanned sources have no text to lift, and are the
only case where OCR would add anything.
"""

from __future__ import annotations

import io
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
import pymupdf
from PIL import Image

from .documents import PDF_UNITS_PER_INCH, page_words
from .markup import draw_on_page
from .project import Project

#: Render mode 3 in the PDF spec is "fill none, stroke none" -- glyphs are
#: positioned and selectable but never painted.
INVISIBLE = 3

ProgressFn = Callable[[int, int], None]


@dataclass(slots=True)
class ExportOptions:
    dpi: float | None = None
    #: Embed an invisible, searchable text layer lifted from the sources.
    text_layer: bool = True
    #: JPEG quality, or None to embed lossless PNG. Line art compresses well
    #: as PNG and JPEG would ring around the strokes, so PNG is the default.
    jpeg_quality: int | None = None
    #: Stamp annotations onto the exported pages as vector content.
    markups: bool = True
    title: str = "Redliner comparison"


def _encode(rgb: np.ndarray, quality: int | None) -> bytes:
    image = Image.fromarray(rgb)
    buffer = io.BytesIO()
    if quality is None:
        image.save(buffer, format="PNG", optimize=True)
    else:
        image.save(buffer, format="JPEG", quality=quality, subsampling=0)
    return buffer.getvalue()


def _text_layer(pdf_page: pymupdf.Page, project: Project, index: int) -> None:
    """Stamp invisible text from every source page feeding this output page.

    Words from all revisions are included, so a search hits text that was
    removed as well as text that was added. Identical words at identical
    positions are emitted once.
    """
    seen: set[tuple[int, int, str]] = set()
    for doc, page_index in project.slot_sources(index):
        if page_index is None:
            continue
        for word in page_words(doc, page_index):
            key = (round(word.x0), round(word.y0), word.text)
            if key in seen:
                continue
            seen.add(key)

            height = max(1.0, word.y1 - word.y0)
            try:
                pdf_page.insert_text(
                    pymupdf.Point(word.x0, word.y1 - height * 0.2),
                    word.text,
                    fontsize=height * 0.8,
                    render_mode=INVISIBLE,
                )
            except Exception:
                # A single unmappable glyph must not sink the whole export;
                # the page still renders, it just loses that word from search.
                continue


def build_pdf(project: Project, indices: Iterable[int] | None = None,
              options: ExportOptions | None = None,
              progress: ProgressFn | None = None) -> bytes:
    """Render the chosen output pages and return a PDF as bytes."""
    options = options or ExportOptions()
    page_indices = list(indices) if indices is not None else project.export_indices()
    if not page_indices:
        raise ValueError("no pages selected for export")

    dpi = options.dpi or project.dpi
    out = pymupdf.open()
    out.set_metadata({"title": options.title, "producer": "Redliner"})

    for done, index in enumerate(page_indices):
        rgb = project.render(index, dpi=dpi)
        width_pt, height_pt = project.page_size_points(index)

        # Keep the page's aspect ratio locked to the raster's. Sources of
        # differing sizes are padded to a common canvas by the compositor, so
        # the raster can be taller or wider than the nominal page box.
        raster_aspect = rgb.shape[0] / rgb.shape[1]
        height_pt = width_pt * raster_aspect

        page = out.new_page(width=width_pt, height=height_pt)
        page.insert_image(
            pymupdf.Rect(0, 0, width_pt, height_pt),
            stream=_encode(rgb, options.jpeg_quality),
        )
        if options.text_layer:
            _text_layer(page, project, index)
        if options.markups and project.pages[index].markups:
            # Drawn last so annotations sit above the comparison, and as vectors
            # rather than being burned into the raster.
            draw_on_page(page, project.pages[index].markups)

        if progress:
            progress(done + 1, len(page_indices))

    data = out.tobytes(garbage=4, deflate=True)
    out.close()
    return data


def estimate_pixels(project: Project, indices: Iterable[int], dpi: float) -> int:
    """Total pixel count for a planned export, for warning about huge jobs."""
    scale = dpi / PDF_UNITS_PER_INCH
    total = 0
    for index in indices:
        width_pt, height_pt = project.page_size_points(index)
        total += round(width_pt * scale) * round(height_pt * scale)
    return total
