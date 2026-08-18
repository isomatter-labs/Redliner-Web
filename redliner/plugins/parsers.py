"""Turning files into something comparable.

A parser answers one question: given some files, how do I produce comparable
pages? Everything downstream -- the compositor, magic align, the exporter --
talks only to :class:`DocumentSource`, so a new format needs no changes outside
its own module.

A parser takes a *list* of paths, not one path, because plenty of real formats
are a set of files: a folder of Gerbers is one board, not one document per
layer. Single-file formats simply ignore the extra entries.

Sub-pixel accuracy is declared rather than assumed. Magic align applies offsets
by re-rendering at a shifted clip window, which is exact for vector sources but
meaningless for a scanned TIFF; a parser that says ``supports_subpixel = False``
gets whole-pixel shifting instead. See :mod:`redliner.core.align`.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pymupdf

from ..core.units import PDF_UNITS_PER_INCH
from . import Registry

log = logging.getLogger("redliner.parsers")

PARSERS: Registry["type[SourceParser]"] = Registry("parser", "redliner.parsers")


@dataclass(frozen=True, slots=True)
class TextWord:
    """One positioned word, in PDF user-space points."""

    x0: float
    y0: float
    x1: float
    y1: float
    text: str


class DocumentSource(ABC):
    """An opened document: pages that can be measured, rendered and searched."""

    #: False if the format cannot honour a fractional-point render offset.
    supports_subpixel = True

    @property
    @abstractmethod
    def page_count(self) -> int:
        ...

    @property
    @abstractmethod
    def page_sizes(self) -> list[tuple[float, float]]:
        """Width and height of each page, in points."""

    @abstractmethod
    def render(self, page_index: int, dpi: float,
               clip: tuple[float, float, float, float] | None = None,
               offset: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
        """Render to an (H, W) uint8 grayscale array where 255 is bare paper.

        `clip` is a window in page points; None means the whole page. `offset`
        displaces the content by that many points, and must be applied without
        disturbing the destination pixel grid -- see the PDF implementation.
        """

    def words(self, page_index: int) -> list[TextWord]:
        """Positioned text, for the searchable export layer. Empty is fine."""
        return []

    def close(self) -> None:
        return None


class SourceParser(ABC):
    """Recognises a set of files and opens them as one document."""

    #: Registry key.
    name: str = ""
    #: Shown in the UI.
    label: str = ""
    #: Lower runs first when several parsers claim the same files.
    priority: int = 100
    #: Lower-case suffixes this parser recognises, for the upload filter.
    extensions: frozenset[str] = frozenset()

    @classmethod
    def matches(cls, paths: list[Path]) -> bool:
        """Whether this parser should handle `paths`."""
        return any(p.suffix.lower() in cls.extensions for p in paths)

    @classmethod
    @abstractmethod
    def open(cls, paths: list[Path]) -> DocumentSource:
        ...


def accepted_extensions() -> list[str]:
    """Every suffix any registered parser accepts, for the upload widget."""
    found: set[str] = set()
    for parser in PARSERS.all():
        found.update(parser.extensions)
    return sorted(found)


def parser_for(paths: list[Path]) -> type[SourceParser]:
    """The first registered parser claiming these files."""
    for parser in PARSERS.all():
        try:
            if parser.matches(paths):
                return parser
        except Exception:
            log.exception("parser %s failed while matching", parser.name)
    raise ValueError(
        "No parser recognised "
        + ", ".join(p.name for p in paths[:3])
        + ("..." if len(paths) > 3 else "")
    )


# -- PDF and anything else PyMuPDF opens --------------------------------

class PyMuPDFSource(DocumentSource):
    """Backed by PyMuPDF, which covers PDF, XPS, EPUB and common raster formats."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        with self.open_document() as doc:
            self._page_count = doc.page_count
            self._page_sizes = [(p.rect.width, p.rect.height) for p in doc]

    # Handles are not safe to share across threads and rendering runs in a
    # worker pool, so the document is reopened per call rather than cached.
    def open_document(self) -> pymupdf.Document:
        return pymupdf.open(self._path)

    @property
    def path(self) -> Path:
        return self._path

    @property
    def page_count(self) -> int:
        return self._page_count

    @property
    def page_sizes(self) -> list[tuple[float, float]]:
        return self._page_sizes

    def render(self, page_index: int, dpi: float,
               clip: tuple[float, float, float, float] | None = None,
               offset: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
        scale = dpi / PDF_UNITS_PER_INCH
        dx, dy = offset

        with self.open_document() as pdf:
            page = pdf.load_page(page_index)
            if clip is None and dx == 0 and dy == 0:
                pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale),
                                      colorspace=pymupdf.csGRAY, alpha=False)
                return np.frombuffer(pix.samples, dtype=np.uint8) \
                    .reshape(pix.height, pix.width).copy()

            window = clip if clip is not None else (
                0.0, 0.0, page.rect.width, page.rect.height)

            # The offset goes into the render matrix while the clip is
            # pre-shifted into that matrix's input space. That pair is what
            # makes a fractional shift exact: the destination pixel grid is
            # untouched, so content displaced by 6.25 px rasterizes to bytes
            # identical to the un-drifted page. Shifting the clip window alone
            # is not equivalent -- PyMuPDF snaps a clip to whole device pixels,
            # so a fractional window shift changes the sampling phase and
            # leaves anti-alias fringing along every edge.
            matrix = pymupdf.Matrix(scale, scale).pretranslate(dx, dy)
            target = (pymupdf.Rect(*window) * pymupdf.Matrix(scale, scale)).irect
            out = np.full((max(1, target.height), max(1, target.width)), 255,
                          dtype=np.uint8)

            source = pymupdf.Rect(window[0] - dx, window[1] - dy,
                                  window[2] - dx, window[3] - dy) & page.rect
            if source.is_empty:
                return out

            pix = page.get_pixmap(matrix=matrix, clip=source,
                                  colorspace=pymupdf.csGRAY, alpha=False)
            if pix.width == 0 or pix.height == 0:
                return out
            rendered = np.frombuffer(pix.samples, dtype=np.uint8) \
                .reshape(pix.height, pix.width)
            left, top = pix.x - target.x0, pix.y - target.y0

        src_x, src_y = max(0, -left), max(0, -top)
        dst_x, dst_y = max(0, left), max(0, top)
        copy_w = min(rendered.shape[1] - src_x, out.shape[1] - dst_x)
        copy_h = min(rendered.shape[0] - src_y, out.shape[0] - dst_y)
        if copy_w > 0 and copy_h > 0:
            out[dst_y : dst_y + copy_h, dst_x : dst_x + copy_w] = \
                rendered[src_y : src_y + copy_h, src_x : src_x + copy_w]
        return out

    def words(self, page_index: int) -> list[TextWord]:
        with self.open_document() as pdf:
            page = pdf.load_page(page_index)
            return [TextWord(x0=w[0], y0=w[1], x1=w[2], y1=w[3], text=w[4])
                    for w in page.get_text("words")]


@PARSERS.register
class PdfParser(SourceParser):
    name = "pdf"
    label = "PDF"
    priority = 0
    extensions = frozenset({".pdf", ".xps", ".oxps", ".epub", ".cbz"})

    @classmethod
    def open(cls, paths: list[Path]) -> DocumentSource:
        if len(paths) > 1:
            log.info("%s takes one file at a time; using %s",
                     cls.name, paths[0].name)
        return PyMuPDFSource(paths[0])


@PARSERS.register
class ImageParser(PdfParser):
    """Raster images, which PyMuPDF opens as a one-page document.

    Separate from the PDF parser so the UI can label it, and so that
    `supports_subpixel` can be answered honestly per format later.
    """

    name = "image"
    label = "Image"
    priority = 10
    extensions = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff",
                            ".bmp", ".gif", ".pnm", ".webp"})
