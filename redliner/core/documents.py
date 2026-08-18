"""Source document handles and rasterization.

The formats themselves live in :mod:`redliner.plugins.parsers`; this module is
the thin layer the rest of the core talks to. It owns the raster cache, the
thumbnail/data-URL helpers, and the :class:`SourceDoc` handle that carries a
document's identity and colour through the project.

A document is a *set* of files, not one file. That is what lets a folder of
Gerbers be one document rather than one document per layer -- the parser
registry decides how to interpret the set.
"""

from __future__ import annotations

import base64
import io
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from ..plugins.parsers import DocumentSource, TextWord, parser_for
from .units import PDF_UNITS_PER_INCH

__all__ = [
    "PDF_UNITS_PER_INCH", "TextWord", "SourceDoc", "load_document",
    "rasterize", "rasterize_clip", "page_words", "blank_like",
    "raster_to_data_url", "thumbnail", "clear_cache", "RasterCache",
]

THUMBNAIL_DPI = 12.0
THUMBNAIL_MAX = 160

#: Byte budget for the raster cache. Engineering sheets are large -- a single
#: E-size page at 300 DPI is ~67 MB grayscale -- so this holds only a handful.
CACHE_BYTES = 1_500_000_000


@dataclass(slots=True)
class SourceDoc:
    """A user-supplied document plus the color its unique ink renders in."""

    doc_id: str
    name: str
    #: Every file making up this document. Usually one; a folder of Gerbers or
    #: a multi-sheet export is many.
    paths: list[Path] = field(default_factory=list)
    #: None until the project assigns one; see Project.add_document.
    color: tuple[float, float, float] | None = None
    page_count: int = 0
    #: Page sizes in PDF points, indexed by page number.
    page_sizes: list[tuple[float, float]] = field(default_factory=list)
    #: Which parser opened it, for display and for reopening.
    parser: str = ""
    _source: DocumentSource | None = None

    @property
    def path(self) -> Path:
        """The first file. Convenient for single-file formats and messages."""
        return self.paths[0]

    @property
    def source(self) -> DocumentSource:
        if self._source is None:
            self._source = parser_for(self.paths).open(self.paths)
        return self._source

    @property
    def supports_subpixel(self) -> bool:
        """Whether alignment offsets can be placed between pixels."""
        return self.source.supports_subpixel

    def cache_key(self) -> tuple:
        stamps = []
        for path in self.paths:
            try:
                stamps.append((str(path), path.stat().st_mtime_ns))
            except OSError:
                stamps.append((str(path), 0))
        return tuple(stamps)


def load_document(paths: Path | list[Path], doc_id: str, name: str,
                  color: tuple[float, float, float] | None = None) -> SourceDoc:
    """Open `paths` with whichever registered parser claims them."""
    if isinstance(paths, (str, Path)):
        paths = [Path(paths)]
    paths = [Path(p) for p in paths]
    if not paths:
        raise ValueError("a document needs at least one file")

    parser = parser_for(paths)
    source = parser.open(paths)
    return SourceDoc(
        doc_id=doc_id, name=name, paths=paths, color=color,
        page_count=source.page_count, page_sizes=list(source.page_sizes),
        parser=parser.name, _source=source,
    )


class RasterCache:
    """Thread-safe LRU cache of rendered grayscale pages, bounded by bytes."""

    def __init__(self, budget: int = CACHE_BYTES) -> None:
        self._budget = budget
        self._entries: OrderedDict[tuple, np.ndarray] = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()

    def get(self, key: tuple) -> np.ndarray | None:
        with self._lock:
            if key not in self._entries:
                return None
            self._entries.move_to_end(key)
            return self._entries[key]

    def put(self, key: tuple, value: np.ndarray) -> None:
        with self._lock:
            if key in self._entries:
                self._bytes -= self._entries.pop(key).nbytes
            self._entries[key] = value
            self._bytes += value.nbytes
            while self._bytes > self._budget and len(self._entries) > 1:
                _, evicted = self._entries.popitem(last=False)
                self._bytes -= evicted.nbytes

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._bytes = 0


_cache = RasterCache()


def rasterize(doc: SourceDoc, page_index: int, dpi: float) -> np.ndarray:
    """Render a whole page to an (H, W) uint8 grayscale array. 255 is paper."""
    key = (doc.cache_key(), page_index, round(dpi, 3))
    cached = _cache.get(key)
    if cached is not None:
        return cached

    raster = doc.source.render(page_index, dpi)
    _cache.put(key, raster)
    return raster


def rasterize_clip(doc: SourceDoc, page_index: int, dpi: float,
                   clip: tuple[float, float, float, float],
                   offset: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    """Render a sub-rectangle of a page, moving the content by `offset` points.

    Deliberately uncached: clip windows are effectively unique per call, so
    caching them would evict the full-page rasters that actually get reused.
    """
    return doc.source.render(page_index, dpi, clip, offset)


def page_words(doc: SourceDoc, page_index: int) -> list[TextWord]:
    """Positioned words, for building a searchable export text layer."""
    return doc.source.words(page_index)


def blank_like(shape: tuple[int, int]) -> np.ndarray:
    """A bare-paper raster, used where a document has no page for a slot."""
    return np.full(shape, 255, dtype=np.uint8)


def raster_to_data_url(raster: np.ndarray, max_edge: int | None = None) -> str:
    """Encode a grayscale or RGB array as a PNG data URL for the browser."""
    image = Image.fromarray(raster)
    if max_edge and max(image.size) > max_edge:
        image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=False, compress_level=1)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def thumbnail(doc: SourceDoc, page_index: int) -> str:
    """A small PNG data URL preview of one source page."""
    return raster_to_data_url(rasterize(doc, page_index, THUMBNAIL_DPI), THUMBNAIL_MAX)


def clear_cache() -> None:
    _cache.clear()
