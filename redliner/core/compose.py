"""N-way per-pixel document compositing.

This is a NumPy port of the ``res/diff.frag`` shader from Redliner v1.5,
generalized from 2 documents to N.

Everything happens in *ink space*: ``ink = 1 - luminance``, so 0.0 is bare
paper and 1.0 is saturated ink. Working in ink space (rather than in
luminance) is what makes the composite linear and therefore extensible to
any number of layers.

For a stack of N ink layers the decomposition is::

    same_i    = min(ink_0 .. ink_N-1)     # ink every document agrees on
    excess_i  = ink_i - same              # ink unique to document i

and the output pixel is::

    px = 1 - same - sum_i (1 - color_i) * excess_i

``same`` is subtracted from all three channels, so shared content renders
black exactly as it appears in the sources. Each document's excess ink pulls
the pixel toward that document's color. For N == 2 this reduces algebraically
to the v1.5 shader (see ``tests/test_compose.py``), where ``excess`` of the
old doc is the shader's ``removed`` and ``excess`` of the new doc is ``added``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Sequence

import numpy as np
from scipy.ndimage import uniform_filter

RGB = tuple[float, float, float]

#: Rows per band when compositing large sheets. An E-size sheet at 300 DPI is
#: ~67 megapixels; materializing float32 intermediates for the whole page at
#: once costs close to a gigabyte, so the compositor works in horizontal bands.
BAND_ROWS = 1024


@dataclass(slots=True)
class DiffSettings:
    """User-facing knobs for the diff composite."""

    highlight: bool = True
    highlight_color: RGB = (1.0, 0.95, 0.2)
    #: Mean absolute ink disagreement inside the sample window, above which a
    #: pixel is highlighted. Resolution-independent (see ``_highlight_mask``).
    highlight_threshold: float = 0.04
    #: Highlight sample window radius, in pixels at the rendered resolution.
    highlight_size: int = 10
    #: Ink below this is treated as paper. Suppresses JPEG/anti-alias noise.
    ink_floor: float = 0.0
    #: When False, ink the documents agree on is dropped instead of drawn black,
    #: leaving only the differences in their document colors. This is the view
    #: the align dialog uses: with shared content gone, a few pixels of drift
    #: read as two coloured ghosts of the same shape, which is exactly the
    #: signal you are trying to null out.
    show_unchanged: bool = True


@dataclass(slots=True)
class Layer:
    """One document's contribution to a composite page."""

    ink: np.ndarray  # (H, W) float32 in [0, 1], or uint8 in [0, 255]
    color: RGB


def to_ink(gray: np.ndarray) -> np.ndarray:
    """Convert an 8-bit grayscale raster to float32 ink coverage."""
    return 1.0 - (gray.astype(np.float32) / 255.0)


def _stack(layers: Sequence[Layer], row0: int, row1: int) -> np.ndarray:
    """Gather a horizontal band of every layer into one (N, rows, W) array."""
    band = []
    for layer in layers:
        chunk = layer.ink[row0:row1]
        if chunk.dtype == np.uint8:
            chunk = to_ink(chunk)
        else:
            chunk = chunk.astype(np.float32, copy=False)
        band.append(chunk)
    return np.stack(band)


def _highlight_mask(stack: np.ndarray, settings: DiffSettings) -> np.ndarray:
    """Solid mask covering neighbourhoods where the documents disagree.

    The v1.5 shader summed ``abs(lhs - rhs)`` over a ``(2 * size)^2`` window and
    compared that raw sum against a threshold, which made the threshold depend
    on both the window size and the render DPI -- retuning was needed whenever
    either changed. Here the same window is averaged instead of summed, so the
    threshold means "average fraction of disagreeing ink" and stays put when
    size or DPI move. It is also O(n) rather than O(n * size^2), since a box
    mean is separable.
    """
    disagreement = stack.max(axis=0) - stack.min(axis=0)
    window = max(1, settings.highlight_size * 2)
    score = uniform_filter(disagreement, size=window, mode="nearest")
    return score > settings.highlight_threshold


def _composite_band(stack: np.ndarray, colors: np.ndarray, settings: DiffSettings) -> np.ndarray:
    """Composite one (N, rows, W) ink band into (rows, W, 3) float32 RGB."""
    if settings.ink_floor > 0.0:
        stack = np.where(stack < settings.ink_floor, 0.0, stack)

    same = stack.min(axis=0)

    px = np.ones((*same.shape, 3), dtype=np.float32)
    if settings.show_unchanged:
        px -= same[..., None]

    # Each layer's unique ink tints the pixel toward that layer's color.
    for i in range(stack.shape[0]):
        excess = stack[i] - same
        px -= (1.0 - colors[i]) * excess[..., None]

    if settings.highlight:
        mask = _highlight_mask(stack, settings)
        px -= (1.0 - np.asarray(settings.highlight_color, dtype=np.float32)) * mask[..., None]

    return np.clip(px, 0.0, 1.0, out=px)


def composite(layers: Sequence[Layer], settings: DiffSettings | None = None) -> np.ndarray:
    """Composite aligned ink layers into an (H, W, 3) uint8 RGB image.

    All layers must already share a shape; use :func:`align` to pad rasters of
    differing page sizes onto a common canvas first.
    """
    if not layers:
        raise ValueError("composite() needs at least one layer")
    settings = settings or DiffSettings()

    shapes = {layer.ink.shape for layer in layers}
    if len(shapes) != 1:
        raise ValueError(f"layers must share a shape, got {sorted(shapes)}")

    height, width = layers[0].ink.shape
    colors = np.asarray([layer.color for layer in layers], dtype=np.float32)
    out = np.empty((height, width, 3), dtype=np.uint8)

    # The highlight box filter reaches `highlight_size` rows past a band edge,
    # so bands are read with a halo and written without it. Without this,
    # highlight regions would show seams at every band boundary.
    halo = settings.highlight_size if settings.highlight else 0

    for row0 in range(0, height, BAND_ROWS):
        row1 = min(row0 + BAND_ROWS, height)
        read0, read1 = max(0, row0 - halo), min(height, row1 + halo)

        band = _composite_band(_stack(layers, read0, read1), colors, settings)
        keep = band[row0 - read0 : row0 - read0 + (row1 - row0)]
        out[row0:row1] = np.rint(keep * 255.0).astype(np.uint8)

    return out


def align(rasters: Sequence[np.ndarray], anchor: str = "topleft") -> list[np.ndarray]:
    """Pad grayscale rasters of differing sizes onto one common canvas.

    Padding is white (bare paper), so padded regions contribute no ink and are
    correctly read as "absent from this document".
    """
    if not rasters:
        return []
    height = max(r.shape[0] for r in rasters)
    width = max(r.shape[1] for r in rasters)

    padded = []
    for raster in rasters:
        if raster.shape == (height, width):
            padded.append(raster)
            continue
        canvas = np.full((height, width), 255, dtype=raster.dtype)
        rows, cols = raster.shape
        if anchor == "center":
            top, left = (height - rows) // 2, (width - cols) // 2
        else:
            top, left = 0, 0
        canvas[top : top + rows, left : left + cols] = raster
        padded.append(canvas)
    return padded
