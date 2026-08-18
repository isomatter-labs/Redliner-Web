"""Local realignment of documents that a CAD exporter nudged out of place.

Producers routinely re-emit a text box or symbol a few pixels off between saves.
That is not a design change, but a per-pixel diff cannot tell the difference, and
a handful of stray pixels can bury the changes that actually matter.

A :class:`AlignPatch` records "inside this rectangle, shift document X by
(dx, dy)". Offsets are **region-scoped**: only pixels inside the rectangle move,
so correcting one drifting label never disturbs the rest of the sheet. The first
document in the project is the anchor and never moves; everything else is
measured against it.

Offsets are stored in PDF points for the same reason markup is -- an alignment
fixed against a 200 DPI preview stays correct in a 600 DPI export.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw

from .units import PDF_UNITS_PER_INCH

Rect = tuple[float, float, float, float]

#: Re-renders one document over a destination rectangle (points), with its
#: content displaced by an offset (points).
Sampler = Callable[[int, Rect, tuple[float, float]], np.ndarray]

#: Search radius as a fraction of the region's smaller side.
SEARCH_FRACTION = 0.25
MIN_SEARCH_PX = 4
MAX_SEARCH_PX = 120

#: Below this ink coverage a region is treated as blank -- correlating against
#: bare paper produces a confident-looking but meaningless peak.
MIN_INK = 1e-4


@dataclass(slots=True)
class AlignPatch:
    """A region, in PDF points, plus a per-document offset in points.

    `rect` is always the bounding box. `polygon` optionally narrows the region
    to a freehand outline inside it -- a lasso drawn around one drifting label
    rather than a box that also swallows the frame line beside it. Only pixels
    inside the polygon move.
    """

    rect: Rect
    offsets: dict[str, tuple[float, float]] = field(default_factory=dict)
    polygon: list[tuple[float, float]] | None = None

    def to_dict(self) -> dict:
        return {
            "rect": list(self.rect),
            "offsets": {k: list(v) for k, v in self.offsets.items()},
            "polygon": None if self.polygon is None
            else [list(p) for p in self.polygon],
        }

    @classmethod
    def from_dict(cls, data: dict) -> AlignPatch:
        polygon = data.get("polygon")
        return cls(
            rect=tuple(float(v) for v in data["rect"]),  # type: ignore[arg-type]
            offsets={k: (float(v[0]), float(v[1]))
                     for k, v in data.get("offsets", {}).items()},
            polygon=None if polygon is None
            else [(float(x), float(y)) for x, y in polygon],
        )

    def is_identity(self) -> bool:
        return all(dx == 0 and dy == 0 for dx, dy in self.offsets.values())


def bounds_of(points: list[tuple[float, float]]) -> Rect:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), min(ys), max(xs), max(ys))


def polygon_mask(polygon: list[tuple[float, float]], origin: tuple[float, float],
                 dpi: float, shape: tuple[int, int]) -> np.ndarray:
    """Rasterize a point-space polygon into a boolean mask of `shape`.

    `origin` is the top-left of the destination window in points, so the
    polygon's own coordinates are translated into that window.
    """
    height, width = shape
    if height <= 0 or width <= 0 or len(polygon) < 3:
        return np.ones((max(0, height), max(0, width)), dtype=bool)

    scale = dpi / PDF_UNITS_PER_INCH
    flat = [((x - origin[0]) * scale, (y - origin[1]) * scale) for x, y in polygon]

    image = Image.new("1", (width, height), 0)
    ImageDraw.Draw(image).polygon(flat, fill=1, outline=1)
    return np.array(image, dtype=bool)


def normalize_rect(rect: Rect) -> Rect:
    x0, y0, x1, y1 = rect
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def search_radius(width_px: int, height_px: int) -> int:
    """How far auto-align looks, scaled to the region the user drew.

    A big blob implies the user is chasing a big drift; a tight one implies a
    couple of pixels. Scaling with the region also keeps the search from
    latching onto a neighbouring feature -- in a dense table, a wide search will
    happily align row N to row N+1.
    """
    span = max(1, min(width_px, height_px))
    return int(np.clip(round(span * SEARCH_FRACTION), MIN_SEARCH_PX, MAX_SEARCH_PX))


def window(source: np.ndarray, left: int, top: int, width: int, height: int,
           fill: int = 255) -> np.ndarray:
    """Read a width x height window from `source`, padding out-of-bounds with paper."""
    out = np.full((height, width), fill, dtype=source.dtype)
    if width <= 0 or height <= 0:
        return out

    src_top, src_left = max(0, top), max(0, left)
    src_bottom = min(source.shape[0], top + height)
    src_right = min(source.shape[1], left + width)
    if src_bottom <= src_top or src_right <= src_left:
        return out

    out[src_top - top : src_bottom - top, src_left - left : src_right - left] = \
        source[src_top:src_bottom, src_left:src_right]
    return out


def _parabolic_offset(before: float, peak: float, after: float) -> float:
    """Sub-sample peak position from three correlation samples.

    The correlation peak rarely sits exactly on a pixel: a real 3.0 pt drift is
    11.875 px at 285 DPI. Rounding to whole pixels leaves a sliver of residue
    that still trips the highlight threshold, and the error grows when the
    offset is re-applied at a higher export DPI. Fitting a parabola through the
    peak and its two neighbours recovers the fraction.
    """
    denominator = before - 2.0 * peak + after
    if abs(denominator) < 1e-12:
        return 0.0
    return float(np.clip(0.5 * (before - after) / denominator, -0.5, 0.5))


def estimate_shift(reference: np.ndarray, moving: np.ndarray,
                   max_shift: int, subpixel: bool = False) -> tuple[float, float]:
    """Find the (dx, dy) in pixels that best aligns `moving` onto `reference`.

    Both inputs are ink coverage (0 = paper). Uses FFT cross-correlation rather
    than a brute-force shift sweep: correlation is O(n log n) against
    O(n * max_shift^2), which matters because the search radius scales with the
    region the user drew.

    The inputs are mean-subtracted first. Raw correlation of sparse ink is
    dominated by however much ink overlaps at all, which biases the peak toward
    whichever shift keeps the most ink inside the window rather than toward the
    shift that actually matches features.
    """
    if reference.shape != moving.shape:
        raise ValueError("estimate_shift needs matching shapes")
    if reference.mean() < MIN_INK or moving.mean() < MIN_INK:
        return (0, 0)

    height, width = reference.shape
    max_shift = max(1, min(max_shift, height - 1, width - 1))

    ref = reference.astype(np.float32) - reference.mean()
    mov = moving.astype(np.float32) - moving.mean()
    if not ref.any() or not mov.any():
        return (0, 0)

    # Zero-pad so the circular correlation cannot wrap real content into the
    # shift range we are about to inspect.
    pad_h, pad_w = height + max_shift, width + max_shift
    spectrum = np.fft.rfft2(ref, s=(pad_h, pad_w)) * \
        np.conj(np.fft.rfft2(mov, s=(pad_h, pad_w)))
    corr = np.fft.irfft2(spectrum, s=(pad_h, pad_w))

    # corr[k] = sum_n ref[n] * mov[n - k], so the peak index is the shift to
    # apply to `moving`. Negative shifts live at the far end of each axis.
    offsets = np.arange(-max_shift, max_shift + 1)
    patch = corr[np.ix_(offsets % pad_h, offsets % pad_w)]
    row, col = np.unravel_index(int(np.argmax(patch)), patch.shape)
    dx, dy = float(offsets[col]), float(offsets[row])

    if subpixel:
        if 0 < col < patch.shape[1] - 1:
            dx += _parabolic_offset(patch[row, col - 1], patch[row, col],
                                    patch[row, col + 1])
        if 0 < row < patch.shape[0] - 1:
            dy += _parabolic_offset(patch[row - 1, col], patch[row, col],
                                    patch[row + 1, col])
    return (dx, dy)


def auto_align(inks: list[np.ndarray], max_shift: int) -> list[tuple[float, float]]:
    """Shifts in pixels aligning every layer onto the first one.

    Sub-pixel, because the result is stored in points and re-applied at export
    resolution -- rounding here would bake in an error that grows with DPI.

    The first document is the anchor and always returns (0, 0). If the anchor's
    region is blank there is nothing to align against, so everything returns
    zero and the user can nudge by hand.
    """
    if not inks:
        return []
    shifts: list[tuple[float, float]] = [(0.0, 0.0)]
    blank_anchor = inks[0].mean() < MIN_INK
    for layer in inks[1:]:
        shifts.append((0.0, 0.0) if blank_anchor
                      else estimate_shift(inks[0], layer, max_shift, subpixel=True))
    return shifts


def apply_patches(rasters: list[np.ndarray], doc_ids: list[str],
                  patches: list[AlignPatch], dpi: float,
                  sample: Sampler | None = None) -> list[np.ndarray]:
    """Return rasters with every patch's region-scoped offset applied.

    `sample(index, clip_rect_in_points) -> raster` re-renders a shifted window
    straight from the source document, which places the offset exactly even
    when it lands between pixels. Without it the offset is rounded to whole
    pixels by shifting the existing raster -- correct, but it can leave up to
    half a pixel of residue, which is enough to keep the highlighter firing on
    content the user just aligned. Rasterized (non-PDF) inputs have no vector
    source to re-render, so they take the rounding path.

    Each patch samples from the *original* rasters, so overlapping patches
    cannot compound their shifts -- the last one to cover a pixel wins, which is
    predictable, rather than the two stacking into a shift neither user asked
    for.
    """
    live = [p for p in patches if not p.is_identity()]
    if not live:
        return rasters

    scale = dpi / PDF_UNITS_PER_INCH
    originals = rasters
    out = [r.copy() for r in rasters]

    for patch in live:
        x0, y0, x1, y1 = normalize_rect(patch.rect)
        left, top = round(x0 * scale), round(y0 * scale)
        right, bottom = round(x1 * scale), round(y1 * scale)

        for index, doc_id in enumerate(doc_ids):
            dx, dy = patch.offsets.get(doc_id, (0.0, 0.0))
            if dx == 0 and dy == 0:
                continue
            shift_x, shift_y = round(dx * scale), round(dy * scale)

            height, width = out[index].shape
            clip_left, clip_top = max(0, left), max(0, top)
            clip_right, clip_bottom = min(width, right), min(height, bottom)
            if clip_right <= clip_left or clip_bottom <= clip_top:
                continue

            if sample is not None:
                # Hand over the destination rect and the offset separately; the
                # sampler applies the shift in the render matrix so the
                # fractional part survives.
                patch_raster = sample(index, (
                    clip_left / scale, clip_top / scale,
                    clip_right / scale, clip_bottom / scale,
                ), (dx, dy))
            else:
                patch_raster = window(
                    originals[index],
                    clip_left - shift_x, clip_top - shift_y,
                    clip_right - clip_left, clip_bottom - clip_top,
                )

            rows = min(patch_raster.shape[0], clip_bottom - clip_top)
            cols = min(patch_raster.shape[1], clip_right - clip_left)
            destination = out[index][clip_top : clip_top + rows,
                                     clip_left : clip_left + cols]

            if patch.polygon is None:
                destination[...] = patch_raster[:rows, :cols]
            else:
                # A lasso moves only what it encloses. Copying through the mask
                # leaves everything else exactly as it was, which is the whole
                # point of drawing a blob rather than a box: a frame line
                # passing beside the drifting label stays put.
                mask = polygon_mask(patch.polygon,
                                    (clip_left / scale, clip_top / scale),
                                    dpi, (rows, cols))
                np.copyto(destination, patch_raster[:rows, :cols], where=mask)
    return out
