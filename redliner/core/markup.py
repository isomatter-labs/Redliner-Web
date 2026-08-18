"""Annotation shapes drawn on top of a composed page.

Shapes are stored in **PDF points**, in the output page's coordinate space with
the origin at the top-left. Storing them in page space rather than in screen or
raster pixels means a markup drawn at a 100 DPI preview lands in exactly the
same place in a 600 DPI export, and stays correct when the preview re-renders
at a different resolution mid-session.

The same shape list drives both the on-screen SVG overlay and the exported PDF,
where the shapes are written as real vector operators -- so annotations stay
crisp and selectable no matter how the underlying comparison was rasterized.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

import pymupdf

RGB = tuple[float, float, float]

KINDS = ("line", "arrow", "rect", "cloud", "pencil", "text")

#: Nominal width of one revision-cloud scallop, in points (~1/6 inch).
CLOUD_ARC = 12.0

#: Padding between text and its box, in points.
TEXT_BOX_PAD = 3.0

#: Freehand simplification tolerance, in points. Pointer moves arrive far denser
#: than the drawn line's actual detail; keeping every sample would bloat both
#: the overlay and the exported PDF for no visible gain.
PENCIL_TOLERANCE = 0.6

#: Selectable fonts, mapped to a PDF base-14 face and an SVG family. Base-14
#: needs no embedding, so exports stay small and open anywhere.
FONTS: dict[str, tuple[str, str]] = {
    "sans": ("helv", "Helvetica, Arial, sans-serif"),
    "serif": ("tiro", "'Times New Roman', Times, serif"),
    "mono": ("cour", "'Courier New', Courier, monospace"),
}
DEFAULT_FONT = "sans"


@dataclass(slots=True)
class Shape:
    """One annotation. `points` holds (x, y) pairs in PDF points, top-left origin."""

    kind: str
    points: list[tuple[float, float]] = field(default_factory=list)
    color: RGB = (0.85, 0.0, 0.0)
    width: float = 1.5
    text: str = ""
    font_size: float = 11.0
    font: str = DEFAULT_FONT
    #: Interior colour, or None for transparent.
    fill: RGB | None = None
    #: Text only: draw a box around the text using `fill` and `color`.
    boxed: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Shape:
        fill = data.get("fill")
        return cls(
            kind=data["kind"],
            points=[(float(x), float(y)) for x, y in data.get("points", [])],
            color=tuple(data.get("color", (0.85, 0.0, 0.0))),  # type: ignore[arg-type]
            width=float(data.get("width", 1.5)),
            text=str(data.get("text", "")),
            font_size=float(data.get("font_size", 11.0)),
            font=str(data.get("font", DEFAULT_FONT)),
            fill=None if fill is None else tuple(fill),  # type: ignore[arg-type]
            boxed=bool(data.get("boxed", False)),
        )

    def bounds(self) -> tuple[float, float, float, float]:
        """Extent of the shape's control points."""
        xs = [p[0] for p in self.points] or [0.0]
        ys = [p[1] for p in self.points] or [0.0]
        return min(xs), min(ys), max(xs), max(ys)

    def bbox(self) -> tuple[float, float, float, float]:
        """Visual bounding box, including text extent and stroke thickness."""
        if self.kind == "text":
            return text_box(self)
        x0, y0, x1, y1 = self.bounds()
        pad = self.width / 2 + (6.0 if self.kind == "cloud" else 0.0)
        return x0 - pad, y0 - pad, x1 + pad, y1 + pad

    def translate(self, dx: float, dy: float) -> None:
        self.points = [(x + dx, y + dy) for x, y in self.points]


def font_faces(name: str) -> tuple[str, str]:
    return FONTS.get(name, FONTS[DEFAULT_FONT])


def text_size(shape: Shape) -> tuple[float, float]:
    """Width and height of a text shape's glyphs, in points."""
    pdf_font, _ = font_faces(shape.font)
    lines = shape.text.split("\n") or [""]
    width = max(
        (pymupdf.get_text_length(line, fontname=pdf_font, fontsize=shape.font_size)
         for line in lines),
        default=0.0,
    )
    return width, len(lines) * shape.font_size * 1.2


def text_box(shape: Shape) -> tuple[float, float, float, float]:
    """The box around a text shape, whether or not it is drawn."""
    if not shape.points:
        return (0.0, 0.0, 0.0, 0.0)
    x, y = shape.points[0]
    width, height = text_size(shape)
    return (x - TEXT_BOX_PAD, y - TEXT_BOX_PAD,
            x + width + TEXT_BOX_PAD, y + height + TEXT_BOX_PAD)


def _hex(color: RGB) -> str:
    return "#" + "".join(f"{round(max(0.0, min(1.0, c)) * 255):02x}" for c in color)


def simplify(points: list[tuple[float, float]],
             tolerance: float = PENCIL_TOLERANCE) -> list[tuple[float, float]]:
    """Ramer-Douglas-Peucker reduction of a freehand stroke."""
    if len(points) < 3:
        return list(points)

    keep = [False] * len(points)
    keep[0] = keep[-1] = True
    stack = [(0, len(points) - 1)]

    while stack:
        start, end = stack.pop()
        if end <= start + 1:
            continue
        ax, ay = points[start]
        bx, by = points[end]
        dx, dy = bx - ax, by - ay
        span = math.hypot(dx, dy)

        worst, worst_index = -1.0, start
        for i in range(start + 1, end):
            px, py = points[i]
            if span < 1e-9:
                distance = math.hypot(px - ax, py - ay)
            else:
                distance = abs(dy * px - dx * py + bx * ay - by * ax) / span
            if distance > worst:
                worst, worst_index = distance, i

        if worst > tolerance:
            keep[worst_index] = True
            stack.append((start, worst_index))
            stack.append((worst_index, end))

    return [p for p, k in zip(points, keep) if k]


def arrow_head(start: tuple[float, float], end: tuple[float, float],
               width: float) -> list[tuple[float, float]]:
    """The three points of an arrowhead at `end`, pointing away from `start`."""
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return []
    ux, uy = dx / length, dy / length
    # Sized generously relative to the stroke: at a 1.5pt line weight a head
    # scaled purely by stroke width is too small to read on a full sheet.
    size = min(max(10.0, width * 5.0), length * 0.5)
    bx, by = end[0] - ux * size, end[1] - uy * size
    return [end, (bx - uy * size * 0.4, by + ux * size * 0.4),
            (bx + uy * size * 0.4, by - ux * size * 0.4)]


def cloud_points(x0: float, y0: float, x1: float, y1: float,
                 arc: float = CLOUD_ARC) -> list[tuple[float, float]]:
    """Approximate a revision cloud as a closed polyline of scallops.

    Returned as a dense polyline rather than arcs so the same point list can be
    handed to both SVG and PyMuPDF without either needing arc support.
    """
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    width, height = x1 - x0, y1 - y0
    if width < 1 or height < 1:
        return []

    def run(start, end, count, normal):
        """Scallops along one edge, bulging in the `normal` direction."""
        points = []
        count = max(1, count)
        for i in range(count):
            ax = start[0] + (end[0] - start[0]) * i / count
            ay = start[1] + (end[1] - start[1]) * i / count
            bx = start[0] + (end[0] - start[0]) * (i + 1) / count
            by = start[1] + (end[1] - start[1]) * (i + 1) / count
            mx, my = (ax + bx) / 2, (ay + by) / 2
            # A quadratic Bezier only reaches halfway to its control point, so
            # the offset is the full segment length to get a scallop that bulges
            # by half its width -- i.e. a semicircle, the conventional cloud.
            offset = math.hypot(bx - ax, by - ay)
            cx, cy = mx + normal[0] * offset, my + normal[1] * offset
            for step in range(1, 7):
                t = step / 6
                inv = 1 - t
                points.append((inv * inv * ax + 2 * inv * t * cx + t * t * bx,
                               inv * inv * ay + 2 * inv * t * cy + t * t * by))
        return points

    across = max(1, round(width / arc))
    down = max(1, round(height / arc))
    result: list[tuple[float, float]] = [(x0, y0)]
    result += run((x0, y0), (x1, y0), across, (0, -1))
    result += run((x1, y0), (x1, y1), down, (1, 0))
    result += run((x1, y1), (x0, y1), across, (0, 1))
    result += run((x0, y1), (x0, y0), down, (-1, 0))
    return result


def geometry(shape: Shape) -> list[tuple[float, float]]:
    """The polyline a shape renders as, for hit testing and export."""
    if shape.kind == "cloud":
        return cloud_points(*shape.bounds())
    if shape.kind == "rect":
        x0, y0, x1, y1 = shape.bounds()
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
    return list(shape.points)


# -- per-kind rendering -------------------------------------------------

#: How each shape kind draws itself, keyed by `Shape.kind`. Plugins add entries
#: here to introduce a new kind of annotation without touching this module; see
#: `redliner/plugins/tools.py` and EXTENDING.md.
#:
#: The registry lives here rather than in the plugins package so that markup has
#: no dependency on plugin machinery -- shapes render the same whether or not
#: anything has been discovered yet.
SHAPE_SVG: dict[str, "Callable[[Shape], str]"] = {}
SHAPE_PDF: dict[str, "Callable[[pymupdf.Page, Shape], None]"] = {}


def register_shape(kind: str, svg=None, pdf=None) -> None:
    """Register how `kind` renders to the screen overlay and to an exported PDF."""
    if svg is not None:
        SHAPE_SVG[kind] = svg
    if pdf is not None:
        SHAPE_PDF[kind] = pdf


# -- SVG ----------------------------------------------------------------

def to_svg(shapes: list[Shape], width_pt: float, height_pt: float,
           selected: int | list[int] | None = None) -> str:
    """Render shapes as an SVG whose user units are PDF points.

    Each shape is wrapped in a group carrying its index and bounding box so the
    browser can hit-test and drag it without asking the server where things are.

    `selected` may be one index or several. Resize handles are offered only for
    a single selection -- with several picked, the browser marks the last one as
    the handle owner, and a corner drag would otherwise have to mean something
    for every shape at once.
    """
    if selected is None:
        chosen: list[int] = []
    elif isinstance(selected, int):
        chosen = [selected]
    else:
        chosen = list(selected)
    primary = chosen[-1] if len(chosen) == 1 else None
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width_pt:.2f} {height_pt:.2f}" '
        f'width="100%" height="100%" style="position:absolute;inset:0;overflow:visible">'
    ]
    for index, shape in enumerate(shapes):
        x0, y0, x1, y1 = shape.bbox()
        # Endpoints travel with the group so the browser can place resize
        # handles on a line without asking the server where its ends are.
        endpoints = ""
        if shape.kind in ("line", "arrow") and len(shape.points) >= 2:
            ends = (shape.points[0], shape.points[-1])
            endpoints = ' data-points="' + " ".join(
                f"{x:.2f},{y:.2f}" for x, y in ends) + '"'
        marker = ""
        if index == primary:
            marker = ' data-selected="1"'
        elif index in chosen:
            marker = ' data-picked="1"'
        parts.append(
            f'<g data-idx="{index}" data-bbox="{x0:.2f},{y0:.2f},{x1:.2f},{y1:.2f}" '
            f'data-kind="{shape.kind}"{endpoints}{marker}>'
            f"{_hit_svg(shape)}{_shape_svg(shape)}</g>"
        )
        if index in chosen:
            parts.append(
                f'<rect class="rl-ants" x="{x0:.2f}" y="{y0:.2f}" '
                f'width="{max(0.1, x1 - x0):.2f}" height="{max(0.1, y1 - y0):.2f}" '
                f'fill="none" stroke="#2196f3" stroke-width="1" '
                f'vector-effect="non-scaling-stroke" pointer-events="none"/>'
            )
    parts.append("</svg>")
    return "".join(parts)


def _hit_svg(shape: Shape) -> str:
    """An invisible fat stroke so thin shapes are still easy to click."""
    if shape.kind == "text":
        x0, y0, x1, y1 = text_box(shape)
        return (f'<rect x="{x0:.2f}" y="{y0:.2f}" width="{max(0.1, x1 - x0):.2f}" '
                f'height="{max(0.1, y1 - y0):.2f}" fill="transparent" stroke="none"/>')

    points = geometry(shape)
    if len(points) < 2:
        return ""
    path = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return (f'<polyline points="{path}" fill="none" stroke="transparent" '
            f'stroke-width="{max(6.0, shape.width * 4):.2f}" stroke-linejoin="round"/>')


def _paint(shape: Shape) -> str:
    fill = "none" if shape.fill is None else _hex(shape.fill)
    return (f'stroke="{_hex(shape.color)}" stroke-width="{shape.width}" '
            f'fill="{fill}" stroke-linecap="round" stroke-linejoin="round"')


def _shape_svg(shape: Shape) -> str:
    custom = SHAPE_SVG.get(shape.kind)
    if custom is not None:
        return custom(shape)
    return _builtin_svg(shape)


def _builtin_svg(shape: Shape) -> str:
    color = _hex(shape.color)
    paint = _paint(shape)
    stroke_only = (f'stroke="{color}" stroke-width="{shape.width}" fill="none" '
                   f'stroke-linecap="round" stroke-linejoin="round"')

    if shape.kind == "line" and len(shape.points) >= 2:
        (x0, y0), (x1, y1) = shape.points[0], shape.points[-1]
        return f'<line x1="{x0:.2f}" y1="{y0:.2f}" x2="{x1:.2f}" y2="{y1:.2f}" {stroke_only}/>'

    if shape.kind == "arrow" and len(shape.points) >= 2:
        start, end = shape.points[0], shape.points[-1]
        head = arrow_head(start, end, shape.width)
        pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in head)
        return (f'<line x1="{start[0]:.2f}" y1="{start[1]:.2f}" '
                f'x2="{end[0]:.2f}" y2="{end[1]:.2f}" {stroke_only}/>'
                f'<polygon points="{pts}" fill="{color}"/>')

    if shape.kind == "rect" and len(shape.points) >= 2:
        x0, y0, x1, y1 = shape.bounds()
        return (f'<rect x="{x0:.2f}" y="{y0:.2f}" width="{x1 - x0:.2f}" '
                f'height="{y1 - y0:.2f}" {paint}/>')

    if shape.kind == "cloud" and len(shape.points) >= 2:
        pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in cloud_points(*shape.bounds()))
        return f'<polygon points="{pts}" {paint}/>' if pts else ""

    if shape.kind == "pencil" and len(shape.points) >= 2:
        pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in shape.points)
        return f'<polyline points="{pts}" {stroke_only}/>'

    if shape.kind == "text" and shape.points and shape.text:
        x, y = shape.points[0]
        _, svg_font = font_faces(shape.font)
        prefix = ""
        if shape.boxed:
            bx0, by0, bx1, by1 = text_box(shape)
            prefix = (f'<rect x="{bx0:.2f}" y="{by0:.2f}" '
                      f'width="{bx1 - bx0:.2f}" height="{by1 - by0:.2f}" {paint}/>')
        spans = "".join(
            f'<tspan x="{x:.2f}" dy="{0 if i == 0 else shape.font_size * 1.2:.2f}">'
            f"{_escape(line)}</tspan>"
            for i, line in enumerate(shape.text.split("\n"))
        )
        return (f'{prefix}<text x="{x:.2f}" y="{y + shape.font_size:.2f}" fill="{color}" '
                f'font-size="{shape.font_size}" font-family="{svg_font}" '
                f'stroke="none">{spans}</text>')

    return ""


def _escape(text: str) -> str:
    return (text.replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


# -- PDF ----------------------------------------------------------------

def draw_on_page(page: pymupdf.Page, shapes: list[Shape]) -> None:
    """Stamp shapes onto an exported PDF page as vector content."""
    for shape in shapes:
        custom = SHAPE_PDF.get(shape.kind)
        if custom is not None:
            custom(page, shape)
            continue
        _builtin_pdf(page, shape)


def _builtin_pdf(page: pymupdf.Page, shape: Shape) -> None:
    color = tuple(float(c) for c in shape.color)
    fill = None if shape.fill is None else tuple(float(c) for c in shape.fill)

    if shape.kind in ("line", "arrow") and len(shape.points) >= 2:
        start, end = shape.points[0], shape.points[-1]
        page.draw_line(pymupdf.Point(*start), pymupdf.Point(*end),
                       color=color, width=shape.width)
        if shape.kind == "arrow":
            head = arrow_head(start, end, shape.width)
            if head:
                page.draw_polyline([pymupdf.Point(*p) for p in head] +
                                   [pymupdf.Point(*head[0])],
                                   color=color, fill=color, width=shape.width)

    elif shape.kind == "rect" and len(shape.points) >= 2:
        x0, y0, x1, y1 = shape.bounds()
        page.draw_rect(pymupdf.Rect(x0, y0, x1, y1), color=color,
                       fill=fill, width=shape.width)

    elif shape.kind == "cloud" and len(shape.points) >= 2:
        pts = cloud_points(*shape.bounds())
        if pts:
            page.draw_polyline([pymupdf.Point(*p) for p in pts] +
                               [pymupdf.Point(*pts[0])],
                               color=color, fill=fill, width=shape.width)

    elif shape.kind == "pencil" and len(shape.points) >= 2:
        page.draw_polyline([pymupdf.Point(*p) for p in shape.points],
                           color=color, width=shape.width)

    elif shape.kind == "text" and shape.points and shape.text:
        x, y = shape.points[0]
        pdf_font, _ = font_faces(shape.font)
        if shape.boxed:
            page.draw_rect(pymupdf.Rect(*text_box(shape)), color=color,
                           fill=fill, width=shape.width)
        for i, line in enumerate(shape.text.split("\n")):
            page.insert_text(
                pymupdf.Point(x, y + shape.font_size * (1 + i * 1.2)),
                line, fontsize=shape.font_size, fontname=pdf_font, color=color,
            )
