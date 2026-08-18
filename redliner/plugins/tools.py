"""Markup tools.

A tool is the bridge between a pointer gesture and a :class:`Shape`. Declaring
one gives you a toolbar button and the drawing behaviour behind it; the shape it
produces is rendered by the shape registry in :mod:`redliner.core.markup`, which
a plugin can also extend to introduce an entirely new kind of annotation.

The gesture vocabulary is deliberately small, because it has to be implemented
in the browser as well as here:

``drag``      two points -- press, move, release (line, box, cloud)
``click``     one point (text)
``freehand``  every sampled point along the drag (pencil)

Tools whose completion needs more than a shape -- text, which must ask for its
content, and magic align, which opens a dialog -- set ``opens_dialog`` and are
completed by the application. That distinction is honest rather than tidy: a
plugin can add shape tools with no knowledge of the UI, but a tool that needs
its own dialog has to cooperate with the app.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

from ..core.markup import Shape, simplify
from . import Registry

TOOLS: Registry["type[MarkupTool]"] = Registry("tool", "redliner.tools")

Gesture = Literal["drag", "click", "freehand"]


class MarkupTool(ABC):
    """One entry in the markup toolbar."""

    #: Registry key, and the value sent to the browser as the active tool.
    name: str = ""
    #: Material Icons ligature. Must be a real one: an invalid name does not
    #: fail loudly, it renders a partial glyph over the neighbouring button.
    icon: str = "edit"
    tooltip: str = ""
    #: Lower sorts earlier in the toolbar.
    priority: int = 100
    gesture: Gesture = "drag"
    #: Set when the application, not the tool, finishes the interaction.
    opens_dialog: bool = False

    @classmethod
    def create(cls, points: list[tuple[float, float]], style: dict) -> Shape | None:
        """Build a shape from the gesture, or None to discard it.

        `style` carries the markup panel's current settings: color, width, fill,
        font, font_size, boxed.
        """
        return None

    @classmethod
    def handles(cls, shape: Shape) -> list[str]:
        """Resize handles this shape offers. ``nw/ne/se/sw`` for boxes,
        ``p0``/``p1`` for endpoints, empty for move-only."""
        return []


def _shape(kind: str, points: list[tuple[float, float]], style: dict) -> Shape:
    return Shape(
        kind=kind, points=points,
        color=style.get("color", (0.85, 0.0, 0.0)),
        width=style.get("width", 1.5),
        fill=style.get("fill"),
        font=style.get("font", "sans"),
        font_size=style.get("font_size", 11.0),
        boxed=style.get("boxed", False),
    )


# -- built-in tools -----------------------------------------------------

@TOOLS.register
class PanTool(MarkupTool):
    name = "pan"
    icon = "pan_tool"
    tooltip = "Pan and zoom"
    priority = 0
    gesture = "drag"


@TOOLS.register
class SelectTool(MarkupTool):
    name = "select"
    icon = "highlight_alt"
    tooltip = "Select — move, resize, edit or delete markup"
    priority = 10


@TOOLS.register
class PencilTool(MarkupTool):
    name = "pencil"
    icon = "gesture"
    tooltip = "Freehand"
    priority = 20
    gesture = "freehand"

    @classmethod
    def create(cls, points, style):
        thinned = simplify(points)
        return _shape("pencil", thinned, style) if len(thinned) >= 2 else None


@TOOLS.register
class LineTool(MarkupTool):
    name = "line"
    icon = "horizontal_rule"
    tooltip = "Line"
    priority = 30

    @classmethod
    def create(cls, points, style):
        return _shape("line", points[:2], style) if len(points) >= 2 else None

    @classmethod
    def handles(cls, shape):
        return ["p0", "p1"]


@TOOLS.register
class ArrowTool(LineTool):
    name = "arrow"
    icon = "north_east"
    tooltip = "Arrow"
    priority = 40

    @classmethod
    def create(cls, points, style):
        return _shape("arrow", points[:2], style) if len(points) >= 2 else None


@TOOLS.register
class RectTool(MarkupTool):
    name = "rect"
    icon = "crop_square"
    tooltip = "Rectangle"
    priority = 50

    @classmethod
    def create(cls, points, style):
        return _shape("rect", points[:2], style) if len(points) >= 2 else None

    @classmethod
    def handles(cls, shape):
        return ["nw", "ne", "se", "sw"]


@TOOLS.register
class CloudTool(RectTool):
    name = "cloud"
    icon = "cloud_queue"
    tooltip = "Revision cloud"
    priority = 60

    @classmethod
    def create(cls, points, style):
        return _shape("cloud", points[:2], style) if len(points) >= 2 else None


@TOOLS.register
class TextTool(MarkupTool):
    name = "text"
    icon = "title"
    tooltip = "Text box"
    priority = 70
    gesture = "click"
    opens_dialog = True


@TOOLS.register
class AlignTool(MarkupTool):
    name = "align"
    icon = "open_with"
    tooltip = "Magic align — correct a local misalignment"
    priority = 80
    opens_dialog = True


def handles_for(shape: Shape) -> list[str]:
    """Resize handles for a shape, asking the tool that owns its kind."""
    for tool in TOOLS.all():
        if tool.name == shape.kind:
            return tool.handles(shape)
    return []
