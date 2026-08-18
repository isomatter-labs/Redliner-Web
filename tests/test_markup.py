"""Markup geometry, SVG output, and vector export."""

from __future__ import annotations

import math

import pymupdf
import pytest

from redliner.core.export import ExportOptions, build_pdf
from redliner.core.markup import (FONTS, Shape, arrow_head, cloud_points,
                                  simplify, text_box, text_size, to_svg)

from .test_pipeline import make_project


def test_shape_round_trips_through_a_dict() -> None:
    shape = Shape(kind="arrow", points=[(1.0, 2.0), (3.0, 4.0)],
                  color=(1.0, 0.0, 0.0), width=2.0, text="hi", font_size=14.0)
    assert Shape.from_dict(shape.to_dict()) == shape


def test_arrow_head_points_away_from_the_start() -> None:
    head = arrow_head((0.0, 0.0), (10.0, 0.0), width=1.5)
    assert len(head) == 3
    assert head[0] == (10.0, 0.0)
    # The two barbs sit behind the tip and straddle the shaft.
    assert all(p[0] < 10.0 for p in head[1:])
    assert head[1][1] * head[2][1] < 0


def test_arrow_head_is_empty_for_a_zero_length_arrow() -> None:
    assert arrow_head((5.0, 5.0), (5.0, 5.0), width=1.5) == []


def test_cloud_stays_near_its_box_and_closes() -> None:
    points = cloud_points(10.0, 20.0, 110.0, 80.0)
    assert len(points) > 20

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    # Scallops bulge outward, so the outline exceeds the box, but only by about
    # one arc radius -- a runaway control point would show up here.
    assert min(xs) > 10.0 - 20 and max(xs) < 110.0 + 20
    assert min(ys) > 20.0 - 20 and max(ys) < 80.0 + 20

    start, end = points[0], points[-1]
    assert math.hypot(end[0] - start[0], end[1] - start[1]) < 20


def test_degenerate_cloud_produces_nothing() -> None:
    assert cloud_points(5.0, 5.0, 5.0, 5.0) == []


@pytest.mark.parametrize("kind,expect", [
    ("line", "<line"), ("arrow", "<polygon"), ("rect", "<rect"), ("cloud", "<polygon"),
])
def test_each_shape_kind_renders_svg(kind: str, expect: str) -> None:
    svg = to_svg([Shape(kind=kind, points=[(10.0, 10.0), (60.0, 40.0)])], 612, 792)
    assert svg.startswith("<svg")
    assert 'viewBox="0 0 612.00 792.00"' in svg
    assert expect in svg


def test_text_shape_escapes_markup() -> None:
    svg = to_svg([Shape(kind="text", points=[(5.0, 5.0)], text='a <b> & "c"')], 100, 100)
    assert "&lt;b&gt;" in svg and "&amp;" in svg
    assert "<b>" not in svg


def test_multiline_text_becomes_separate_spans() -> None:
    svg = to_svg([Shape(kind="text", points=[(5.0, 5.0)], text="one\ntwo\nthree")], 100, 100)
    assert svg.count("<tspan") == 3


def test_empty_text_shape_renders_nothing() -> None:
    assert to_svg([Shape(kind="text", points=[(5.0, 5.0)], text="")], 100, 100) \
        .count("<text") == 0


def test_markup_is_exported_as_vectors_not_pixels() -> None:
    project = make_project("ab")
    project.pages[0].markups = [
        Shape(kind="rect", points=[(100.0, 100.0), (300.0, 200.0)],
              color=(1.0, 0.0, 0.0), width=2.0),
        Shape(kind="text", points=[(120.0, 240.0)], text="see note 4",
              color=(1.0, 0.0, 0.0), font_size=12.0),
    ]
    data = build_pdf(project, [0], ExportOptions(dpi=100))

    with pymupdf.open(stream=data, filetype="pdf") as pdf:
        page = pdf[0]
        assert "see note 4" in page.get_text()
        # Vector drawings, distinct from the embedded comparison raster.
        drawings = page.get_drawings()
        assert drawings, "markup was not written as vector content"
        assert any(abs(d["rect"].width - 200) < 5 for d in drawings)


def test_markups_can_be_left_out_of_the_export() -> None:
    project = make_project("ab")
    project.pages[0].markups = [
        Shape(kind="rect", points=[(100.0, 100.0), (300.0, 200.0)])
    ]
    data = build_pdf(project, [0], ExportOptions(dpi=100, markups=False))
    with pymupdf.open(stream=data, filetype="pdf") as pdf:
        assert not pdf[0].get_drawings()


# -- freehand -----------------------------------------------------------

def test_simplify_thins_a_dense_stroke_but_keeps_its_ends() -> None:
    """Pointer samples arrive far denser than the stroke's real detail."""
    straight = [(float(i) * 0.5, 10.0) for i in range(200)]
    thinned = simplify(straight)

    assert len(thinned) < 10, "a straight drag should collapse to a few points"
    assert thinned[0] == straight[0]
    assert thinned[-1] == straight[-1]


def test_simplify_preserves_genuine_corners() -> None:
    corner = [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0), (20.0, 10.0), (20.0, 20.0)]
    thinned = simplify(corner)
    assert (20.0, 0.0) in thinned, "the corner is the whole shape of this stroke"


def test_simplify_leaves_short_strokes_alone() -> None:
    assert simplify([(0.0, 0.0), (1.0, 1.0)]) == [(0.0, 0.0), (1.0, 1.0)]


def test_pencil_renders_and_exports_as_a_polyline() -> None:
    stroke = Shape(kind="pencil",
                   points=[(10.0, 10.0), (20.0, 30.0), (40.0, 20.0), (60.0, 50.0)])
    assert "<polyline" in to_svg([stroke], 612, 792)

    project = make_project("ab")
    project.pages[0].markups = [stroke]
    data = build_pdf(project, [0], ExportOptions(dpi=100))
    with pymupdf.open(stream=data, filetype="pdf") as pdf:
        assert pdf[0].get_drawings(), "freehand stroke missing from the export"


# -- fill ---------------------------------------------------------------

def test_transparent_fill_is_the_default() -> None:
    svg = to_svg([Shape(kind="rect", points=[(0.0, 0.0), (10.0, 10.0)])], 100, 100)
    assert 'fill="none"' in svg


def test_fill_colour_reaches_svg_and_pdf() -> None:
    box = Shape(kind="rect", points=[(20.0, 20.0), (120.0, 80.0)],
                fill=(1.0, 1.0, 0.0), color=(1.0, 0.0, 0.0))
    assert 'fill="#ffff00"' in to_svg([box], 612, 792)

    project = make_project("ab")
    project.pages[0].markups = [box]
    data = build_pdf(project, [0], ExportOptions(dpi=100))
    with pymupdf.open(stream=data, filetype="pdf") as pdf:
        fills = [d.get("fill") for d in pdf[0].get_drawings()]
    assert any(f is not None for f in fills), "fill did not survive export"


def test_cloud_accepts_a_fill() -> None:
    cloud = Shape(kind="cloud", points=[(10.0, 10.0), (90.0, 60.0)], fill=(0.9, 0.9, 0.9))
    assert 'fill="#e6e6e6"' in to_svg([cloud], 200, 200)


# -- fonts --------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(FONTS))
def test_each_font_renders_and_exports(name: str) -> None:
    note = Shape(kind="text", points=[(50.0, 50.0)], text="TORQUE 12 IN-LB", font=name)
    _, svg_family = FONTS[name]
    assert svg_family in to_svg([note], 612, 792)

    project = make_project("ab")
    project.pages[0].markups = [note]
    data = build_pdf(project, [0], ExportOptions(dpi=100))
    with pymupdf.open(stream=data, filetype="pdf") as pdf:
        assert "TORQUE 12 IN-LB" in pdf[0].get_text()


def test_font_choice_changes_measured_width() -> None:
    """Monospace is wider than proportional for the same string, so the box
    around the text has to be measured per font rather than assumed."""
    sans = Shape(kind="text", points=[(0.0, 0.0)], text="iiiii", font="sans")
    mono = Shape(kind="text", points=[(0.0, 0.0)], text="iiiii", font="mono")
    assert text_size(mono)[0] > text_size(sans)[0]


# -- boxed text ---------------------------------------------------------

def test_boxed_text_draws_a_box_sized_to_the_text() -> None:
    note = Shape(kind="text", points=[(100.0, 100.0)], text="SEE NOTE 4",
                 boxed=True, fill=(1.0, 1.0, 1.0))
    svg = to_svg([note], 612, 792)
    assert svg.count("<rect") >= 1

    x0, y0, x1, y1 = text_box(note)
    width, _ = text_size(note)
    assert x1 - x0 == pytest.approx(width + 6.0, abs=0.1)
    assert x0 < 100.0 and y0 < 100.0, "the box should surround the anchor point"


def test_unboxed_text_draws_no_box() -> None:
    """Both variants carry one invisible rect as a hit target, so the test is
    that boxing adds exactly one more rect -- the visible one."""
    plain = Shape(kind="text", points=[(10.0, 10.0)], text="plain", boxed=False)
    boxed = Shape(kind="text", points=[(10.0, 10.0)], text="plain", boxed=True)

    assert to_svg([boxed], 200, 200).count("<rect") == \
        to_svg([plain], 200, 200).count("<rect") + 1


def test_boxed_text_exports_a_box() -> None:
    project = make_project("ab")
    project.pages[0].markups = [
        Shape(kind="text", points=[(100.0, 200.0)], text="SEE NOTE 4",
              boxed=True, fill=(1.0, 1.0, 0.6))
    ]
    data = build_pdf(project, [0], ExportOptions(dpi=100))
    with pymupdf.open(stream=data, filetype="pdf") as pdf:
        assert pdf[0].get_drawings(), "text box missing from the export"
        assert "SEE NOTE 4" in pdf[0].get_text()


def test_text_bbox_covers_the_glyphs_not_just_the_anchor() -> None:
    note = Shape(kind="text", points=[(10.0, 10.0)], text="a longish note here")
    x0, y0, x1, y1 = note.bbox()
    assert x1 - x0 > 20, "bbox collapsed to the anchor point"
    assert y1 - y0 > 5


# -- selection metadata -------------------------------------------------

def test_groups_carry_index_and_bbox_for_hit_testing() -> None:
    svg = to_svg([Shape(kind="rect", points=[(0.0, 0.0), (10.0, 10.0)])], 100, 100)
    assert 'data-idx="0"' in svg
    assert "data-bbox=" in svg
    assert 'data-kind="rect"' in svg


def test_selected_group_is_marked_and_outlined() -> None:
    shapes = [Shape(kind="rect", points=[(0.0, 0.0), (10.0, 10.0)]),
              Shape(kind="rect", points=[(20.0, 20.0), (30.0, 30.0)])]
    svg = to_svg(shapes, 100, 100, selected=1)
    assert 'data-selected="1"' in svg
    assert "rl-ants" in svg, "selection outline should use the marching-ants class"
    assert svg.count('data-selected="1"') == 1


def test_line_endpoints_travel_with_the_group() -> None:
    """Handles for a line sit on its ends, which the bbox alone cannot give."""
    svg = to_svg([Shape(kind="line", points=[(10.0, 90.0), (80.0, 20.0)])], 100, 100)
    assert 'data-points="10.00,90.00 80.00,20.00"' in svg


def test_thin_shapes_get_a_fat_invisible_hit_target() -> None:
    svg = to_svg([Shape(kind="line", points=[(0.0, 0.0), (50.0, 0.0)], width=0.5)],
                 100, 100)
    assert 'stroke="transparent"' in svg


def test_translate_moves_every_point() -> None:
    shape = Shape(kind="line", points=[(1.0, 2.0), (3.0, 4.0)])
    shape.translate(10.0, -5.0)
    assert shape.points == [(11.0, -3.0), (13.0, -1.0)]


def test_markup_position_is_independent_of_render_dpi() -> None:
    """Shapes are stored in points, so the same annotation must land in the
    same place whether the page was previewed at 100 or exported at 400 DPI."""
    rects = []
    for dpi in (100, 400):
        project = make_project("ab")
        project.pages[0].markups = [
            Shape(kind="rect", points=[(100.0, 100.0), (300.0, 200.0)], width=2.0)
        ]
        data = build_pdf(project, [0], ExportOptions(dpi=dpi))
        with pymupdf.open(stream=data, filetype="pdf") as pdf:
            rects.append(pdf[0].get_drawings()[0]["rect"])

    assert abs(rects[0].x0 - rects[1].x0) < 1
    assert abs(rects[0].y0 - rects[1].y0) < 1
    assert abs(rects[0].width - rects[1].width) < 1
