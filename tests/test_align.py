"""Magic-align: shift estimation and region-scoped patch application."""

from __future__ import annotations

import numpy as np
import pytest

from redliner.core.align import (AlignPatch, apply_patches, auto_align, bounds_of,
                                 estimate_shift, normalize_rect, search_radius,
                                 window)
from redliner.core.compose import DiffSettings, Layer, composite
from redliner.core.project import Project

from .test_pipeline import make_project


def ink_with_mark(shape=(80, 80), top=30, left=30, size=10) -> np.ndarray:
    """An ink-coverage array with one solid block."""
    a = np.zeros(shape, dtype=np.float32)
    a[top : top + size, left : left + size] = 1.0
    return a


def paper_with_mark(shape=(80, 80), top=30, left=30, size=10) -> np.ndarray:
    """A grayscale raster (255 = paper) with one solid black block."""
    a = np.full(shape, 255, dtype=np.uint8)
    a[top : top + size, left : left + size] = 0
    return a


# -- shift estimation ---------------------------------------------------

@pytest.mark.parametrize("dx,dy", [(5, 0), (0, 4), (-6, 3), (7, -5), (0, 0)])
def test_estimate_shift_recovers_a_known_offset(dx: int, dy: int) -> None:
    """The returned shift is the one to APPLY to `moving` to land on reference."""
    reference = ink_with_mark(top=30, left=30)
    moving = ink_with_mark(top=30 + dy, left=30 + dx)

    assert estimate_shift(reference, moving, max_shift=20) == (-dx, -dy)


def test_estimated_shift_actually_aligns_the_layers() -> None:
    """Round-trip: feeding the estimate back through the patch machinery must
    make the two layers agree, which is the only thing that really matters."""
    reference = paper_with_mark(top=30, left=30)
    moving = paper_with_mark(top=34, left=37)

    dx, dy = estimate_shift(255 - reference.astype(np.float32),
                            255 - moving.astype(np.float32), max_shift=20)

    # 72 pt per inch at 72 dpi means 1 point == 1 pixel, keeping the test in
    # pixel units without hiding the point<->pixel conversion.
    patch = AlignPatch(rect=(0, 0, 80, 80), offsets={"b": (float(dx), float(dy))})
    fixed = apply_patches([reference, moving], ["a", "b"], [patch], dpi=72)

    assert np.array_equal(fixed[0], fixed[1])


def test_estimate_shift_is_blank_safe() -> None:
    blank = np.zeros((40, 40), dtype=np.float32)
    assert estimate_shift(blank, ink_with_mark((40, 40), 10, 10, 5), 10) == (0, 0)
    assert estimate_shift(ink_with_mark((40, 40), 10, 10, 5), blank, 10) == (0, 0)


def test_estimate_shift_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="matching shapes"):
        estimate_shift(np.zeros((4, 4), np.float32), np.zeros((5, 4), np.float32), 2)


def test_shift_is_not_biased_by_ink_falling_out_of_the_window() -> None:
    """Raw correlation of sparse ink peaks wherever the most ink overlaps, which
    drags the answer toward zero shift. Mean-subtraction is what prevents that,
    so a mark near the edge must still be measured correctly."""
    reference = ink_with_mark((60, 60), top=25, left=4, size=8)
    moving = ink_with_mark((60, 60), top=25, left=12, size=8)
    assert estimate_shift(reference, moving, max_shift=15) == (-8, 0)


# -- search radius ------------------------------------------------------

def test_search_radius_scales_with_the_drawn_region() -> None:
    assert search_radius(400, 400) > search_radius(80, 80)


def test_search_radius_follows_the_smaller_side() -> None:
    assert search_radius(1000, 60) == search_radius(60, 1000)


def test_search_radius_is_clamped_at_both_ends() -> None:
    assert search_radius(1, 1) >= 4
    assert search_radius(100_000, 100_000) <= 120


# -- window -------------------------------------------------------------

def test_window_pads_out_of_bounds_with_paper() -> None:
    source = np.zeros((10, 10), dtype=np.uint8)
    out = window(source, -3, -3, 6, 6)
    assert out.shape == (6, 6)
    assert (out[:3, :3] == 255).all(), "off-page area must read as paper"
    assert (out[3:, 3:] == 0).all()


def test_window_entirely_outside_is_all_paper() -> None:
    source = np.zeros((10, 10), dtype=np.uint8)
    assert (window(source, 50, 50, 4, 4) == 255).all()


# -- patch application --------------------------------------------------

def test_patch_moves_only_pixels_inside_its_region() -> None:
    raster = np.full((80, 80), 255, dtype=np.uint8)
    raster[10:20, 10:20] = 0          # inside the patch region
    raster[60:70, 60:70] = 0          # outside it
    anchor = np.full((80, 80), 255, dtype=np.uint8)

    patch = AlignPatch(rect=(0, 0, 40, 40), offsets={"b": (5.0, 0.0)})
    out = apply_patches([anchor, raster], ["a", "b"], [patch], dpi=72)

    assert (out[1][10:20, 15:25] == 0).all(), "in-region mark did not shift"
    assert (out[1][60:70, 60:70] == 0).all(), "out-of-region mark must not move"
    assert np.array_equal(out[0], anchor), "anchor document must never move"


def test_identity_patches_are_a_no_op() -> None:
    rasters = [np.full((20, 20), 255, np.uint8), np.full((20, 20), 255, np.uint8)]
    patch = AlignPatch(rect=(0, 0, 10, 10), offsets={"b": (0.0, 0.0)})
    out = apply_patches(rasters, ["a", "b"], [patch], dpi=72)
    assert out[0] is rasters[0] and out[1] is rasters[1]


def test_overlapping_patches_do_not_compound() -> None:
    """Both patches sample the original, so a pixel covered twice is shifted
    once -- not by the sum of the two offsets."""
    raster = np.full((60, 60), 255, dtype=np.uint8)
    raster[20:26, 20:26] = 0
    anchor = np.full((60, 60), 255, dtype=np.uint8)

    patches = [
        AlignPatch(rect=(0, 0, 60, 60), offsets={"b": (4.0, 0.0)}),
        AlignPatch(rect=(0, 0, 60, 60), offsets={"b": (4.0, 0.0)}),
    ]
    out = apply_patches([anchor, raster], ["a", "b"], patches, dpi=72)

    assert (out[1][20:26, 24:30] == 0).all()
    assert (out[1][20:26, 28:34] == 255).any(), "offset was applied twice"


def test_patch_offsets_are_in_points_so_dpi_scales_them() -> None:
    raster = np.full((200, 200), 255, dtype=np.uint8)
    raster[100:110, 100:110] = 0
    anchor = np.full((200, 200), 255, dtype=np.uint8)
    patch = AlignPatch(rect=(0, 0, 200, 200), offsets={"b": (10.0, 0.0)})

    at_72 = apply_patches([anchor, raster], ["a", "b"], [patch], dpi=72)
    at_144 = apply_patches([anchor, raster], ["a", "b"], [patch], dpi=144)

    # 10 points is 10 px at 72 dpi and 20 px at 144 dpi.
    assert (at_72[1][100:110, 110:120] == 0).all()
    assert (at_144[1][100:110, 120:130] == 0).all()


def test_patch_clips_to_the_raster_bounds() -> None:
    raster = np.full((30, 30), 255, dtype=np.uint8)
    anchor = raster.copy()
    patch = AlignPatch(rect=(-100, -100, 500, 500), offsets={"b": (3.0, 3.0)})
    out = apply_patches([anchor, raster], ["a", "b"], [patch], dpi=72)
    assert out[1].shape == (30, 30)


# -- auto align ---------------------------------------------------------

def test_auto_align_anchors_the_first_document() -> None:
    layers = [ink_with_mark(left=30), ink_with_mark(left=36), ink_with_mark(left=27)]
    shifts = auto_align(layers, max_shift=20)

    assert shifts[0] == (0, 0), "first document is the anchor and must not move"
    assert shifts[1][0] == pytest.approx(-6, abs=0.2)
    assert shifts[2][0] == pytest.approx(3, abs=0.2)
    assert shifts[1][1] == pytest.approx(0, abs=0.2)


def test_auto_align_gives_up_when_the_anchor_region_is_blank() -> None:
    layers = [np.zeros((40, 40), np.float32), ink_with_mark((40, 40), 10, 10, 5)]
    assert auto_align(layers, max_shift=10) == [(0, 0), (0, 0)]


def test_subpixel_estimate_recovers_a_fractional_shift() -> None:
    """A real 3.0 pt drift is 11.875 px at 285 DPI. Whole-pixel rounding leaves
    residue that grows when the offset is re-applied at export resolution, so
    the estimate has to resolve fractions."""
    reference = ink_with_mark((80, 80), top=30, left=30, size=12)
    # Half-way between a 5 px and a 6 px shift.
    at_5 = ink_with_mark((80, 80), top=30, left=35, size=12)
    at_6 = ink_with_mark((80, 80), top=30, left=36, size=12)
    moving = 0.5 * at_5 + 0.5 * at_6

    dx, _ = estimate_shift(reference, moving, max_shift=15, subpixel=True)
    assert dx == pytest.approx(-5.5, abs=0.35)

    whole, _ = estimate_shift(reference, moving, max_shift=15)
    assert whole == round(whole), "the default path must stay whole-pixel"


def test_subpixel_does_not_disturb_an_exact_shift() -> None:
    reference = ink_with_mark((80, 80), left=30)
    moving = ink_with_mark((80, 80), left=37)
    dx, dy = estimate_shift(reference, moving, max_shift=20, subpixel=True)
    assert dx == pytest.approx(-7, abs=0.05)
    assert dy == pytest.approx(0, abs=0.05)


def test_auto_align_of_nothing() -> None:
    assert auto_align([], max_shift=10) == []


# -- difference-only view -----------------------------------------------

def test_difference_only_view_drops_shared_ink() -> None:
    black = np.zeros((1, 1), dtype=np.uint8)
    layers = [Layer(black, (1.0, 0.0, 0.0)), Layer(black, (0.0, 0.0, 1.0))]

    normal = composite(layers, DiffSettings(highlight=False, show_unchanged=True))
    diff_only = composite(layers, DiffSettings(highlight=False, show_unchanged=False))

    assert tuple(normal[0, 0]) == (0, 0, 0), "shared ink should render black"
    assert tuple(diff_only[0, 0]) == (255, 255, 255), "shared ink should drop out"


def test_difference_only_view_keeps_unique_ink() -> None:
    black = np.zeros((1, 1), dtype=np.uint8)
    white = np.full((1, 1), 255, dtype=np.uint8)
    out = composite([Layer(black, (1.0, 0.0, 0.0)), Layer(white, (0.0, 0.0, 1.0))],
                    DiffSettings(highlight=False, show_unchanged=False))
    assert tuple(out[0, 0]) == (255, 0, 0)


# -- integration --------------------------------------------------------

def test_align_patch_cleans_up_a_shifted_page() -> None:
    """The whole point: a document nudged a few points off should compare clean
    once a patch corrects it."""
    project = make_project("aa", dpi=100)
    page_w, page_h = project.page_size_points(0)

    def chromatic(rgb: np.ndarray) -> int:
        flat = rgb.reshape(-1, 3).astype(int)
        return int(((flat.max(axis=1) - flat.min(axis=1)) > 40).sum())

    assert chromatic(project.render(0)) == 0, "identical docs should start clean"

    drift = AlignPatch(rect=(0, 0, page_w, page_h),
                       offsets={project.docs[1].doc_id: (2.0, 1.0)})
    project.pages[0].patches = [drift]
    assert chromatic(project.render(0)) > 100, "the shift should show up as colour"

    project.pages[0].patches = [
        AlignPatch(rect=(0, 0, page_w, page_h),
                   offsets={project.docs[1].doc_id: (0.0, 0.0)})
    ]
    assert chromatic(project.render(0)) == 0, "removing the shift should clean up"


def _jitter_pair(tmp_path, offset=(3.0, 2.0)):
    """Two PDFs identical except one text block nudged, as a CAD export would."""
    import pymupdf

    paths = []
    for name, (jx, jy) in (("base", (0.0, 0.0)), ("drift", offset)):
        doc = pymupdf.open()
        page = doc.new_page(width=612, height=792)
        page.draw_rect(pymupdf.Rect(40, 40, 572, 752), color=(0, 0, 0), width=1.2)
        page.insert_text(pymupdf.Point(380 + jx, 200 + jy),
                         "NOTE 3: TORQUE TO 12 IN-LB", fontsize=11)
        path = tmp_path / f"{name}.pdf"
        doc.save(path)
        doc.close()
        paths.append(path)
    return paths


def test_subpixel_offset_reaches_the_renderer(tmp_path) -> None:
    """A 0.3 pt offset is well under one pixel at 100 DPI (0.72 pt). If offsets
    were applied by shifting finished pixels it would round away to nothing; the
    clip window is continuous, so it must still change the render."""
    from redliner.core.documents import load_document

    base, drift = _jitter_pair(tmp_path)
    project = Project(dpi=100)
    project.add_document(load_document(base, "base", "base"))
    project.add_document(load_document(drift, "drift", "drift"))

    rect = (360.0, 180.0, 570.0, 215.0)
    unshifted = project.region_rasters(0, rect, dpi=100, offsets={})
    nudged = project.region_rasters(0, rect, dpi=100, offsets={"drift": (0.3, 0.0)})

    assert not np.array_equal(unshifted[1], nudged[1]), \
        "sub-pixel offset was rounded away instead of reaching the renderer"


def test_exact_offset_cancels_a_jittered_block(tmp_path) -> None:
    """The end-to-end promise: correcting the drift removes the false positive."""
    from redliner.core.documents import load_document

    base, drift = _jitter_pair(tmp_path, offset=(3.0, 2.0))
    project = Project(dpi=150)
    project.settings = DiffSettings(highlight=False)
    project.add_document(load_document(base, "base", "base"))
    project.add_document(load_document(drift, "drift", "drift"))

    def chromatic(rgb: np.ndarray) -> int:
        flat = rgb.reshape(-1, 3).astype(int)
        return int(((flat.max(axis=1) - flat.min(axis=1)) > 40).sum())

    # The region's edges must sit on blank paper. Shifting the window pulls in
    # whatever lies just outside it, so an edge grazing the page frame at
    # x=572 would drag that frame into one document and not the other --
    # manufacturing a difference at the seam. This is the same reason the tool
    # tells users to draw the blob clear of surrounding geometry.
    rect = (360.0, 178.0, 555.0, 214.0)
    before = chromatic(project.render_region(0, rect, dpi=150))
    assert before > 200, "the injected drift should read as a large difference"

    corrected = chromatic(project.render_region(
        0, rect, dpi=150, offsets={"drift": (-3.0, -2.0)}))
    assert corrected == 0, (
        f"correcting the drift should eliminate it entirely: {before} -> {corrected}"
    )


@pytest.mark.parametrize("dpi", [100, 150, 203, 300, 601])
def test_correction_is_exact_at_any_resolution(tmp_path, dpi: int) -> None:
    """A 3 pt shift is 4.17 px at 100 DPI and 8.46 px at 203 -- rarely a whole
    number. Applying the offset through the render matrix rather than by moving
    pixels is what makes every one of these cancel completely; whole-pixel
    shifting leaves fringing that scales with how far the offset sits from an
    integer."""
    from redliner.core.documents import load_document

    base, drift = _jitter_pair(tmp_path, offset=(3.0, 2.0))
    project = Project(dpi=float(dpi))
    project.settings = DiffSettings(highlight=False)
    project.add_document(load_document(base, "base", "base"))
    project.add_document(load_document(drift, "drift", "drift"))

    rect = (360.0, 178.0, 555.0, 214.0)

    def chromatic(rgb: np.ndarray) -> int:
        flat = rgb.reshape(-1, 3).astype(int)
        return int(((flat.max(axis=1) - flat.min(axis=1)) > 40).sum())

    assert chromatic(project.render_region(0, rect, dpi=dpi)) > 100
    assert chromatic(project.render_region(
        0, rect, dpi=dpi, offsets={"drift": (-3.0, -2.0)})) == 0


def test_auto_align_finds_the_jitter_end_to_end(tmp_path) -> None:
    from redliner.core.compose import to_ink
    from redliner.core.documents import load_document

    base, drift = _jitter_pair(tmp_path, offset=(3.0, 2.0))
    project = Project(dpi=150)
    project.add_document(load_document(base, "base", "base"))
    project.add_document(load_document(drift, "drift", "drift"))

    rect = (360.0, 180.0, 570.0, 215.0)
    dpi = 300.0
    crops = project.region_rasters(0, rect, dpi=dpi)
    inks = [to_ink(c) for c in crops]
    shifts = auto_align(inks, search_radius(inks[0].shape[1], inks[0].shape[0]))

    step = 72.0 / dpi  # points per pixel
    assert shifts[1][0] * step == pytest.approx(-3.0, abs=0.3)
    assert shifts[1][1] * step == pytest.approx(-2.0, abs=0.3)


def test_region_render_is_cropped_and_difference_only() -> None:
    project = make_project("ab", dpi=100)
    rect = (50.0, 50.0, 250.0, 150.0)
    region = project.render_region(0, rect, dpi=100)

    # 200 x 100 points at 100 dpi.
    assert region.shape[0] == pytest.approx(139, abs=2)
    assert region.shape[1] == pytest.approx(278, abs=2)
    flat = region.reshape(-1, 3)
    assert (flat.max(axis=1) < 40).sum() == 0, "no black: shared ink must drop out"


# -- lasso regions ------------------------------------------------------

def test_polygon_mask_covers_only_the_outline() -> None:
    from redliner.core.align import polygon_mask

    triangle = [(0.0, 0.0), (72.0, 0.0), (0.0, 72.0)]
    mask = polygon_mask(triangle, (0.0, 0.0), dpi=72, shape=(72, 72))

    assert mask.shape == (72, 72)
    assert mask[2, 2], "just inside the right angle should be covered"
    assert not mask[60, 60], "well outside the hypotenuse should not be"
    # A right triangle is about half the box.
    assert 0.35 < mask.mean() < 0.65


def test_polygon_mask_translates_to_the_window_origin() -> None:
    from redliner.core.align import polygon_mask

    square = [(100.0, 100.0), (140.0, 100.0), (140.0, 140.0), (100.0, 140.0)]
    mask = polygon_mask(square, (100.0, 100.0), dpi=72, shape=(40, 40))
    assert mask.mean() > 0.9, "the window starts at the polygon, so it fills it"


def test_degenerate_polygon_covers_everything() -> None:
    """Fewer than three points is not an outline; treat it as the whole box
    rather than silently masking the correction away to nothing."""
    from redliner.core.align import polygon_mask

    mask = polygon_mask([(0.0, 0.0), (10.0, 10.0)], (0.0, 0.0), 72, (10, 10))
    assert mask.all()


def test_lasso_moves_only_what_it_encloses() -> None:
    """The reason for a blob over a box: a line passing beside the drifting
    element must stay exactly where it was."""
    raster = np.full((80, 80), 255, dtype=np.uint8)
    raster[20:30, 20:30] = 0        # inside the lasso
    raster[20:30, 60:70] = 0        # outside it, must not move
    anchor = np.full((80, 80), 255, dtype=np.uint8)

    lasso = [(15.0, 15.0), (40.0, 15.0), (40.0, 40.0), (15.0, 40.0)]
    patch = AlignPatch(rect=bounds_of(lasso), offsets={"b": (5.0, 0.0)},
                       polygon=lasso)
    out = apply_patches([anchor, raster], ["a", "b"], [patch], dpi=72)

    assert (out[1][20:30, 25:35] == 0).all(), "enclosed mark should have shifted"
    assert (out[1][20:30, 60:70] == 0).all(), "mark outside the lasso must not move"


def test_patch_without_a_polygon_still_moves_the_whole_box() -> None:
    raster = np.full((60, 60), 255, dtype=np.uint8)
    raster[10:20, 10:20] = 0
    anchor = np.full((60, 60), 255, dtype=np.uint8)

    patch = AlignPatch(rect=(0.0, 0.0, 60.0, 60.0), offsets={"b": (4.0, 0.0)})
    out = apply_patches([anchor, raster], ["a", "b"], [patch], dpi=72)
    assert (out[1][10:20, 14:24] == 0).all()


def test_align_patch_round_trips_its_polygon() -> None:
    lasso = [(1.0, 2.0), (3.0, 4.0), (5.0, 1.0)]
    patch = AlignPatch(rect=bounds_of(lasso), offsets={"b": (1.0, 2.0)},
                       polygon=lasso)
    assert AlignPatch.from_dict(patch.to_dict()) == patch


def test_bounds_of_is_the_enclosing_box() -> None:
    assert bounds_of([(5.0, 9.0), (1.0, 2.0), (3.0, 7.0)]) == (1.0, 2.0, 5.0, 9.0)


def test_lasso_avoids_the_collateral_a_box_causes(tmp_path) -> None:
    """The motivating case. A box region drawn near the sheet frame drags that
    frame into one document and not the other, manufacturing a difference where
    there was none. A lasso around the label alone does not."""
    from redliner.core.documents import load_document

    base, drift = _jitter_pair(tmp_path, offset=(3.0, 2.0))
    project = Project(dpi=150)
    project.settings = DiffSettings(highlight=False)
    project.add_document(load_document(base, "base", "base"))
    project.add_document(load_document(drift, "drift", "drift"))
    drift_id = project.docs[1].doc_id

    def chromatic(rgb: np.ndarray) -> int:
        flat = rgb.reshape(-1, 3).astype(int)
        return int(((flat.max(axis=1) - flat.min(axis=1)) > 40).sum())

    # A region whose right edge runs past the frame line at x=572.
    greedy = (360.0, 178.0, 590.0, 214.0)
    project.pages[0].patches = [
        AlignPatch(rect=greedy, offsets={drift_id: (-3.0, -2.0)})
    ]
    with_box = chromatic(project.render(0))

    # The same correction, but enclosing only the note text.
    lasso = [(362.0, 180.0), (556.0, 180.0), (556.0, 212.0), (362.0, 212.0)]
    project.pages[0].patches = [
        AlignPatch(rect=greedy, offsets={drift_id: (-3.0, -2.0)}, polygon=lasso)
    ]
    with_lasso = chromatic(project.render(0))

    assert with_box > 0, "the greedy box should drag the frame and show colour"
    assert with_lasso < with_box, (
        f"the lasso should leave less collateral: box={with_box} lasso={with_lasso}"
    )
