"""The N-way compositor must not change v1.5's output for the 2-document case.

``shader_reference`` below is a literal transcription of the diff math in
``res/diff.frag`` from Redliner v1.5. The generalized compositor is expected to
agree with it exactly (to 8-bit rounding) whenever there are two layers.
"""

from __future__ import annotations

import numpy as np
import pytest

from redliner.core.compose import DiffSettings, Layer, align, composite

COL_REM = (1.0, 0.5, 0.0)  # orange, "old" document
COL_ADD = (0.0, 0.4, 1.0)  # blue, "new" document


def shader_reference(gray_lhs: np.ndarray, gray_rhs: np.ndarray,
                     col_rem=COL_REM, col_add=COL_ADD) -> np.ndarray:
    """Literal port of the diff.frag inner loop (highlight branch omitted)."""
    inv_lhs = 1.0 - gray_lhs.astype(np.float32) / 255.0
    inv_rhs = 1.0 - gray_rhs.astype(np.float32) / 255.0

    added = np.clip(inv_rhs - inv_lhs, 0, 1)
    removed = np.clip(inv_lhs - inv_rhs, 0, 1)
    same = np.clip(inv_lhs - removed, 0, 1)

    px = np.ones((*gray_lhs.shape, 3), dtype=np.float32)
    px -= (1.0 - np.float32(col_add)) * added[..., None]
    px -= (1.0 - np.float32(col_rem)) * removed[..., None]
    px -= same[..., None]
    return np.clip(px, 0, 1)


def _pair(seed: int, shape=(64, 96)) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return (rng.integers(0, 256, shape, dtype=np.uint8),
            rng.integers(0, 256, shape, dtype=np.uint8))


@pytest.mark.parametrize("seed", range(6))
def test_two_layers_match_v15_shader(seed: int) -> None:
    """Full-range random pixels, which hit every branch of the clamp algebra.

    Tolerance is 1/255 rather than exact: the compositor subtracts `same`
    before the per-layer terms and the shader subtracts it last, so float32
    accumulation rounds differently for pixels landing exactly on an 8-bit
    .5 boundary. In float64 the two forms agree to ~7e-8, and on realistic
    page content (below) they are bit-identical.
    """
    lhs, rhs = _pair(seed)
    settings = DiffSettings(highlight=False)

    got = composite([Layer(lhs, COL_REM), Layer(rhs, COL_ADD)], settings).astype(int)
    want = np.rint(shader_reference(lhs, rhs) * 255.0).astype(int)

    assert np.abs(got - want).max() <= 1


def test_two_layers_match_shader_on_realistic_ink() -> None:
    """Random noise exercises the algebra; this exercises the actual use case:
    mostly-white pages with sparse black marks and anti-aliased edges."""
    rng = np.random.default_rng(0)
    lhs = np.full((128, 128), 255, dtype=np.uint8)
    rhs = np.full((128, 128), 255, dtype=np.uint8)

    lhs[20:60, 10:100] = 0            # shared block
    rhs[20:60, 10:100] = 0
    lhs[70:80, 10:50] = 0             # only in old  -> orange
    rhs[90:100, 60:120] = 0           # only in new  -> blue
    lhs[105:115, 10:40] = rng.integers(0, 256, (10, 30), dtype=np.uint8)  # soft edges

    got = composite([Layer(lhs, COL_REM), Layer(rhs, COL_ADD)], DiffSettings(highlight=False))
    want = np.rint(shader_reference(lhs, rhs) * 255.0).astype(np.uint8)
    assert np.array_equal(got, want)


def test_shared_ink_renders_black_and_unique_ink_takes_layer_color() -> None:
    black, white = np.uint8(0), np.uint8(255)
    lhs = np.array([[black, black, white]], dtype=np.uint8)
    rhs = np.array([[black, white, black]], dtype=np.uint8)

    out = composite([Layer(lhs, COL_REM), Layer(rhs, COL_ADD)], DiffSettings(highlight=False))

    assert tuple(out[0, 0]) == (0, 0, 0)                       # in both -> black
    assert tuple(out[0, 1]) == (255, 128, 0)                   # old only -> orange
    assert tuple(out[0, 2]) == (0, 102, 255)                   # new only -> blue


def test_single_layer_is_passthrough_grayscale() -> None:
    """With one document there is no excess ink, so the composite collapses to
    the original page regardless of the assigned color."""
    gray = np.random.default_rng(3).integers(0, 256, (32, 32), dtype=np.uint8)
    out = composite([Layer(gray, COL_ADD)], DiffSettings(highlight=False))
    assert np.array_equal(out, np.repeat(gray[..., None], 3, axis=2))


def test_three_layers_show_partial_agreement() -> None:
    """A mark present in two of three documents is still a change and must stay
    visible. A pure pairwise-exclusive decomposition would erase it."""
    white = np.full((1, 1), 255, dtype=np.uint8)
    black = np.zeros((1, 1), dtype=np.uint8)

    out = composite(
        [Layer(black, (1.0, 0.0, 0.0)), Layer(black, (0.0, 1.0, 0.0)), Layer(white, (0.0, 0.0, 1.0))],
        DiffSettings(highlight=False),
    )
    assert tuple(out[0, 0]) != (255, 255, 255), "change present in 2 of 3 docs vanished"


def test_banding_is_seamless_with_highlight_enabled() -> None:
    """Band-wise compositing must produce bit-identical output to a single pass,
    including across band seams where the highlight box filter reaches."""
    from redliner.core import compose as compose_mod

    rng = np.random.default_rng(7)
    lhs = np.full((3000, 64), 255, dtype=np.uint8)
    rhs = lhs.copy()
    for row in rng.integers(0, 3000, 40):
        rhs[row : row + 3, 20:40] = 0

    settings = DiffSettings(highlight=True, highlight_size=10)
    layers = [Layer(lhs, COL_REM), Layer(rhs, COL_ADD)]

    banded = composite(layers, settings)
    original = compose_mod.BAND_ROWS
    try:
        compose_mod.BAND_ROWS = 1 << 30  # force a single band
        whole = composite(layers, settings)
    finally:
        compose_mod.BAND_ROWS = original

    assert np.array_equal(banded, whole)


def test_align_pads_with_paper_not_ink() -> None:
    small = np.zeros((10, 10), dtype=np.uint8)
    big = np.zeros((20, 30), dtype=np.uint8)

    padded = align([small, big])

    assert all(p.shape == (20, 30) for p in padded)
    assert padded[0][15, 20] == 255, "padding must be paper, or it reads as shared ink"


def test_composite_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="share a shape"):
        composite([Layer(np.zeros((4, 4), np.uint8), COL_REM),
                   Layer(np.zeros((5, 4), np.uint8), COL_ADD)])
