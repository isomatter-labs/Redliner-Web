"""End-to-end tests over the real example PDFs from Redliner v1.5.

a.pdf / b.pdf / c.pdf each contain the same four lines with different words
changed, which makes them a good fixture for checking that shared content
stays black and per-document changes pick up that document's color.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pymupdf
import pytest

from redliner.core.compose import DiffSettings
from redliner.core.documents import load_document
from redliner.core.export import ExportOptions, build_pdf
from redliner.core.project import DEFAULT_COLORS, Project

DATA = Path(__file__).parent / "data"


def make_project(names: str = "ab", dpi: float = 120.0) -> Project:
    """Build a project from example PDFs, one per character in `names`.

    Ids are position-based so that repeating a name (e.g. "aa", comparing a
    document against itself) still yields distinct documents, the way two
    separate uploads of the same file would.
    """
    project = Project(dpi=dpi, preview_dpi=dpi)
    project.settings = DiffSettings(highlight=False)
    for position, name in enumerate(names):
        project.add_document(
            load_document(DATA / f"{name}.pdf", f"{name}{position}", name)
        )
    return project


def test_documents_get_distinct_default_colors() -> None:
    project = make_project("abc")
    colors = [doc.color for doc in project.docs]
    assert colors == DEFAULT_COLORS[:3]
    assert len(set(colors)) == 3


def test_shared_text_is_black_and_changes_are_colored() -> None:
    project = make_project("ab")
    rgb = project.render(0)

    flat = rgb.reshape(-1, 3).astype(int)
    chroma = flat.max(axis=1) - flat.min(axis=1)

    assert (chroma > 40).any(), "no colored pixels: changes were not detected"
    assert (flat.max(axis=1) < 40).any(), "no black pixels: shared ink was lost"

    # Both documents' colors must actually appear -- orange-dominant pixels for
    # the old revision and blue-dominant for the new.
    orange = ((flat[:, 0] > flat[:, 2] + 40) & (chroma > 40)).sum()
    blue = ((flat[:, 2] > flat[:, 0] + 40) & (chroma > 40)).sum()
    assert orange > 100 and blue > 100, f"orange={orange} blue={blue}"


def test_identical_documents_produce_no_color() -> None:
    """Comparing a document against itself must come out as plain black-on-white."""
    project = make_project("aa")
    flat = project.render(0).reshape(-1, 3).astype(int)
    chroma = flat.max(axis=1) - flat.min(axis=1)
    assert chroma.max() == 0, "identical inputs produced a colored difference"


def test_highlight_marks_changes_and_spares_unchanged_text() -> None:
    project = make_project("ab")
    project.settings = DiffSettings(highlight=True, highlight_color=(1.0, 1.0, 0.0),
                                    highlight_threshold=0.02, highlight_size=8)
    rgb = project.render(0)

    # Highlighter is yellow: red and green high, blue suppressed.
    flat = rgb.reshape(-1, 3).astype(int)
    yellow = ((flat[:, 0] > 200) & (flat[:, 1] > 200) & (flat[:, 2] < 100)).sum()
    assert yellow > 500, f"expected a highlighter region, found {yellow} px"

    plain = make_project("ab").render(0)
    assert not np.array_equal(rgb, plain), "highlight setting had no effect"


def test_blank_slots_render_as_that_document_being_absent() -> None:
    """A blank pushes a document's page down; the vacated row must still render
    from the remaining documents rather than erroring or coming out empty."""
    project = make_project("ab")
    project.insert_blank("a0", 0)

    assert project.page_count == 2
    assert project.sequences["a0"] == [None, 0]
    assert project.sequences["b1"] == [0, None]

    first = project.render(0)
    assert first.reshape(-1, 3).min() < 200, "page with only b is blank"

    second = project.render(1)
    assert second.reshape(-1, 3).min() < 200, "page with only a is blank"


def test_delete_output_page_removes_the_row_everywhere() -> None:
    project = make_project("ab")
    project.insert_blank("a0", 0)
    project.delete_output_page(0)

    assert project.page_count == 1
    assert project.sequences["a0"] == [0]
    assert project.sequences["b1"] == [None]


def test_removing_a_document_keeps_the_project_renderable() -> None:
    project = make_project("abc")
    project.remove_document(project.docs[1].doc_id)
    assert [d.name for d in project.docs] == ["a", "c"]
    assert project.render(0).shape[2] == 3


def test_export_produces_a_searchable_pdf() -> None:
    project = make_project("ab")
    data = build_pdf(project, [0], ExportOptions(dpi=120, text_layer=True))

    with pymupdf.open(stream=data, filetype="pdf") as pdf:
        assert pdf.page_count == 1
        text = pdf[0].get_text()
        # Text common to both revisions, and text unique to one of them.
        assert "same" in text
        assert "DOCUMENT B" in text.upper()
        assert pdf[0].get_images(), "no composited image was embedded"


def test_export_without_text_layer_has_no_text() -> None:
    project = make_project("ab")
    data = build_pdf(project, [0], ExportOptions(dpi=100, text_layer=False))
    with pymupdf.open(stream=data, filetype="pdf") as pdf:
        assert pdf[0].get_text().strip() == ""


def test_export_respects_page_selection() -> None:
    project = make_project("ab")
    project.insert_blank("a0", 0)
    project.pages[0].export = False

    assert project.export_indices() == [1]
    data = build_pdf(project, options=ExportOptions(dpi=100))
    with pymupdf.open(stream=data, filetype="pdf") as pdf:
        assert pdf.page_count == 1


def test_export_with_nothing_selected_is_an_error() -> None:
    project = make_project("ab")
    for page in project.pages:
        page.export = False
    with pytest.raises(ValueError, match="no pages selected"):
        build_pdf(project)


def test_exported_page_keeps_the_source_page_size() -> None:
    project = make_project("ab")
    data = build_pdf(project, [0], ExportOptions(dpi=100))
    with pymupdf.open(stream=data, filetype="pdf") as pdf:
        rect = pdf[0].rect
    assert rect.width == pytest.approx(612, abs=1)
    assert rect.height == pytest.approx(792, abs=1)


def test_render_dpi_scales_the_raster() -> None:
    project = make_project("ab")
    low = project.render(0, dpi=72)
    high = project.render(0, dpi=144)
    assert high.shape[0] == pytest.approx(low.shape[0] * 2, abs=2)
