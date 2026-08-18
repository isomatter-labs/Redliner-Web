"""The extension points.

These tests are written the way a plugin author would use the system: define a
class, register it, and check the core picks it up without any other change.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pymupdf
import pytest

from redliner.core.markup import (SHAPE_PDF, SHAPE_SVG, Shape, register_shape,
                                  to_svg)
from redliner.plugins import Registry
from redliner.plugins.fetchers import (FETCHERS, Fetcher, FetchResult,
                                       FolderFetcher, UploadFetcher)
from redliner.plugins.parsers import (PARSERS, DocumentSource, SourceParser,
                                      accepted_extensions, parser_for)
from redliner.plugins.tools import TOOLS, MarkupTool, handles_for

DATA = Path(__file__).parent / "data"


# -- registry -----------------------------------------------------------

def test_registry_registers_and_finds_by_name() -> None:
    registry: Registry = Registry("thing", "redliner.nothing")
    registry._discovered = True   # skip discovery for an isolated registry

    class Thing:
        name = "thing-a"

    registry.register(Thing)
    assert registry.get("thing-a") is Thing
    assert registry.names() == ["thing-a"]


def test_registry_rejects_a_nameless_plugin() -> None:
    registry: Registry = Registry("thing", "redliner.nothing")

    class Nameless:
        pass

    with pytest.raises(ValueError, match="needs a `name`"):
        registry.register(Nameless)


def test_registry_orders_by_priority() -> None:
    registry: Registry = Registry("thing", "redliner.nothing")
    registry._discovered = True

    class Late:
        name = "late"
        priority = 90

    class Early:
        name = "early"
        priority = 10

    registry.register(Late)
    registry.register(Early)
    assert registry.names() == ["early", "late"]


def test_a_broken_extension_does_not_stop_discovery(monkeypatch, caplog) -> None:
    """One bad plugin must not take the server down -- the person who can fix it
    is usually not the person who needs the server running."""
    registry: Registry = Registry("thing", "redliner.nothing")

    def explode(name: str):
        raise RuntimeError("plugin is broken")

    monkeypatch.setattr("redliner.plugins.importlib.import_module", explode)
    registry.discover()          # must not raise
    assert registry.all() == []


def test_discovery_runs_only_once() -> None:
    registry: Registry = Registry("thing", "redliner.nothing")
    calls = []
    registry._load_local_extensions = lambda: calls.append(1)  # type: ignore[method-assign]
    registry._load_entry_points = lambda: None                 # type: ignore[method-assign]
    registry.discover()
    registry.discover()
    assert len(calls) == 1


# -- parsers ------------------------------------------------------------

def test_builtin_parsers_are_registered() -> None:
    assert "pdf" in PARSERS.names()
    assert "image" in PARSERS.names()


def test_pdf_files_route_to_the_pdf_parser() -> None:
    assert parser_for([DATA / "a.pdf"]).name == "pdf"


def test_accepted_extensions_covers_every_parser() -> None:
    accepted = accepted_extensions()
    assert ".pdf" in accepted
    assert ".png" in accepted
    assert all(e.startswith(".") for e in accepted)


def test_unknown_files_report_a_useful_error() -> None:
    with pytest.raises(ValueError, match="No parser recognised"):
        parser_for([Path("mystery.qqq")])


def test_a_custom_parser_takes_over_its_extension() -> None:
    """The whole point: a new format needs no changes outside its own module."""

    class FakeSource(DocumentSource):
        supports_subpixel = False

        @property
        def page_count(self) -> int:
            return 2

        @property
        def page_sizes(self):
            return [(100.0, 200.0), (100.0, 200.0)]

        def render(self, page_index, dpi, clip=None, offset=(0.0, 0.0)):
            return np.full((10, 10), 128, dtype=np.uint8)

    class WidgetParser(SourceParser):
        name = "test-widget"
        label = "Widget"
        priority = 5
        extensions = frozenset({".widget"})

        @classmethod
        def open(cls, paths):
            return FakeSource()

    PARSERS.register(WidgetParser)
    try:
        assert parser_for([Path("board.widget")]) is WidgetParser

        from redliner.core.documents import load_document
        doc = load_document([Path("board.widget")], "w", "widget")
        assert doc.page_count == 2
        assert doc.parser == "test-widget"
        assert doc.supports_subpixel is False
    finally:
        PARSERS._items.pop("test-widget", None)


def test_a_parser_receives_every_file_of_a_multi_file_document() -> None:
    """A folder of Gerbers is one board, not one document per layer."""
    seen: list[list[Path]] = []

    class SetSource(DocumentSource):
        @property
        def page_count(self) -> int:
            return 1

        @property
        def page_sizes(self):
            return [(612.0, 792.0)]

        def render(self, page_index, dpi, clip=None, offset=(0.0, 0.0)):
            return np.full((4, 4), 255, dtype=np.uint8)

    class SetParser(SourceParser):
        name = "test-set"
        priority = 5
        extensions = frozenset({".layer"})

        @classmethod
        def open(cls, paths):
            seen.append(list(paths))
            return SetSource()

    PARSERS.register(SetParser)
    try:
        from redliner.core.documents import load_document
        paths = [Path(f"board.{n}.layer") for n in ("top", "bottom", "mask")]
        doc = load_document(paths, "s", "board")

        assert seen == [paths], "the parser must see the whole set at once"
        assert doc.paths == paths
        assert doc.path == paths[0], "`path` should still address the first file"
    finally:
        PARSERS._items.pop("test-set", None)


def test_gerber_stub_claims_gerber_files_and_explains_itself() -> None:
    parser = parser_for([Path("board.gbr"), Path("board.gtl")])
    assert parser.name == "gerber"

    source = parser.open([Path("board.gtl"), Path("board.gbl")])
    assert source.page_count == 1
    assert source.supports_subpixel is False, \
        "a raster renderer cannot place content between pixels"

    with pytest.raises(NotImplementedError, match="not implemented"):
        source.render(0, 100.0)


def test_gerber_stub_does_not_hijack_a_pdf() -> None:
    assert parser_for([DATA / "a.pdf"]).name == "pdf"


# -- fetchers -----------------------------------------------------------

def test_builtin_fetchers_are_registered() -> None:
    assert {"upload", "url", "folder"} <= set(FETCHERS.names())


def test_upload_fetcher_resolves_inside_the_workspace(tmp_path) -> None:
    import asyncio

    target = tmp_path / "drawing.pdf"
    target.write_bytes(b"%PDF-1.4\n")
    fetcher = UploadFetcher(tmp_path)

    name, paths = asyncio.run(fetcher.fetch("drawing.pdf"))
    assert name == "drawing"
    assert paths == [target]


def test_folder_fetcher_refuses_to_escape_its_root(tmp_path) -> None:
    """A fetcher that serves arbitrary server paths is a file disclosure hole."""
    import asyncio

    root = tmp_path / "vault"
    root.mkdir()
    (root / "ok.pdf").write_bytes(b"%PDF-1.4\n")
    (tmp_path / "secret.txt").write_text("do not serve me")

    fetcher = FolderFetcher(tmp_path / "work")
    (tmp_path / "work").mkdir()
    fetcher.root = root

    with pytest.raises(ValueError, match="escapes"):
        asyncio.run(fetcher.fetch("../secret.txt"))


def test_folder_fetcher_returns_a_whole_directory_as_one_document(tmp_path) -> None:
    import asyncio

    root = tmp_path / "vault"
    (root / "board").mkdir(parents=True)
    for layer in ("top.gtl", "bottom.gbl", "mask.gts"):
        (root / "board" / layer).write_text("G04 stub*")

    work = tmp_path / "work"
    work.mkdir()
    fetcher = FolderFetcher(work)
    fetcher.root = root

    name, paths = asyncio.run(fetcher.fetch("board"))
    assert name == "board"
    assert len(paths) == 3, "every layer should arrive as one document"
    assert all(p.parent == work for p in paths)


def test_a_custom_fetcher_is_discovered() -> None:
    class VaultFetcher(Fetcher):
        name = "test-vault"
        label = "Vault"
        icon = "inventory_2"

        async def search(self, query):
            return [FetchResult(ref="DWG-1/B", title="DWG-1", subtitle="rev B")]

        async def fetch(self, ref):
            return ref, []

    FETCHERS.register(VaultFetcher)
    try:
        assert FETCHERS.get("test-vault") is VaultFetcher
    finally:
        FETCHERS._items.pop("test-vault", None)


# -- tools and shapes ---------------------------------------------------

def test_builtin_tools_are_registered_in_toolbar_order() -> None:
    names = TOOLS.names()
    assert names[0] == "pan"
    assert {"select", "pencil", "line", "arrow", "rect", "cloud", "text",
            "align"} <= set(names)


def test_tools_create_the_shape_they_advertise() -> None:
    style = {"color": (1.0, 0.0, 0.0), "width": 2.0}
    for kind in ("line", "arrow", "rect", "cloud"):
        shape = TOOLS.get(kind).create([(0.0, 0.0), (10.0, 10.0)], style)
        assert shape is not None and shape.kind == kind
        assert shape.width == 2.0


def test_dialog_tools_create_nothing_directly() -> None:
    """Text and align are completed by the app, not by the tool."""
    for kind in ("text", "align"):
        assert TOOLS.get(kind).opens_dialog is True
        assert TOOLS.get(kind).create([(0.0, 0.0)], {}) is None


def test_pencil_tool_thins_its_own_stroke() -> None:
    dense = [(float(i) * 0.5, 5.0) for i in range(100)]
    shape = TOOLS.get("pencil").create(dense, {})
    assert shape is not None
    assert len(shape.points) < len(dense)


def test_pencil_tool_discards_a_degenerate_stroke() -> None:
    assert TOOLS.get("pencil").create([(1.0, 1.0)], {}) is None


def test_handles_come_from_the_owning_tool() -> None:
    assert handles_for(Shape(kind="rect", points=[(0.0, 0.0), (1.0, 1.0)])) == \
        ["nw", "ne", "se", "sw"]
    assert handles_for(Shape(kind="line", points=[(0.0, 0.0), (1.0, 1.0)])) == \
        ["p0", "p1"]
    assert handles_for(Shape(kind="pencil", points=[(0.0, 0.0)])) == []


def test_a_custom_tool_and_shape_render_end_to_end() -> None:
    """A plugin can add a whole new annotation type: a tool to draw it, plus how
    it renders on screen and in the exported PDF."""

    def star_svg(shape: Shape) -> str:
        x0, y0, x1, y1 = shape.bounds()
        return f'<circle cx="{(x0+x1)/2}" cy="{(y0+y1)/2}" r="5" fill="gold"/>'

    def star_pdf(page, shape: Shape) -> None:
        x0, y0, x1, y1 = shape.bounds()
        page.draw_circle(pymupdf.Point((x0 + x1) / 2, (y0 + y1) / 2), 5,
                         color=(1, 0.8, 0))

    class StarTool(MarkupTool):
        name = "star"
        icon = "star"
        tooltip = "Star"

        @classmethod
        def create(cls, points, style):
            return Shape(kind="star", points=points[:2])

    TOOLS.register(StarTool)
    register_shape("star", svg=star_svg, pdf=star_pdf)
    try:
        shape = TOOLS.get("star").create([(10.0, 10.0), (30.0, 30.0)], {})
        assert "<circle" in to_svg([shape], 612, 792)

        # And it exports as real vector content.
        doc = pymupdf.open()
        page = doc.new_page(width=612, height=792)
        from redliner.core.markup import draw_on_page
        draw_on_page(page, [shape])
        assert page.get_drawings(), "custom shape did not reach the PDF"
        doc.close()
    finally:
        TOOLS._items.pop("star", None)
        SHAPE_SVG.pop("star", None)
        SHAPE_PDF.pop("star", None)


def test_orphaned_session_directories_are_swept(tmp_path, monkeypatch) -> None:
    """Uploads are deleted on disconnect, but that never runs if the process is
    killed -- so controlled documents could sit in temp indefinitely."""
    import time

    from redliner.ui.app import Session

    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))

    stale = tmp_path / "redliner-old"
    stale.mkdir()
    (stale / "drawing.pdf").write_bytes(b"%PDF-1.4\n")
    old = time.time() - 48 * 3600
    import os
    os.utime(stale, (old, old))

    fresh = tmp_path / "redliner-live"
    fresh.mkdir()
    (fresh / "drawing.pdf").write_bytes(b"%PDF-1.4\n")

    unrelated = tmp_path / "something-else"
    unrelated.mkdir()

    assert Session.sweep_orphans(max_age_hours=12.0) == 1
    assert not stale.exists()
    assert fresh.exists(), "a concurrent instance's uploads must survive"
    assert unrelated.exists(), "only redliner directories may be touched"


def test_builtin_shapes_still_render_without_any_registration() -> None:
    """The registry is an override point, not a requirement."""
    assert "rect" not in SHAPE_SVG
    assert "<rect" in to_svg([Shape(kind="rect", points=[(0.0, 0.0), (9.0, 9.0)])],
                             100, 100)


# -- page preview cache -------------------------------------------------

def _preview_session(tmp_path):
    """A Session-like object exercising only the thumbnail key logic."""
    from redliner.core.documents import load_document
    from redliner.core.project import Project
    from redliner.ui.app import Session

    project = Project(dpi=150)
    for name in ("a", "b"):
        project.add_document(load_document(DATA / f"{name}.pdf", name, name))

    class Fake:
        pass

    fake = Fake()
    fake.project = project
    fake.thumbnail_key = Session.thumbnail_key.__get__(fake)
    fake.thumbnail_dpi = Session.thumbnail_dpi.__get__(fake)
    return fake


def test_preview_key_is_stable_when_nothing_changes(tmp_path) -> None:
    session = _preview_session(tmp_path)
    assert session.thumbnail_key(0) == session.thumbnail_key(0)


def test_preview_key_follows_document_colour(tmp_path) -> None:
    """A stale preview showing the old colours is worse than none: the panel is
    what you use to decide which pages to export."""
    session = _preview_session(tmp_path)
    before = session.thumbnail_key(0)
    session.project.docs[0].color = (0.0, 1.0, 0.0)
    assert session.thumbnail_key(0) != before


def test_preview_key_follows_diff_settings(tmp_path) -> None:
    from redliner.core.compose import DiffSettings

    session = _preview_session(tmp_path)
    before = session.thumbnail_key(0)
    session.project.settings = DiffSettings(highlight=False)
    assert session.thumbnail_key(0) != before


def test_preview_key_follows_alignment(tmp_path) -> None:
    from redliner.core.align import AlignPatch

    session = _preview_session(tmp_path)
    before = session.thumbnail_key(0)
    session.project.pages[0].patches = [
        AlignPatch(rect=(0.0, 0.0, 100.0, 100.0), offsets={"b": (2.0, 1.0)})
    ]
    assert session.thumbnail_key(0) != before


def test_preview_key_follows_page_alignment_sequence(tmp_path) -> None:
    session = _preview_session(tmp_path)
    before = session.thumbnail_key(0)
    session.project.insert_blank("a", 0)
    assert session.thumbnail_key(0) != before


def test_preview_dpi_bounds_cost_across_sheet_sizes(tmp_path) -> None:
    """A fixed DPI would make an E-size sheet cost 16x a letter one; the target
    is a pixel width so every sheet renders about the same amount."""
    from redliner.ui.app import PAGE_THUMB_RENDER_PX

    session = _preview_session(tmp_path)
    widths = []
    for size in ((612.0, 792.0), (1584.0, 2448.0), (2448.0, 1584.0)):
        session.project.docs[0].page_sizes = [size]
        session.project.docs[1].page_sizes = [size]
        dpi = session.thumbnail_dpi(0)
        widths.append(size[0] * dpi / 72.0)

    for width in widths:
        assert width == pytest.approx(PAGE_THUMB_RENDER_PX, rel=0.02)
