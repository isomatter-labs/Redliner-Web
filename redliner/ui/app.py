"""Redliner web UI.

One :class:`Session` per browser connection. Nothing is shared between users,
and uploads live in a per-session temp directory that is removed on disconnect,
so a shared server never leaks one team's drawings into another's view.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
import time
import uuid
from contextlib import nullcontext
from pathlib import Path

from nicegui import app, events, run, ui

from ..core.compose import DiffSettings
from ..core.documents import (PDF_UNITS_PER_INCH, SourceDoc, clear_cache,
                              load_document, raster_to_data_url, thumbnail)
from ..core.align import (AlignPatch, auto_align, normalize_rect,
                          search_radius)
from ..core.compose import to_ink
from ..core.export import ExportOptions, build_pdf, estimate_pixels
from ..core.markup import FONTS, Shape, to_svg
from ..core.project import Project, hex_to_rgb, rgb_to_hex
from ..core import shares
from .share_routes import SHARE_PREFIX, register as register_share_routes
from ..plugins import discover_all
from ..plugins.fetchers import FETCHERS
from ..plugins.parsers import accepted_extensions
from ..plugins.tools import TOOLS

#: Target width in pixels for the align dialog's preview render. Wide enough to
#: judge a couple of pixels of drift, small enough to re-render on every nudge.
ALIGN_PREVIEW_PX = 800
ALIGN_DPI_RANGE = (100.0, 400.0)

#: Stroke widths offered in the markup panel, in points.
WIDTHS = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]

#: Preview renders below this are fast enough to feel live while dragging a
#: slider. The viewer re-renders sharper on demand when the user zooms in.
BASE_PREVIEW_DPI = 100.0
MAX_PREVIEW_DPI = 400.0

#: Refuse export jobs above this many pixels; past here a browser tab is more
#: likely to run the server out of memory than to produce a usable file.
MAX_EXPORT_PIXELS = 1_600_000_000


class Session:
    """All per-client state and the widgets bound to it."""

    def __init__(self) -> None:
        self.project = Project(dpi=200.0, preview_dpi=BASE_PREVIEW_DPI)
        self.tempdir = Path(tempfile.mkdtemp(prefix="redliner-"))
        #: Treat a multi-file upload as one document (a folder of Gerbers).
        self.combine_uploads = False
        self._pending: list[tuple[Path, str]] = []
        self._pending_timer = None
        # One fetcher instance per session, so any credentials a fetcher holds
        # belong to that user and vanish when they disconnect.
        self.fetchers = [cls(self.tempdir) for cls in FETCHERS.all()]
        self.index = 0
        self.render_dpi = BASE_PREVIEW_DPI
        self.busy = False
        self._generation = 0
        self.viewer: object = None
        self.tool = "pan"
        self.markup_color = "#d90000"
        self.markup_width = 1.5
        self.markup_fill: str | None = None   # None == transparent
        self.markup_font = "sans"
        self.markup_boxed = False
        #: Index into the current page's markups, or None.
        self.selected: int | None = None
        self._shape_drag: dict = {"id": None, "points": None}
        # Set while an align dialog is open. The drag and keyboard listeners are
        # registered once per client and routed here, rather than re-registered
        # per dialog, which would leave a dead handler behind on every open.
        self._align_drag = None
        self._align_keys = None
        #: A never-refreshed container that owns deferred timers.
        self._timer_host = None

    def dispose(self) -> None:
        shutil.rmtree(self.tempdir, ignore_errors=True)

    @staticmethod
    def sweep_orphans(max_age_hours: float = 12.0) -> int:
        """Delete session directories left behind by an unclean shutdown.

        Uploads are removed when a client disconnects, but that never runs if
        the process is killed or crashes -- so controlled documents can sit in
        the system temp directory indefinitely. Sweeping at startup bounds how
        long anything can outlive its session.

        Only directories older than `max_age_hours` are touched, so a restart
        cannot delete uploads belonging to a concurrently running instance.
        """
        cutoff = time.time() - max_age_hours * 3600
        # Shared PDFs are long-lived by design and must never be swept as if
        # they were an abandoned session. They are stored under a name that
        # does not match the glob below, and this check keeps that true even if
        # someone points REDLINER_SHARE_DIR somewhere that does.
        protected = shares.default_root().resolve()

        removed = 0
        for path in Path(tempfile.gettempdir()).glob("redliner-*"):
            try:
                if not path.is_dir():
                    continue
                if path.resolve() == protected:
                    continue
                if path.stat().st_mtime < cutoff:
                    shutil.rmtree(path, ignore_errors=True)
                    removed += 1
            except OSError:
                continue
        return removed

    # -- documents -------------------------------------------------------

    async def handle_upload(self, event: events.UploadEventArguments) -> None:
        upload = event.file
        # `save` streams to disk in chunks; a large sheet set should never be
        # held in memory in full just to land it in the temp directory.
        target = self.tempdir / f"{uuid.uuid4().hex}-{upload.name}"
        await upload.save(target)

        if self.combine_uploads:
            # Hold the batch and open it as one document once uploads settle.
            # NiceGUI reports each file separately with no "batch finished"
            # signal, so a short debounce is what defines the group.
            self._pending.append((target, upload.name))
            if self._pending_timer is not None:
                self._pending_timer.cancel()
            self._pending_timer = ui.timer(0.6, self.flush_pending, once=True)
            return

        await self.add_document([target], Path(upload.name).stem)

    async def flush_pending(self) -> None:
        """Open everything collected during a combined upload as one document."""
        batch, self._pending = self._pending, []
        self._pending_timer = None
        if not batch:
            return
        paths = [path for path, _ in batch]
        name = Path(batch[0][1]).stem if len(batch) == 1 else \
            f"{Path(batch[0][1]).stem} +{len(batch) - 1}"
        await self.add_document(paths, name)

    async def add_document(self, paths: list[Path], name: str) -> None:
        try:
            doc = await run.io_bound(load_document, paths, uuid.uuid4().hex, name)
        except Exception as exc:
            ui.notify(f"Could not read {name}: {exc}", type="negative")
            return

        if doc.page_count == 0:
            ui.notify(f"{name} has no pages", type="warning")
            return

        self.project.add_document(doc)
        files = "" if len(paths) == 1 else f", {len(paths)} files"
        ui.notify(f"Added {doc.name} ({doc.page_count} page"
                  f"{'s' if doc.page_count != 1 else ''}{files})", type="positive")
        self.refresh_all()

    async def fetch_document(self, fetcher, ref: str) -> None:
        """Pull a document from a fetcher and add it to the project."""
        notification = ui.notification(f"Fetching from {fetcher.label}...",
                                       spinner=True, timeout=None)
        try:
            name, paths = await fetcher.fetch(ref)
        except NotImplementedError as exc:
            notification.dismiss()
            ui.notify(str(exc), type="warning", multi_line=True,
                      classes="max-w-xl")
            return
        except Exception as exc:
            notification.dismiss()
            ui.notify(f"Fetch failed: {exc}", type="negative")
            return
        notification.dismiss()
        await self.add_document(paths, name)

    async def share(self) -> None:
        """Publish the selected pages to a temporary link."""
        indices = self.project.export_indices()
        if not indices:
            ui.notify("No pages are marked for export", type="warning")
            return

        ttl = {"seconds": shares.DEFAULT_TTL}
        dpi = {"value": max(self.project.dpi, 200.0)}
        label = {"text": " vs ".join(d.name for d in self.project.docs) or "redline"}

        with ui.dialog() as dialog, ui.card().classes("w-[520px] max-w-full gap-3"):
            ui.label("Share a link").classes("text-base font-medium")
            ui.label(f"{len(indices)} page{'s' if len(indices) != 1 else ''} "
                     "will be exported and hosted at a temporary URL.") \
                .classes("text-sm opacity-70")

            name = ui.input("Name", value=label["text"]) \
                .props("dense outlined").classes("w-full") \
                .tooltip("Only used for the download filename; the link itself "
                         "is a random token")

            with ui.row().classes("w-full gap-2 no-wrap"):
                ui.select({s: t for t, s in shares.TTL_CHOICES},
                          label="Expires after", value=ttl["seconds"],
                          on_change=lambda e: ttl.__setitem__("seconds", int(e.value))) \
                    .props("dense outlined").classes("flex-grow")
                ui.number("DPI", value=dpi["value"], min=72, max=1200, step=25,
                          on_change=lambda e: dpi.__setitem__(
                              "value", float(e.value or 200))) \
                    .props("dense outlined").classes("w-28") \
                    .tooltip("Resolution of the shared PDF. The raster is baked "
                             "in, so pick enough for whoever reviews it.")

            result = ui.column().classes("w-full gap-1")

            async def publish() -> None:
                label["text"] = name.value or "redline"
                notification = ui.notification("Building the shared PDF...",
                                               spinner=True, timeout=None)
                try:
                    data = await run.io_bound(
                        build_pdf, self.project, indices,
                        ExportOptions(dpi=dpi["value"],
                                      text_layer=self.text_layer.value,
                                      title=label["text"]),
                    )
                    share = await run.io_bound(
                        shares.store().create, data, ttl["seconds"],
                        label["text"], len(indices))
                except Exception as exc:
                    notification.dismiss()
                    ui.notify(f"Could not create the share: {exc}", type="negative")
                    return
                notification.dismiss()

                url = f"{await ui.run_javascript('window.location.origin')}" \
                      f"{SHARE_PREFIX}/{share.token}"
                result.clear()
                with result:
                    ui.label("Link ready").classes("text-sm font-medium")
                    with ui.row().classes("w-full items-center gap-2 no-wrap"):
                        field = ui.input(value=url).props("dense outlined readonly") \
                            .classes("flex-grow")
                        ui.button(icon="content_copy", on_click=lambda: (
                            ui.run_javascript(
                                f"navigator.clipboard.writeText({url!r})"),
                            ui.notify("Copied", type="positive"))) \
                            .props("flat dense").tooltip("Copy link")
                    ui.label(f"Expires in {share.describe_remaining()} · "
                             f"{share.size / 1e6:.1f} MB · {share.pages} pages") \
                        .classes("text-xs opacity-60")

            with ui.row().classes("w-full justify-end gap-2"):
                ui.button("Close", on_click=dialog.close).props("flat no-caps")
                ui.button("Create link", icon="link", on_click=publish) \
                    .props("unelevated no-caps")
        dialog.open()

    def remove_document(self, doc: SourceDoc) -> None:
        self.project.remove_document(doc.doc_id)
        self.index = min(self.index, max(0, self.project.page_count - 1))
        self.refresh_all()

    def pick_color(self, doc: SourceDoc, swatch, value: str) -> None:
        doc.color = hex_to_rgb(value)
        swatch.style(f"background:{rgb_to_hex(doc.color)};"
                     "width:24px;height:24px;min-height:0;min-width:0")
        self.align_panel.refresh()
        self.schedule_render()

    def move_document(self, doc: SourceDoc, delta: int) -> None:
        docs = self.project.docs
        i = docs.index(doc)
        j = max(0, min(len(docs) - 1, i + delta))
        if i != j:
            docs[i], docs[j] = docs[j], docs[i]
            self.refresh_all()

    # -- rendering -------------------------------------------------------

    def schedule_render(self, keep_view: bool = True) -> None:
        """Queue a re-render just after the current handler returns.

        The timer is parented to a container that is never refreshed, because
        `ui.timer` attaches to whatever slot is current and a timer whose parent
        gets deleted never fires. Refreshing a panel deletes its children --
        including the button being clicked -- so scheduling from inside, say,
        the document list would silently drop the render. That was the "deleting
        a document doesn't regenerate the diff" bug.
        """
        host = self._timer_host
        with host if host is not None else nullcontext():
            ui.timer(0.01, lambda: self.render(keep_view), once=True)

    async def render(self, keep_view: bool = False) -> None:
        if self.viewer is None:
            return
        if not self.project.page_count:
            self.viewer.set_source("")
            self.page_label.refresh()
            return

        self.index = max(0, min(self.index, self.project.page_count - 1))
        self._generation += 1
        token = self._generation
        self.busy = True
        self.progress.set_visibility(True)
        try:
            image = await run.io_bound(self.project.render, self.index, self.render_dpi)
            url = await run.io_bound(raster_to_data_url, image)
        except Exception as exc:
            ui.notify(f"Render failed: {exc}", type="negative")
            return
        finally:
            self.busy = False
            self.progress.set_visibility(False)

        # A newer render started while this one was in flight; drop this result
        # so a slow high-DPI pass cannot overwrite a fresher low-DPI one.
        if token != self._generation:
            return
        self.viewer.set_source(url, keep_view=keep_view)
        self.refresh_markup()
        self.page_label.refresh()

    # -- markup ----------------------------------------------------------

    def current_page(self):
        pages = self.project.pages
        return pages[self.index] if 0 <= self.index < len(pages) else None

    def refresh_markup(self) -> None:
        """Push the current page's annotations and page size to the overlay."""
        page = self.current_page()
        if self.viewer is None or page is None:
            return
        if self.selected is not None and self.selected >= len(page.markups):
            self.selected = None
        width_pt, height_pt = self.project.page_size_points(self.index)
        self.viewer.set_page_size(width_pt, height_pt)
        self.viewer.set_markup(
            to_svg(page.markups, width_pt, height_pt, selected=self.selected))

    def select_tool(self, tool: str) -> None:
        self.tool = tool
        if tool != "select":
            self.selected = None
            self.refresh_markup()
        self.viewer.set_tool(tool, self.markup_color, self.markup_width)
        self.tool_panel.refresh()
        self.markup_panel.refresh()

    # -- selection -------------------------------------------------------

    def selected_shape(self) -> Shape | None:
        page = self.current_page()
        if page is None or self.selected is None:
            return None
        if 0 <= self.selected < len(page.markups):
            return page.markups[self.selected]
        return None

    def pick_markup(self, index: int | None) -> None:
        self.selected = index
        # Loading the shape's own properties into the panel means the controls
        # describe what is selected, so editing one does not silently reset the
        # others to whatever was last used for drawing.
        self.adopt_selection_properties()
        self.refresh_markup()
        self.markup_panel.refresh()

    def drag_markup(self, payload: dict) -> None:
        """Move the selected shape, or one of its handles, by a live delta."""
        shape = self.selected_shape()
        if shape is None:
            return

        # A new drag id means a new gesture: snapshot the geometry it started
        # from, because the deltas that follow are cumulative from that point.
        if self._shape_drag["id"] != payload.get("drag"):
            self._shape_drag = {"id": payload.get("drag"),
                                "points": list(shape.points)}

        origin = self._shape_drag["points"] or []
        dx, dy = float(payload.get("dx", 0.0)), float(payload.get("dy", 0.0))
        handle = payload.get("handle")

        if not handle:
            shape.points = [(x + dx, y + dy) for x, y in origin]
        elif handle.startswith("p") and len(origin) >= 2:
            # Line and arrow endpoints.
            end = 0 if handle == "p0" else len(origin) - 1
            shape.points = list(origin)
            shape.points[end] = (origin[end][0] + dx, origin[end][1] + dy)
        elif len(origin) >= 2:
            # Box corners: move the two edges the grabbed corner belongs to.
            x0, y0 = min(p[0] for p in origin), min(p[1] for p in origin)
            x1, y1 = max(p[0] for p in origin), max(p[1] for p in origin)
            if "n" in handle:
                y0 += dy
            else:
                y1 += dy
            if "w" in handle:
                x0 += dx
            else:
                x1 += dx
            shape.points = [(x0, y0), (x1, y1)]

        self.refresh_markup()

    def delete_selected(self) -> None:
        page = self.current_page()
        if page is None or self.selected is None:
            return
        if 0 <= self.selected < len(page.markups):
            page.markups.pop(self.selected)
        self.selected = None
        self.refresh_markup()
        self.markup_panel.refresh()

    def nudge_selected(self, dx: float, dy: float) -> None:
        shape = self.selected_shape()
        if shape is not None:
            shape.translate(dx, dy)
            self.refresh_markup()

    def apply_to_selection(self) -> None:
        """Push the current markup properties onto the selected shape."""
        shape = self.selected_shape()
        if shape is None:
            return
        shape.color = hex_to_rgb(self.markup_color)
        shape.width = self.markup_width
        shape.fill = None if self.markup_fill is None else hex_to_rgb(self.markup_fill)
        shape.font = self.markup_font
        shape.boxed = self.markup_boxed
        self.refresh_markup()

    def adopt_selection_properties(self) -> None:
        """Load the selected shape's properties into the markup panel."""
        shape = self.selected_shape()
        if shape is None:
            return
        self.markup_color = rgb_to_hex(shape.color)
        self.markup_width = shape.width
        self.markup_fill = None if shape.fill is None else rgb_to_hex(shape.fill)
        self.markup_font = shape.font
        self.markup_boxed = shape.boxed

    def on_shape(self, payload: dict) -> None:
        page = self.current_page()
        if page is None:
            return
        points = [(float(x), float(y)) for x, y in payload.get("points", [])]
        kind = payload.get("kind", "line")

        if kind == "text":
            self.ask_for_text(points[0] if points else (0.0, 0.0))
            return

        if kind == "align":
            if len(points) >= 2:
                self.open_align_dialog((points[0][0], points[0][1],
                                        points[1][0], points[1][1]))
            return

        # Everything else is the registered tool's business: it decides what
        # shape the gesture produces, or discards it by returning None.
        tool = TOOLS.get(kind)
        if tool is None:
            return
        shape = tool.create(points, self.style())
        if shape is None:
            return

        page.markups.append(shape)
        self.refresh_markup()

    def style(self) -> dict:
        """The markup panel's current settings, handed to tools."""
        return {
            "color": hex_to_rgb(self.markup_color),
            "width": self.markup_width,
            "fill": None if self.markup_fill is None else hex_to_rgb(self.markup_fill),
            "font": self.markup_font,
            "boxed": self.markup_boxed,
        }

    def new_shape(self, kind: str, points: list[tuple[float, float]]) -> Shape:
        """A shape carrying the markup panel's current properties."""
        return Shape(kind=kind, points=points, **{
            k: v for k, v in self.style().items()
        })

    def ask_for_text(self, position: tuple[float, float] | None = None,
                     shape: Shape | None = None) -> None:
        """Editor for a text annotation, used for both placing and editing."""
        page = self.current_page()
        if page is None:
            return
        editing = shape is not None

        with ui.dialog() as dialog, ui.card().classes("w-96"):
            ui.label("Edit text" if editing else "Text box") \
                .classes("text-base font-medium")
            field = ui.textarea(placeholder="Note text",
                                value=shape.text if editing else "") \
                .props("autofocus outlined").classes("w-full")
            with ui.row().classes("w-full items-center gap-2"):
                size = ui.number("Size (pt)",
                                 value=shape.font_size if editing else 11,
                                 min=4, max=96, step=1).props("dense outlined")
                font = ui.select({k: k.title() for k in FONTS}, label="Font",
                                 value=shape.font if editing else self.markup_font) \
                    .props("dense outlined").classes("flex-grow")
            boxed = ui.switch("Draw a box around it",
                              value=shape.boxed if editing else self.markup_boxed)

            with ui.row().classes("w-full justify-end gap-2"):
                ui.button("Cancel", on_click=dialog.close).props("flat")

                def commit() -> None:
                    if field.value:
                        target = shape if editing else self.new_shape("text", [position])
                        target.text = field.value
                        target.font_size = float(size.value or 11)
                        target.font = font.value
                        target.boxed = bool(boxed.value)
                        if not editing:
                            page.markups.append(target)
                        self.refresh_markup()
                    dialog.close()

                ui.button("Save" if editing else "Place", on_click=commit) \
                    .props("unelevated")
        dialog.open()

    def edit_markup(self, index: int) -> None:
        """Double-click handler: text opens its editor, shapes adopt properties."""
        page = self.current_page()
        if page is None or not 0 <= index < len(page.markups):
            return
        self.selected = index
        shape = page.markups[index]
        if shape.kind == "text":
            self.ask_for_text(shape=shape)
        else:
            self.adopt_selection_properties()
            self.markup_panel.refresh()
        self.refresh_markup()

    # -- magic align -----------------------------------------------------

    def _align_dpi(self, rect: tuple[float, float, float, float]) -> float:
        """Preview resolution for a region, targeting a readable pixel width."""
        width_pt = max(1.0, rect[2] - rect[0])
        dpi = ALIGN_PREVIEW_PX * PDF_UNITS_PER_INCH / width_pt
        return float(min(max(dpi, ALIGN_DPI_RANGE[0]), ALIGN_DPI_RANGE[1]))

    def dispatch_align_drag(self, event) -> None:
        if self._align_drag is not None:
            self._align_drag(event.args or {})

    def dispatch_key(self, event) -> None:
        # The align dialog takes the keyboard while it is open.
        if self._align_keys is not None:
            self._align_keys(event)
            return
        if not event.action.keydown or self.selected_shape() is None:
            return
        # Typing in a dialog does not reach here: ui.keyboard ignores events
        # originating from input, textarea, select and button by default.
        if event.key == "Delete" or event.key == "Backspace":
            self.delete_selected()
            return
        if not event.key.is_cursorkey:
            return
        steps = {"ArrowLeft": (-1.0, 0.0), "ArrowRight": (1.0, 0.0),
                 "ArrowUp": (0.0, -1.0), "ArrowDown": (0.0, 1.0)}
        for name, (dx, dy) in steps.items():
            if event.key == name:
                scale = 5.0 if event.modifiers.shift else 1.0
                self.nudge_selected(dx * scale, dy * scale)
                return

    def open_align_dialog(self, rect: tuple[float, float, float, float]) -> None:
        """Correct a local misalignment inside `rect`.

        The preview is difference-only: shared ink drops out, so a few pixels of
        CAD drift show up as two coloured ghosts of the same shape and the job
        is simply to make them cancel.
        """
        page = self.current_page()
        docs = self.project.docs
        if page is None or len(docs) < 2:
            ui.notify("Magic align needs at least two documents", type="warning")
            return

        rect = normalize_rect(rect)
        if rect[2] - rect[0] < 4 or rect[3] - rect[1] < 4:
            ui.notify("Draw a larger region to align", type="warning")
            return

        # Re-drawing over an existing correction edits it rather than stacking a
        # second patch on top, which would compound in ways nobody intends.
        centre = ((rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2)
        existing = next(
            (p for p in page.patches
             if p.rect[0] <= centre[0] <= p.rect[2] and p.rect[1] <= centre[1] <= p.rect[3]),
            None,
        )
        if existing is not None:
            rect = normalize_rect(existing.rect)

        dpi = self._align_dpi(rect)
        step = PDF_UNITS_PER_INCH / dpi  # one preview pixel, in points

        offsets: dict[str, tuple[float, float]] = {d.doc_id: (0.0, 0.0) for d in docs}
        if existing is not None:
            offsets.update({k: (float(v[0]), float(v[1]))
                            for k, v in existing.offsets.items()})

        selected = {"doc": docs[1].doc_id}
        drag = {"id": None, "base": (0.0, 0.0)}
        generation = {"n": 0}

        preview_id = f"align-preview-{uuid.uuid4().hex[:8]}"

        async def redraw() -> None:
            generation["n"] += 1
            token = generation["n"]
            try:
                rgb = await run.io_bound(self.project.render_region, self.index,
                                         rect, dpi, dict(offsets))
                url = await run.io_bound(raster_to_data_url, rgb)
            except Exception as exc:
                ui.notify(f"Align preview failed: {exc}", type="negative")
                return
            # Drag fires faster than a region renders; drop stale frames so the
            # preview cannot jump backwards to an older offset.
            if token != generation["n"]:
                return
            preview._props["src"] = url
            preview.update()

        def schedule() -> None:
            rows.refresh()
            ui.timer(0.01, redraw, once=True)

        def nudge(doc_id: str, dx: int, dy: int, scale: int = 1) -> None:
            x, y = offsets[doc_id]
            offsets[doc_id] = (x + dx * step * scale, y + dy * step * scale)
            schedule()

        def on_drag(args: dict) -> None:
            if drag["id"] != args.get("drag"):
                drag["id"] = args.get("drag")
                drag["base"] = offsets[selected["doc"]]
            base_x, base_y = drag["base"]
            offsets[selected["doc"]] = (base_x + float(args.get("dx", 0)) * step,
                                        base_y + float(args.get("dy", 0)) * step)
            schedule()

        def on_key(event) -> None:
            if not event.action.keydown or not event.key.is_cursorkey:
                return
            steps = {"ArrowLeft": (-1, 0), "ArrowRight": (1, 0),
                     "ArrowUp": (0, -1), "ArrowDown": (0, 1)}
            for name, (dx, dy) in steps.items():
                if event.key == name:
                    nudge(selected["doc"], dx, dy, 5 if event.modifiers.shift else 1)
                    return

        async def run_auto() -> None:
            crops = await run.io_bound(self.project.region_rasters,
                                       self.index, rect, dpi, None)
            inks = [to_ink(c) for c in crops]
            radius = search_radius(inks[0].shape[1], inks[0].shape[0])
            shifts = await run.io_bound(auto_align, inks, radius)
            for doc, (dx, dy) in zip(docs, shifts):
                offsets[doc.doc_id] = (dx * step, dy * step)
            schedule()

        def reset() -> None:
            for doc in docs:
                offsets[doc.doc_id] = (0.0, 0.0)
            schedule()

        def close() -> None:
            self._align_drag = None
            self._align_keys = None
            dialog.close()

        def apply() -> None:
            if existing is not None:
                page.patches.remove(existing)
            patch = AlignPatch(rect=rect, offsets=dict(offsets))
            if not patch.is_identity():
                page.patches.append(patch)
            close()
            self.select_tool("pan")
            self.schedule_render(keep_view=True)

        def remove() -> None:
            if existing is not None:
                page.patches.remove(existing)
            close()
            self.schedule_render(keep_view=True)

        with ui.dialog() as dialog, ui.card().classes("w-[900px] max-w-full gap-2"):
            with ui.row().classes("items-center w-full"):
                ui.label("Magic align").classes("text-base font-medium")
                ui.label("shared ink is hidden — nudge until the colours cancel") \
                    .classes("text-xs opacity-60")
                ui.space()
                ui.label(f"{dpi:.0f} DPI").classes("text-xs opacity-60")

            preview = ui.element("img").props(f'id={preview_id}') \
                .classes("w-full rounded") \
                .style("image-rendering: pixelated; background: #ffffff;"
                       "max-height: 46vh; object-fit: contain")

            @ui.refreshable
            def rows() -> None:
                for position, doc in enumerate(docs):
                    dx, dy = offsets[doc.doc_id]
                    is_anchor = position == 0
                    active = (not is_anchor) and selected["doc"] == doc.doc_id
                    row_class = "items-center gap-2 w-full no-wrap rounded px-2 py-1"
                    with ui.row().classes(row_class + (" bg-primary/20" if active else "")):
                        ui.element("div").style(
                            f"width:12px;height:12px;border-radius:2px;"
                            f"background:{rgb_to_hex(doc.color)}"
                        )
                        ui.label(doc.name).classes("text-sm truncate w-40")

                        if is_anchor:
                            ui.label("anchor — does not move") \
                                .classes("text-xs opacity-60 flex-grow")
                            continue

                        ui.button("Move this",
                                  on_click=lambda _, d=doc.doc_id: (
                                      selected.__setitem__("doc", d), rows.refresh())) \
                            .props("flat dense no-caps" if not active
                                   else "unelevated dense no-caps color=primary") \
                            .classes("text-xs")
                        ui.label(f"{round(dx / step):+d}, {round(dy / step):+d} px") \
                            .classes("text-xs font-mono w-24 text-center")

                        for icon, (ddx, ddy) in (
                            ("chevron_left", (-1, 0)), ("chevron_right", (1, 0)),
                            ("expand_less", (0, -1)), ("expand_more", (0, 1)),
                        ):
                            ui.button(icon=icon,
                                      on_click=lambda _, d=doc.doc_id, x=ddx, y=ddy:
                                      nudge(d, x, y)) \
                                .props("flat dense size=sm")
                        ui.button(icon="restart_alt",
                                  on_click=lambda _, d=doc.doc_id: (
                                      offsets.__setitem__(d, (0.0, 0.0)), schedule())) \
                            .props("flat dense size=sm").tooltip("Reset this document")

            rows()

            ui.label("Drag the preview to move the selected document, or use the "
                     "arrow keys — hold Shift for 5 px steps.") \
                .classes("text-xs opacity-60")

            with ui.row().classes("w-full justify-end gap-2 items-center"):
                if existing is not None:
                    ui.button("Remove alignment", on_click=remove) \
                        .props("flat color=negative no-caps")
                ui.space()
                ui.button("Auto", icon="auto_fix_high", on_click=run_auto) \
                    .props("flat no-caps").tooltip("Re-run automatic alignment")
                ui.button("Reset", on_click=reset).props("flat no-caps")
                ui.button("Cancel", on_click=close).props("flat no-caps")
                ui.button("Apply", on_click=apply).props("unelevated no-caps")

        self._align_drag = on_drag
        self._align_keys = on_key
        dialog.on("hide", lambda _: close())
        dialog.open()

        async def start() -> None:
            await ui.run_javascript(f"RedlinerAlignDrag.attach('{preview_id}')")
            # Auto-align runs on open: the common case is that the guess is
            # right and the user only confirms it.
            if existing is None:
                await run_auto()
            else:
                await redraw()

        ui.timer(0.05, start, once=True)

    def undo_markup(self) -> None:
        page = self.current_page()
        if page and page.markups:
            page.markups.pop()
            self.refresh_markup()

    def clear_markup(self) -> None:
        page = self.current_page()
        if page and page.markups:
            page.markups.clear()
            self.refresh_markup()

    def on_zoom(self, relative: float) -> None:
        """Re-render to match the zoom level, so detail resolves as you go in."""
        wanted = min(MAX_PREVIEW_DPI, max(BASE_PREVIEW_DPI,
                                          self.project.preview_dpi * relative))
        # Snap to coarse steps: without this, every wheel tick is a new DPI and
        # the server re-renders continuously for no visible gain.
        wanted = round(wanted / 50.0) * 50.0
        if abs(wanted - self.render_dpi) < 1.0 or self.busy:
            return
        self.render_dpi = wanted
        self.schedule_render(keep_view=True)

    def go_to(self, index: int) -> None:
        if 0 <= index < self.project.page_count and index != self.index:
            self.index = index
            self.render_dpi = BASE_PREVIEW_DPI
            self.schedule_render(keep_view=False)
            self.pages_panel.refresh()

    def step(self, delta: int) -> None:
        self.go_to(self.index + delta)

    def refresh_all(self) -> None:
        self.docs_panel.refresh()
        self.align_panel.refresh()
        self.pages_panel.refresh()
        self.schedule_render(keep_view=False)

    # -- export ----------------------------------------------------------

    async def export(self) -> None:
        indices = self.project.export_indices()
        if not indices:
            ui.notify("No pages are marked for export", type="warning")
            return

        pixels = estimate_pixels(self.project, indices, self.project.dpi)
        if pixels > MAX_EXPORT_PIXELS:
            ui.notify(
                f"That export is about {pixels / 1e9:.1f} gigapixels. "
                "Lower the export DPI or select fewer pages.", type="negative",
            )
            return

        notification = ui.notification(f"Exporting {len(indices)} pages...",
                                       spinner=True, timeout=None)
        try:
            data = await run.io_bound(
                build_pdf, self.project, indices,
                ExportOptions(dpi=self.project.dpi, text_layer=self.text_layer.value),
            )
        except Exception as exc:
            notification.dismiss()
            ui.notify(f"Export failed: {exc}", type="negative")
            return
        notification.dismiss()

        ui.download.content(data, "redline.pdf")
        ui.notify(f"Exported {len(indices)} pages ({len(data) / 1e6:.1f} MB)",
                  type="positive")


def build_page(session: Session) -> None:
    """Lay out the whole interface for one session."""
    from .viewer import ZoomPanViewer

    project = session.project

    # Deferred renders are scheduled against this element rather than whatever
    # slot happens to be open. It is created once here and never refreshed, so a
    # timer parented to it always survives long enough to fire.
    session._timer_host = ui.element("div").style("display:none")

    # Registered once per client; both are inert until an align dialog claims
    # them. The drag events are throttled and carry a cumulative delta, so a
    # dropped frame costs nothing.
    ui.on("align_drag", session.dispatch_align_drag, throttle=0.05)
    ui.keyboard(on_key=session.dispatch_key)

    # -- header ----------------------------------------------------------
    with ui.header().classes("items-center gap-4 px-4 py-2"):
        ui.icon("difference", size="1.6rem")
        ui.label("Redliner").classes("text-xl font-semibold")
        ui.label("per-pixel document comparison").classes("text-sm opacity-70")
        ui.space()
        session.progress = ui.spinner(size="sm")
        session.progress.set_visibility(False)
        ui.button("Share", icon="link", on_click=session.share) \
            .props("flat").tooltip("Publish to a temporary link")
        ui.button("Export PDF", icon="download", on_click=session.export) \
            .props("unelevated color=primary")

    # -- left drawer: documents and settings ------------------------------
    with ui.left_drawer(value=True).props("width=340 bordered").classes("gap-0 p-0"):
        with ui.column().classes("w-full p-3 gap-3"):
            ui.label("Documents").classes("text-base font-medium")
            # The uploader's own file list duplicates the Documents list below
            # and never clears, so hide it and keep only the progress header.
            ui.add_css(".redliner-upload .q-uploader__list { display: none; }")
            accept = ",".join(accepted_extensions())
            ui.upload(
                multiple=True, auto_upload=True, max_files=64,
                on_upload=session.handle_upload,
            ).props(f'accept="{accept}" flat bordered') \
                .classes("w-full redliner-upload")

            ui.switch(
                "Combine files into one document",
                value=session.combine_uploads,
                on_change=lambda e: setattr(session, "combine_uploads", e.value),
            ).props("dense").tooltip(
                "For formats that are a set of files rather than one — a folder "
                "of Gerbers is a single board, not one document per layer"
            )

            _build_fetcher_panel(session)

            @ui.refreshable
            def docs_panel() -> None:
                if not project.docs:
                    ui.label("Add two or more documents to compare.") \
                        .classes("text-sm opacity-60 py-2")
                    return
                for position, doc in enumerate(project.docs):
                    with ui.card().classes("w-full p-2 gap-1"):
                        with ui.row().classes("items-center gap-2 w-full no-wrap"):
                            # A filled swatch, not an icon: which color belongs
                            # to which document is the thing you read this list
                            # for, so it has to be visible at a glance.
                            swatch = ui.button().props("flat dense round") \
                                .style(f"background:{rgb_to_hex(doc.color)};"
                                       "width:24px;height:24px;min-height:0;min-width:0") \
                                .tooltip("Color for ink unique to this document")
                            with swatch:
                                ui.color_picker(
                                    on_pick=lambda e, d=doc, b=swatch:
                                    session.pick_color(d, b, e.color)
                                )
                            ui.label(doc.name).classes("text-sm truncate flex-grow") \
                                .tooltip(doc.name)
                            ui.label(f"{doc.page_count}p").classes("text-xs opacity-60")
                            ui.button(icon="arrow_upward", on_click=lambda _, d=doc:
                                      session.move_document(d, -1)) \
                                .props("flat dense size=sm").set_enabled(position > 0) \
                                .tooltip("Move earlier in the stacking order")
                            ui.button(icon="arrow_downward", on_click=lambda _, d=doc:
                                      session.move_document(d, 1)) \
                                .props("flat dense size=sm") \
                                .set_enabled(position < len(project.docs) - 1) \
                                .tooltip("Move later in the stacking order")
                            ui.button(icon="close", on_click=lambda _, d=doc:
                                      session.remove_document(d)) \
                                .props("flat dense size=sm color=negative")

            session.docs_panel = docs_panel
            docs_panel()

            ui.separator()
            ui.label("Diff settings").classes("text-base font-medium")

            def touch(_=None) -> None:
                session.schedule_render()

            highlight = ui.switch("Highlight changes", value=project.settings.highlight,
                                  on_change=lambda e: (
                                      setattr(project.settings, "highlight", e.value), touch()))
            highlight.tooltip("Draw a highlighter mark around every changed region")

            with ui.column().classes("w-full gap-1 pl-1").bind_visibility_from(highlight, "value"):
                ui.label("Highlight color").classes("text-xs opacity-70")
                ui.color_input(
                    value=rgb_to_hex(project.settings.highlight_color),
                    on_change=lambda e: (setattr(project.settings, "highlight_color",
                                                 hex_to_rgb(e.value)), touch()),
                ).props("dense borderless").classes("w-full")

                # `label-always` floats the value badge above the track, so each
                # slider needs headroom or the badge sits on top of its caption.
                ui.label("Sensitivity").classes("text-xs opacity-70")
                ui.slider(min=0.005, max=0.3, step=0.005,
                          value=project.settings.highlight_threshold,
                          on_change=lambda e: (setattr(project.settings,
                                                       "highlight_threshold", e.value), touch())) \
                    .props("label-always dense").classes("mt-7") \
                    .tooltip("Lower catches subtler changes, at the cost of noise")

                ui.label("Spread").classes("text-xs opacity-70")
                ui.slider(min=1, max=40, step=1, value=project.settings.highlight_size,
                          on_change=lambda e: (setattr(project.settings,
                                                       "highlight_size", int(e.value)), touch())) \
                    .props("label-always dense").classes("mt-7") \
                    .tooltip("How far the highlighter extends past a change, in pixels")

            ui.label("Ink threshold").classes("text-xs opacity-70")
            ui.slider(min=0.0, max=0.4, step=0.01, value=project.settings.ink_floor,
                      on_change=lambda e: (setattr(project.settings,
                                                   "ink_floor", e.value), touch())) \
                .props("label-always dense").classes("mt-7") \
                .tooltip("Ignore ink fainter than this. Raise it to suppress "
                         "scanner speckle and JPEG noise.")

            ui.separator()

            @ui.refreshable
            def markup_panel() -> None:
                selected = session.selected_shape()
                with ui.row().classes("items-center gap-2 w-full"):
                    ui.label("Markup").classes("text-base font-medium")
                    if selected is not None:
                        ui.badge(f"editing {selected.kind}").props("color=primary")

                def changed() -> None:
                    # Property edits retarget the selection when there is one,
                    # and otherwise set the defaults for the next shape drawn.
                    session.apply_to_selection()
                    session.viewer.set_tool(session.tool, session.markup_color,
                                            session.markup_width)

                with ui.row().classes("items-center gap-2 w-full no-wrap"):
                    ui.label("Stroke").classes("text-xs opacity-70 w-12")
                    ui.color_input(
                        value=session.markup_color,
                        on_change=lambda e: (setattr(session, "markup_color", e.value),
                                             changed()),
                    ).props("dense borderless").classes("flex-grow")

                with ui.row().classes("items-center gap-2 w-full no-wrap"):
                    ui.label("Fill").classes("text-xs opacity-70 w-12")
                    fill_on = ui.switch(
                        value=session.markup_fill is not None,
                        on_change=lambda e: (
                            setattr(session, "markup_fill",
                                    "#ffff00" if e.value else None),
                            changed(), markup_panel.refresh()),
                    ).props("dense").tooltip("Off means a transparent interior")
                    if session.markup_fill is not None:
                        ui.color_input(
                            value=session.markup_fill,
                            on_change=lambda e: (setattr(session, "markup_fill", e.value),
                                                 changed()),
                        ).props("dense borderless").classes("flex-grow")
                    else:
                        ui.label("transparent").classes("text-xs opacity-50")

                with ui.row().classes("items-center gap-2 w-full no-wrap"):
                    ui.label("Width").classes("text-xs opacity-70 w-12")
                    ui.select({w: f"{w:g} pt" for w in WIDTHS},
                              value=session.markup_width,
                              on_change=lambda e: (
                                  setattr(session, "markup_width", float(e.value)),
                                  changed()),
                              ).props("dense outlined").classes("flex-grow")

                with ui.row().classes("items-center gap-2 w-full no-wrap"):
                    ui.label("Font").classes("text-xs opacity-70 w-12")
                    ui.select({k: k.title() for k in FONTS},
                              value=session.markup_font,
                              on_change=lambda e: (
                                  setattr(session, "markup_font", e.value), changed()),
                              ).props("dense outlined").classes("flex-grow")

                ui.switch("Box around text", value=session.markup_boxed,
                          on_change=lambda e: (setattr(session, "markup_boxed", e.value),
                                               changed())) \
                    .props("dense")

                if selected is not None:
                    with ui.row().classes("gap-1 w-full"):
                        ui.button("Delete", icon="delete",
                                  on_click=session.delete_selected) \
                            .props("flat dense no-caps color=negative")
                        ui.label("or press Delete; arrows nudge") \
                            .classes("text-xs opacity-50 self-center")

            session.markup_panel = markup_panel
            markup_panel()

            ui.separator()
            ui.label("Output").classes("text-base font-medium")
            ui.number("Export DPI", value=project.dpi, min=50, max=1200, step=25,
                      on_change=lambda e: setattr(project, "dpi", float(e.value or 200))) \
                .props("dense outlined").classes("w-full") \
                .tooltip("Resolution of the exported PDF. 200-300 suits most drawings.")
            session.text_layer = ui.switch("Searchable text layer", value=True) \
                .tooltip("Copy positioned text from the sources into the export, "
                         "so the result is searchable without OCR")

    # -- right drawer: output pages ---------------------------------------
    with ui.right_drawer(value=True).props("width=200 bordered").classes("p-2"):
        ui.label("Pages").classes("text-base font-medium pb-1")

        @ui.refreshable
        def pages_panel() -> None:
            if not project.page_count:
                ui.label("No pages yet").classes("text-sm opacity-60")
                return
            with ui.column().classes("w-full gap-1"):
                for i, page in enumerate(project.pages):
                    active = "bg-primary/20" if i == session.index else ""
                    with ui.row().classes(
                        f"items-center gap-1 w-full no-wrap rounded px-1 {active}"
                    ):
                        ui.checkbox(
                            value=page.export,
                            on_change=lambda e, p=page: setattr(p, "export", e.value),
                        ).props("dense").tooltip("Include this page in the export")
                        ui.button(f"Page {i + 1}", on_click=lambda _, n=i: session.go_to(n)) \
                            .props("flat dense no-caps align=left").classes("flex-grow")

        session.pages_panel = pages_panel
        pages_panel()

        ui.separator().classes("my-2")
        with ui.row().classes("gap-1"):
            ui.button("All", on_click=lambda: (
                [setattr(p, "export", True) for p in project.pages], pages_panel())) \
                .props("flat dense no-caps")
            ui.button("None", on_click=lambda: (
                [setattr(p, "export", False) for p in project.pages], pages_panel())) \
                .props("flat dense no-caps")

    # -- main area --------------------------------------------------------
    # NiceGUI's default content wrapper adds padding and sizes to its children;
    # the viewer needs to claim the whole viewport below the header instead.
    ui.query(".nicegui-content").classes("p-0 gap-0 w-full")

    with ui.column().classes("w-full p-0 gap-0"):
        with ui.tabs().classes("w-full") as tabs:
            view_tab = ui.tab("Compare", icon="visibility")
            align_tab = ui.tab("Align pages", icon="view_column")

        # Header (50px) plus the tab strip (48px) is all that sits above this,
        # so claim the rest of the viewport exactly -- any slack turns into a
        # second scrollbar next to the viewer's own pan surface.
        with ui.tab_panels(tabs, value=view_tab).classes("w-full p-0 overflow-hidden") \
                .style("height: calc(100vh - 98px)"):
            with ui.tab_panel(view_tab).classes("p-0 h-full overflow-hidden"):
                with ui.column().classes("w-full h-full gap-0"):
                    with ui.row().classes("items-center gap-2 px-3 py-1 border-b w-full"):
                        ui.button(icon="chevron_left",
                                  on_click=lambda: session.step(-1)).props("flat dense")

                        @ui.refreshable
                        def page_label() -> None:
                            total = project.page_count
                            text = (f"Page {session.index + 1} of {total}"
                                    if total else "Nothing loaded")
                            ui.label(text).classes("text-sm w-32 text-center")

                        session.page_label = page_label
                        page_label()

                        ui.button(icon="chevron_right",
                                  on_click=lambda: session.step(1)).props("flat dense")
                        ui.separator().props("vertical")
                        ui.button(icon="zoom_out",
                                  on_click=lambda: session.viewer.zoom_by(1 / 1.4)) \
                            .props("flat dense")
                        ui.button(icon="zoom_in",
                                  on_click=lambda: session.viewer.zoom_by(1.4)) \
                            .props("flat dense")
                        ui.button(icon="fit_screen",
                                  on_click=lambda: session.viewer.fit()) \
                            .props("flat dense").tooltip("Fit to window (or double-click)")
                        ui.separator().props("vertical")

                        @ui.refreshable
                        def tool_panel() -> None:
                            # Every button stays `flat`; the active one is marked
                            # with a background class instead of a filled
                            # variant, because switching Quasar button variants
                            # changes their width and shifts the whole row out
                            # from under the pointer as you pick a tool.
                            for tool in TOOLS.all():
                                highlight = ("bg-primary text-white"
                                             if session.tool == tool.name else "")
                                ui.button(icon=tool.icon,
                                          on_click=lambda _, v=tool.name:
                                          session.select_tool(v)) \
                                    .props("flat dense").classes(highlight) \
                                    .tooltip(tool.tooltip)

                        session.tool_panel = tool_panel
                        tool_panel()

                        ui.button(icon="undo", on_click=session.undo_markup) \
                            .props("flat dense").tooltip("Undo last markup")
                        ui.button(icon="delete_sweep", on_click=session.clear_markup) \
                            .props("flat dense").tooltip("Clear this page's markup")
                        ui.space()
                        ui.label().bind_text_from(
                            session, "render_dpi", lambda d: f"{d:.0f} DPI preview",
                        ).classes("text-xs opacity-60")

                    # min-h-0 is required: a flex child defaults to min-height
                    # auto, so an oversized page raster would stretch this box
                    # past the viewport and hand scrolling back to the browser
                    # instead of to the viewer's own pan.
                    with ui.element("div").classes("w-full flex-grow relative min-h-0"):
                        session.viewer = ZoomPanViewer(
                            on_zoom=session.on_zoom, on_shape=session.on_shape,
                            on_pick=session.pick_markup, on_drag=session.drag_markup,
                            on_edit=session.edit_markup,
                        )

            with ui.tab_panel(align_tab).classes("p-3 h-full overflow-auto"):
                with ui.row().classes("items-center gap-2 pb-2"):
                    ui.label("Line up the pages that should be compared "
                             "against each other.").classes("text-sm opacity-70")
                    ui.space()
                    ui.button("Reset to page order", icon="restart_alt",
                              on_click=lambda: (project.auto_align(),
                                                session.refresh_all())) \
                        .props("flat dense no-caps")

                @ui.refreshable
                def align_panel() -> None:
                    if not project.docs:
                        ui.label("Upload documents first.").classes("opacity-60")
                        return
                    _build_align_grid(session)

                session.align_panel = align_panel
                align_panel()


def _build_fetcher_panel(session: Session) -> None:
    """Search-and-fetch UI for every registered fetcher except plain upload.

    Rendered generically from the registry, so a fetcher plugin gets a working
    interface without shipping any UI code of its own.
    """
    remote = [f for f in session.fetchers if f.name != "upload"]
    if not remote:
        return

    with ui.expansion("Fetch from a system", icon="cloud_download") \
            .classes("w-full").props("dense"):
        for fetcher in remote:
            with ui.column().classes("w-full gap-1 pb-2"):
                with ui.row().classes("items-center gap-2 w-full no-wrap"):
                    ui.icon(fetcher.icon).classes("text-sm")
                    ui.label(fetcher.label).classes("text-sm font-medium")
                    ui.space()
                    for text, action in fetcher.actions().items():
                        ui.button(text, on_click=_fetcher_action(action)) \
                            .props("flat dense no-caps size=sm")

                status = fetcher.status()
                if status:
                    ui.label(status).classes("text-xs opacity-60")

                results = ui.column().classes("w-full gap-0")

                async def run_search(f=fetcher, box=results, field=None) -> None:
                    box.clear()
                    try:
                        hits = await f.search(field.value or "")
                    except NotImplementedError as exc:
                        with box:
                            ui.label(str(exc)).classes("text-xs opacity-60")
                        return
                    except Exception as exc:
                        ui.notify(f"Search failed: {exc}", type="negative")
                        return
                    with box:
                        if not hits:
                            ui.label("No matches").classes("text-xs opacity-60")
                        for hit in hits[:25]:
                            ui.button(
                                f"{hit.title}  {hit.subtitle}".strip(),
                                on_click=lambda _, r=hit.ref, ff=f:
                                session.fetch_document(ff, r),
                            ).props("flat dense no-caps align=left").classes("w-full")

                if fetcher.searchable:
                    field = ui.input(placeholder="Search…") \
                        .props("dense outlined clearable").classes("w-full")
                    field.on("keydown.enter",
                             lambda _, f=fetcher, b=results, q=field:
                             run_search(f, b, q))
                    ui.button("Search",
                              on_click=lambda _, f=fetcher, b=results, q=field:
                              run_search(f, b, q)) \
                        .props("flat dense no-caps size=sm")
                else:
                    reference = ui.input(placeholder="Paste a URL or reference") \
                        .props("dense outlined").classes("w-full")
                    ui.button("Fetch",
                              on_click=lambda _, f=fetcher, r=reference:
                              session.fetch_document(f, r.value or "")) \
                        .props("flat dense no-caps size=sm")


def _fetcher_action(action):
    """Wrap a fetcher's action so a stub's NotImplementedError reads as guidance
    rather than as a crash."""
    async def run() -> None:
        try:
            result = action()
            if hasattr(result, "__await__"):
                await result
        except NotImplementedError as exc:
            ui.notify(str(exc), type="warning", multi_line=True,
                      classes="max-w-xl")
        except Exception as exc:
            ui.notify(f"{exc}", type="negative")
    return run


def _build_align_grid(session: Session) -> None:
    """A row per output page, a column per document, thumbnails in the cells."""
    project = session.project
    columns = f"60px repeat({len(project.docs)}, minmax(120px, 1fr)) 60px"

    with ui.element("div").style(f"display:grid; grid-template-columns:{columns};"
                                 "gap:6px; align-items:stretch; width:100%"):
        ui.label("#").classes("text-xs font-medium opacity-70 self-center")
        for doc in project.docs:
            with ui.row().classes("items-center gap-1 no-wrap"):
                ui.element("div").style(
                    f"width:10px;height:10px;border-radius:2px;"
                    f"background:{rgb_to_hex(doc.color)}"
                )
                ui.label(doc.name).classes("text-xs font-medium truncate")
        ui.label("").classes("text-xs")

        for row in range(project.page_count):
            ui.label(str(row + 1)).classes("text-sm opacity-70 self-center text-center")

            for doc in project.docs:
                page_index = project.sequences[doc.doc_id][row]
                _align_cell(session, doc, row, page_index)

            with ui.column().classes("items-center justify-center gap-0"):
                ui.button(icon="delete_outline",
                          on_click=lambda _, r=row: (project.delete_output_page(r),
                                                     session.refresh_all())) \
                    .props("flat dense size=sm color=negative") \
                    .tooltip("Delete this output page from every document")


def _align_cell(session: Session, doc: SourceDoc, row: int, page_index: int | None) -> None:
    project = session.project

    classes = "w-full items-center justify-center p-1 rounded border cursor-pointer"
    if page_index is None:
        classes += " border-dashed opacity-50"

    with ui.column().classes(classes):
        if page_index is None:
            ui.icon("crop_free", size="1.2rem")
            ui.label("blank").classes("text-xs")
        else:
            try:
                ui.image(thumbnail(doc, page_index)).classes("w-full max-h-24 object-contain")
            except Exception:
                ui.icon("broken_image", size="1.2rem")
            ui.label(f"p{page_index + 1}").classes("text-xs opacity-70")

        with ui.menu():
            ui.menu_item("Insert blank here",
                         lambda d=doc, r=row: (project.insert_blank(d.doc_id, r),
                                               session.refresh_all()))
            ui.menu_item("Remove from this document",
                         lambda d=doc, r=row: (project.remove_slot(d.doc_id, r),
                                               session.refresh_all()))
            ui.separator()
            ui.menu_item("Move up",
                         lambda d=doc, r=row: (project.move_slot(d.doc_id, r, r - 1),
                                               session.refresh_all()))
            ui.menu_item("Move down",
                         lambda d=doc, r=row: (project.move_slot(d.doc_id, r, r + 1),
                                               session.refresh_all()))


@ui.page("/")
async def index() -> None:
    ui.dark_mode().enable()
    session = Session()
    build_page(session)

    await ui.context.client.connected()
    await session.viewer.start()

    ui.context.client.on_disconnect(session.dispose)


def _start_share_sweeper(interval: float = 600.0) -> None:
    """Drop expired shares periodically.

    Expiry is also checked when a link is opened, but a share nobody visits
    again would otherwise sit on disk forever. Sweeping on a timer bounds how
    long an expired comparison survives regardless of traffic.
    """
    store = shares.store()
    removed = store.sweep()
    if removed:
        logging.getLogger("redliner").info("removed %d expired shares", removed)

    @app.on_startup
    async def _sweeper() -> None:
        async def loop() -> None:
            while True:
                await asyncio.sleep(interval)
                try:
                    store.sweep()
                except Exception:
                    logging.getLogger("redliner").exception("share sweep failed")
        asyncio.create_task(loop())


def main(host: str = "0.0.0.0", port: int = 8080, reload: bool = False) -> None:
    # Up front, so a broken plugin shows up in the startup log rather than
    # surprising the first person who tries to upload something.
    discover_all()
    register_share_routes()
    _start_share_sweeper()
    orphans = Session.sweep_orphans()
    if orphans:
        logging.getLogger("redliner").info(
            "removed %d orphaned session directories", orphans)
    app.on_shutdown(clear_cache)
    ui.run(host=host, port=port, title="Redliner", reload=reload,
           favicon="\N{LARGE RED SQUARE}", dark=True, show=False)
