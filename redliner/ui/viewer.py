"""Zoom/pan canvas for the composed page, with a markup overlay.

Panning and zooming run entirely in the browser -- a server round-trip per
mouse-move would make dragging a large sheet unusable. Python is only told
about the zoom level, throttled, so it can re-render at a matching resolution:
CSS scaling a 100 DPI raster to 400% turns fine drawing detail to mush, which
defeats the purpose of a redline.

The annotation overlay is an SVG whose user units are PDF points, carried by
the same transform as the image. Because it is vector and point-based, markup
stays sharp at any zoom and lands in the right place regardless of the DPI the
preview happens to be rendered at.
"""

from __future__ import annotations

from collections.abc import Callable

from nicegui import ui

VIEWER_JS = """
window.RedlinerViewer = (function () {
  const views = {};

  function apply(v) {
    if (!v.img) return;
    const t = `translate(${v.tx}px, ${v.ty}px) scale(${v.scale})`;
    v.img.style.transform = t;
    for (const layer of [v.overlay, v.preview]) {
      if (!layer) continue;
      // Sized in image pixels and carrying the same transform, so their SVG
      // (whose viewBox is in PDF points) tracks the page exactly.
      layer.style.width = v.img.naturalWidth + 'px';
      layer.style.height = v.img.naturalHeight + 'px';
      layer.style.transform = t;
    }
    drawHandles(v);
  }

  /* ---- coordinate helpers ------------------------------------------ */

  // Screen pixels per PDF point, and where the page's origin sits on screen.
  function mapping(v) {
    const perPt = v.img.naturalWidth / (v.wPt || v.img.naturalWidth);
    return {k: perPt * v.scale, tx: v.tx, ty: v.ty};
  }

  // A gesture freezes the mapping it started with. A re-render at a new DPI
  // changes naturalWidth and v.scale together, and the two do not update in
  // the same instant -- naturalWidth flips when the new image decodes, v.scale
  // only in the load handler. A pointermove landing between those two reads a
  // mismatched pair and converts to a point miles away, which is what made a
  // freehand stroke jump when the zoom-triggered re-render fired mid-drag.
  //
  // Freezing is not an approximation: keep-view re-renders preserve the page's
  // on-screen position and size exactly, so the mapping captured at
  // pointerdown stays visually correct for the whole stroke.
  function activeMapping(v) {
    return v.frozen || mapping(v);
  }

  function freezeMapping(v) {
    v.frozen = mapping(v);
  }

  function thawMapping(v) {
    v.frozen = null;
  }

  function pointsPerPixel(v) {
    return 1 / activeMapping(v).k;
  }

  function toPoints(v, clientX, clientY) {
    const box = v.root.getBoundingClientRect();
    const m = activeMapping(v);
    return [(clientX - box.left - m.tx) / m.k,
            (clientY - box.top - m.ty) / m.k];
  }

  function toScreen(v, xPt, yPt) {
    const box = v.root.getBoundingClientRect();
    const perPt = v.img.naturalWidth / (v.wPt || v.img.naturalWidth);
    return [box.left + v.tx + xPt * perPt * v.scale,
            box.top + v.ty + yPt * perPt * v.scale];
  }

  /* ---- selection handles ------------------------------------------- */

  // Handles are drawn in screen space rather than inside the transformed
  // overlay, so they stay a constant grab size no matter how far you zoom in.
  function handleSpec(group) {
    const [x0, y0, x1, y1] = group.dataset.bbox.split(',').map(Number);
    const kind = group.dataset.kind;
    if (kind === 'line' || kind === 'arrow') {
      const pts = (group.dataset.points || '').split(' ')
        .map(p => p.split(',').map(Number)).filter(p => p.length === 2);
      return pts.map((p, i) => ({name: 'p' + i, x: p[0], y: p[1]}));
    }
    if (kind === 'rect' || kind === 'cloud') {
      return [{name: 'nw', x: x0, y: y0}, {name: 'ne', x: x1, y: y0},
              {name: 'se', x: x1, y: y1}, {name: 'sw', x: x0, y: y1}];
    }
    return [];  // text and pencil move but do not resize
  }

  function drawHandles(v) {
    if (!v.handles) return;
    const group = v.overlay && v.overlay.querySelector('g[data-selected="1"]');
    if (!group || v.tool !== 'select') { v.handles.innerHTML = ''; return; }

    const box = v.root.getBoundingClientRect();
    v.handles.innerHTML = handleSpec(group).map(h => {
      const [sx, sy] = toScreen(v, h.x, h.y);
      return `<div data-handle="${h.name}" style="position:absolute;
        left:${sx - box.left - 5}px; top:${sy - box.top - 5}px;
        width:10px; height:10px; background:#2196f3; border:1px solid #fff;
        border-radius:2px; pointer-events:auto; cursor:pointer;"></div>`;
    }).join('');
  }

  function fit(id) {
    const v = views[id];
    if (!v || !v.img || !v.img.naturalWidth) return;
    const box = v.root.getBoundingClientRect();
    const pad = 24;
    v.scale = Math.min((box.width - pad) / v.img.naturalWidth,
                       (box.height - pad) / v.img.naturalHeight);
    v.tx = (box.width - v.img.naturalWidth * v.scale) / 2;
    v.ty = (box.height - v.img.naturalHeight * v.scale) / 2;
    v.base = v.scale;
    apply(v);
    report(v, true);
  }

  function report(v, immediate) {
    // Zoom relative to fit-to-window, which is what "how much detail does the
    // server need to send" actually depends on.
    const rel = v.scale / (v.base || v.scale);
    clearTimeout(v.timer);
    const send = () => {
      // Never swap the image out from under an in-progress gesture. The debounce
      // means a zoom just before drawing would otherwise land mid-stroke.
      // Re-armed when the gesture ends, so the sharper render still arrives.
      if (v.gesture) { v.pendingReport = true; return; }
      emitEvent('viewer_zoom', {id: v.id, zoom: rel});
    };
    if (immediate) send(); else v.timer = setTimeout(send, 350);
  }

  function onLoad(v) {
    // A re-render at a different DPI is the same page at a new pixel size:
    // rescale so the sheet stays exactly where the user put it. Only a
    // genuinely new page falls through to fit().
    //
    // The flag rides on the image element itself rather than arriving as a
    // separate call, because the two would be separate socket messages with no
    // ordering guarantee -- and in practice the src landed first, so the image
    // loaded while the flag was still unset and every zoom snapped back to fit.
    const keep = v.img.getAttribute('data-keep') === '1';
    if (keep && v.lastW && v.img.naturalWidth) {
      const ratio = v.lastW / v.img.naturalWidth;
      v.scale *= ratio;
      v.base *= ratio;
      apply(v);
    } else {
      fit(v.id);
    }
    v.lastW = v.img.naturalWidth;
  }

  // The image lives inside an <img> that NiceGUI re-renders on update, and may
  // swap the node. Rebinding from a MutationObserver keeps the viewer working
  // whether the node is reused or replaced -- binding once at attach time
  // silently stops working after the first render.
  function bindImage(v) {
    const img = v.root.querySelector('img');
    if (!img || img === v.img) return;
    v.img = img;
    img.style.transformOrigin = '0 0';
    img.style.position = 'absolute';
    img.style.willChange = 'transform';
    img.draggable = false;
    img.addEventListener('load', () => onLoad(v));
    if (img.complete && img.naturalWidth) onLoad(v);
  }

  /* ---- markup drawing ---------------------------------------------- */

  function previewSvg(v, a, b, trail) {
    const [x0, y0] = a, [x1, y1] = b;
    const col = v.toolColor, w = v.toolWidth;
    // `rl-ants` animates the dash offset, so an in-progress shape reads as
    // "still being drawn" rather than as something already committed.
    const stroke = `stroke="${col}" stroke-width="${w}" fill="none"
                    stroke-linecap="round" stroke-linejoin="round" class="rl-ants"`;
    let body = '';
    if (v.tool === 'pencil') {
      const path = (trail || []).map(p => `${p[0]},${p[1]}`).join(' ');
      body = `<polyline points="${path}" stroke="${col}" stroke-width="${w}"
               fill="none" stroke-linecap="round" stroke-linejoin="round"/>`;
    } else if (v.tool === 'line' || v.tool === 'arrow') {
      body = `<line x1="${x0}" y1="${y0}" x2="${x1}" y2="${y1}" ${stroke}/>`;
      if (v.tool === 'arrow') {
        const dx = x1 - x0, dy = y1 - y0, len = Math.hypot(dx, dy);
        if (len > 1) {
          const ux = dx / len, uy = dy / len,
                s = Math.min(Math.max(10, w * 5), len * 0.5);
          const bx = x1 - ux * s, by = y1 - uy * s;
          body += `<polygon fill="${col}" points="${x1},${y1}
                   ${bx - uy * s * 0.4},${by + ux * s * 0.4}
                   ${bx + uy * s * 0.4},${by - ux * s * 0.4}"/>`;
        }
      }
    } else {
      // Rectangle, cloud and align share a box preview. The latter two are
      // dashed: a cloud's scalloped outline is generated on release, and an
      // align region is a selection rather than something that gets drawn.
      const dash = (v.tool === 'cloud' || v.tool === 'align')
        ? 'stroke-dasharray="6 4"' : '';
      body = `<rect x="${Math.min(x0, x1)}" y="${Math.min(y0, y1)}"
               width="${Math.abs(x1 - x0)}" height="${Math.abs(y1 - y0)}"
               ${stroke} ${dash}/>`;
    }
    return `<svg xmlns="http://www.w3.org/2000/svg"
             viewBox="0 0 ${v.wPt} ${v.hPt}" width="100%" height="100%"
             style="position:absolute;inset:0;overflow:visible">${body}</svg>`;
  }

  return {
    attach(id) {
      const root = document.getElementById(id);
      if (!root || views[id]) return;
      const v = {id, root, img: null, overlay: null, preview: null,
                 scale: 1, tx: 0, ty: 0, base: 0, timer: null,
                 tool: null, toolColor: '#d90000', toolWidth: 1.5,
                 wPt: 0, hPt: 0,
                 // Gesture state: `frozen` pins the screen<->points mapping for
                 // the duration of a drag, `gesture` defers zoom reports.
                 frozen: null, gesture: false, pendingReport: false};
      views[id] = v;

      v.overlay = root.querySelector('.rl-overlay');
      v.preview = root.querySelector('.rl-preview');
      v.handles = root.querySelector('.rl-handles');
      for (const el of [v.overlay, v.preview]) {
        if (el) {
          el.style.position = 'absolute';
          el.style.top = '0';
          el.style.left = '0';
          el.style.transformOrigin = '0 0';
          el.style.pointerEvents = 'none';
        }
      }
      if (v.handles) {
        // Screen-space layer: never transformed, so handles keep a constant
        // grab size. Individual handles re-enable pointer events.
        v.handles.style.position = 'absolute';
        v.handles.style.inset = '0';
        v.handles.style.pointerEvents = 'none';
      }

      bindImage(v);
      new MutationObserver(() => bindImage(v))
        .observe(root, {childList: true, subtree: true, attributes: true,
                        attributeFilter: ['src']});

      root.addEventListener('wheel', (e) => {
        e.preventDefault();
        const box = root.getBoundingClientRect();
        const cx = e.clientX - box.left, cy = e.clientY - box.top;
        const factor = Math.exp(-e.deltaY * 0.0015);
        const next = Math.min(Math.max(v.scale * factor, (v.base || 1) * 0.2), 60);
        // Keep the point under the cursor pinned while scaling.
        v.tx = cx - (cx - v.tx) * (next / v.scale);
        v.ty = cy - (cy - v.ty) * (next / v.scale);
        v.scale = next;
        apply(v);
        report(v, false);
      }, {passive: false});

      let mode = null, lx = 0, ly = 0, from = null, trail = [];
      let dragId = 0, dragHandle = null, ox = 0, oy = 0;

      root.addEventListener('pointerdown', (e) => {
        if (e.button !== 0) return;
        // Capture is an optimisation, not a requirement: losing the pointer
        // outside the element only cuts a drag short. Letting it throw here
        // would abort the handler and break drawing entirely.
        try { root.setPointerCapture(e.pointerId); } catch (_) {}

        if (v.tool === 'select') {
          const handle = e.target.closest && e.target.closest('[data-handle]');
          if (handle) {
            mode = 'handle'; dragHandle = handle.dataset.handle;
            dragId += 1; ox = e.clientX; oy = e.clientY;
            v.gesture = true;
            freezeMapping(v);
            return;
          }
          const group = e.target.closest && e.target.closest('g[data-idx]');
          emitEvent('markup_pick', {id: v.id, idx: group ? +group.dataset.idx : null});
          if (group) {
            mode = 'shape'; dragHandle = null;
            dragId += 1; ox = e.clientX; oy = e.clientY;
            v.gesture = true;
            freezeMapping(v);
          }
          return;
        }

        if (v.tool && v.tool !== 'pan') {
          if (v.tool === 'text') {
            const p = toPoints(v, e.clientX, e.clientY);
            emitEvent('viewer_shape', {id: v.id, kind: 'text', points: [p]});
            return;
          }
          mode = 'draw';
          v.gesture = true;
          freezeMapping(v);
          from = toPoints(v, e.clientX, e.clientY);
          trail = [from];
        } else {
          mode = 'pan';
          lx = e.clientX; ly = e.clientY;
          root.style.cursor = 'grabbing';
        }
      });

      root.addEventListener('pointermove', (e) => {
        if (mode === 'pan') {
          v.tx += e.clientX - lx; v.ty += e.clientY - ly;
          lx = e.clientX; ly = e.clientY;
          apply(v);
        } else if (mode === 'draw' && v.preview) {
          const now = toPoints(v, e.clientX, e.clientY);
          if (v.tool === 'pencil') trail.push(now);
          v.preview.innerHTML = previewSvg(v, from, now, trail);
        } else if (mode === 'shape' || mode === 'handle') {
          // Cumulative delta, so a throttled or dropped update never loses
          // movement -- the next one carries the whole offset.
          const per = pointsPerPixel(v);
          emitEvent('markup_drag', {id: v.id, drag: dragId, handle: dragHandle,
                                    dx: (e.clientX - ox) * per,
                                    dy: (e.clientY - oy) * per});
        }
      });

      const stop = (e) => {
        try { root.releasePointerCapture(e.pointerId); } catch (_) {}
        if (mode === 'draw' && from) {
          const to = toPoints(v, e.clientX, e.clientY);
          if (v.preview) v.preview.innerHTML = '';
          if (v.tool === 'pencil') {
            if (trail.length > 2) {
              emitEvent('viewer_shape', {id: v.id, kind: 'pencil', points: trail});
            }
          } else if (Math.hypot(to[0] - from[0], to[1] - from[1]) > 2) {
            emitEvent('viewer_shape', {id: v.id, kind: v.tool, points: [from, to]});
          }
        }
        if (mode === 'pan') root.style.cursor = v.tool && v.tool !== 'pan' ? 'crosshair' : 'grab';
        mode = null; from = null; trail = []; dragHandle = null;

        // The gesture is over: unfreeze, and deliver any zoom report that was
        // held back so the sharper render still arrives.
        v.gesture = false;
        thawMapping(v);
        if (v.pendingReport) {
          v.pendingReport = false;
          report(v, true);
        }
      };
      root.addEventListener('pointerup', stop);
      root.addEventListener('pointercancel', stop);

      root.addEventListener('dblclick', (e) => {
        const group = e.target.closest && e.target.closest('g[data-idx]');
        if (v.tool === 'select' && group) {
          emitEvent('markup_edit', {id: v.id, idx: +group.dataset.idx});
        } else if (!v.tool || v.tool === 'pan') {
          fit(id);
        }
      });
      new ResizeObserver(() => { if (!v.base) fit(id); }).observe(root);
    },
    fit,
    setPage(id, wPt, hPt) {
      const v = views[id];
      if (!v) return;
      v.wPt = wPt; v.hPt = hPt;
      apply(v);
    },
    setTool(id, tool, color, width) {
      const v = views[id];
      if (!v) return;
      v.tool = tool; v.toolColor = color; v.toolWidth = width;
      v.root.style.cursor = tool === 'select' ? 'default'
        : (tool && tool !== 'pan' ? 'crosshair' : 'grab');
      // The overlay only intercepts clicks while selecting; otherwise it would
      // swallow the pointer and break panning and drawing.
      if (v.overlay) v.overlay.style.pointerEvents = tool === 'select' ? 'auto' : 'none';
      drawHandles(v);
    },
    refresh(id) {
      const v = views[id];
      if (v) drawHandles(v);
    },
    zoomBy(id, factor) {
      const v = views[id];
      if (!v) return;
      const box = v.root.getBoundingClientRect();
      const cx = box.width / 2, cy = box.height / 2;
      const next = Math.min(Math.max(v.scale * factor, (v.base || 1) * 0.2), 60);
      v.tx = cx - (cx - v.tx) * (next / v.scale);
      v.ty = cy - (cy - v.ty) * (next / v.scale);
      v.scale = next;
      apply(v);
      report(v, false);
    },
  };
})();

// Dragging inside the align dialog's preview nudges the selected document.
window.RedlinerAlignDrag = {
  attach(imgId) {
    const img = document.getElementById(imgId);
    if (!img || img.dataset.rlDrag) return;
    img.dataset.rlDrag = '1';
    img.style.touchAction = 'none';
    img.style.cursor = 'move';

    let dragging = false, ox = 0, oy = 0, dragId = 0;
    img.addEventListener('pointerdown', (e) => {
      dragging = true;
      ox = e.clientX; oy = e.clientY; dragId += 1;
      try { img.setPointerCapture(e.pointerId); } catch (_) {}
      e.preventDefault();
    });
    img.addEventListener('pointermove', (e) => {
      if (!dragging) return;
      // Report the cumulative delta since pointerdown, in source pixels, not a
      // per-move increment. Re-renders are throttled, so dropped events would
      // silently eat movement if each one only carried its own step.
      const scale = img.naturalWidth / Math.max(1, img.clientWidth);
      emitEvent('align_drag', {drag: dragId,
                               dx: (e.clientX - ox) * scale,
                               dy: (e.clientY - oy) * scale});
    });
    const stop = (e) => {
      dragging = false;
      try { img.releasePointerCapture(e.pointerId); } catch (_) {}
    };
    img.addEventListener('pointerup', stop);
    img.addEventListener('pointercancel', stop);
  }
};
"""


def install() -> None:
    """Add the viewer script to this page's head, once per client.

    The guard has to be per-client, not per-process: head HTML is emitted into
    each page as it is served, so a process-wide "already installed" flag would
    give the script to whichever client connected first and leave every
    subsequent visitor with a dead viewer.
    """
    client = ui.context.client
    if getattr(client, "_redliner_viewer_installed", False):
        return
    ui.add_head_html(f"<script>{VIEWER_JS}</script>")
    ui.add_head_html(
        "<style>"
        "@keyframes rl-ants { to { stroke-dashoffset: -12; } }"
        ".rl-ants { stroke-dasharray: 6 6; animation: rl-ants 0.5s linear infinite; }"
        "</style>"
    )
    client._redliner_viewer_installed = True


class ZoomPanViewer:
    """An image that can be panned, zoomed and annotated."""

    _counter = 0

    def __init__(self, on_zoom: Callable[[float], None] | None = None,
                 on_shape: Callable[[dict], None] | None = None,
                 on_pick: Callable[[int | None], None] | None = None,
                 on_drag: Callable[[dict], None] | None = None,
                 on_edit: Callable[[int], None] | None = None) -> None:
        install()
        ZoomPanViewer._counter += 1
        self.element_id = f"redliner-view-{ZoomPanViewer._counter}"
        self._on_zoom = on_zoom
        self._on_shape = on_shape
        self._on_pick = on_pick
        self._on_drag = on_drag
        self._on_edit = on_edit

        self.root = ui.element("div").props(f'id={self.element_id}').classes(
            "relative w-full h-full overflow-hidden bg-neutral-200 "
            "dark:bg-neutral-800 cursor-grab select-none"
        )
        with self.root:
            # A raw <img>, deliberately not ui.image: ui.image renders a Quasar
            # q-img whose wrapper sizes and clips the inner image to the
            # container. The pan/zoom transform has to act on the image's own
            # natural pixel size, which only a plain <img> gives us.
            self.image = ui.element("img").classes("max-w-none")
            self.overlay = ui.html("").classes("rl-overlay")
            self.preview = ui.html("").classes("rl-preview")
            self.handles = ui.html("").classes("rl-handles")

        ui.on("viewer_zoom", self._handle_zoom)
        ui.on("viewer_shape", self._handle_shape)
        ui.on("markup_pick", self._handle_pick)
        ui.on("markup_drag", self._handle_drag, throttle=0.05)
        ui.on("markup_edit", self._handle_edit)

    async def start(self) -> None:
        """Wire up the browser-side handlers. Call once the client is connected."""
        await ui.run_javascript(f"RedlinerViewer.attach('{self.element_id}')")

    def _mine(self, event) -> dict | None:
        payload = event.args or {}
        return payload if payload.get("id") == self.element_id else None

    def _handle_zoom(self, event) -> None:
        payload = self._mine(event)
        if payload and self._on_zoom:
            self._on_zoom(float(payload.get("zoom", 1.0)))

    def _handle_shape(self, event) -> None:
        payload = self._mine(event)
        if payload and self._on_shape:
            self._on_shape(payload)

    def _handle_pick(self, event) -> None:
        payload = self._mine(event)
        if payload and self._on_pick:
            index = payload.get("idx")
            self._on_pick(None if index is None else int(index))

    def _handle_drag(self, event) -> None:
        payload = self._mine(event)
        if payload and self._on_drag:
            self._on_drag(payload)

    def _handle_edit(self, event) -> None:
        payload = self._mine(event)
        if payload and self._on_edit and payload.get("idx") is not None:
            self._on_edit(int(payload["idx"]))

    def set_source(self, data_url: str, keep_view: bool = False) -> None:
        """Swap the displayed image.

        `keep_view` marks the swap as a resolution change of the same page, so
        the browser holds the current pan and zoom instead of refitting.
        """
        # Both assigned straight into the prop dict rather than via props(): a
        # PNG data URL contains '=' and ',', which the props string parser
        # splits on. Setting them together means the flag and the image reach
        # the browser in one atomic element update.
        self.image._props["src"] = data_url
        self.image._props["data-keep"] = "1" if keep_view else "0"
        self.image.update()

    def set_page_size(self, width_pt: float, height_pt: float) -> None:
        ui.run_javascript(
            f"RedlinerViewer.setPage('{self.element_id}', {width_pt}, {height_pt})"
        )

    def set_markup(self, svg: str) -> None:
        self.overlay.set_content(svg)
        # The overlay's DOM was just replaced, so the selection handles that
        # were positioned against the old nodes have to be rebuilt.
        ui.run_javascript(f"RedlinerViewer.refresh('{self.element_id}')")

    def set_tool(self, tool: str, color: str, width: float) -> None:
        ui.run_javascript(
            f"RedlinerViewer.setTool('{self.element_id}', "
            f"'{tool}', '{color}', {width})"
        )

    def fit(self) -> None:
        ui.run_javascript(f"RedlinerViewer.fit('{self.element_id}')")

    def zoom_by(self, factor: float) -> None:
        ui.run_javascript(f"RedlinerViewer.zoomBy('{self.element_id}', {factor})")
