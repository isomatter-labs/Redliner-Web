# Redliner 2.0

Per-pixel visual diff for PDFs and drawings, served over the web. Stand up one
server and users compare revisions in a browser with nothing to install.

This is a from-scratch remake of Redliner 1.5 (PySide6 + OpenGL desktop app).
The comparison math is a NumPy port of that version's `res/diff.frag` shader,
generalized from 2 documents to N.

## Running it

```bash
python -m venv .venv
.venv/bin/pip install -e .          # Windows: .venv\Scripts\pip
.venv/bin/python -m redliner        # --host, --port, --reload
```

Then open <http://localhost:8080>.

Tests: `python -m pytest`

## How the comparison works

Everything happens in **ink space**: `ink = 1 - luminance`, so 0.0 is bare paper
and 1.0 is saturated ink. Working in ink rather than luminance is what makes the
composite linear, and therefore extensible past two documents.

For a stack of N aligned ink layers:

```
same     = min(ink_0 … ink_N-1)      # ink every document agrees on
excess_i = ink_i - same              # ink unique to document i

px = 1 - same - Σ (1 - color_i) · excess_i
```

`same` is subtracted from all three channels, so shared content renders black
exactly as it appears in the sources. Each document's excess ink pulls the pixel
toward that document's assigned color.

For N = 2 this reduces algebraically to the v1.5 shader — `excess` of the old
document is the shader's `removed`, `excess` of the new one is `added`.
`tests/test_compose.py` pins this against a literal transcription of the shader.

### Differences from the original, and why

- **N documents, not 2.** The decomposition above generalizes; a naive pairwise
  version would erase a mark present in two of three documents, since it is a
  change but is unique to neither.
- **Highlight threshold is resolution-independent.** The shader summed
  `abs(lhs - rhs)` over a `(2·size)²` window and compared the raw sum against a
  threshold, so the threshold had to be retuned whenever window size or DPI
  changed. Here the window is averaged instead of summed, so the setting means
  "average fraction of disagreeing ink" and stays put. It is also O(n) rather
  than O(n·size²), because a box mean is separable.
- **Composite is banded.** An E-size sheet at 300 DPI is ~67 megapixels;
  materializing float32 intermediates for a whole page at once approaches a
  gigabyte. Pages are composited in horizontal bands with a halo so the
  highlight filter does not seam at band edges.

### The prose spec vs. the shader

The original description of the algorithm (average the colorized layers, then
darken with a separately-built "unmodified" layer) and the shipped shader are
not the same algorithm. The shader's approach is better and is what is
implemented here: averaging washes unique content out toward white in
proportion to the number of documents, so with three inputs a change unique to
one of them renders at a third of its saturation. The subtractive ink form has
no such falloff.

## Extending it

Parsers (new file formats), fetchers (PLM systems and document vaults) and
markup tools are behind registries — adding one means writing a class, with no
core changes. Plugins install either by dropping a module into
`redliner/extensions/` or by declaring entry points from a separate pip package.

See **[EXTENDING.md](EXTENDING.md)**. Worked stubs live in
`redliner/extensions/`: a Gerber parser showing the multi-file case, and a PLM
fetcher showing search-and-authenticate.

## Architecture

```
redliner/
  core/
    compose.py     the diff engine (NumPy port of diff.frag, N-way)
    documents.py   document handles, bounded LRU raster cache
    project.py     documents + page alignment + render orchestration
    align.py       magic align: shift estimation and region-scoped offsets
    markup.py      annotation shapes, SVG and vector-PDF rendering
    export.py      composite -> PDF, with a searchable text layer
  plugins/
    __init__.py    registry and plugin discovery
    parsers.py     file formats -> comparable pages
    fetchers.py    where documents come from
    tools.py       markup tools
  extensions/      drop-in plugins, imported at startup
  ui/
    app.py         NiceGUI application, one Session per browser connection
    viewer.py      zoom/pan canvas and markup overlay
```

**Page alignment.** Each document has a *sequence*: a list as long as the output,
whose entries are a source page index or `None` for "nothing here". Inserting a
blank into one document's sequence is how a user re-aligns a revision that
gained or lost a sheet. All sequences are kept the same length, so output page
`i` composites `sequences[doc][i]` across every document.

**Searchable output without OCR.** The composite is a raster, but the source
PDFs already carry positioned text, so exported pages get an invisible text
layer (PDF render mode 3) lifted straight from them. That is both faster and
far more accurate than re-recognizing text we already have in vector form.
Words from every revision are included, so a search finds text that was removed
as well as text that was added. Scanned sources have no text to lift and are the
only case where OCR would add anything.

**Markup is vector, in page space.** Shapes are stored in PDF points, so an
annotation drawn over a 100 DPI preview lands identically in a 600 DPI export.
The same shape list drives the on-screen SVG overlay and the exported PDF, where
shapes are written as real vector operators rather than burned into the raster.

Tools: freehand, line, arrow, rectangle, revision cloud and text. Stroke colour,
fill (with a transparent option), line width and font are set in the Markup
panel; changing one retargets the current selection if there is one, and
otherwise sets the default for the next shape drawn. Text can be boxed, and the
box is measured with the chosen font's real metrics rather than an assumed
character width — monospace is wider than proportional for the same string.

Freehand strokes are thinned with Ramer–Douglas–Peucker before being stored.
Pointer samples arrive far denser than a stroke's actual detail, and keeping
every one bloats both the overlay and the exported PDF for no visible gain.

**Editing existing markup.** The select tool picks a shape, drags it, and
resizes it from corner handles (box shapes) or endpoint handles (lines and
arrows). Double-click a text annotation to edit it; Delete removes the
selection and the arrow keys nudge it. Two details worth knowing:

- Only a shape's *stroke* is a hit target, plus its interior if it has a fill —
  clicking inside an unfilled rectangle deliberately hits nothing, as in most
  vector editors. Thin shapes carry an invisible fat stroke so they stay easy
  to grab.
- Handles are drawn in screen space rather than inside the transformed overlay,
  so they keep a constant grab size however far you zoom in.

In-progress shapes and the selection outline both use an animated marching-ants
dash, so what is being drawn reads differently from what is already committed.

**Page previews.** The Pages panel shows a thumbnail of each composed page, so
you can see which sheets actually changed before deciding what to export. Pages
left out of the export are dimmed. Previews are rendered to a target pixel width
rather than a fixed DPI, so an E-size sheet costs about the same as a letter one,
and cached against a fingerprint of everything that affects a render -- document
colours, diff settings, page alignment, magic-align patches -- rather than a
manual dirty flag that someone eventually forgets to set.

**Progressive preview.** The viewer renders at 100 DPI for responsiveness and
re-renders at up to 400 DPI as you zoom in, holding your pan and zoom across the
swap. Pan and zoom themselves are pure browser-side transforms — a server
round-trip per mouse-move would make dragging a large sheet unusable.

## Sharing

**Share** publishes the pages you have marked for export to a temporary URL —
`/s/<token>` — with a chosen lifetime, from 15 minutes to 30 days.

The link serves a PDF rather than a live viewer. That is deliberate: a live
share would have to keep your source documents on the server for the whole TTL
in order to re-render, whereas a PDF exposes only the one composite you chose.
It is also what reviewers actually want — they open it in Bluebeam or Acrobat,
mark it up further and attach it to a change order. The searchable text layer
comes along, so Ctrl+F works in their reader.

Tokens are `uuid4` hex: 122 bits, so shares cannot be found by guessing, and
validating the shape also stops a token ever being a path traversal. Expiry is
enforced when a link is opened *and* by a sweeper, so a share nobody revisits
still goes away.

Shares live on disk (`REDLINER_SHARE_DIR`, default `<temp>/redliner_shares`) so
a link you have already emailed survives a restart.

**Expiry is hygiene, not revocation.** Anyone who opened a share has the PDF and
keeps it. Redliner is expected to sit behind a login; that is what controls who
gets in.

## Magic align

CAD exporters routinely re-emit a text box or symbol a few points off between
saves. That is not a design change, but a per-pixel diff cannot tell the
difference, and the drift is often visually louder than the revision you are
actually looking for.

Lasso the drifting element with the align tool and a dialog opens showing just
that region, rendered **difference-only**: ink both documents agree on drops out entirely, so
drift appears as two coloured ghosts of the same shape and the job is simply to
make them cancel. Auto-align runs on open; you can then drag the preview or use
the arrow keys (Shift for 5 px steps) to adjust. The first document is the
anchor and never moves.

Three things make it work:

- **Offsets are region-scoped, and the region is a freehand blob.** Only pixels
  inside the outline move, so correcting one drifting label never disturbs the
  rest of the sheet -- not even a frame line running right beside it.
- **Auto-align is FFT cross-correlation**, not a brute-force shift sweep —
  O(n log n) against O(n·r²), which matters because the search radius scales
  with the region you drew. The inputs are mean-subtracted first: raw
  correlation of sparse ink peaks wherever the most ink overlaps at all, which
  drags the answer toward zero shift. The peak is then refined to sub-pixel by
  fitting a parabola through its neighbours.
- **The offset is applied in vector space, not by moving pixels.** This is the
  part that decides whether the feature works. A 3 pt drift is 6.25 px at 150
  DPI and 8.46 px at 203 — rarely a whole number, so a pixel shift can only ever
  get within half a pixel, and that residue is bright enough to keep tripping
  the change highlighter on content you just aligned. Instead the offset goes
  into the *render matrix* while the clip window is pre-shifted into that
  matrix's input space, which leaves the destination pixel grid untouched.
  Corrected regions then come out **bit-identical** — zero differing pixels, at
  every resolution tested (see `test_correction_is_exact_at_any_resolution`).

  Shifting the clip window alone is not equivalent, and this is the trap:
  PyMuPDF snaps a clip to whole device pixels, so a fractional window shift
  silently changes the sampling phase and leaves anti-alias fringing on every
  edge.

Rasterized (non-PDF) inputs have no vector source to re-render, so they fall
back to whole-pixel shifting.

**Enclose everything that drifted together, and nothing else.** Only what the
lasso contains moves, so an outline cutting through the middle of a box will
correct the text inside it and leave the box edges behind. The difference-only
preview shows this immediately: anything left behind stays coloured.

This is why the region is freehand rather than a rectangle. On the test fixture,
correcting a 3 pt drift with a box whose edge grazes the sheet frame leaves 450
differing pixels — the frame itself, dragged out of alignment. The same
correction inside a lasso around the label alone leaves **zero**.

## Not built yet

- **Vector-space diffing (the "moonshot").** Everything above rasterizes. A true
  vector diff would need to compare page content streams — matching paths and
  text runs between revisions and emitting recolored vector output. It is
  genuinely hard: PDF producers reorder and re-emit operators between saves, so
  "same path" has to be decided geometrically, not by comparing streams. Worth
  scoping separately. Note that the searchable-text-layer work above already
  delivers one of its main benefits (text-searchable output).
- Grouping markup, and rotating it.
- Rotating markup, and resizing freehand strokes (they move but do not scale).
- Whole-page alignment. Offsets are region-scoped only; a sheet that drifted
  bodily needs one region covering it.
- Freehand align regions. The tool draws a rectangle, not an arbitrary blob.
- Persistence. A session lives in server memory and its uploads are deleted on
  disconnect; there are no saved projects.
- Automatic page matching (aligning revisions by content rather than by hand).

## License

GPL-3.0-or-later, the same as [Redliner 1.5](https://github.com/isomatter-labs/Redliner).
See [LICENSE](LICENSE).

Worth knowing if you are here to fork: the GPL's copyleft reaches anything you
distribute that is derived from this code. Running a modified Redliner on your
own internal server is not distribution, so in-house parsers, fetchers and tools
carry no obligation to publish. Handing a modified build to another company does.
The plugin entry-point route in [EXTENDING.md](EXTENDING.md) also lets you keep
proprietary integrations in a separate package rather than as a fork.
