# Extending Redliner

Redliner expects to be adapted. Three things differ most between organisations,
so each is behind a registry:

| Extension point | Answers | Module |
| --- | --- | --- |
| **Parsers** | How does this file become comparable pages? | `redliner/plugins/parsers.py` |
| **Fetchers** | Where do documents come from? | `redliner/plugins/fetchers.py` |
| **Tools** | What can you draw on a comparison? | `redliner/plugins/tools.py` |

Adding any of them means writing one class. No core file changes.

## Two ways to install a plugin

**Drop-in** — put a module in `redliner/extensions/`. Every module there is
imported at startup. Simplest if you have forked the repo.

**Entry points** — ship a separate package, no fork required:

```toml
# your-plugin/pyproject.toml
[project.entry-points."redliner.parsers"]
odb = "acme_redliner.odb:OdbParser"

[project.entry-points."redliner.fetchers"]
windchill = "acme_redliner.windchill:WindchillFetcher"

[project.entry-points."redliner.tools"]
balloon = "acme_redliner.balloon:BalloonTool"
```

`pip install your-plugin` next to Redliner and it is picked up. This is the
better route for anything you want to maintain across Redliner upgrades.

A plugin that raises on import is logged and skipped. One broken extension must
not stop the server from starting, because whoever can fix the plugin usually
isn't whoever needs the server up.

## Parsers — new file formats

A parser answers one question: given some files, how do I produce comparable
pages? Everything downstream — the compositor, magic align, the exporter — talks
only to `DocumentSource`.

```python
from pathlib import Path
import numpy as np
from redliner.plugins.parsers import PARSERS, DocumentSource, SourceParser

class OdbSource(DocumentSource):
    supports_subpixel = False        # see "Sub-pixel accuracy" below

    def __init__(self, paths: list[Path]) -> None:
        self.paths = paths

    @property
    def page_count(self) -> int:
        return 1

    @property
    def page_sizes(self) -> list[tuple[float, float]]:
        return [(612.0, 792.0)]      # points

    def render(self, page_index, dpi, clip=None, offset=(0.0, 0.0)):
        """(H, W) uint8 grayscale, 255 = bare paper."""
        ...

    def words(self, page_index):
        return []                    # optional; feeds the searchable export

@PARSERS.register
class OdbParser(SourceParser):
    name = "odb"
    label = "ODB++"
    priority = 40                    # lower wins when parsers overlap
    extensions = frozenset({".tgz", ".odb"})

    @classmethod
    def open(cls, paths: list[Path]) -> DocumentSource:
        return OdbSource(paths)
```

### One document, many files

`open()` takes a **list** of paths because plenty of real formats are a set of
files. A folder of Gerbers is one board; comparing it layer-by-layer as separate
documents would bury the change you care about.

Users get this either by ticking **Combine files into one document** before
uploading, or by fetching a directory through a fetcher that returns several
paths. Single-file formats just use `paths[0]`.

`redliner/extensions/gerber_parser.py` is a worked skeleton of this case:
it groups layer files, reports what it found, and raises with instructions
rather than guessing. A half-right board render is worse than an honest error,
because a diff of two wrong renders looks like a real difference.

### Sub-pixel accuracy

Magic align corrects CAD drift by re-rendering at a shifted clip window. For
vector sources that is exact at any fraction of a pixel. For a raster source it
is not, so declare it:

```python
supports_subpixel = False
```

Redliner then falls back to whole-pixel shifting. Being honest costs a little
alignment precision; claiming it wrongly produces silent anti-alias fringing on
every edge — see the PDF implementation in `parsers.py` for why.

## Fetchers — pulling from a vault or PLM

Most engineering organisations keep drawings behind authentication, and the
useful workflow is "compare part 4471 rev B against rev C", not "find the file
on disk".

```python
from pathlib import Path
from redliner.plugins.fetchers import FETCHERS, Fetcher, FetchResult

@FETCHERS.register
class WindchillFetcher(Fetcher):
    name = "windchill"
    label = "Windchill"
    icon = "inventory_2"             # a real Material Icons ligature

    def __init__(self, workspace: Path) -> None:
        super().__init__(workspace)
        self._token = None

    def actions(self):               # rendered as buttons
        return {"Sign out": self.sign_out} if self._token \
            else {"Sign in": self.sign_in}

    def status(self):
        return "signed in" if self._token else "not signed in"

    async def search(self, query: str) -> list[FetchResult]:
        return [FetchResult(ref="4471/B", title="ASSY 4471", subtitle="rev B")]

    async def fetch(self, ref: str) -> tuple[str, list[Path]]:
        target = self.workspace / f"{ref.replace('/', '-')}.pdf"
        target.write_bytes(await self._download(ref))
        return ref, [target]
```

Only `fetch` is required. A fetcher with no `search` still works — the user
pastes a document number straight in.

**This differs from the desktop version deliberately.** There, a fetcher opened
its own modal file picker. A server cannot do that, so `search()` returns
results and Redliner renders them. The upside is that a fetcher no longer knows
anything about the UI toolkit.

Notes that matter in practice:

- **One fetcher instance per session.** Any token you store on `self` belongs to
  that user and vanishes when they disconnect.
- **Write only into `self.workspace`.** It is deleted on disconnect, which is
  what keeps controlled documents from lingering on the server.
- **Return several paths** when one document is several files. The parser
  registry takes it from there.
- **Validate paths** if you serve anything from the filesystem. `FolderFetcher`
  rejects refs that escape its root; a fetcher that serves arbitrary server
  paths is a file disclosure hole.

`redliner/extensions/plm_fetcher.py` is a documented stub to copy.

## Tools — new markup

A tool turns a pointer gesture into a `Shape`. The gesture vocabulary is small
because it is implemented in the browser too:

| Gesture | Points delivered | Used by |
| --- | --- | --- |
| `drag` | press and release | line, box, cloud |
| `click` | one point | text |
| `freehand` | every sample along the drag | pencil |

```python
from redliner.core.markup import Shape, register_shape
from redliner.plugins.tools import TOOLS, MarkupTool

@TOOLS.register
class BalloonTool(MarkupTool):
    name = "balloon"
    icon = "trip_origin"
    tooltip = "Balloon callout"
    priority = 55                    # position in the toolbar
    gesture = "drag"

    @classmethod
    def create(cls, points, style):
        if len(points) < 2:
            return None
        return Shape(kind="balloon", points=points[:2],
                     color=style["color"], width=style["width"],
                     fill=style.get("fill"))

    @classmethod
    def handles(cls, shape):
        return ["nw", "ne", "se", "sw"]
```

`style` carries the markup panel's current settings: `color`, `width`, `fill`,
`font`, `font_size`, `boxed`.

To give a new shape kind its own appearance, register renderers for it. Both are
optional — omit them and the built-in geometry handling applies.

```python
def balloon_svg(shape) -> str:
    x0, y0, x1, y1 = shape.bounds()
    return f'<ellipse cx="{(x0+x1)/2}" cy="{(y0+y1)/2}" .../>'

def balloon_pdf(page, shape) -> None:
    page.draw_oval(pymupdf.Rect(*shape.bounds()), color=shape.color)

register_shape("balloon", svg=balloon_svg, pdf=balloon_pdf)
```

Markup is stored in **PDF points**, so a shape drawn over a 100 DPI preview
lands identically in a 600 DPI export. Keep your geometry in points and this is
free.

### Icon names

Icons must be real Material Icons ligatures. An invalid name does not fail
loudly — the font resolves ligatures greedily, so `cloud_outlined` renders the
`cloud` glyph *plus* a literal `_outlined` that paints over the neighbouring
button. `_outlined` / `_round` / `_sharp` are separate font families, never name
suffixes. `tests/test_icons.py` checks every registered plugin's icon for this.

## Testing a plugin

`tests/test_plugins.py` is written the way a plugin author would use the system
— define a class, register it, check the core picks it up. Copy the patterns
there. Registries are plain objects, so a test can register into one and pop the
entry afterwards:

```python
PARSERS.register(MyParser)
try:
    assert parser_for([Path("x.myfmt")]) is MyParser
finally:
    PARSERS._items.pop("myfmt", None)
```
