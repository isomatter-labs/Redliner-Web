"""Extension points.

Redliner is meant to be forked and adapted, so the three things a company is
most likely to need to change are pulled behind registries:

* **parsers**  -- how a file becomes comparable pages (PDF today, Gerber, ODB++,
  a raster format, whatever your CAD tool emits)
* **fetchers** -- where documents come from (local upload, a URL, a PLM system
  or document vault that needs authentication)
* **tools**    -- what you can draw on a comparison

Plugins are discovered from two places, and neither requires touching the core:

1. ``redliner/extensions/`` -- drop a module in and it is imported at startup.
   Simplest route when you have forked the repo.
2. Python entry points -- ship a separate package that declares e.g.
   ``[project.entry-points."redliner.parsers"]``. Nothing to fork; the plugin
   installs alongside Redliner with pip.

A plugin that raises on import is logged and skipped. One bad extension must
not stop the server from starting, because the person who can fix the plugin is
usually not the person who needs the server up.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from importlib.metadata import entry_points
from typing import Generic, TypeVar

log = logging.getLogger("redliner.plugins")

T = TypeVar("T")


class Registry(Generic[T]):
    """A named collection of plugin classes of one kind."""

    def __init__(self, kind: str, entry_point_group: str) -> None:
        self.kind = kind
        self.entry_point_group = entry_point_group
        self._items: dict[str, T] = {}
        self._discovered = False

    def register(self, item: T) -> T:
        """Decorator. Registers by the class's ``name`` attribute."""
        name = getattr(item, "name", None)
        if not name:
            raise ValueError(f"{self.kind} plugin {item!r} needs a `name`")
        if name in self._items and self._items[name] is not item:
            log.warning("%s plugin %r is being replaced", self.kind, name)
        self._items[name] = item
        return item

    def get(self, name: str) -> T | None:
        self.discover()
        return self._items.get(name)

    def all(self) -> list[T]:
        """Every registered plugin, ordered by `priority` then registration."""
        self.discover()
        return sorted(self._items.values(),
                      key=lambda item: getattr(item, "priority", 100))

    def names(self) -> list[str]:
        return [getattr(item, "name") for item in self.all()]

    def discover(self) -> None:
        """Import extension modules once, on first use."""
        if self._discovered:
            return
        self._discovered = True  # set first: a failure must not retry forever
        self._load_local_extensions()
        self._load_entry_points()

    def _load_local_extensions(self) -> None:
        try:
            package = importlib.import_module("redliner.extensions")
        except ModuleNotFoundError:
            return
        except Exception:
            # The package's own __init__ can fail too, not just the modules
            # inside it. Letting that propagate would stop the server from
            # starting over a plugin problem.
            log.exception("could not import redliner.extensions")
            return
        for module in pkgutil.iter_modules(package.__path__):
            if module.name.startswith("_"):
                continue
            full = f"redliner.extensions.{module.name}"
            try:
                importlib.import_module(full)
            except Exception:
                log.exception("could not load extension %s", full)

    def _load_entry_points(self) -> None:
        try:
            found = entry_points(group=self.entry_point_group)
        except Exception:
            log.exception("could not read entry points for %s", self.entry_point_group)
            return
        for point in found:
            try:
                self.register(point.load())
            except Exception:
                log.exception("could not load %s plugin %r", self.kind, point.name)


def discover_all() -> None:
    """Force discovery of every registry. Called once at server startup so that
    plugin import errors surface in the log immediately rather than on the first
    upload."""
    from . import fetchers, parsers, tools

    parsers.PARSERS.discover()
    fetchers.FETCHERS.discover()
    tools.TOOLS.discover()
