"""Where documents come from.

Local upload is one source among several. Most engineering organisations keep
drawings in a PLM system or document vault behind authentication, and the
useful workflow is "search for part 4471, compare rev B against rev C" rather
than "find the file on disk and drag it in".

A fetcher provides two things: a way to *search* for documents, and a way to
*fetch* one into local temporary storage. It may also expose named actions --
sign in, switch site, clear a token -- which the UI renders as buttons.

This differs from the desktop version deliberately. There, a fetcher opened its
own modal file picker. A browser cannot do that from the server side, so
``search()`` returns results and Redliner renders them. The upside is that a
fetcher no longer has to know anything about the UI toolkit.
"""

from __future__ import annotations

import logging
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from . import Registry

log = logging.getLogger("redliner.fetchers")

FETCHERS: Registry["type[Fetcher]"] = Registry("fetcher", "redliner.fetchers")


@dataclass(slots=True)
class FetchResult:
    """One hit from a search, ready to be shown and then fetched."""

    #: Opaque to Redliner; only the fetcher that produced it has to parse it.
    ref: str
    title: str
    subtitle: str = ""
    #: Extra refs offered alongside, e.g. other revisions of the same drawing.
    revisions: list[str] = field(default_factory=list)


class Fetcher(ABC):
    """A source of documents."""

    name: str = ""
    label: str = ""
    icon: str = "folder"
    priority: int = 100
    #: False hides the fetcher's search box (local upload has nothing to search).
    searchable: bool = True

    def __init__(self, workspace: Path) -> None:
        #: Per-session scratch directory. Anything written here is deleted when
        #: the user disconnects, so fetched documents never outlive the session.
        self.workspace = workspace

    def actions(self) -> dict[str, callable]:
        """Named buttons, e.g. ``{"Sign in": self.authenticate}``."""
        return {}

    def status(self) -> str:
        """Short line shown under the fetcher, e.g. "signed in as ...."."""
        return ""

    async def search(self, query: str) -> list[FetchResult]:
        return []

    @abstractmethod
    async def fetch(self, ref: str) -> tuple[str, list[Path]]:
        """Retrieve `ref`, returning a display name and the local file paths.

        Returning a *list* of paths is what lets one logical document be many
        files -- a folder of Gerbers, a multi-sheet export. The parser registry
        decides what to make of them.
        """


@FETCHERS.register
class UploadFetcher(Fetcher):
    """Files the user drags in from their own machine.

    Fetching is a copy: the browser has already written the upload into the
    session workspace, so `ref` is simply a path inside it.
    """

    name = "upload"
    label = "Upload"
    icon = "upload_file"
    priority = 0
    searchable = False

    async def fetch(self, ref: str) -> tuple[str, list[Path]]:
        path = Path(ref)
        if not path.is_absolute():
            path = self.workspace / path
        if not path.exists():
            raise FileNotFoundError(f"{path} is not in this session's workspace")
        return path.stem, [path]


@FETCHERS.register
class UrlFetcher(Fetcher):
    """Documents fetched over HTTP(S).

    Deliberately unauthenticated. Anything that needs credentials should be its
    own fetcher so the auth lives with the system it belongs to, rather than
    being bolted onto a generic downloader.
    """

    name = "url"
    label = "URL"
    icon = "link"
    priority = 10
    searchable = False

    async def fetch(self, ref: str) -> tuple[str, list[Path]]:
        import httpx

        url = ref.strip()
        if not url.lower().startswith(("http://", "https://")):
            raise ValueError("Only http:// and https:// URLs are supported")

        name = Path(url.split("?")[0]).name or "download"
        target = self.workspace / name
        async with httpx.AsyncClient(follow_redirects=True, timeout=60.0) as client:
            async with client.stream("GET", url) as response:
                response.raise_for_status()
                with target.open("wb") as handle:
                    async for chunk in response.aiter_bytes():
                        handle.write(chunk)
        return target.stem, [target]


@FETCHERS.register
class FolderFetcher(Fetcher):
    """A server-side directory, for documents that live next to the server.

    Handy for a shared drive mounted on the Redliner host. Searching walks the
    root; fetching a directory returns every file inside it, which is how a
    folder of Gerbers arrives as one document.
    """

    name = "folder"
    label = "Server folder"
    icon = "folder_open"
    priority = 20

    #: Set to a Path to enable. Left None so the default deployment exposes
    #: nothing: a fetcher that serves arbitrary server paths is a file
    #: disclosure hole unless someone has deliberately chosen the root.
    root: Path | None = None

    def status(self) -> str:
        return f"root: {self.root}" if self.root else "no root configured"

    async def search(self, query: str) -> list[FetchResult]:
        if not self.root:
            return []
        needle = query.lower().strip()
        results: list[FetchResult] = []
        for path in sorted(self.root.rglob("*")):
            if len(results) >= 50:
                break
            if needle and needle not in path.name.lower():
                continue
            relative = path.relative_to(self.root)
            if path.is_dir():
                results.append(FetchResult(ref=str(relative), title=path.name,
                                           subtitle="folder"))
            elif path.is_file():
                results.append(FetchResult(ref=str(relative), title=path.name,
                                           subtitle=str(relative.parent)))
        return results

    async def fetch(self, ref: str) -> tuple[str, list[Path]]:
        if not self.root:
            raise RuntimeError("FolderFetcher.root is not configured")

        source = (self.root / ref).resolve()
        # Refuse anything that escapes the root, however it was spelled.
        if not source.is_relative_to(self.root.resolve()):
            raise ValueError("path escapes the configured root")
        if not source.exists():
            raise FileNotFoundError(str(source))

        if source.is_dir():
            copied = []
            for item in sorted(source.iterdir()):
                if item.is_file():
                    target = self.workspace / item.name
                    shutil.copy2(item, target)
                    copied.append(target)
            if not copied:
                raise ValueError(f"{source.name} contains no files")
            return source.name, copied

        target = self.workspace / source.name
        shutil.copy2(source, target)
        return source.stem, [target]
