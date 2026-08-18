"""PLM / document-vault fetcher -- STUB.

The example to copy when wiring Redliner to whatever system holds your
controlled drawings: Windchill, Teamcenter, Agile, Arena, SharePoint, an
internal REST service.

The shape of the problem is always the same:

1. authenticate, and keep the token for the session
2. search by part number or title, returning candidate documents
3. fetch one revision's file(s) into the session workspace

Only step 3 is mandatory. A fetcher with no search still works -- the user
pastes a document number straight in.

Credentials
-----------
`authenticate` below takes no password argument on purpose. Prompting for a
secret is the application's job, and where it comes from is a deployment
decision: an environment variable, a service account, an OIDC redirect, a
per-user token typed into the sign-in dialog. Keep the token on the instance --
one fetcher instance per session -- so signing in never leaks across users.
"""

from __future__ import annotations

import os
from pathlib import Path

from ..plugins.fetchers import FETCHERS, Fetcher, FetchResult


@FETCHERS.register
class PlmFetcher(Fetcher):
    name = "plm"
    label = "PLM vault (stub)"
    icon = "inventory_2"
    priority = 30

    def __init__(self, workspace: Path) -> None:
        super().__init__(workspace)
        self.base_url = os.environ.get("REDLINER_PLM_URL", "")
        self._token: str | None = None

    # -- session ---------------------------------------------------------

    def actions(self) -> dict[str, callable]:
        """Buttons Redliner renders next to this fetcher."""
        if self._token:
            return {"Sign out": self.sign_out}
        return {"Sign in": self.authenticate}

    def status(self) -> str:
        if not self.base_url:
            return "set REDLINER_PLM_URL to enable"
        return "signed in" if self._token else "not signed in"

    def authenticate(self) -> None:
        raise NotImplementedError(
            "Implement PlmFetcher.authenticate: obtain a token from "
            f"{self.base_url or '<REDLINER_PLM_URL>'} and store it on self._token."
        )

    def sign_out(self) -> None:
        self._token = None

    # -- lookup ----------------------------------------------------------

    async def search(self, query: str) -> list[FetchResult]:
        """Return candidates for `query` (a part number, title, drawing number).

        Populate `revisions` so the UI can offer "compare rev B against rev C"
        without a second round trip.
        """
        raise NotImplementedError(
            "Implement PlmFetcher.search: GET your vault's search endpoint and "
            "map each hit to a FetchResult(ref=..., title=..., subtitle=...)."
        )

    async def fetch(self, ref: str) -> tuple[str, list[Path]]:
        """Download `ref` into `self.workspace` and return (name, paths).

        Write files under `self.workspace`; it is deleted when the session ends,
        which is what keeps controlled documents from lingering on the server.

        Return several paths when one document is several files -- a Gerber
        package, a multi-sheet export. The parser registry takes it from there.
        """
        raise NotImplementedError(
            "Implement PlmFetcher.fetch: download the document for `ref` into "
            "self.workspace and return (display_name, [paths])."
        )
