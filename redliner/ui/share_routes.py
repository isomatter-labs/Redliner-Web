"""HTTP routes for shared comparisons.

Separate from the page so the share URL is a plain HTTP endpoint with no
NiceGUI session behind it: a recipient opening a link should get a PDF, not a
websocket and a server-side session.
"""

from __future__ import annotations

import re

from fastapi import Response
from fastapi.responses import FileResponse, HTMLResponse
from nicegui import app

from ..core import shares

SHARE_PREFIX = "/s"

#: Filenames are echoed into a Content-Disposition header, so anything outside
#: this set is dropped rather than escaped -- a quote or newline there is a
#: header injection.
SAFE_FILENAME = re.compile(r"[^A-Za-z0-9._ -]")

EXPIRED_PAGE = """
<!doctype html>
<title>Link expired</title>
<style>
  body { font-family: system-ui, sans-serif; background: #14171a; color: #e8eaed;
         display: grid; place-items: center; height: 100vh; margin: 0; }
  .card { text-align: center; max-width: 32rem; padding: 2rem; }
  h1 { font-weight: 600; font-size: 1.4rem; margin: 0 0 .5rem; }
  p { opacity: .7; line-height: 1.5; margin: 0; }
</style>
<div class="card">
  <h1>This link has expired</h1>
  <p>Shared comparisons are temporary. Ask whoever sent it to share it again.</p>
</div>
"""


def download_name(label: str) -> str:
    cleaned = SAFE_FILENAME.sub("", label).strip() or "redline"
    return f"{cleaned[:80]}.pdf"


def register() -> None:
    """Attach the share routes to the running app."""

    @app.get(f"{SHARE_PREFIX}/{{token}}")
    def serve_share(token: str) -> Response:
        found = shares.store().get(token)
        if found is None:
            # Same response whether the token never existed or has expired, so
            # the endpoint cannot be used to probe which tokens are real.
            return HTMLResponse(EXPIRED_PAGE, status_code=404)

        share, path = found
        return FileResponse(
            path,
            media_type="application/pdf",
            headers={
                "Content-Disposition":
                    f'inline; filename="{download_name(share.label)}"',
                # Expiring content should not be held by a shared cache.
                "Cache-Control": "private, no-store",
                "X-Content-Type-Options": "nosniff",
            },
        )
