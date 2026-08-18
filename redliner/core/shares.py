"""Temporary shareable links to an exported comparison.

A share is a composed PDF, a token, and an expiry. Someone hands a colleague a
link, the colleague opens the redline, and the link stops working after a
chosen interval.

Expiry is hygiene, not access control. A recipient who opens a share can save
the PDF, and nothing here can take that copy back. Redliner is expected to sit
behind a login, and that is what actually controls who gets in; the TTL just
stops links accumulating forever.

Shares live on disk rather than in memory so that a link already sent by email
still works after the server restarts -- an in-memory share would die on deploy
and look like a broken link rather than an expired one.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

log = logging.getLogger("redliner.shares")

#: Tokens are uuid4 hex: 122 bits of randomness, so a share cannot be found by
#: guessing. Validating the shape also keeps a token from ever being a path
#: traversal, since it is used as a directory name.
TOKEN_PATTERN = re.compile(r"^[0-9a-f]{32}$")

PDF_NAME = "redline.pdf"
MANIFEST_NAME = "manifest.json"

#: Offered in the UI, as (label, seconds).
TTL_CHOICES: list[tuple[str, int]] = [
    ("15 minutes", 15 * 60),
    ("1 hour", 60 * 60),
    ("8 hours", 8 * 60 * 60),
    ("24 hours", 24 * 60 * 60),
    ("7 days", 7 * 24 * 60 * 60),
    ("30 days", 30 * 24 * 60 * 60),
]
DEFAULT_TTL = 24 * 60 * 60
MAX_TTL = 30 * 24 * 60 * 60


@dataclass(frozen=True, slots=True)
class Share:
    token: str
    label: str
    created: float
    expires: float
    size: int
    pages: int

    @property
    def seconds_left(self) -> float:
        return max(0.0, self.expires - time.time())

    def is_expired(self, now: float | None = None) -> bool:
        return (now if now is not None else time.time()) >= self.expires

    def describe_remaining(self) -> str:
        # Rounded before flooring: the microseconds that elapse between creating
        # a share and rendering this string are enough to turn a 24 hour TTL
        # into "23 hours", which reads like the setting was ignored.
        seconds = round(self.seconds_left)
        if seconds <= 0:
            return "expired"
        for unit, size in (("day", 86400), ("hour", 3600), ("minute", 60)):
            if seconds >= size:
                count = seconds // size
                return f"{count} {unit}{'s' if count != 1 else ''}"
        return "under a minute"


def default_root() -> Path:
    """Where shares live. Override with REDLINER_SHARE_DIR.

    Deliberately not named ``redliner-<something>`` matching the session
    directory prefix: the orphaned-session sweeper globs those, and would
    cheerfully delete every live share.
    """
    import os

    configured = os.environ.get("REDLINER_SHARE_DIR")
    if configured:
        return Path(configured)
    return Path(tempfile.gettempdir()) / "redliner_shares"


class ShareStore:
    """Persistent, expiring store of shared PDFs."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else default_root()

    def _dir(self, token: str) -> Path | None:
        if not TOKEN_PATTERN.match(token):
            return None
        return self.root / token

    def create(self, data: bytes, ttl_seconds: float = DEFAULT_TTL,
               label: str = "", pages: int = 0) -> Share:
        """Store `data` and return its share record."""
        ttl = max(60.0, min(float(ttl_seconds), float(MAX_TTL)))
        token = uuid.uuid4().hex
        now = time.time()
        share = Share(token=token, label=label.strip(), created=now,
                      expires=now + ttl, size=len(data), pages=pages)

        target = self.root / token
        target.mkdir(parents=True, exist_ok=True)
        (target / PDF_NAME).write_bytes(data)
        (target / MANIFEST_NAME).write_text(json.dumps(asdict(share), indent=1),
                                            encoding="utf-8")
        return share

    def get(self, token: str) -> tuple[Share, Path] | None:
        """The share and its PDF path, or None if unknown or expired.

        An expired share is deleted here as well as by the sweeper, so a link
        that has run out stops serving immediately even if the sweeper has not
        come round yet.
        """
        folder = self._dir(token)
        if folder is None or not folder.is_dir():
            return None

        share = self._read_manifest(folder)
        if share is None:
            return None
        if share.is_expired():
            self.delete(token)
            return None

        pdf = folder / PDF_NAME
        return (share, pdf) if pdf.is_file() else None

    def delete(self, token: str) -> bool:
        folder = self._dir(token)
        if folder is None or not folder.is_dir():
            return False
        shutil.rmtree(folder, ignore_errors=True)
        return True

    def _read_manifest(self, folder: Path) -> Share | None:
        try:
            data = json.loads((folder / MANIFEST_NAME).read_text(encoding="utf-8"))
            return Share(**data)
        except (OSError, ValueError, TypeError):
            log.warning("unreadable share manifest in %s", folder)
            return None

    def all(self) -> list[Share]:
        """Every live share, soonest to expire first."""
        if not self.root.is_dir():
            return []
        shares = []
        for folder in self.root.iterdir():
            if not folder.is_dir() or not TOKEN_PATTERN.match(folder.name):
                continue
            share = self._read_manifest(folder)
            if share is not None and not share.is_expired():
                shares.append(share)
        return sorted(shares, key=lambda s: s.expires)

    def sweep(self) -> int:
        """Delete expired shares. Returns how many went."""
        if not self.root.is_dir():
            return 0
        removed = 0
        now = time.time()
        for folder in self.root.iterdir():
            if not folder.is_dir():
                continue
            # A directory that is not a valid token, or whose manifest is
            # unreadable, is junk rather than a share; leave it alone rather
            # than deleting something that might not be ours.
            if not TOKEN_PATTERN.match(folder.name):
                continue
            share = self._read_manifest(folder)
            if share is None or share.is_expired(now):
                shutil.rmtree(folder, ignore_errors=True)
                removed += 1
        return removed


_store: ShareStore | None = None


def store() -> ShareStore:
    """The process-wide share store."""
    global _store
    if _store is None:
        _store = ShareStore()
    return _store
