"""Temporary shared links."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from redliner.core.shares import (MAX_TTL, TOKEN_PATTERN, Share, ShareStore,
                                  default_root)
from redliner.ui.share_routes import download_name


def make_store(tmp_path: Path) -> ShareStore:
    return ShareStore(tmp_path / "shares")


# -- tokens -------------------------------------------------------------

def test_tokens_are_unguessable_hex(tmp_path) -> None:
    store = make_store(tmp_path)
    tokens = {store.create(b"%PDF-1.4\n").token for _ in range(20)}
    assert len(tokens) == 20, "tokens must not repeat"
    assert all(TOKEN_PATTERN.match(t) for t in tokens)
    assert all(len(t) == 32 for t in tokens)


@pytest.mark.parametrize("token", [
    "../../etc/passwd", "..", "a" * 31, "A" * 32, "", "x/y",
    "0123456789abcdef0123456789abcdeg",     # 'g' is not hex
])
def test_malformed_tokens_are_rejected(tmp_path, token: str) -> None:
    """The token becomes a directory name, so anything but hex is a traversal
    risk rather than merely a miss."""
    assert make_store(tmp_path).get(token) is None


# -- lifecycle ----------------------------------------------------------

def test_a_share_round_trips(tmp_path) -> None:
    store = make_store(tmp_path)
    share = store.create(b"%PDF-1.4\npayload", ttl_seconds=3600,
                         label="ASSY 4471", pages=3)

    found = store.get(share.token)
    assert found is not None
    got, path = found
    assert got.label == "ASSY 4471"
    assert got.pages == 3
    assert path.read_bytes() == b"%PDF-1.4\npayload"


def test_an_expired_share_stops_serving_and_is_removed(tmp_path) -> None:
    store = make_store(tmp_path)
    share = store.create(b"%PDF-1.4\n", ttl_seconds=60)

    # Reach into the manifest to age it, rather than sleeping.
    manifest = store.root / share.token / "manifest.json"
    data = json.loads(manifest.read_text())
    data["expires"] = time.time() - 1
    manifest.write_text(json.dumps(data))

    assert store.get(share.token) is None
    assert not (store.root / share.token).exists(), \
        "an expired share should be deleted on access, not just hidden"


def test_ttl_is_clamped(tmp_path) -> None:
    store = make_store(tmp_path)
    forever = store.create(b"x", ttl_seconds=10 ** 9)
    assert forever.expires - forever.created == pytest.approx(MAX_TTL, abs=1)

    instant = store.create(b"x", ttl_seconds=0)
    assert instant.expires - instant.created >= 60, "a share must outlive its creation"


def test_sweep_removes_only_expired_shares(tmp_path) -> None:
    store = make_store(tmp_path)
    live = store.create(b"x", ttl_seconds=3600)
    dead = store.create(b"x", ttl_seconds=3600)

    manifest = store.root / dead.token / "manifest.json"
    data = json.loads(manifest.read_text())
    data["expires"] = time.time() - 1
    manifest.write_text(json.dumps(data))

    assert store.sweep() == 1
    assert (store.root / live.token).exists()
    assert not (store.root / dead.token).exists()


def test_sweep_ignores_directories_that_are_not_shares(tmp_path) -> None:
    """Junk in the share root might belong to something else; deleting it is
    not this sweeper's business."""
    store = make_store(tmp_path)
    store.root.mkdir(parents=True, exist_ok=True)
    stranger = store.root / "not-a-token"
    stranger.mkdir()
    (stranger / "important.txt").write_text("keep me")

    assert store.sweep() == 0
    assert stranger.exists()


def test_all_lists_live_shares_soonest_first(tmp_path) -> None:
    store = make_store(tmp_path)
    later = store.create(b"x", ttl_seconds=3600, label="later")
    sooner = store.create(b"x", ttl_seconds=120, label="sooner")

    assert [s.label for s in store.all()] == ["sooner", "later"]
    assert {s.token for s in store.all()} == {later.token, sooner.token}


def test_a_share_survives_a_new_store_instance(tmp_path) -> None:
    """Links get emailed; a server restart must not break them."""
    share = make_store(tmp_path).create(b"%PDF-1.4\n", ttl_seconds=3600)
    assert make_store(tmp_path).get(share.token) is not None


def test_unreadable_manifest_is_not_served(tmp_path) -> None:
    store = make_store(tmp_path)
    share = store.create(b"x", ttl_seconds=3600)
    (store.root / share.token / "manifest.json").write_text("{ not json")
    assert store.get(share.token) is None


# -- remaining-time wording ---------------------------------------------

@pytest.mark.parametrize("seconds,expected", [
    (0, "expired"), (30, "under a minute"), (90, "1 minute"),
    (3600 * 2, "2 hours"), (86400 * 3, "3 days"),
])
def test_remaining_time_reads_naturally(seconds: int, expected: str) -> None:
    now = time.time()
    share = Share(token="0" * 32, label="", created=now,
                  expires=now + seconds, size=0, pages=0)
    assert share.describe_remaining() == expected


# -- download filename --------------------------------------------------

@pytest.mark.parametrize("label", [
    'evil"; drop', "line\nbreak", "../../escape", "", "   ",
])
def test_download_names_cannot_inject_a_header(label: str) -> None:
    """The label lands in Content-Disposition, where a quote or newline is a
    header injection rather than a cosmetic problem."""
    name = download_name(label)
    assert '"' not in name
    assert "\n" not in name and "\r" not in name
    assert "/" not in name and "\\" not in name
    assert name.endswith(".pdf")


def test_download_name_keeps_readable_labels() -> None:
    assert download_name("ASSY 4471 rev B") == "ASSY 4471 rev B.pdf"


# -- interaction with the session sweeper -------------------------------

def test_session_sweeper_does_not_eat_shares(tmp_path, monkeypatch) -> None:
    """The orphaned-session sweeper globs `redliner-*` in the temp directory.
    A share directory caught by that glob would delete every live link."""
    import os

    from redliner.ui.app import Session

    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))

    share_root = default_root()
    share_root.mkdir(parents=True, exist_ok=True)
    store = ShareStore(share_root)
    share = store.create(b"%PDF-1.4\n", ttl_seconds=3600)

    # Age the share root well past the session cutoff.
    old = time.time() - 48 * 3600
    os.utime(share_root, (old, old))

    stale_session = tmp_path / "redliner-abandoned"
    stale_session.mkdir()
    os.utime(stale_session, (old, old))

    Session.sweep_orphans(max_age_hours=12.0)

    assert store.get(share.token) is not None, "a live share was swept away"
    assert not stale_session.exists(), "the stale session should still be removed"


def test_default_share_root_avoids_the_session_glob() -> None:
    """Belt and braces: the default name must not match `redliner-*` at all."""
    assert not default_root().name.startswith("redliner-")
