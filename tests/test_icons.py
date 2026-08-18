"""Guard against Material Icons names that fail silently.

The icon font resolves ligatures greedily, so an invalid name renders whatever
prefix *is* a valid ligature plus the remaining characters as literal text --
which escapes the icon's fixed-size box and paints over adjacent controls. It
never raises, so a typo here shows up only as a visual glitch.

The variant suffixes below are Material *font families* (Material Icons
Outlined, Round, Sharp, Two Tone), never suffixes on a name. Writing
"cloud_outlined" instead of "cloud_queue" is the exact mistake this catches.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

UI_SOURCE = Path(__file__).resolve().parents[1] / "redliner" / "ui" / "app.py"

VARIANT_SUFFIXES = ("_outlined", "_filled", "_round", "_rounded", "_sharp", "_two_tone")

ICON_PATTERN = re.compile(r'icon="([a-z0-9_]+)"|ui\.icon\("([a-z0-9_]+)"')


def icon_names() -> list[str]:
    source = UI_SOURCE.read_text(encoding="utf-8")
    names = [a or b for a, b in ICON_PATTERN.findall(source)]

    # Tool icons come from the plugin registry, which includes anything a
    # third-party or drop-in extension has registered -- exactly the icons most
    # likely to be typo'd, since their author never sees this file.
    from redliner.plugins.tools import TOOLS
    names += [tool.icon for tool in TOOLS.all()]

    from redliner.plugins.fetchers import FETCHERS
    names += [fetcher.icon for fetcher in FETCHERS.all()]
    return names


def test_icons_were_found_at_all() -> None:
    assert len(icon_names()) > 10, "icon scan found nothing; the pattern is stale"


@pytest.mark.parametrize("name", sorted(set(icon_names())))
def test_icon_name_is_not_a_font_variant(name: str) -> None:
    for suffix in VARIANT_SUFFIXES:
        assert not name.endswith(suffix), (
            f'"{name}" looks like a Material Icons font-variant name. '
            f'Drop "{suffix}" or use the real ligature (e.g. cloud_queue).'
        )
