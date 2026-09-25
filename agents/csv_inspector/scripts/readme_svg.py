"""What the README's two generated pictures share: palettes, fonts and SVG primitives.

``render_readme_hero.py`` (the animated hero) and ``render_walkthrough.py``
(the "How it works" storyboard) import from here, never from each other.
Standard library only.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path

ASSETS = Path(__file__).resolve().parent.parent / "docs" / "assets"
FONT = "ui-monospace,SFMono-Regular,'SF Mono',Menlo,Consolas,'Liberation Mono',monospace"
SANS = "-apple-system,BlinkMacSystemFont,'Segoe UI','Noto Sans',Helvetica,Arial,sans-serif"


@dataclass(frozen=True)
class Palette:
    """One color theme, taken from GitHub's Primer palette."""

    name: str
    canvas: str
    panel: str
    border: str
    text: str
    muted: str
    blue: str
    green: str
    red: str
    purple: str


LIGHT = Palette(
    "light", "#ffffff", "#f6f8fa", "#d0d7de", "#1f2328", "#656d76",
    "#0969da", "#1a7f37", "#cf222e", "#8250df",
)  # fmt: skip
DARK = Palette(
    "dark", "#0d1117", "#161b22", "#30363d", "#e6edf3", "#8b949e",
    "#4493f8", "#3fb950", "#f85149", "#a371f7",
)  # fmt: skip


def text(x: float, y: float, content: str, cls: str = "", anchor: str = "start") -> str:
    """Return an SVG ``<text>`` element with escaped content; tabs show as arrows."""
    extra = f' class="{cls}"' if cls else ""
    align = f' text-anchor="{anchor}"' if anchor != "start" else ""
    body = escape(content).replace("\t", '<tspan class="tab"> → </tspan>')
    return f'<text x="{x}" y="{y}"{extra}{align}>{body}</text>'


def pill_width(label: str) -> float:
    """Return the drawn width of a tag holding ``label``."""
    return 6.6 * len(label) + 14


def pill(x: float, y: float, label: str, color: str, cls: str) -> str:
    """Return a right-aligned rounded tag ending at ``x``, baseline ``y``."""
    width = pill_width(label)
    return (
        f'<g class="{cls}"><rect x="{x - width}" y="{y - 12}" width="{width}" height="17" '
        f'rx="8.5" fill="{color}" fill-opacity=".14" stroke="{color}" stroke-opacity=".5"/>'
        f'<text x="{x - 7}" y="{y}" class="tag" style="fill:{color}" text-anchor="end">'
        f"{escape(label)}</text></g>"
    )


def wrap_names(names: list[str], width: int) -> list[str]:
    """Return ``names`` as a JSON-style list wrapped to lines of ``width`` characters."""
    lines = ["["]
    for i, name in enumerate(names):
        item = f'"{name}"' + ("," if i < len(names) - 1 else "]")
        if len(lines[-1]) + len(item) + 1 > width:
            lines.append(" ")
        lines[-1] += ("" if lines[-1] in ("[", " ") else " ") + item
    return lines
