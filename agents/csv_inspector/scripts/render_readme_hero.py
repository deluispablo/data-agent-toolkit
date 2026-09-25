"""Render the README's animated hero and the demo files it depicts.

Writes four files to ``docs/assets/``:

- ``demo_sales.csv``: a Windows-1252 export with a preamble banner, ``;``
  delimiter, decimal commas, a totals row and an end marker.
- ``demo_stock.tsv``: a UTF-16 (BOM) tab-separated export with doubled
  quotes inside quoted fields, a totals row and an export stamp after the
  data.
- ``hero-light.svg`` and ``hero-dark.svg``: the same CSS-animated picture
  in GitHub's light and dark palettes, inspecting one file after the other.
  The README picks one with ``<picture>`` and ``prefers-color-scheme``.

The README's terminal recording (``docs/assets/demo.tape``) inspects the
same two files. The result panel shows real results: after a prompt,
grounding or default-model change, rerun the CLI on both files and update
each scene's ``result``, ``columns`` and ``facts`` from its output::

    csv-inspector docs/assets/demo_sales.csv --stats
    csv-inspector docs/assets/demo_stock.tsv --stats

Each scene holds its finished result for several seconds before the next
one starts. Viewers with ``prefers-reduced-motion`` see the first scene,
finished and still. Standard library only.

Usage:
    python scripts/render_readme_hero.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from html import escape
from pathlib import Path

ASSETS = Path(__file__).resolve().parent.parent / "docs" / "assets"


@dataclass(frozen=True)
class Scene:
    """One demo file, how much of it the picture shows, and its real result."""

    filename: str
    codec: str
    lines: list[str]
    header_row_index: int
    footer_count: int
    head_count: int  # lines drawn in the head window; the tail shows 3
    result: list[tuple[str, str]]
    columns: list[str]
    facts: str


SCENES = [
    Scene(
        filename="demo_sales.csv",
        codec="cp1252",
        lines=[
            "Sales report – ACME Europe Ltd.",  # noqa: RUF001 (a Windows-1252-only byte)
            "Generated: 2026-09-24 08:15",
            "",
            "Date;Store;Product;Units;Amount (€);Returned",
            "2026-07-01;München;Ground coffee 1kg;12;143,40;no",
            "2026-07-01;Zürich;Green tea;7;38,50;no",
            "2026-07-02;Malmö;Coffee beans;3;41,85;yes",
            "2026-07-02;Kraków;Pure cocoa;20;96,00;no",
            "2026-07-03;Besançon;Ground coffee 1kg;9;107,55;no",
            "TOTAL;;;51;427,30;",
            "*** End of report ***",
        ],
        header_row_index=3,
        footer_count=2,
        head_count=6,
        result=[
            ("encoding", '"Windows-1252"'),
            ("delimiter", '";"'),
            ("quotechar", '"\\""'),
            ("escapechar", "null"),
            ("doublequote", "true"),
            ("has_header", "true"),
            ("header_row_index", "3"),
            ("footer_rows_to_skip", "2"),
        ],
        columns=["Date", "Store", "Product", "Units", "Amount (€)", "Returned"],
        facts="qwen2.5-coder:7b on local Ollama · 969 tokens · confidence 0.95 · $0",
    ),
    Scene(
        filename="demo_stock.tsv",
        codec="utf-16",
        lines=[
            "SKU\tItem\tQty\tUnit price\tUpdated",
            'A-1001\t"Hex bolt ""M8"""\t1200\t0.12\t2026-08-30',
            "A-1002\tWasher, zinc\t5400\t0.03\t2026-08-30",
            'B-2040\t"Hinge ""heavy duty"""\t75\t4.80\t2026-08-29',
            "C-3100\tCable tie 200 mm\t9800\t0.02\t2026-08-28",
            'D-0007\t"Sealant ""clear"""\t140\t6.25\t2026-08-27',
            "TOTAL\t\t16615\t\t",
            "Exported 2026-08-30 18:00 by WMS",
        ],
        header_row_index=0,
        footer_count=2,
        head_count=4,
        result=[
            ("encoding", '"UTF-16"'),
            ("delimiter", '"\\t"'),
            ("quotechar", '"\\""'),
            ("escapechar", "null"),
            ("doublequote", "true"),
            ("has_header", "true"),
            ("header_row_index", "0"),
            ("footer_rows_to_skip", "2"),
        ],
        columns=["SKU", "Item", "Qty", "Unit price", "Updated"],
        facts="qwen2.5-coder:7b on local Ollama · 985 tokens · confidence 0.95 · $0",
    ),
]

SCENE_SECONDS = 16.0  # each scene, including its hold and fade-out
CYCLE = SCENE_SECONDS * len(SCENES)
FADE = 0.35  # seconds per fade in or out
OUT = 15.2  # when a scene fades out, relative to its start

WIDTH = 900
HEIGHT = 392
FX, FY, FW, FH = 12, 12, 506, 340  # file panel; the result panel shares FY and FH
RX = FX + FW + 64  # result panel
RW = WIDTH - RX - 12
LH = 24  # line height of the file panel
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


def write_demo_files() -> list[Path]:
    """Write each scene's demo file, byte for byte.

    Returns:
        The paths written.
    """
    paths = []
    for scene in SCENES:
        path = ASSETS / scene.filename
        path.write_bytes(("\r\n".join(scene.lines) + "\r\n").encode(scene.codec))
        paths.append(path)
    return paths


@dataclass
class Timeline:
    """Collects one ``@keyframes`` rule per animated element."""

    rules: list[str] = field(default_factory=list)

    def _pct(self, seconds: float) -> str:
        return f"{100 * seconds / CYCLE:.2f}%"

    def fade(self, start: float, end: float) -> str:
        """Fade an element in at ``start`` and out at ``end`` (absolute seconds).

        Returns:
            The class name to put on the element.
        """
        name = f"a{len(self.rules)}"
        p = self._pct
        self.rules.append(
            f"@keyframes {name}{{0%,{p(start)}{{opacity:0}}"
            f"{p(start + FADE)},{p(end)}{{opacity:1}}"
            f"{p(end + FADE)},100%{{opacity:0}}}}"
            f".{name}{{animation:{name} {CYCLE}s linear infinite both}}"
        )
        return name

    def sweep(self, start: float, end: float, y0: float, y1: float) -> str:
        """Move an element down from ``y0`` to ``y1``, visible only meanwhile.

        Returns:
            The class names to put on the element.
        """
        name = f"a{len(self.rules)}"
        p = self._pct
        self.rules.append(
            f"@keyframes {name}{{0%,{p(start)}{{opacity:0;transform:translateY({y0}px)}}"
            f"{p(start + 0.1)}{{opacity:1}}"
            f"{p(end - 0.1)}{{opacity:1}}"
            f"{p(end)},100%{{opacity:0;transform:translateY({y1}px)}}}}"
            f".{name}{{animation:{name} {CYCLE}s ease-in-out infinite both}}"
        )
        return f"{name} sweep"


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


def file_panel(scene: Scene, t0: float, pal: Palette, t: Timeline) -> list[str]:
    """Return one scene's file content, sample windows and tags, starting at ``t0``."""
    body: list[str] = []
    fx, fy, fw, lh = FX, FY, FW, LH
    out = t0 + OUT
    gx, tx = fx + 14, fx + 42  # gutter and text x

    head = scene.lines[: scene.head_count]
    tail = scene.lines[-3:]
    head_y = [fy + 58 + i * lh for i in range(len(head))]
    gap_y = head_y[-1] + lh + 6
    tail_y = [gap_y + lh + 8 + i * lh for i in range(len(tail))]

    content = t.fade(t0 + 0.1, out)
    body.append(f'<g class="{content}">')
    body.append(text(fx + 16, fy + 22, scene.filename, "title"))

    # Head and tail sample windows, drawn under the text.
    win_x, win_w = fx + 6, fw - 12
    h0, h1 = head_y[0] - 15, head_y[-1] + 7
    w0, w1 = tail_y[0] - 15, tail_y[-1] + 7
    head_cls, tail_cls = t.fade(t0 + 0.8, out), t.fade(t0 + 2.1, out)
    for y0, y1, cls in ((h0, h1, head_cls), (w0, w1, tail_cls)):
        body.append(
            f'<rect class="{cls}" x="{win_x}" y="{y0}" width="{win_w}" height="{y1 - y0}" '
            f'rx="6" fill="{pal.blue}" fill-opacity=".07" stroke="{pal.blue}" '
            f'stroke-opacity=".55" stroke-dasharray="4 3"/>'
        )
    for start, y0, y1 in ((t0 + 0.8, h0, h1), (t0 + 2.1, w0, w1)):
        body.append(
            f'<line class="{t.sweep(start, start + 1.0, y0, y1)}" x1="{win_x}" '
            f'x2="{win_x + win_w}" y1="0" y2="0" stroke="{pal.blue}" stroke-width="2"/>'
        )

    # Classified rows: preamble, header and footer bands.
    hri = scene.header_row_index
    pre_cls, hdr_cls, ftr_cls = (t.fade(t0 + s, out) for s in (4.6, 5.0, 5.4))
    if hri:
        body.append(
            f'<rect class="{pre_cls}" x="{win_x + 2}" y="{head_y[0] - 14}" '
            f'width="{win_w - 4}" height="{hri * lh}" rx="4" fill="{pal.muted}" '
            f'fill-opacity=".12"/>'
        )
    body.append(
        f'<rect class="{hdr_cls}" x="{win_x + 2}" y="{head_y[hri] - 14}" width="{win_w - 4}" '
        f'height="{lh - 2}" rx="4" fill="{pal.green}" fill-opacity=".16"/>'
    )
    footer_top = tail_y[len(tail) - scene.footer_count]
    body.append(
        f'<rect class="{ftr_cls}" x="{win_x + 2}" y="{footer_top - 14}" width="{win_w - 4}" '
        f'height="{scene.footer_count * lh - 2}" rx="4" fill="{pal.red}" fill-opacity=".12"/>'
    )

    for i, (line, y) in enumerate(zip(head, head_y, strict=True)):
        body.append(text(gx, y, str(i), "gutter"))
        body.append(text(tx, y, line, "line"))
    body.append(f'<g class="{hdr_cls}">{text(tx, head_y[hri], head[hri], "hdr")}</g>')
    body.append(text(fx + fw / 2, gap_y + 4, "⋯  the middle is never read  ⋯", "gap", "middle"))
    body.append(text(gx, tail_y[0], "⋮", "gutter"))
    for line, y in zip(tail, tail_y, strict=True):
        body.append(text(tx, y, line, "line"))

    # Window labels sit on the gap row, left and right of its caption, so
    # they never cover a line; classification tags are right-aligned.
    right = fx + fw - 12
    label = "▲ head sample"
    body.append(pill(win_x + 6 + pill_width(label), gap_y + 4, label, pal.blue, head_cls))
    body.append(pill(right, gap_y + 4, "▼ tail sample", pal.blue, tail_cls))
    if hri:
        label = f"preamble: skip {hri} line{'s' if hri > 1 else ''}"
        body.append(pill(right, head_y[min(1, hri - 1)] + 1, label, pal.muted, pre_cls))
        hdr_line, label = hri - 1, f"▼ header_row_index = {hri}"
    else:
        hdr_line, label = 0, "header_row_index = 0"
    body.append(pill(right, head_y[hdr_line] + 1, label, pal.green, hdr_cls))
    label = f"footer: skip {scene.footer_count} lines"
    body.append(pill(right, tail_y[-1] + 1, label, pal.red, ftr_cls))
    body.append("</g>")
    return body


def result_panel(scene: Scene, t0: float, t: Timeline) -> list[str]:
    """Return one scene's model step, result rows and run facts, starting at ``t0``."""
    body: list[str] = []
    out = t0 + OUT
    mx, my = FX + FW + 32, FY + FH / 2

    body.append(
        f'<g class="{t.fade(t0 + 3.2, out)}"><path d="M{mx - 22} {my}h40" '
        f'class="flow arrow"/><path d="M{mx + 14} {my - 5}l6 5l-6 5" class="arrow"/>'
        f'<circle cx="{mx - 2}" cy="{my - 34}" r="15" class="bubble"/>'
        f"{text(mx - 2, my - 30, 'LLM', 'llm', 'middle')}</g>"
    )
    body.append(
        f'<g class="{t.fade(t0 + 4.2, out)}">{text(mx - 2, my + 26, "grounded", "tiny", "middle")}'
        f"{text(mx - 2, my + 39, 'in the', 'tiny', 'middle')}"
        f"{text(mx - 2, my + 52, 'bytes', 'tiny', 'middle')}</g>"
    )

    kx, vx = RX + 16, RX + 170
    y = FY + 58.0
    start = t0 + 5.2
    for key, value in scene.result:
        body.append(
            f'<g class="{t.fade(start, out)}">{text(kx, y, key, "key")}'
            f"{text(vx, y, value, 'val')}</g>"
        )
        y += 20
        start += 0.25
    names = wrap_names(scene.columns, 38)
    body.append(f'<g class="{t.fade(start, out)}">{text(kx, y, "columns", "key")}')
    for line in names:
        y += 19
        body.append(text(kx + 14, y, line, "col"))
    body.append("</g>")
    body.append(
        f'<g class="{t.fade(start + 0.6, out)}">'
        f"{text(WIDTH / 2, HEIGHT - 14, scene.facts, 'facts', 'middle')}</g>"
    )
    return body


def frame(pal: Palette) -> list[str]:
    """Return the static panels shared by every scene."""
    body: list[str] = []
    for x, w in ((FX, FW), (RX, RW)):
        body.append(
            f'<rect x="{x}" y="{FY}" width="{w}" height="{FH}" rx="10" '
            f'fill="{pal.panel}" stroke="{pal.border}"/>'
        )
        body.append(f'<path d="M{x} {FY + 34}h{w}" stroke="{pal.border}"/>')
    body.append(text(FX + FW - 16, FY + 22, "encoding? delimiter? header?", "muted", "end"))
    body.append(text(RX + 16, FY + 22, "CSVInspectionResult", "title"))
    return body


def render(pal: Palette) -> str:
    """Return the hero SVG in one palette."""
    t = Timeline()
    body = frame(pal)
    for i, scene in enumerate(SCENES):
        t0 = i * SCENE_SECONDS
        body.append(f'<g class="s{i}">')
        body += file_panel(scene, t0, pal, t) + result_panel(scene, t0, t)
        body.append("</g>")

    # Reduced motion: the first scene, finished and still.
    hidden = ",".join([".sweep", *(f".s{i}" for i in range(1, len(SCENES)))])
    still = "@media (prefers-reduced-motion:reduce){*{animation:none!important}"
    still += f"{hidden}{{display:none}}}}"
    style = f"""
text{{font-family:{FONT};font-size:12.5px;fill:{pal.text};white-space:pre}}
.title{{font-family:{SANS};font-size:13px;font-weight:600}}
.muted{{font-family:{SANS};font-size:12px;fill:{pal.muted}}}
.gutter{{fill:{pal.muted};font-size:11px}}
.tab{{fill:{pal.muted}}}
.hdr{{fill:{pal.green};font-weight:600}}
.gap{{font-family:{SANS};font-size:11.5px;fill:{pal.muted};font-style:italic}}
.tag{{font-family:{SANS};font-size:11px;font-weight:600}}
.llm{{font-family:{SANS};font-size:10.5px;font-weight:700;fill:{pal.purple}}}
.tiny{{font-family:{SANS};font-size:10px;fill:{pal.muted}}}
.key{{fill:{pal.muted}}}
.val{{fill:{pal.blue}}}
.col{{fill:{pal.purple}}}
.facts{{font-family:{SANS};font-size:12px;fill:{pal.muted}}}
.arrow{{fill:none;stroke:{pal.purple};stroke-width:2}}
.bubble{{fill:{pal.purple};fill-opacity:.14;stroke:{pal.purple}}}
.flow{{stroke-dasharray:5 4;animation:flow .8s linear infinite}}
@keyframes flow{{to{{stroke-dashoffset:-9}}}}
{"".join(t.rules)}
{still}
"""
    title = "csv-inspector reads the head and tail of messy CSV files and returns their structure"
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" '
        f'width="{WIDTH}" height="{HEIGHT}" role="img" aria-labelledby="t">'
        f'<title id="t">{title}</title><style>{style}</style>'
        f'<rect width="{WIDTH}" height="{HEIGHT}" rx="12" fill="{pal.canvas}"/>'
        + "".join(body)
        + "</svg>\n"
    )


def main() -> None:
    """Write the demo files and both hero SVGs."""
    ASSETS.mkdir(parents=True, exist_ok=True)
    for path in write_demo_files():
        print(path)
    for pal in (LIGHT, DARK):
        path = ASSETS / f"hero-{pal.name}.svg"
        path.write_text(render(pal), encoding="utf-8", newline="\n")
        print(path)


if __name__ == "__main__":
    main()
