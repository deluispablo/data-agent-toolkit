"""Render the README's animated hero and the demo CSV it depicts.

Writes three files to ``docs/assets/``:

- ``demo_sales.csv``: a small, messy Windows-1252 export (preamble banner,
  ``;`` delimiter, decimal commas, a totals row and an end marker). The
  README's terminal recording (``docs/assets/demo.tape``) inspects it.
- ``hero-light.svg`` and ``hero-dark.svg``: the same CSS-animated picture
  in GitHub's light and dark palettes. The README picks one with
  ``<picture>`` and ``prefers-color-scheme``.

The right-hand panel shows a real result. After a prompt, grounding or
default-model change, rerun the CLI on the demo file and update
``RESULT_ROWS``, ``COLUMNS`` and ``RUN_FACTS`` from its output::

    csv-inspector docs/assets/demo_sales.csv --stats

The animation has a visible resting state: every element is drawn in its
final position, and viewers with ``prefers-reduced-motion`` see only that.
Standard library only.

Usage:
    python scripts/render_readme_hero.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from html import escape
from pathlib import Path

ASSETS = Path(__file__).resolve().parent.parent / "docs" / "assets"

PREAMBLE = ["Informe de ventas - ACME Iberia S.A.", "Generado: 2026-09-24 08:15", ""]
HEADER = "Fecha;Tienda;Producto;Unidades;Importe (€);Devuelto"
DATA = [
    "2026-07-01;Madrid Centro;Café molido 1kg;12;143,40;no",
    "2026-07-01;Sevilla;Té verde;7;38,50;no",
    "2026-07-02;A Coruña;Café en grano;3;41,85;sí",
    "2026-07-02;Málaga;Cacao puro;20;96,00;no",
    "2026-07-03;Bilbao;Café molido 1kg;9;107,55;no",
]
FOOTER = ["TOTAL;;;51;427,30;", "*** Fin del informe ***"]

# Copied from a real run of the CLI on demo_sales.csv (see the module docstring).
RESULT_ROWS = [
    ("encoding", '"Windows-1252"'),
    ("delimiter", '";"'),
    ("has_header", "true"),
    ("header_row_index", "3"),
    ("footer_rows_to_skip", "2"),
]
COLUMNS = [
    ("Fecha", "date"),
    ("Tienda", "string"),
    ("Producto", "string"),
    ("Unidades", "integer"),
    ("Importe (€)", "float"),
    ("Devuelto", "boolean"),
]
RUN_FACTS = "qwen2.5-coder:7b on local Ollama · 7.0 s · 1,672 tokens · $0"

CYCLE = 13.0  # seconds per animation loop
FADE = 0.35  # seconds per fade in or out
OUT = 12.2  # when the animated layer fades out before the loop restarts

WIDTH = 900
HEIGHT = 392
FX, FY, FW, FH = 12, 12, 506, 340  # file panel; the result panel shares FY and FH
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


def write_demo_csv() -> Path:
    """Write the demo export the hero depicts, byte for byte.

    Returns:
        The path written.
    """
    lines = [*PREAMBLE, HEADER, *DATA, *FOOTER]
    path = ASSETS / "demo_sales.csv"
    path.write_bytes(("\r\n".join(lines) + "\r\n").encode("cp1252"))
    return path


@dataclass
class Timeline:
    """Collects one ``@keyframes`` rule per animated element."""

    rules: list[str] = field(default_factory=list)

    def _pct(self, seconds: float) -> str:
        return f"{100 * seconds / CYCLE:.2f}%"

    def fade(self, start: float, end: float = OUT) -> str:
        """Fade an element in at ``start`` and out at ``end``.

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
            The class name to put on the element.
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
        return name


def text(x: float, y: float, content: str, cls: str = "", anchor: str = "start") -> str:
    """Return an SVG ``<text>`` element with escaped content."""
    extra = f' class="{cls}"' if cls else ""
    align = f' text-anchor="{anchor}"' if anchor != "start" else ""
    return f'<text x="{x}" y="{y}"{extra}{align}>{escape(content)}</text>'


def pill(x: float, y: float, label: str, color: str, cls: str) -> str:
    """Return a right-aligned rounded tag ending at ``x``, baseline ``y``."""
    width = 6.6 * len(label) + 14
    return (
        f'<g class="{cls}"><rect x="{x - width}" y="{y - 12}" width="{width}" height="17" '
        f'rx="8.5" fill="{color}" fill-opacity=".14" stroke="{color}" stroke-opacity=".5"/>'
        f'<text x="{x - 7}" y="{y}" class="tag" style="fill:{color}" text-anchor="end">'
        f"{escape(label)}</text></g>"
    )


def file_panel(pal: Palette, t: Timeline) -> list[str]:
    """Return the file panel: the sampled lines, the windows and the tags."""
    body: list[str] = []
    fx, fy, fw, fh, lh = FX, FY, FW, FH, LH
    body.append(
        f'<rect x="{fx}" y="{fy}" width="{fw}" height="{fh}" rx="10" '
        f'fill="{pal.panel}" stroke="{pal.border}"/>'
    )
    body.append(f'<path d="M{fx} {fy + 34}h{fw}" stroke="{pal.border}"/>')
    body.append(text(fx + 16, fy + 22, "demo_sales.csv", "title"))
    body.append(text(fx + fw - 16, fy + 22, "encoding? delimiter? header?", "muted", "end"))

    gx, tx = fx + 14, fx + 42  # gutter and text x
    head_top = fy + 58
    head = [*PREAMBLE, HEADER, DATA[0], DATA[1]]
    head_y = [head_top + i * lh for i in range(len(head))]
    gap_y = head_y[-1] + lh + 6
    tail = [DATA[-1], *FOOTER]
    tail_y = [gap_y + lh + 8 + i * lh for i in range(len(tail))]

    # Head and tail sample windows, drawn under the text.
    win_x, win_w = fx + 6, fw - 12
    h0, h1 = head_y[0] - 15, head_y[-1] + 7
    t0, t1 = tail_y[0] - 15, tail_y[-1] + 7
    head_cls, tail_cls = t.fade(0.6), t.fade(1.9)
    for y0, y1, cls in ((h0, h1, head_cls), (t0, t1, tail_cls)):
        body.append(
            f'<rect class="{cls}" x="{win_x}" y="{y0}" width="{win_w}" height="{y1 - y0}" '
            f'rx="6" fill="{pal.blue}" fill-opacity=".07" stroke="{pal.blue}" '
            f'stroke-opacity=".55" stroke-dasharray="4 3"/>'
        )
    body.append(
        f'<line class="{t.sweep(0.6, 1.7, h0, h1)}" x1="{win_x}" x2="{win_x + win_w}" '
        f'y1="0" y2="0" stroke="{pal.blue}" stroke-width="2"/>'
    )
    body.append(
        f'<line class="{t.sweep(1.9, 2.8, t0, t1)}" x1="{win_x}" x2="{win_x + win_w}" '
        f'y1="0" y2="0" stroke="{pal.blue}" stroke-width="2"/>'
    )

    # Classified rows: preamble, header and footer bands.
    pre_cls, hdr_cls, ftr_cls = t.fade(4.4), t.fade(4.8), t.fade(5.2)
    header_y = head_y[len(PREAMBLE)]
    body.append(
        f'<rect class="{pre_cls}" x="{win_x + 2}" y="{head_y[0] - 14}" width="{win_w - 4}" '
        f'height="{len(PREAMBLE) * lh}" rx="4" fill="{pal.muted}" fill-opacity=".12"/>'
    )
    body.append(
        f'<rect class="{hdr_cls}" x="{win_x + 2}" y="{header_y - 14}" width="{win_w - 4}" '
        f'height="{lh - 2}" rx="4" fill="{pal.green}" fill-opacity=".16"/>'
    )
    body.append(
        f'<rect class="{ftr_cls}" x="{win_x + 2}" y="{tail_y[1] - 14}" width="{win_w - 4}" '
        f'height="{len(FOOTER) * lh - 2}" rx="4" fill="{pal.red}" fill-opacity=".12"/>'
    )

    for i, (line, y) in enumerate(zip(head, head_y, strict=True)):
        body.append(text(gx, y, str(i), "gutter"))
        body.append(text(tx, y, line, "line"))
    body.append(f'<g class="{hdr_cls}">{text(tx, header_y, HEADER, "hdr")}</g>')
    body.append(text(fx + fw / 2, gap_y + 4, "⋯  the middle is never read  ⋯", "gap", "middle"))
    body.append(text(gx, tail_y[0], "⋮", "gutter"))
    for line, y in zip(tail, tail_y, strict=True):
        body.append(text(tx, y, line, "line"))

    # Window labels and classification tags, right-aligned inside the panel.
    right = fx + fw - 12
    body.append(pill(right, head_y[0] + 1, "head sample", pal.blue, head_cls))
    body.append(pill(right, tail_y[0] + 1, "tail sample", pal.blue, tail_cls))
    body.append(pill(right, head_y[1] + 1, "preamble: skip 3 lines", pal.muted, pre_cls))
    body.append(pill(right, head_y[2] + 1, "▼ header_row_index = 3", pal.green, hdr_cls))
    body.append(pill(right, tail_y[2] + 1, "footer: skip 2 lines", pal.red, ftr_cls))
    return body


def result_panel(pal: Palette, t: Timeline) -> list[str]:
    """Return the model step, the result panel and the run facts."""
    body: list[str] = []
    fx, fy, fw, fh = FX, FY, FW, FH

    # Model step between the panels.
    mx = fx + fw + 32
    my = fy + fh / 2
    llm_cls = t.fade(3.0)
    body.append(
        f'<g class="{llm_cls}"><path d="M{mx - 22} {my}h40" stroke="{pal.purple}" '
        f'stroke-width="2" class="flow"/><path d="M{mx + 14} {my - 5}l6 5l-6 5" '
        f'fill="none" stroke="{pal.purple}" stroke-width="2"/>'
        f'<circle cx="{mx - 2}" cy="{my - 34}" r="15" fill="{pal.purple}" fill-opacity=".14" '
        f'stroke="{pal.purple}"/>{text(mx - 2, my - 30, "LLM", "llm", "middle")}</g>'
    )
    body.append(
        f'<g class="{t.fade(4.0)}">{text(mx - 2, my + 26, "grounded", "tiny", "middle")}'
        f"{text(mx - 2, my + 39, 'in the', 'tiny', 'middle')}"
        f"{text(mx - 2, my + 52, 'bytes', 'tiny', 'middle')}</g>"
    )

    # Result panel.
    rx, rw = fx + fw + 64, WIDTH - (fx + fw + 64) - 12
    body.append(
        f'<rect x="{rx}" y="{fy}" width="{rw}" height="{fh}" rx="10" '
        f'fill="{pal.panel}" stroke="{pal.border}"/>'
    )
    body.append(f'<path d="M{rx} {fy + 34}h{rw}" stroke="{pal.border}"/>')
    body.append(text(rx + 16, fy + 22, "CSVInspectionResult", "title"))
    kx, vx = rx + 16, rx + 176
    y = fy + 58
    start = 5.0
    for key, value in RESULT_ROWS:
        body.append(
            f'<g class="{t.fade(start)}">{text(kx, y, key, "key")}{text(vx, y, value, "val")}</g>'
        )
        y += 21
        start += 0.3
    body.append(f'<g class="{t.fade(start)}">{text(kx, y, "columns", "key")}</g>')
    y += 21
    for name, kind in COLUMNS:
        start += 0.2
        body.append(
            f'<g class="{t.fade(start)}">{text(kx + 14, y, name, "col")}'
            f"{text(vx, y, kind, 'type')}</g>"
        )
        y += 19

    body.append(
        f'<g class="{t.fade(start + 0.6)}">'
        f"{text(WIDTH / 2, HEIGHT - 14, RUN_FACTS, 'facts', 'middle')}</g>"
    )
    return body


def render(pal: Palette) -> str:
    """Return the hero SVG in one palette."""
    t = Timeline()
    body = file_panel(pal, t) + result_panel(pal, t)

    style = f"""
text{{font-family:{FONT};font-size:12.5px;fill:{pal.text};white-space:pre}}
.title{{font-family:{SANS};font-size:13px;font-weight:600}}
.muted{{font-family:{SANS};font-size:12px;fill:{pal.muted}}}
.gutter{{fill:{pal.muted};font-size:11px}}
.hdr{{fill:{pal.green};font-weight:600}}
.gap{{font-family:{SANS};font-size:11.5px;fill:{pal.muted};font-style:italic}}
.tag{{font-family:{SANS};font-size:11px;font-weight:600}}
.llm{{font-family:{SANS};font-size:10.5px;font-weight:700;fill:{pal.purple}}}
.tiny{{font-family:{SANS};font-size:10px;fill:{pal.muted}}}
.key{{fill:{pal.muted}}}
.val{{fill:{pal.blue}}}
.col{{fill:{pal.text}}}
.type{{fill:{pal.purple}}}
.facts{{font-family:{SANS};font-size:12px;fill:{pal.muted}}}
.flow{{stroke-dasharray:5 4;animation:flow .8s linear infinite}}
@keyframes flow{{to{{stroke-dashoffset:-9}}}}
{"".join(t.rules)}
@media (prefers-reduced-motion:reduce){{*{{animation:none!important}}}}
"""
    title = "csv-inspector reads the head and tail of a messy CSV and returns its structure"
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" '
        f'width="{WIDTH}" height="{HEIGHT}" role="img" aria-labelledby="t">'
        f'<title id="t">{title}</title><style>{style}</style>'
        f'<rect width="{WIDTH}" height="{HEIGHT}" rx="12" fill="{pal.canvas}"/>'
        + "".join(body)
        + "</svg>\n"
    )


def main() -> None:
    """Write the demo CSV and both hero SVGs."""
    ASSETS.mkdir(parents=True, exist_ok=True)
    print(write_demo_csv())
    for pal in (LIGHT, DARK):
        path = ASSETS / f"hero-{pal.name}.svg"
        path.write_text(render(pal), encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
