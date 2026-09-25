"""Render the README's "How it works" storyboard from the captured walkthrough.

Reads ``docs/assets/walkthrough.json`` (written by
``scripts/capture_walkthrough.py``) and writes ``walkthrough-light.svg`` and
``walkthrough-dark.svg`` next to it: five panels (sample, prompt, model
answer, grounding, result) with the real data of one inspection. With
``--markdown`` it also prints the storyboard's alt text and the five
``<details>`` blocks for the README's steps; paste them in by hand.

Order: capture first, then render. Rendering needs no model and is
deterministic. Palette, fonts and SVG helpers come from
``render_readme_hero.py``. Standard library only.

Usage:
    python scripts/render_walkthrough.py [--markdown]
"""

from __future__ import annotations

import argparse
import json
from html import escape
from typing import Any

from render_readme_hero import ASSETS, DARK, FONT, LIGHT, SANS, Palette, pill, text, wrap_names

WALKTHROUGH = ASSETS / "walkthrough.json"

WIDTH = 900
MARGIN = 12
GAP = 24
TOP_Y = 12
ROW1_H = 262  # the answer panel's thirteen lines set the first row's height
ROW2_H = 244  # the result panel's eleven lines set the second row's height
ROW2_Y = TOP_Y + ROW1_H + 40
HEIGHT = ROW2_Y + ROW2_H + 36
LH = 16  # line height inside a panel
CHAR_W = 6.9  # width of one 11.5 px monospace character
W1 = (WIDTH - 2 * MARGIN - 2 * GAP) / 3  # first row: three panels
W2 = (WIDTH - 2 * MARGIN - GAP) / 2  # second row: two panels
X1 = [MARGIN + i * (W1 + GAP) for i in range(3)]
X2 = [MARGIN + i * (W2 + GAP) for i in range(2)]

Lines = list[tuple[str, str]]  # (text, css class)


def clip(line: str, width: int) -> str:
    """Return ``line`` without carriage returns, cut to ``width`` drawn characters with ``…``.

    A tab counts as three characters, the width ``text()`` draws it at (`` → ``).
    """
    line = line.replace("\r", "")
    drawn = [3 if char == "\t" else 1 for char in line]
    if sum(drawn) <= width:
        return line
    used = 0
    for end, size in enumerate(drawn):
        if used + size > width - 1:
            return line[:end] + "…"
        used += size
    return line  # unreachable: the sum exceeded width


def _fmt(value: object) -> str:
    """Return ``value`` as JSON, the way the result prints it."""
    return json.dumps(value, ensure_ascii=False)


def _columns(width: float) -> int:
    """Return how many monospace characters fit in a panel ``width`` wide."""
    return int((width - 32) / CHAR_W)


def _frame(
    x: float,
    y: float,
    w: float,
    pal: Palette,
    *,
    h: float,
    number: int,
    title: str,
    subtitle: str,
    highlight: bool = False,
) -> list[str]:
    """Return a panel's box, number badge, title and one-line subtitle."""
    stroke, width = (pal.green, 2) if highlight else (pal.border, 1)
    return [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{pal.panel}" '
        f'stroke="{stroke}" stroke-width="{width}"/>',
        f'<circle cx="{x + 24}" cy="{y + 22}" r="11" class="badge"/>',
        text(x + 24, y + 26, str(number), "num", "middle"),
        text(x + 44, y + 27, title, "title"),
        text(x + 16, y + 48, subtitle, "sub"),
    ]


def _limit(h: float) -> int:
    """Return how many text rows fit below a panel's subtitle when it is ``h`` tall."""
    return int((h - 72 - 8) / LH) + 1


def _lines(x: float, y: float, lines: Lines, width: int, limit: int) -> list[str]:
    """Return ``lines`` as clipped ``<text>`` rows, at most ``limit`` of them."""
    if len(lines) > limit:
        lines = [*lines[: limit - 1], ("⋯", "sub")]
    return [text(x, y + i * LH, clip(t, width), cls) for i, (t, cls) in enumerate(lines)]


def _kv(key: str, value: object, width: int, cls: str) -> Lines:
    """Return ``key: value`` on one line, or on two when it does not fit."""
    one = f"{key}: {_fmt(value)}"
    return [(one, cls)] if len(one) <= width else [(f"{key}:", cls), ("  " + _fmt(value), cls)]


def _sample_panel(data: dict[str, Any], pal: Palette) -> list[str]:
    s, size = data["samples"], data["file"]["size_bytes"]
    x, y, w = X1[0], TOP_Y, W1
    body = _frame(
        x,
        y,
        w,
        pal,
        h=ROW1_H,
        number=1,
        title="Sample",
        subtitle="Two bounded reads, one at each end.",
    )
    bx, by, bw = x + 16, y + 62, w - 32
    head_w, tail_w = bw * s["head_bytes"] / size, bw * s["tail_bytes"] / size
    body.append(f'<rect x="{bx}" y="{by}" width="{bw}" height="12" rx="3" class="unread"/>')
    body.append(f'<rect x="{bx}" y="{by}" width="{head_w}" height="12" rx="3" class="window"/>')
    body.append(text(bx, by + 28, f"head {s['head_bytes']} B", "key"))
    lines: Lines = [(line, "code") for line in s["head_text"].splitlines()[:4]]
    if s["tail_text"] is None:
        lines.append(("no tail: the head covers the file", "sub"))
    else:
        body.append(
            f'<rect x="{bx + bw - tail_w}" y="{by}" width="{tail_w}" height="12" rx="3" '
            f'class="window"/>'
        )
        body.append(text(bx + bw, by + 28, f"tail {s['tail_bytes']} B", "key", "end"))
        if s["unread_bytes"]:
            body.append(
                text(bx + bw / 2, by + 44, f"{s['unread_bytes']} B never read", "sub", "middle")
            )
        lines.append(("⋯", "sub"))
        lines += [(line, "code") for line in s["tail_text"].splitlines()[:2]]
    return body + _lines(x + 16, y + 132, lines, _columns(w), _limit(ROW1_H) - 4)


def _prompt_panel(data: dict[str, Any], pal: Palette) -> list[str]:
    p = data["prompt"]
    x, y, w = X1[1], TOP_Y, W1
    body = _frame(
        x,
        y,
        w,
        pal,
        h=ROW1_H,
        number=2,
        title="Prompt",
        subtitle="One prompt; a JSON Schema fixes the answer.",
    )
    excerpt: Lines = [(line, "code") for line in p["text"].splitlines() if line.strip()][:9]
    body += _lines(x + 16, y + 72, [*excerpt, ("⋯", "sub")], _columns(w), _limit(ROW1_H) - 2)
    label = f"{p['tokens']} tokens · {p['version']} · schema"
    body.append(pill(x + w - 12, y + ROW1_H - 16, label, pal.purple, "tokens"))
    return body


def _answer_panel(data: dict[str, Any], pal: Palette) -> list[str]:
    a = data["answer"]
    changed = {d["field"] for d in data["grounding"]}
    if "footer_lines" in changed:
        changed.add("footer_first_line")
    x, y, w = X1[2], TOP_Y, W1
    body = _frame(
        x,
        y,
        w,
        pal,
        h=ROW1_H,
        number=3,
        title="Model answer",
        subtitle="Raw JSON; red fields get corrected next.",
    )
    lines: Lines = []
    for key, value in a["fields"].items():
        lines += _kv(key, value, _columns(w), "bad" if key in changed else "code")
    return body + _lines(x + 16, y + 72, lines, _columns(w), _limit(ROW1_H))


def _ground_panel(data: dict[str, Any], pal: Palette) -> list[str]:
    x, y, w = X2[0], ROW2_Y, W2
    subtitle = "Positions and text are recomputed from the real bytes."
    body = _frame(
        x, y, w, pal, h=ROW2_H, number=4, title="Ground", subtitle=subtitle, highlight=True
    )
    lines: Lines = []
    if not data["grounding"]:
        lines.append(("nothing to correct: the model's answer matched the bytes", "good"))
    for change in data["grounding"]:
        model, final = change["model"], change["result"]
        model_values = model if isinstance(model, list) else [model]
        final_values = final if isinstance(final, list) else [final]
        lines.append((change["field"], "key"))
        lines += [("  model  " + _fmt(v), "bad") for v in model_values] or [("  model  []", "bad")]
        lines += [
            (("  bytes  " if i == 0 else "         ") + _fmt(v), "good")
            for i, v in enumerate(final_values)
        ]
    return body + _lines(x + 16, y + 72, lines, _columns(w), _limit(ROW2_H))


def _result_panel(data: dict[str, Any], pal: Palette) -> list[str]:
    r = data["result"]
    x, y, w = X2[1], ROW2_Y, W2
    width = _columns(w)
    body = _frame(
        x,
        y,
        w,
        pal,
        h=ROW2_H,
        number=5,
        title="Result",
        subtitle="Validated, then ready for your reader.",
    )
    lines: Lines = [
        (f"encoding {_fmt(r['encoding'])}  delimiter {_fmt(r['delimiter'])}", "code"),
        (
            f"quotechar {_fmt(r['quotechar'])}  escapechar {_fmt(r['escapechar'])}  "
            f"doublequote {_fmt(r['doublequote'])}",
            "code",
        ),
        (f"has_header {_fmt(r['has_header'])}  header_row_index {r['header_row_index']}", "code"),
        (f"footer_rows_to_skip {r['footer_rows_to_skip']}", "code"),
    ]
    names = wrap_names(r["columns"], width - 8)
    lines += [(("columns " if i == 0 else "        ") + n, "code") for i, n in enumerate(names)]
    lines.append((f"confidence {r['confidence']}", "code"))
    lines += [
        ("", "code"),
        ("df = pd.read_csv(path, sep=r.delimiter,", "py"),
        ("    encoding=r.encoding, skiprows=r.header_row_index,", "py"),
        ('    skipfooter=r.footer_rows_to_skip, engine="python")', "py"),
    ]
    return body + _lines(x + 16, y + 72, lines, width, _limit(ROW2_H))


def _arrows() -> list[str]:
    """Return the arrows joining the panels, in reading order."""
    mid = TOP_Y + ROW1_H / 2
    body = [
        f'<path d="M{X1[i] + W1 + 4} {mid}H{X1[i + 1] - 6}" class="arrow"/>'
        f'<path d="M{X1[i + 1] - 11} {mid - 5}l5 5l-5 5" class="arrow"/>'
        for i in range(2)
    ]
    sx, dx = X1[2] + W1 / 2, X2[0] + W2 / 2
    turn = TOP_Y + ROW1_H + 20
    body.append(
        f'<path d="M{sx} {TOP_Y + ROW1_H + 2}V{turn}H{dx}V{ROW2_Y - 4}" class="arrow"/>'
        f'<path d="M{dx - 5} {ROW2_Y - 9}l5 5l5 -5" class="arrow"/>'
    )
    mid2 = ROW2_Y + ROW2_H / 2
    body.append(
        f'<path d="M{X2[0] + W2 + 4} {mid2}H{X2[1] - 6}" class="arrow"/>'
        f'<path d="M{X2[1] - 11} {mid2 - 5}l5 5l-5 5" class="arrow"/>'
    )
    return body


def _caption(data: dict[str, Any]) -> str:
    u, s = data["usage"], data["samples"]
    tokens = (u["prompt_tokens"] or 0) + (u["completion_tokens"] or 0)
    return (
        f"Real run: {u['model']}, local Ollama · {tokens:,} tokens · "
        f"{u['latency_seconds']} s · $0 · windows {s['head_bytes']} + {s['tail_bytes']} B "
        f"for a {data['file']['size_bytes']}-byte file (default 4 KiB each)"
    )


def render(pal: Palette, data: dict[str, Any]) -> str:
    """Return the storyboard SVG in one palette."""
    body = [
        *_arrows(),
        *_sample_panel(data, pal),
        *_prompt_panel(data, pal),
        *_answer_panel(data, pal),
        *_ground_panel(data, pal),
        *_result_panel(data, pal),
        text(WIDTH / 2, HEIGHT - 12, _caption(data), "facts", "middle"),
    ]
    style = f"""
text{{font-family:{FONT};font-size:11.5px;fill:{pal.text};white-space:pre}}
.title{{font-family:{SANS};font-size:14px;font-weight:600}}
.sub{{font-family:{SANS};font-size:11px;fill:{pal.muted}}}
.num{{font-family:{SANS};font-size:11px;font-weight:700;fill:{pal.canvas}}}
.badge{{fill:{pal.blue}}}
.key{{fill:{pal.muted}}}
.bad{{fill:{pal.red}}}
.good{{fill:{pal.green}}}
.py{{fill:{pal.purple}}}
.tab{{fill:{pal.muted}}}
.tag{{font-family:{SANS};font-size:11px;font-weight:600}}
.facts{{font-family:{SANS};font-size:12px;fill:{pal.muted}}}
.window{{fill:{pal.blue};fill-opacity:.45}}
.unread{{fill:{pal.muted};fill-opacity:.18}}
.arrow{{fill:none;stroke:{pal.muted};stroke-width:1.5}}
"""
    title = f"How csv-inspector handled {data['file']['name']}, step by step"
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" '
        f'width="{WIDTH}" height="{HEIGHT}" role="img" aria-labelledby="t">'
        f'<title id="t">{escape(title)}</title><style>{style}</style>'
        f'<rect width="{WIDTH}" height="{HEIGHT}" rx="12" fill="{pal.canvas}"/>'
        + "".join(body)
        + "</svg>\n"
    )


def alt_text(data: dict[str, Any]) -> str:
    """Return the storyboard's alt text: the five steps with their real values."""
    s, r, u = data["samples"], data["result"], data["usage"]
    if s["tail_text"] is None:
        sample = f"the {s['head_bytes']}-byte head covers the whole file"
    else:
        sample = (
            f"{s['head_bytes']} bytes of head and {s['tail_bytes']} bytes of tail are read, "
            f"{s['unread_bytes']} bytes in between are not"
        )
    fixes = "; ".join(
        f"{c['field']} {_fmt(c['model'])} corrected to {_fmt(c['result'])}"
        for c in data["grounding"]
    )
    return (
        f"How csv-inspector handled {data['file']['name']} ({data['file']['size_bytes']} bytes). "
        f"1 Sample: {sample}. "
        f"2 Prompt: {data['prompt']['tokens']} tokens, prompt version {data['prompt']['version']}, "
        f"answer shape fixed by a JSON Schema. "
        f"3 Model answer from {data['answer']['model']}. "
        f"4 Ground: {fixes or 'nothing to correct'}. "
        f"5 Result: encoding {r['encoding']}, delimiter {_fmt(r['delimiter'])}, "
        f"header_row_index {r['header_row_index']}, footer_rows_to_skip "
        f"{r['footer_rows_to_skip']}, columns {', '.join(r['columns'])}. "
        f"{u['model']} on local Ollama, $0."
    )


def _fence(body: str, lang: str = "text") -> str:
    """Return ``body`` in a four-backtick fence (samples may hold backticks)."""
    return f"````{lang}\n{body.replace(chr(13), '').rstrip()}\n````"


def _details(summary: str, body: str) -> str:
    return f"<details>\n<summary>{summary}</summary>\n\n{body}\n\n</details>"


def _cell(value: object) -> str:
    """Return ``value`` as JSON in a table cell, with pipes escaped."""
    return "`" + _fmt(value).replace("|", "\\|") + "`"


def details_blocks(data: dict[str, Any]) -> list[str]:
    """Return the five ``<details>`` blocks, one per README step."""
    s, p, a = data["samples"], data["prompt"], data["answer"]
    counts = (
        f"{s['head_bytes']} bytes of head, {s['tail_bytes']} bytes of tail, "
        f"{s['unread_bytes']} bytes never read, of {data['file']['size_bytes']} "
        f"(encoding {s['encoding']})."
    )
    tail = (
        "No tail: the head covers the file."
        if s["tail_text"] is None
        else f"Tail (it may start mid-line):\n\n{_fence(s['tail_text'])}"
    )
    if data["grounding"]:
        rows = "".join(
            f"| {_cell(c['field']).replace(chr(34), '')} | {_cell(c['model'])} | "
            f"{_cell(c['result'])} |\n"
            for c in data["grounding"]
        )
        grounding = "| Field | Model | Result |\n|---|---|---|\n" + rows.rstrip("\n")
    else:
        grounding = "Nothing: the model's answer matched the bytes."
    return [
        _details(
            "The samples, as decoded", f"{counts}\n\nHead:\n\n{_fence(s['head_text'])}\n\n{tail}"
        ),
        _details(
            f"The exact prompt ({p['tokens']} tokens, prompt version {p['version']})",
            _fence(p["text"]),
        ),
        _details(f"The model's raw answer ({a['model']})", _fence(a["text"], "json")),
        _details("What grounding changed", grounding),
        _details(
            "The final result",
            _fence(json.dumps(data["result"], indent=2, ensure_ascii=False), "json"),
        ),
    ]


def main(argv: list[str] | None = None) -> None:
    """Write both storyboard SVGs; with ``--markdown``, print the README pieces."""
    parser = argparse.ArgumentParser(description="Render the README walkthrough.")
    parser.add_argument(
        "--markdown", action="store_true", help="Print the alt text and the <details> blocks."
    )
    args = parser.parse_args(argv)
    data = json.loads(WALKTHROUGH.read_text(encoding="utf-8"))
    for pal in (LIGHT, DARK):
        path = ASSETS / f"walkthrough-{pal.name}.svg"
        path.write_text(render(pal, data), encoding="utf-8", newline="\n")
        print(path)
    if args.markdown:
        print(f"\n<!-- alt -->\n{escape(alt_text(data))}")
        for number, block in enumerate(details_blocks(data), start=1):
            print(f"\n<!-- step {number} -->\n{block}")


if __name__ == "__main__":
    main()
