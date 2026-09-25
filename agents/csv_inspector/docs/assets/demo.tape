# Terminal recording for the README (https://github.com/charmbracelet/vhs).
#
# Records a real inspection of demo_sales.csv with the local Ollama backend.
# Prerequisites: vhs, ttyd and ffmpeg on PATH; the workspace synced
# (`uv sync --all-packages` at the repository root); Ollama running with
# qwen2.5-coder:7b pulled. Run from agents/csv_inspector:
#
#     vhs docs/assets/demo.tape
#
# Regenerate after a change to the demo file, the prompt or the default
# model, together with scripts/render_readme_hero.py.
#
# VHS 0.12 cannot drive ffmpeg 9 and silently writes no GIF, so the tape
# only saves frames; assemble them (also from agents/csv_inspector) with:
#
#     ffmpeg -y -framerate 50 -i .vhs-frames/frame-text-%05d.png -framerate 50 -i .vhs-frames/frame-cursor-%05d.png -filter_complex "[0][1]overlay,fps=10,split[a][b];[a]palettegen=max_colors=64[p];[b][p]paletteuse=dither=none" docs/assets/demo.gif
#
# then delete .vhs-frames/.

Output .vhs-frames/

Set Shell powershell
Set FontSize 16
Set Width 1100
Set Height 640
Set Padding 20
Set Theme "GitHub Dark"
Set TypingSpeed 35ms

Hide
Type "cd docs/assets; $env:Path = (Resolve-Path ../../../../.venv/Scripts).Path + ';' + $env:Path; function prompt { 'PS> ' }; Clear-Host"
Enter
Sleep 1s
Show

Type "Get-Content demo_sales.csv"
Sleep 300ms
Enter
Sleep 3s

Type "python -q"
Enter
Sleep 1s
Type "from csv_inspector import inspect_csv"
Enter
Type 'r = inspect_csv("demo_sales.csv")'
Enter
Wait+Screen@120s /sales\.csv..\s+>>>/
Type "r.encoding, r.delimiter, r.header_row_index, r.footer_rows_to_skip"
Enter
Sleep 2s
Type 'for c in r.columns: print(f"{c.name:<12} {c.inferred_type}")'
Enter
Enter
Sleep 2s
Type "r.footer_lines"
Enter
Sleep 2s
Type "r.usage.model, round(r.usage.latency_seconds, 1)"
Enter
Sleep 5s
