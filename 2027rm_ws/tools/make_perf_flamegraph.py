#!/usr/bin/env python3
import argparse
import html
import os
import re
import subprocess
import sys
from collections import defaultdict


HEADER_RE = re.compile(r":\s+(\d+)\s+[^:]+:\s*$")


def simplify_symbol(line: str) -> str:
    line = line.strip()
    if not line:
        return ""
    parts = line.split(None, 1)
    if len(parts) == 2 and re.fullmatch(r"[0-9a-fA-F]+", parts[0]):
        line = parts[1]
    if " (" in line:
        line = line.split(" (", 1)[0]
    if "+0x" in line:
        line = line.split("+0x", 1)[0]
    line = line.strip()
    if not line or line == "[unknown]":
        return ""
    return line


def folded_from_perf_script(perf_text: str) -> dict:
    folded = defaultdict(int)
    weight = 1
    stack = []

    def flush():
        nonlocal weight, stack
        if stack:
            folded[";".join(reversed(stack))] += weight
        weight = 1
        stack = []

    for raw in perf_text.splitlines():
        line = raw.rstrip()
        if not line:
            flush()
            continue
        if not raw.startswith("\t") and not raw.startswith(" "):
            flush()
            match = HEADER_RE.search(line)
            weight = int(match.group(1)) if match else 1
            continue
        symbol = simplify_symbol(line)
        if symbol:
            stack.append(symbol)
    flush()
    return folded


class Node:
    __slots__ = ("name", "value", "children")

    def __init__(self, name: str):
        self.name = name
        self.value = 0
        self.children = {}


def build_tree(folded: dict) -> Node:
    root = Node("all")
    for stack, value in folded.items():
        root.value += value
        node = root
        for name in stack.split(";"):
            child = node.children.get(name)
            if child is None:
                child = Node(name)
                node.children[name] = child
            child.value += value
            node = child
    return root


def color_for(name: str) -> str:
    h = 0
    for ch in name:
        h = (h * 131 + ord(ch)) & 0xFFFFFFFF
    r = 200 + (h % 55)
    g = 80 + ((h >> 8) % 100)
    b = 40 + ((h >> 16) % 60)
    if "Yolo" in name or "preprocess" in name:
        g = min(190, g + 45)
    if "openvino" in name or "[JIT]" in name:
        b = min(180, b + 80)
    if "MediaProcess" in name or "Mv" in name:
        r = min(255, r + 25)
    return f"rgb({r},{g},{b})"


def layout(node: Node, x: float, y: int, scale: float, frames: list, depth: int):
    width = node.value * scale
    frames.append((x, y, width, node))
    child_x = x
    for child in sorted(node.children.values(), key=lambda n: n.value, reverse=True):
        layout(child, child_x, y + 1, scale, frames, depth + 1)
        child_x += child.value * scale


def emit_svg(root: Node, out_path: str, title: str, width_px: int):
    frame_h = 18
    top_pad = 42
    bottom_pad = 24
    min_width = 0.5
    frames = []
    scale = (width_px - 20) / max(1, root.value)
    layout(root, 10, 0, scale, frames, 0)
    max_y = max((y for _, y, _, _ in frames), default=0)
    height = top_pad + (max_y + 1) * frame_h + bottom_pad

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"""<?xml version="1.0" standalone="no"?>
<svg version="1.1" width="{width_px}" height="{height}" onload="init(evt)" viewBox="0 0 {width_px} {height}" xmlns="http://www.w3.org/2000/svg">
<style>
text {{ font-family: monospace; font-size: 12px; }}
.frame:hover {{ stroke: #111; stroke-width: 1; }}
</style>
<script type="text/ecmascript"><![CDATA[
function init(evt) {{}}
function s(info) {{ document.getElementById("details").firstChild.nodeValue = info; }}
function c() {{ document.getElementById("details").firstChild.nodeValue = " "; }}
]]></script>
<text x="10" y="22" font-size="17">{html.escape(title)}</text>
<text id="details" x="10" y="{height - 6}"> </text>
""")
        for x, y, w, node in frames:
            if w < min_width:
                continue
            draw_y = top_pad + (max_y - y) * frame_h
            name = html.escape(node.name)
            pct = 100.0 * node.value / max(1, root.value)
            info = html.escape(f"{node.name} ({pct:.2f}%, {node.value} samples)")
            f.write(f'<g class="frame" onmouseover="s(\'{info}\')" onmouseout="c()">\n')
            f.write(f'<title>{info}</title>\n')
            f.write(f'<rect x="{x:.3f}" y="{draw_y}" width="{w:.3f}" height="{frame_h - 1}" fill="{color_for(node.name)}"/>\n')
            if w > 35:
                max_chars = max(1, int(w / 7))
                label = name if len(name) <= max_chars else name[: max(1, max_chars - 2)] + ".."
                f.write(f'<text x="{x + 3:.3f}" y="{draw_y + 13}">{label}</text>\n')
            f.write("</g>\n")
        f.write("</svg>\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a simple flamegraph SVG from perf.data without external FlameGraph scripts.")
    parser.add_argument("-i", "--input", default="2027rm_ws/perf/perf.data", help="perf.data path")
    parser.add_argument("-o", "--output", default="2027rm_ws/perf/flamegraph.svg", help="output SVG path")
    parser.add_argument("--width", type=int, default=1400, help="SVG width")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"perf data not found: {args.input}", file=sys.stderr)
        return 1

    perf_text = subprocess.check_output(["perf", "script", "-i", args.input], text=True, errors="replace")
    folded = folded_from_perf_script(perf_text)
    if not folded:
        print("no stacks found in perf data", file=sys.stderr)
        return 1

    root = build_tree(folded)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    emit_svg(root, args.output, f"perf flamegraph: {os.path.basename(args.input)}", args.width)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
