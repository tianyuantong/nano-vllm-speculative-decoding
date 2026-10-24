"""The README figure (SVG, light and dark): each speculative line's speedup over plain decoding at every concurrency
of one panel, the median as a column and the range over seeds as a whisker, counted as report_v2.py counts them.
No plotting dependency: the SVG is written directly.

  python tools/plot_speedup.py --protocol benchmarks/protocol-v2.json --panel K128 --results DIR [DIR ...] \
      --out docs/assets/speedup-k128
"""
import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_v2 import BASELINE, count_runs, format_range, load_runs  # noqa: E402

THEMES = {
    "light": {"surface": "#fcfcfb", "text": "#0b0b0b", "muted": "#52514e", "grid": "#e1e0d9", "reference": "#898781",
              "series": ["#2a78d6", "#eb6834"]},
    "dark": {"surface": "#1a1a19", "text": "#f0efec", "muted": "#c3c2b7", "grid": "#2c2c2a", "reference": "#898781",
             "series": ["#3987e5", "#d95926"]},
}
LINES = {"dflash_k7": "DFlash draft, k = 7", "model_k3": "Qwen3-0.6B draft, k = 3"}    # drawn in this order
FONT = "font-family='-apple-system, BlinkMacSystemFont, Segoe UI, Helvetica, Arial, sans-serif'"
WIDTH, HEIGHT = 760, 430
LEFT, RIGHT, TOP, BOTTOM = 56, 24, 104, 60
BAR_WIDTH, BAR_PITCH, RADIUS = 22, 32, 4


def speedups(runs: list[tuple[str, dict, dict]], protocol: dict, panel: str) -> dict[str, dict[int, list[float]]]:
    """line -> C -> R for every seed of the panel with valid runs of the line and of k0."""
    _, _, q = count_runs(runs, protocol)
    out = defaultdict(lambda: defaultdict(list))
    for (run_panel, concurrency, _), by_line in sorted(q.items()):
        if run_panel == panel and BASELINE in by_line:
            for line in LINES:
                if line in by_line:
                    out[line][concurrency].append(by_line[line][0] / by_line[BASELINE][0])
    return out


def label(ratios: list[float]) -> str:
    """The report's R without its parenthesis: the range when it spans more than 0.05, otherwise the median."""
    return format_range(ratios).split(" (")[0] + "×"


def text(x: float, y: float, content: str, size: int, fill: str, anchor: str = "start", weight: str = "normal") -> str:
    return (f"<text x='{x:.1f}' y='{y:.1f}' font-size='{size}' fill='{fill}' text-anchor='{anchor}' "
            f"font-weight='{weight}' {FONT}>{content}</text>")


def column(x: float, y_top: float, y_base: float, fill: str) -> str:
    """A column rounded at its data end and square at the baseline."""
    r = min(RADIUS, (y_base - y_top) / 2)
    return (f"<path d='M{x:.1f},{y_base:.1f} V{y_top + r:.1f} Q{x:.1f},{y_top:.1f} {x + r:.1f},{y_top:.1f} "
            f"H{x + BAR_WIDTH - r:.1f} Q{x + BAR_WIDTH:.1f},{y_top:.1f} {x + BAR_WIDTH:.1f},{y_top + r:.1f} "
            f"V{y_base:.1f} Z' fill='{fill}'/>")


def speedup_svg(values: dict[str, dict[int, list[float]]], theme: dict, subtitle: str) -> str:
    concurrencies = sorted({concurrency for by_concurrency in values.values() for concurrency in by_concurrency})
    top_value = max(max(ratios) for by_concurrency in values.values() for ratios in by_concurrency.values())
    y_max = math.ceil((top_value + 0.25) * 2) / 2
    plot_w, plot_h = WIDTH - LEFT - RIGHT, HEIGHT - TOP - BOTTOM
    slot = plot_w / len(concurrencies)

    def y_of(value: float) -> float:
        return TOP + plot_h * (1 - value / y_max)

    title = "Speedup over plain decoding, by concurrency"
    summary = "; ".join(f"{name}: " + ", ".join(f"{label(values[line][c])} at C = {c}" for c in concurrencies)
                        for line, name in LINES.items())
    parts = [f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {WIDTH} {HEIGHT}' width='{WIDTH}' height='{HEIGHT}' "
             f"role='img'><title>{title}</title><desc>{summary}</desc>",
             f"<rect width='{WIDTH}' height='{HEIGHT}' fill='{theme['surface']}' rx='8'/>",
             text(LEFT, 28, title, 18, theme["text"], weight="600"),
             text(LEFT, 49, subtitle, 12, theme["muted"])]

    x = LEFT
    for color, name in zip(theme["series"], LINES.values()):
        parts.append(f"<rect x='{x}' y='68' width='12' height='12' rx='3' fill='{color}'/>")
        parts.append(text(x + 18, 78.5, name, 12, theme["text"]))
        x += 18 + 6.0 * len(name) + 20
    parts.append(f"<line x1='{x}' y1='74' x2='{x + 16}' y2='74' stroke='{theme['reference']}' stroke-width='1.5'/>")
    parts.append(text(x + 22, 78.5, "plain decoding (1×)", 12, theme["text"]))

    tick = 0.0
    while tick <= y_max + 1e-9:
        y = y_of(tick)
        parts.append(f"<line x1='{LEFT}' y1='{y:.1f}' x2='{WIDTH - RIGHT}' y2='{y:.1f}' stroke='{theme['grid']}' stroke-width='1'/>")
        parts.append(text(LEFT - 8, y + 4, f"{tick:g}×", 11, theme["muted"], anchor="end"))
        tick += 0.5
    parts.append(f"<line x1='{LEFT}' y1='{y_of(1):.1f}' x2='{WIDTH - RIGHT}' y2='{y_of(1):.1f}' "
                 f"stroke='{theme['reference']}' stroke-width='1.5'/>")

    for index, concurrency in enumerate(concurrencies):
        center = LEFT + slot * (index + 0.5)
        for position, (line, color) in enumerate(zip(LINES, theme["series"])):
            ratios = values[line].get(concurrency)
            if not ratios:
                continue
            middle = center + (position - (len(LINES) - 1) / 2) * BAR_PITCH
            low, high = min(ratios), max(ratios)
            parts.append(column(middle - BAR_WIDTH / 2, y_of(statistics.median(ratios)), y_of(0), color))
            for y in (y_of(low), y_of(high)):
                parts.append(f"<line x1='{middle - 4:.1f}' y1='{y:.1f}' x2='{middle + 4:.1f}' y2='{y:.1f}' "
                             f"stroke='{theme['muted']}' stroke-width='1.5'/>")
            parts.append(f"<line x1='{middle:.1f}' y1='{y_of(high):.1f}' x2='{middle:.1f}' y2='{y_of(low):.1f}' "
                         f"stroke='{theme['muted']}' stroke-width='1.5'/>")
            parts.append(text(middle, y_of(high) - 8, label(ratios), 11, theme["text"], anchor="middle", weight="600"))
        parts.append(text(center, TOP + plot_h + 20, str(concurrency), 12, theme["muted"], anchor="middle"))
    parts.append(text(LEFT + plot_w / 2, TOP + plot_h + 42, "concurrent requests (C)", 11, theme["muted"], anchor="middle"))
    return "\n".join(parts) + "\n</svg>\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--panel", required=True, help="a key of the protocol's panels")
    parser.add_argument("--results", type=Path, nargs="+", required=True,
                        help="directories of bench-v2 result files; a rerun directory last replaces the runs it repeats")
    parser.add_argument("--out", type=Path, required=True, help="path prefix: writes PREFIX-light.svg and PREFIX-dark.svg")
    args = parser.parse_args(argv)
    protocol = json.loads(args.protocol.read_text())
    values = speedups(load_runs(args.results), protocol, args.panel)
    seeds = max(len(ratios) for by_concurrency in values.values() for ratios in by_concurrency.values())
    gpu = protocol["gpu"]["name"].removeprefix("NVIDIA ")
    subtitle = (f"{protocol['models']['target']} · {protocol['models']['dtype']} · {gpu} · "
                f"{args.panel} panel · median and range over {seeds} seeds")
    for mode, theme in THEMES.items():
        path = args.out.with_name(f"{args.out.name}-{mode}.svg")
        path.write_text(speedup_svg(values, theme, subtitle))
        print(path)


if __name__ == "__main__":
    main()
