"""Render the README figures (SVG, light and dark) from a directory of bench_spec.py results.

Figure 1: recommended-sampling throughput relative to ordinary decoding, by batch size.
Figure 2: where a speculative round's time goes (rec, k = 3) next to an ordinary decode step.
No plotting dependency: the SVG is written directly so the figures regenerate anywhere.
"""
import argparse
import json
import re
from pathlib import Path

from summarize_bench import summarize

THEMES = {
    "light": {"surface": "#fcfcfb", "text": "#0b0b0b", "muted": "#52514e", "grid": "#e1e0d9",
              "series": ["#2a78d6", "#eb6834", "#1baf7a"], "neutral": "#9a9890"},
    "dark": {"surface": "#1a1a19", "text": "#ffffff", "muted": "#c3c2b7", "grid": "#2c2c2a",
             "series": ["#3987e5", "#d95926", "#199e70"], "neutral": "#6f6e69"},
}
LINES = [("rec", 3, "Qwen3 recommended sampling, k=3"), ("flat", 3, "T=1.0, no truncation, k=3"), ("greedy", 4, "greedy, k=4")]
BATCH_SIZES = [1, 2, 4, 8, 16]
FONT = "font-family='-apple-system, BlinkMacSystemFont, Segoe UI, Helvetica, Arial, sans-serif'"
NAME = re.compile(r"(?P<line>[a-z]+)-B(?P<batch>\d+)-k(?P<k>\d+)\.json")


def load(results_dir: Path) -> dict:
    results = {}
    for path in results_dir.glob("*.json"):
        match = NAME.fullmatch(path.name)
        if match:
            results[(match["line"], int(match["batch"]), int(match["k"]))] = summarize(json.loads(path.read_text()))
    return results


def speedups(results: dict) -> dict:
    return {line: [results[(line, b, k)]["throughput_tok_s"] / results[(line, b, 0)]["throughput_tok_s"] for b in BATCH_SIZES]
            for line, k, _ in LINES}


def text(x, y, content, size=13, fill="#000", anchor="start", weight="normal", extra=""):
    return (f"<text x='{x:.1f}' y='{y:.1f}' font-size='{size}' fill='{fill}' text-anchor='{anchor}' "
            f"font-weight='{weight}' {FONT} {extra}>{content}</text>")


def speedup_svg(values: dict, theme: dict) -> str:
    width, height = 760, 410
    left, right, top, bottom = 64, 24, 110, 56
    plot_w, plot_h = width - left - right, height - top - bottom
    y_max = 1.7
    group_w = plot_w / len(BATCH_SIZES)
    bar_w = group_w * 0.48
    color = theme["series"][0]

    def y_of(value):
        return top + plot_h * (1 - value / y_max)

    parts = [f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {width} {height}' width='{width}' height='{height}'>",
             f"<rect width='{width}' height='{height}' fill='{theme['surface']}' rx='8'/>",
             text(left, 28, "More tokens per second, from batch 1 to batch 16", 18, theme["text"], weight="600"),
             text(left, 49, "Qwen3-8B + Qwen3-0.6B · BF16 · RTX PRO 6000 Blackwell", 12, theme["muted"]),
             text(left, 67, "Qwen3 recommended sampling · k=3 · 48 prompts × up to 512 output tokens", 12, theme["muted"]),
             f"<rect x='{left}' y='83' width='10' height='10' rx='2' fill='{color}'/>",
             text(left + 16, 92, "Speculative decoding", 12, theme["text"]),
             f"<line x1='254' y1='88' x2='276' y2='88' stroke='{theme['text']}' stroke-width='1.5' stroke-dasharray='6 4'/>",
             text(284, 92, "Ordinary decoding = 1.0×", 12, theme["text"])]
    for tick in [0.0, 0.5, 1.0, 1.5]:
        y = y_of(tick)
        parts.append(f"<line x1='{left}' y1='{y:.1f}' x2='{left + plot_w}' y2='{y:.1f}' stroke='{theme['grid']}' stroke-width='1'/>")
        parts.append(text(left - 10, y + 4, f"{tick:.1f}×", 12, theme["muted"], anchor="end"))
    for i, (batch, value) in enumerate(zip(BATCH_SIZES, values["rec"])):
        center = left + group_w * (i + 0.5)
        h = plot_h * value / y_max
        parts.append(f"<rect x='{center - bar_w / 2:.1f}' y='{y_of(value):.1f}' width='{bar_w:.1f}' height='{h:.1f}' fill='{color}' rx='4'/>")
        parts.append(text(center, y_of(value) - 10, f"{value:.2f}×", 18, theme["text"], anchor="middle", weight="600"))
        parts.append(text(center, top + plot_h + 23, f"B = {batch}", 13, theme["muted"], anchor="middle"))
    parts.append(text(left + plot_w / 2, height - 12, "Concurrent requests (max_num_seqs)", 12, theme["muted"], anchor="middle"))
    y1 = y_of(1.0)
    parts.append(f"<line x1='{left}' y1='{y1:.1f}' x2='{left + plot_w}' y2='{y1:.1f}' stroke='{theme['text']}' stroke-width='1.5' stroke-dasharray='6 4'/>")
    parts.append("</svg>")
    return "\n".join(parts)


def round_cost_svg(results: dict, theme: dict) -> str:
    width, height = 760, 450
    left, right, top, bottom = 64, 40, 84, 96
    plot_w, plot_h = width - left - right, height - top - bottom
    y_max = 36.0
    group_w = plot_w / len(BATCH_SIZES)
    bar_w = group_w * 0.28
    phases = [("verify", "target verification (k+1 tokens)"), ("draft", "draft: k steps + catch-up"), ("other", "sampling + host copy")]
    colors = dict(zip(["verify", "draft", "other"], theme["series"]))

    def y_of(value):
        return top + plot_h * (1 - value / y_max)

    parts = [f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {width} {height}' width='{width}' height='{height}'>",
             f"<rect width='{width}' height='{height}' fill='{theme['surface']}' rx='8'/>",
             text(left, 26, "Time per speculative round vs one ordinary decode step", 16, theme["text"], weight="600"),
             text(left, 44, "Stack height = sum of phase medians (CUDA events); whole-round p50 is listed in the table.", 11.5, theme["muted"]),
             text(left, 60, "Qwen3 recommended sampling · k=3 · ≈2.4 tokens per request per round", 11.5, theme["muted"])]
    for tick in range(0, 37, 6):
        y = y_of(tick)
        parts.append(f"<line x1='{left}' y1='{y:.1f}' x2='{left + plot_w}' y2='{y:.1f}' stroke='{theme['grid']}' stroke-width='1'/>")
        parts.append(text(left - 10, y + 4, f"{tick}", 11.5, theme["muted"], anchor="end"))
    parts.append(text(left - 10, top - 10, "ms", 11.5, theme["muted"], anchor="end"))
    for i, batch in enumerate(BATCH_SIZES):
        spec = results[("rec", batch, 3)]
        step = results[("rec", batch, 0)]["step_ms"][f"decode@{batch}"]["p50"]
        phase = spec["phase_ms_p50"]
        stack = [("verify", phase["verify"]), ("draft", phase["propose"] + phase["catch_up"]), ("other", phase["accept"] + phase["commit_copy"])]
        center = left + group_w * (i + 0.5)
        x_round, x_step = center - bar_w - 4, center + 4
        y_cursor = y_of(0)
        for name, value in stack:
            h = plot_h * value / y_max
            parts.append(f"<rect x='{x_round:.1f}' y='{y_cursor - h + 1:.1f}' width='{bar_w:.1f}' height='{max(h - 2, 0):.1f}' fill='{colors[name]}' rx='2'/>")
            if value >= 3:
                parts.append(text(x_round + bar_w / 2, y_cursor - h / 2 + 4, f"{value:.1f}", 11, theme["surface"], anchor="middle", weight="600"))
            y_cursor -= h
        total = sum(v for _, v in stack)
        parts.append(text(x_round + bar_w / 2, y_of(total) - 8, f"{total:.1f}", 12, theme["text"], anchor="middle", weight="600"))
        h_step = plot_h * step / y_max
        parts.append(f"<rect x='{x_step:.1f}' y='{y_of(step):.1f}' width='{bar_w:.1f}' height='{h_step:.1f}' fill='{theme['neutral']}' rx='2'/>")
        parts.append(text(x_step + bar_w / 2, y_of(step) - 8, f"{step:.1f}", 12, theme["text"], anchor="middle", weight="600"))
        parts.append(text(center, top + plot_h + 20, f"B = {batch}", 12, theme["muted"], anchor="middle"))
        parts.append(text(x_round + bar_w / 2, top + plot_h + 36, "round", 10.5, theme["muted"], anchor="middle"))
        parts.append(text(x_step + bar_w / 2, top + plot_h + 36, "step", 10.5, theme["muted"], anchor="middle"))
    entries = [(colors["verify"], phases[0][1]), (colors["draft"], phases[1][1]),
               (colors["other"], phases[2][1]), (theme["neutral"], "ordinary decode step (k=0)")]
    for row, pair in enumerate((entries[:2], entries[2:])):
        legend_x = left
        legend_y = height - 34 + 18 * row
        for color, label in pair:
            parts.append(f"<rect x='{legend_x}' y='{legend_y - 10}' width='10' height='10' rx='2' fill='{color}'/>")
            parts.append(text(legend_x + 15, legend_y - 1, label, 11.5, theme["text"]))
            legend_x += 15 + 6.6 * len(label) + 30
    parts.append("</svg>")
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--out", type=Path, default=Path("docs/assets"))
    args = parser.parse_args()
    results = load(args.results_dir)
    values = speedups(results)
    args.out.mkdir(parents=True, exist_ok=True)
    for mode, theme in THEMES.items():
        (args.out / f"speedup-{mode}.svg").write_text(speedup_svg(values, theme))
        (args.out / f"round-anatomy-{mode}.svg").write_text(round_cost_svg(results, theme))
    for line, series in values.items():
        print(line, [round(v, 3) for v in series])


if __name__ == "__main__":
    main()
