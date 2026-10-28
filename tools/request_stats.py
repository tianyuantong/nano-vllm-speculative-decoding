"""What a request sees in bench-v2 closed-loop results, with runs counted as report_v2.py counts them: each line's
latency (TTFT, the gaps between a request's consecutive deliveries, and round time, p50 and p99), its tokens per round
by prompt category, and the spread of tokens per round over requests. With --figures, two SVG figures (light and dark)
for one panel: the gaps of plain decoding against one speculative line, and tokens per round by category.

  python tools/request_stats.py --protocol benchmarks/protocol-v2.1.json --results DIR [DIR ...] \
      [--figures docs/assets/k128b --panel K128b --line dspark_k7]
"""
import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from itl_stats import itl  # noqa: E402
from plot_speedup import BAR_PITCH, BAR_WIDTH, LINES, THEMES, column, text  # noqa: E402
from report_v2 import BASELINE, count_runs, load_runs  # noqa: E402
from summarize_bench import percentile, round_weighted_tokens, tokens_per_round  # noqa: E402

CATEGORIES = {"code_completion": "code completion", "NIAH": "NIAH", "creative_writing": "creative writing",
              "open_books": "open-book QA"}
WIDTH, LEFT, RIGHT, BOTTOM, PANEL_GAP = 760, 56, 24, 60, 64
GAPS_HEIGHT, GAPS_TOP = 480, 156    # room for two legend rows and the panel headings above the plots
CATEGORIES_HEIGHT, CATEGORIES_TOP = 430, 112
NOTCH_MIN_PX = 4    # a notch closer than this to the column's top is left out


def valid_runs(runs: list[tuple[str, dict, dict]], protocol: dict) -> dict[tuple[str, int, str, int], tuple[dict, dict]]:
    """(panel, C, line, seed) -> (result, summary) of each valid repeat-1 run. A later run with the same key replaces an
    earlier one, as in report_v2.count_runs."""
    invalid, _, _ = count_runs(runs, protocol)
    latest = {}
    for name, result, summary in runs:
        config = result["config"]
        if config["repeat"] == 1:
            latest[(config["panel"], config["concurrency"], config["line"], config["seed"])] = (name, result, summary)
    return {key: (result, summary) for key, (name, result, summary) in latest.items() if not invalid[name]}


def latency(result: dict, summary: dict) -> dict[str, float]:
    """One run's TTFT, gaps between deliveries (with and without the gaps that hold a prefill step), round time, and
    tokens per delivery: tokens per round for a speculative line, 1 for plain decoding."""
    gaps = itl(result, result["config"]["concurrency"])
    stats = result["speculative_stats"]
    return {"ttft_p50": summary["ttft_ms"]["p50"], "ttft_p99": summary["ttft_ms"]["p99"],
            "itl_p50": gaps["p50"], "itl_p99": gaps["p99"], "itl_p99_clean": gaps["p99_clean"],
            "stalled_share": gaps["stalled_share"], "round_p50": summary["round_ms"]["p50"],
            "round_p99": summary["round_ms"]["p99"], "tokens_per_delivery": round_weighted_tokens(stats) if stats else 1.0}


def latency_table(valid: dict) -> dict[tuple[str, int, str], dict[str, float]]:
    """(panel, C, line) -> the median over seeds of each latency measure, and the seed count."""
    per_seed = defaultdict(list)
    for (panel, concurrency, line, _), (result, summary) in valid.items():
        per_seed[(panel, concurrency, line)].append(latency(result, summary))
    return {key: {measure: statistics.median(row[measure] for row in rows) for measure in rows[0]} | {"seeds": len(rows)}
            for key, rows in sorted(per_seed.items())}


def category_tokens(valid: dict) -> tuple[dict[tuple, list[int]], dict[tuple, set[str]]]:
    """(panel, C, line, category) -> [tokens delivered after each request's first delivery, the number of those
    deliveries], summed over requests and seeds, so their ratio weights by rounds as the main tables do; and
    (panel, C, category) -> the prompt ids. The first delivery is the prefill's token. Speculative lines only."""
    totals, prompts = defaultdict(lambda: [0, 0]), defaultdict(set)
    for (panel, concurrency, line, _), (result, _) in valid.items():
        if line == BASELINE:
            continue
        for request in result["requests"]:
            later = request["deliveries"][1:]
            total = totals[(panel, concurrency, line, request["category"])]
            total[0] += sum(n for _, n in later)
            total[1] += len(later)
            prompts[(panel, concurrency, request["category"])].add(request["id"])
    return totals, prompts


def pooled(totals: dict, panel: str, line: str, category: str, concurrencies: list[int]) -> float:
    rows = [totals.get((panel, c, line, category), (0, 0)) for c in concurrencies]
    return sum(tokens for tokens, _ in rows) / sum(deliveries for _, deliveries in rows)


def prompt_count(prompts: dict, panel: str, category: str, concurrencies: list[int]) -> int:
    return len(set().union(*(prompts.get((panel, c, category), set()) for c in concurrencies)))


def per_request_tokens(valid: dict) -> dict[tuple[str, int, str], list[float]]:
    """(panel, C, line) -> each request's mean tokens per delivery after the prefill's, over every seed."""
    spread = defaultdict(list)
    for (panel, concurrency, line, _), (result, _) in valid.items():
        if line != BASELINE:
            spread[(panel, concurrency, line)] += [value for value in tokens_per_round(result["requests"])
                                                   if value is not None]
    return spread


def print_tables(valid: dict) -> None:
    print("| panel | C | line | seeds | TTFT p50 / p99 ms | ITL p50 / p99 ms | ITL p99 without prefill gaps ms "
          "| gaps with a prefill | round p50 / p99 ms | tokens per delivery |")
    print("|---|---:|---|---:|---|---|---:|---:|---|---:|")
    for (panel, concurrency, line), m in latency_table(valid).items():
        print(f"| {panel} | {concurrency} | {line} | {m['seeds']} | {m['ttft_p50']:.0f} / {m['ttft_p99']:.0f} "
              f"| {m['itl_p50']:.1f} / {m['itl_p99']:.1f} | {m['itl_p99_clean']:.1f} | {m['stalled_share']:.0%} "
              f"| {m['round_p50']:.1f} / {m['round_p99']:.1f} | {m['tokens_per_delivery']:.2f} |")

    totals, prompts = category_tokens(valid)
    lines = [line for line in LINES if any(key[2] == line for key in totals)]
    print("\nTokens per round by category, weighted by rounds, over every seed: at C = 1 and over every C.\n")
    print("| panel | category | prompts at C = 1 / every C | " + " | ".join(f"{line} C = 1 / every C" for line in lines) + " |")
    print("|---|---|---|" + "---:|" * len(lines))
    for panel in sorted({key[0] for key in totals}):
        concurrencies = sorted({key[1] for key in totals if key[0] == panel})
        for category in sorted({key[3] for key in totals if key[0] == panel}):
            cells = [f"{pooled(totals, panel, line, category, [1]):.2f} / {pooled(totals, panel, line, category, concurrencies):.2f}"
                     for line in lines]
            counts = f"{prompt_count(prompts, panel, category, [1])} / {prompt_count(prompts, panel, category, concurrencies)}"
            print(f"| {panel} | {category} | {counts} | " + " | ".join(cells) + " |")

    print("\nTokens per round of each request (its deliveries after the prefill's), over every seed.\n")
    print("| panel | C | line | requests | p10 | p50 | p90 |")
    print("|---|---:|---|---:|---:|---:|---:|")
    for (panel, concurrency, line), values in sorted(per_request_tokens(valid).items()):
        print(f"| {panel} | {concurrency} | {line} | {len(values)} | {percentile(values, 10):.2f} "
              f"| {percentile(values, 50):.2f} | {percentile(values, 90):.2f} |")


def svg_open(title: str, summary: str, subtitle: str, theme: dict, height: int) -> list[str]:
    return [f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {WIDTH} {height}' width='{WIDTH}' height='{height}' "
            f"role='img'><title>{title}</title><desc>{summary}</desc>",
            f"<rect width='{WIDTH}' height='{height}' fill='{theme['surface']}' rx='8'/>",
            text(LEFT, 28, title, 18, theme["text"], weight="600"), text(LEFT, 49, subtitle, 12, theme["muted"])]


def legend(entries: list[tuple[str, str]], x: float, y: float, theme: dict) -> list[str]:
    """Swatch and label pairs in one row; a swatch kind of 'line' draws the reference line, any other a color."""
    parts = []
    for swatch, label in entries:
        if swatch == "line":
            parts.append(f"<line x1='{x}' y1='{y - 4}' x2='{x + 16}' y2='{y - 4}' stroke='{theme['reference']}' stroke-width='1.5'/>")
            x += 22
        else:
            parts.append(f"<rect x='{x}' y='{y - 10}' width='12' height='12' rx='3' fill='{swatch}'/>")
            x += 18
        parts.append(text(x, y + 0.5, label, 12, theme["text"]))
        x += 6.0 * len(label) + 20
    return parts


def axis(y_max: float, step: float, x0: float, x1: float, y_of, theme: dict, unit: str = "") -> list[str]:
    parts = []
    for index in range(int(round(y_max / step)) + 1):
        tick, y = index * step, y_of(index * step)
        parts.append(f"<line x1='{x0}' y1='{y:.1f}' x2='{x1}' y2='{y:.1f}' stroke='{theme['grid']}' stroke-width='1'/>")
        parts.append(text(x0 - 8, y + 4, f"{tick:g}{unit}", 11, theme["muted"], anchor="end"))
    return parts


def nice_step(top: float) -> float:
    """A tick step that puts four to eight ticks under top."""
    return next(step for step in (1, 2, 5, 10, 20, 25, 50, 100, 200) if top / step <= 8)


def gaps_panel(table: dict, panel: str, line: str, measure: str, x0: float, width: float, heading: str,
               theme: dict) -> list[str]:
    """Plain decoding and one line, a pair of columns per C, of one gap measure; the p99 panel notches each column at
    the p99 without the gaps that hold a prefill step."""
    concurrencies = sorted({c for (p, c, name) in table if p == panel and name == line})
    top = max(table[(panel, c, name)][measure] for c in concurrencies for name in (BASELINE, line))
    step = nice_step(top)
    y_max = math.ceil(top / step) * step
    plot_h = GAPS_HEIGHT - GAPS_TOP - BOTTOM

    def y_of(value: float) -> float:
        return GAPS_TOP + plot_h * (1 - value / y_max)

    parts = [text(x0, GAPS_TOP - 24, heading, 13, theme["text"], weight="600")]
    parts += axis(y_max, step, x0, x0 + width, y_of, theme)
    slot = width / len(concurrencies)
    for index, concurrency in enumerate(concurrencies):
        center = x0 + slot * (index + 0.5)
        for position, name in enumerate((BASELINE, line)):
            row = table[(panel, concurrency, name)]
            middle = center + (position - 0.5) * BAR_PITCH
            parts.append(column(middle - BAR_WIDTH / 2, y_of(row[measure]), y_of(0), theme["series"][name]))
            notch = y_of(row["itl_p99_clean"])
            if measure == "itl_p99" and notch - y_of(row[measure]) >= NOTCH_MIN_PX:
                parts.append(f"<line x1='{middle - BAR_WIDTH / 2:.1f}' y1='{notch:.1f}' x2='{middle + BAR_WIDTH / 2:.1f}' "
                             f"y2='{notch:.1f}' stroke='{theme['surface']}' stroke-width='2'/>")
            parts.append(text(middle, y_of(row[measure]) - 6, f"{row[measure]:.0f}", 11, theme["text"], anchor="middle"))
        parts.append(text(center, GAPS_HEIGHT - BOTTOM + 20, str(concurrency), 12, theme["muted"], anchor="middle"))
    parts.append(text(x0 + width / 2, GAPS_HEIGHT - BOTTOM + 42, "concurrent requests (C)", 11, theme["muted"],
                      anchor="middle"))
    return parts


def gaps_svg(table: dict, panel: str, line: str, theme: dict, subtitle: str) -> str:
    rows = {c: table[(p, c, name)] for (p, c, name) in table if p == panel and name == line}
    tokens = [row["tokens_per_delivery"] for row in rows.values()]
    title = f"Gaps between a request's deliveries: plain decoding and the {LINES[line].split(',')[0]}"
    summary = "; ".join(f"C = {c}: ITL p50 {table[(panel, c, BASELINE)]['itl_p50']:.1f} and {row['itl_p50']:.1f} ms, "
                        f"p99 {table[(panel, c, BASELINE)]['itl_p99']:.1f} and {row['itl_p99']:.1f} ms"
                        for c, row in sorted(rows.items()))
    parts = svg_open(title, summary, subtitle, theme, GAPS_HEIGHT)
    parts += legend([(theme["series"][BASELINE], "plain decoding: 1 token a delivery"),
                     (theme["series"][line], f"{LINES[line]}: {min(tokens):.2f}–{max(tokens):.2f} tokens a delivery")],
                    LEFT, 78, theme)
    parts.append(f"<rect x='{LEFT}' y='90' width='12' height='12' rx='3' fill='{theme['series'][BASELINE]}'/>"
                 f"<line x1='{LEFT}' y1='96' x2='{LEFT + 12}' y2='96' stroke='{theme['surface']}' stroke-width='2'/>")
    parts.append(text(LEFT + 18, 100.5, "notch: p99 without the gaps that hold a prefill step", 12, theme["text"]))
    width = (WIDTH - LEFT - RIGHT - PANEL_GAP) / 2
    parts += gaps_panel(table, panel, line, "itl_p50", LEFT, width, "Typical gap, ITL p50 (ms)", theme)
    parts += gaps_panel(table, panel, line, "itl_p99", LEFT + width + PANEL_GAP, width, "Longest gaps, ITL p99 (ms)", theme)
    return "\n".join(parts) + "\n</svg>\n"


def categories_svg(totals: dict, prompts: dict, panel: str, theme: dict, subtitle: str) -> str:
    """Tokens per round by category over every C, a column per speculative line, the categories in descending order of
    the last line's value."""
    lines = [line for line in LINES if any(key[0] == panel and key[2] == line for key in totals)]
    concurrencies = sorted({key[1] for key in totals if key[0] == panel})
    values = {(line, category): pooled(totals, panel, line, category, concurrencies)
              for line in lines for category in CATEGORIES if (panel, concurrencies[0], line, category) in totals}
    categories = sorted({category for _, category in values}, key=lambda category: -values[(lines[-1], category)])
    y_max, plot_w, plot_h = 4.0, WIDTH - LEFT - RIGHT, CATEGORIES_HEIGHT - CATEGORIES_TOP - BOTTOM

    def y_of(value: float) -> float:
        return CATEGORIES_TOP + plot_h * (1 - value / y_max)

    title = "Tokens per round by prompt category"
    summary = "; ".join(f"{CATEGORIES[category]}: " + ", ".join(f"{LINES[line]} {values[(line, category)]:.2f}"
                                                                for line in lines) for category in categories)
    parts = svg_open(title, summary, subtitle, theme, CATEGORIES_HEIGHT)
    parts += legend([(theme["series"][line], LINES[line]) for line in lines] + [("line", "plain decoding: 1 token a step")],
                    LEFT, 78, theme)
    parts += axis(y_max, 1.0, LEFT, WIDTH - RIGHT, y_of, theme)
    parts.append(f"<line x1='{LEFT}' y1='{y_of(1):.1f}' x2='{WIDTH - RIGHT}' y2='{y_of(1):.1f}' "
                 f"stroke='{theme['reference']}' stroke-width='1.5'/>")
    slot = plot_w / len(categories)
    for index, category in enumerate(categories):
        center = LEFT + slot * (index + 0.5)
        for position, line in enumerate(lines):
            value = values[(line, category)]
            middle = center + (position - (len(lines) - 1) / 2) * BAR_PITCH
            parts.append(column(middle - BAR_WIDTH / 2, y_of(value), y_of(0), theme["series"][line]))
            parts.append(text(middle, y_of(value) - 6, f"{value:.2f}", 11, theme["text"], anchor="middle", weight="600"))
        count = prompt_count(prompts, panel, category, concurrencies)
        gain = values[(lines[-1], category)] / values[(lines[0], category)] - 1
        parts.append(text(center, CATEGORIES_HEIGHT - BOTTOM + 20, CATEGORIES[category], 12, theme["text"], anchor="middle"))
        parts.append(text(center, CATEGORIES_HEIGHT - BOTTOM + 38, f"{count} prompts · {LINES[lines[-1]].split()[0]} {gain:+.0%}",
                          11, theme["muted"], anchor="middle"))
    return "\n".join(parts) + "\n</svg>\n"


def write_figures(valid: dict, protocol: dict, panel: str, line: str, prefix: Path) -> None:
    seeds = max(len({key[3] for key in valid if key[:3] == (panel, c, line)}) for c in {key[1] for key in valid})
    gpu = protocol["gpu"]["name"].removeprefix("NVIDIA ")
    base = f"{protocol['models']['target']} · {protocol['models']['dtype']} · {gpu} · {panel} panel"
    totals, prompts = category_tokens(valid)
    for mode, theme in THEMES.items():
        figures = {"gaps": gaps_svg(latency_table(valid), panel, line, theme, f"{base} · medians over {seeds} seeds"),
                   "categories": categories_svg(totals, prompts, panel, theme,
                                                f"{base} · every concurrency and seed, weighted by rounds")}
        for name, svg in figures.items():
            path = prefix.with_name(f"{prefix.name}-{name}-{mode}.svg")
            path.write_text(svg)
            print(path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--results", type=Path, nargs="+", required=True,
                        help="directories of bench-v2 result files; a rerun directory last replaces the runs it repeats")
    parser.add_argument("--figures", type=Path, help="path prefix: writes PREFIX-gaps-*.svg and PREFIX-categories-*.svg")
    parser.add_argument("--panel", help="the panel the figures show")
    parser.add_argument("--line", default="dspark_k7", help="the line the gaps figure sets against plain decoding")
    args = parser.parse_args(argv)
    if args.figures and not args.panel:
        parser.error("--figures needs --panel")
    protocol = json.loads(args.protocol.read_text())
    valid = valid_runs(load_runs(args.results), protocol)
    print_tables(valid)
    if args.figures:
        write_figures(valid, protocol, args.panel, args.line, args.figures)


if __name__ == "__main__":
    main()
