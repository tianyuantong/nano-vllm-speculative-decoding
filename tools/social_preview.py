"""Render the 1280×640 GitHub social-preview card (SVG) from a results directory."""
import argparse
from pathlib import Path

from plot_results import BATCH_SIZES, THEMES, load, speedups, text


def card(values: list[float], theme: dict) -> str:
    width, height = 1280, 640
    parts = [f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {width} {height}' width='{width}' height='{height}'>",
             f"<rect width='{width}' height='{height}' fill='{theme['surface']}'/>",
             text(64, 96, "nano-vllm-speculative-decoding", 40, theme["text"], weight="700"),
             text(64, 140, "Batched GPU sampling · CUDA Graph verification · scheduler integration", 20, theme["muted"]),
             text(64, 290, f"{values[0]:.2f}×", 96, theme["series"][0], weight="700"),
             text(64, 326, "throughput at batch 1", 20, theme["muted"]),
             text(340, 290, f"{values[3]:.2f}×", 96, theme["series"][0], weight="700"),
             text(340, 326, "throughput at batch 8", 20, theme["muted"]),
             text(64, 402, "Qwen3-8B target · Qwen3-0.6B draft", 20, theme["text"]),
             text(64, 434, "BF16 · RTX PRO 6000 Blackwell", 18, theme["muted"]),
             text(64, 474, "Recommended sampling · 3 draft tokens per round", 18, theme["muted"]),
             text(64, 506, "48 prompts · up to 512 output tokens per prompt", 18, theme["muted"]),
             text(64, 560, "English · 简体中文 · performance results · implementation design", 18, theme["muted"])]
    # mini bar chart on the right
    x0, y0, w, h = 760, 200, 456, 300
    y_max = 1.7
    bar_w = w / len(BATCH_SIZES) * 0.55
    for i, (batch, value) in enumerate(zip(BATCH_SIZES, values)):
        cx = x0 + w * (i + 0.5) / len(BATCH_SIZES)
        bh = h * value / y_max
        parts.append(f"<rect x='{cx - bar_w / 2:.1f}' y='{y0 + h - bh:.1f}' width='{bar_w:.1f}' height='{bh:.1f}' rx='6' fill='{theme['series'][0]}'/>")
        parts.append(text(cx, y0 + h - bh - 12, f"{value:.2f}×", 20, theme["text"], anchor="middle", weight="600"))
        parts.append(text(cx, y0 + h + 30, f"B={batch}", 18, theme["muted"], anchor="middle"))
    y1 = y0 + h - h * 1.0 / y_max
    parts.append(f"<line x1='{x0}' y1='{y1:.1f}' x2='{x0 + w}' y2='{y1:.1f}' stroke='{theme['text']}' stroke-width='2' stroke-dasharray='8 6'/>")
    parts.append(text(x0, y0 - 36, "Throughput ratio · k=3", 18, theme["text"], weight="600"))
    parts.append(text(x0, y0 - 12, "Dashed: ordinary decoding = 1.0×", 14, theme["muted"]))
    parts.append("</svg>")
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    values = speedups(load(args.results_dir))["rec"]
    args.out.write_text(card(values, THEMES["light"]))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
