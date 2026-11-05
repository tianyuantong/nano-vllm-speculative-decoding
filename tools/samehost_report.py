"""M5's same-machine comparison of nano and vLLM (protocol v2.4 samehost): wall-clock Q_full of each line at each C over
the seeds, and whether their ranges overlap. Where they do not, a gap remains and the other diagnostics run there.

  python tools/samehost_report.py --nano DIR --vllm DIR

nano's lines are named as vLLM's: k0 is compared with k0_v2.
"""
import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_bench import round_weighted_tokens  # noqa: E402

VLLM_NAME = {"k0": "k0_v2", "dflash_k7": "dflash_k7", "dspark_k7": "dspark_k7"}


def nano_runs(directory: Path) -> dict[tuple[str, int], dict[int, dict]]:
    """(vLLM's name of the line, C) -> seed -> wall-clock Q_full and tokens per round."""
    runs = defaultdict(dict)
    for path in sorted(directory.glob("*.json")):
        result = json.loads(path.read_text())
        config = result["config"]
        tokens = sum(len(request["token_ids"]) for request in result["requests"])
        stats = result["speculative_stats"]
        runs[(VLLM_NAME[config["line"]], config["concurrency"])][config["seed"]] = {
            "q": tokens / result["wall_seconds"], "tokens_per_round": round_weighted_tokens(stats) if stats else None}
    return runs


def vllm_runs(directory: Path) -> dict[tuple[str, int], dict[int, dict]]:
    runs = defaultdict(dict)
    for path in sorted(directory.glob("*.json")):
        result = json.loads(path.read_text())
        run = result["runs"][0]
        runs[(result["line"], result["concurrency"])][result["engine"]["seed"]] = {
            "q": run["tokens_per_second"], "tokens_per_round": run["tokens_per_round"]}
    return runs


def gap_remains(nano: list[float], vllm: list[float]) -> bool:
    return max(nano) < min(vllm) or max(vllm) < min(nano)


def report(nano: dict, vllm: dict) -> str:
    out = ["| line | C | seeds | nano tok/s | vLLM tok/s | nano / vLLM | gap remains | nano tokens/round | vLLM tokens/round |",
           "|---|---:|---:|---|---|---:|---|---|---|"]
    for key in sorted(nano):
        line, concurrency = key
        ours, theirs = nano[key], vllm[key]
        assert set(ours) == set(theirs), f"{key}: the seeds differ"
        q_ours, q_theirs = [row["q"] for row in ours.values()], [row["q"] for row in theirs.values()]

        def spread(values: list[float]) -> str:
            return f"{statistics.median(values):.1f} ({min(values):.1f}–{max(values):.1f})"

        def tokens(rows: dict) -> str:
            values = [row["tokens_per_round"] for row in rows.values()]
            return "-" if None in values else f"{statistics.median(values):.2f} ({min(values):.2f}–{max(values):.2f})"

        out.append(f"| {line} | {concurrency} | {len(ours)} | {spread(q_ours)} | {spread(q_theirs)} "
                   f"| {statistics.median(q_ours) / statistics.median(q_theirs):.3f} "
                   f"| {'yes' if gap_remains(q_ours, q_theirs) else 'no'} | {tokens(ours)} | {tokens(theirs)} |")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nano", type=Path, required=True)
    parser.add_argument("--vllm", type=Path, required=True)
    args = parser.parse_args(argv)
    print(report(nano_runs(args.nano), vllm_runs(args.vllm)))


if __name__ == "__main__":
    main()
