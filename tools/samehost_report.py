"""M5's same-machine comparison of nano and vLLM (protocol v2.4 samehost): wall-clock Q_full of each line at each C over
the seeds, and whether their ranges overlap. Where they do not, a gap remains and the other diagnostics run there. The
rule needs the five dev seeds on both sides and no invalid run: a (line, C) short of that shows what it has and no verdict.
A rerun directory listed after the original replaces its runs.

  python tools/samehost_report.py --protocol benchmarks/protocol-v2.4.json --nano DIR [DIR ...] --vllm DIR [DIR ...]

nano's lines are named as vLLM's: k0 is compared with k0_v2.
"""
import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_v2 import count_runs, final_runs, load_runs  # noqa: E402
from summarize_bench import round_weighted_tokens  # noqa: E402

VLLM_NAME = {"k0": "k0_v2", "dflash_k7": "dflash_k7", "dspark_k7": "dspark_k7"}


def nano_rows(runs: list[tuple[str, dict, dict]], protocol: dict) -> tuple[dict, list[str]]:
    """(vLLM's name of the line, C) -> seed -> wall-clock Q_full and tokens per round, for the runs that count and are
    valid; and each invalid one."""
    invalid, _, _ = count_runs(runs, protocol)
    rows, bad = defaultdict(dict), []
    for (_, concurrency, line, seed, _), (name, result, _) in final_runs(runs).items():
        if invalid[name]:
            bad.append(f"nano {Path(name).name}: {', '.join(invalid[name])}")
            continue
        tokens = sum(len(request["token_ids"]) for request in result["requests"])
        stats = result["speculative_stats"]
        rows[(VLLM_NAME[line], concurrency)][seed] = {"q": tokens / result["wall_seconds"],
                                                      "tokens_per_round": round_weighted_tokens(stats) if stats else None}
    return rows, bad


def vllm_rows(directories: list[Path]) -> tuple[dict, list[str]]:
    rows, bad = defaultdict(dict), []
    for path in (path for directory in directories for path in sorted(directory.glob("*.json"))):
        result = json.loads(path.read_text())
        run, key, seed = result["runs"][0], (result["line"], result["concurrency"]), result["engine"]["seed"]
        rows[key].pop(seed, None)                                  # a later directory's run replaces this one
        bad = [line for line in bad if not line.startswith(f"vllm {path.name}:")]
        if run["preemptions"]:
            bad.append(f"vllm {path.name}: preemption > 0")
            continue
        rows[key][seed] = {"q": run["tokens_per_second"], "tokens_per_round": run["tokens_per_round"]}
    return rows, bad


def gap_remains(nano: list[float], vllm: list[float]) -> bool:
    return max(nano) < min(vllm) or max(vllm) < min(nano)


def report(nano: dict, vllm: dict, seeds: list[int], bad: list[str]) -> str:
    out = ["| line | C | seeds | nano tok/s | vLLM tok/s | nano / vLLM | gap remains | nano tokens/round | vLLM tokens/round |",
           "|---|---:|---:|---|---|---:|---|---|---|"]
    for key in sorted(set(nano) | set(vllm)):
        line, concurrency = key
        ours, theirs = nano.get(key, {}), vllm.get(key, {})
        q_ours, q_theirs = [row["q"] for row in ours.values()], [row["q"] for row in theirs.values()]

        def spread(values: list[float]) -> str:
            return f"{statistics.median(values):.1f} ({min(values):.1f}–{max(values):.1f})" if values else "-"

        def tokens(rows: dict) -> str:
            values = [row["tokens_per_round"] for row in rows.values()]
            return "-" if not values or None in values else f"{statistics.median(values):.2f} ({min(values):.2f}–{max(values):.2f})"

        whole = set(ours) == set(theirs) == set(seeds)
        ratio = f"{statistics.median(q_ours) / statistics.median(q_theirs):.3f}" if q_ours and q_theirs else "-"
        verdict = ("yes" if gap_remains(q_ours, q_theirs) else "no") if whole else "incomplete"
        out.append(f"| {line} | {concurrency} | {min(len(ours), len(theirs))} | {spread(q_ours)} | {spread(q_theirs)} | {ratio} "
                   f"| {verdict} | {tokens(ours)} | {tokens(theirs)} |")
    if bad:
        out += ["", "Invalid, rerun their whole (C, seed):", "", *(f"- {line}" for line in bad)]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--nano", type=Path, nargs="+", required=True)
    parser.add_argument("--vllm", type=Path, nargs="+", required=True)
    args = parser.parse_args(argv)
    protocol = json.loads(args.protocol.read_text())
    nano, nano_bad = nano_rows(load_runs(args.nano), protocol)
    vllm, vllm_bad = vllm_rows(args.vllm)
    print(report(nano, vllm, protocol["seeds"]["dev"], nano_bad + vllm_bad))


if __name__ == "__main__":
    main()
