"""Job A's frozen verify schedule (protocol v2.2 verify_schedule): the 7 STS temperatures fitted on the calibration
runs' confidence logs, the confidences' ECE and AUROC before and after, and the replay that predicts the schedule's
round throughput gain at C=64, all written into the verify-schedule-v1 JSON that VerifySchedule.load reads.

Exit status 1 when the replay's gain is below the protocol's stop_if_gain_below: no dev matrix then.

  python tools/fit_confidence_sts.py --protocol benchmarks/protocol-v2.3.json --cost COST.json \
      --logs RUN.confidences.json [RUN.confidences.json ...] --out benchmarks/inputs/verify-schedule-v1.json
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nanovllm.engine.verify_schedule import SCHEMA, VerifySchedule, survival_probabilities  # noqa: E402

COST_SCHEMA = "verify-cost-v1"
LOG_SCHEMA = "confidence-log-v1"
ECE_BINS = 10


def sigmoid(values: np.ndarray) -> np.ndarray:
    return np.exp(-np.logaddexp(0.0, -values))


def outcomes(accepted: np.ndarray, num_drafts: int) -> np.ndarray:
    """[N] accepted counts -> [N, k]: whether each row's first i drafts were all accepted, i = 1..k."""
    return accepted[:, None] >= np.arange(1, num_drafts + 1)


def binary_cross_entropy(probability: np.ndarray, outcome: np.ndarray) -> float:
    probability = np.clip(probability, 1e-12, 1 - 1e-12)
    return float(-np.mean(np.where(outcome, np.log(probability), np.log1p(-probability))))


def fit_temperatures(logits: np.ndarray, accepted: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """STS: for each position in turn, the grid temperature minimizing the binary cross-entropy of the survival
    probability against 'the first i drafts all accepted', the earlier positions' temperatures kept."""
    observed = outcomes(accepted, logits.shape[1])
    temperatures, survival = [], np.ones(len(logits))
    for position in range(logits.shape[1]):
        losses = [binary_cross_entropy(survival * sigmoid(logits[:, position] / temperature), observed[:, position])
                  for temperature in grid]
        best = float(grid[int(np.argmin(losses))])
        temperatures.append(best)
        survival = survival * sigmoid(logits[:, position] / best)
    return np.array(temperatures)


def expected_calibration_error(probability: np.ndarray, outcome: np.ndarray, bins: int = ECE_BINS) -> float:
    """Over equal-width bins of the prediction: |mean prediction − frequency| of each bin, weighted by its size."""
    index = np.minimum((probability * bins).astype(int), bins - 1)
    gaps = [np.sum(index == b) * abs(probability[index == b].mean() - outcome[index == b].mean())
            for b in range(bins) if np.any(index == b)]
    return float(sum(gaps) / len(probability))


def auroc(score: np.ndarray, outcome: np.ndarray) -> float:
    """The chance a random positive scores above a random negative, ties counting half (Mann–Whitney U)."""
    _, inverse, counts = np.unique(score, return_inverse=True, return_counts=True)
    ranks = (np.cumsum(counts) - (counts - 1) / 2)[inverse]       # 1-based, ties at their average rank
    positives = int(outcome.sum())
    negatives = len(outcome) - positives
    return float((ranks[outcome].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def calibration(logits: np.ndarray, accepted: np.ndarray, temperatures: np.ndarray) -> dict:
    """ECE per position and pooled, and AUROC pooled over positions, of the survival probabilities."""
    observed = outcomes(accepted, logits.shape[1])
    survival = survival_probabilities(logits, temperatures)
    return {"ece_by_position": [expected_calibration_error(survival[:, i], observed[:, i]) for i in range(logits.shape[1])],
            "ece_pooled": expected_calibration_error(survival.ravel(), observed.ravel()),
            "auroc_pooled": auroc(survival.ravel(), observed.ravel())}


def replay(rounds: list[dict], schedule: VerifySchedule, concurrency: int) -> dict:
    """The schedule on the recorded rounds of `concurrency` rows. A row that accepted n of its k drafts accepts
    min(n, L − 1) when it verifies L: the round's drafts, target probabilities and uniforms do not depend on L. Both
    rounds are timed by the schedule's cost table, the fixed-length one at k + 1 per row. Returns the predicted round
    throughput gain over verifying every draft and the mean verified length."""
    tokens_fixed = tokens_scheduled = ms_fixed = ms_scheduled = 0.0
    lengths_seen = []
    for entry in rounds:
        accepted = np.array(entry["accepted"])
        if len(accepted) != concurrency:
            continue
        logits = np.array(entry["logits"], dtype=np.float64)
        lengths = np.array(schedule.verify_lengths(logits))
        round_ms = schedule.round_ms(concurrency)
        tokens_fixed += float(np.sum(accepted + 1))
        tokens_scheduled += float(np.sum(np.minimum(accepted, lengths - 1) + 1))
        ms_fixed += round_ms[concurrency * (logits.shape[1] + 1)]
        ms_scheduled += round_ms[lengths.sum()]
        lengths_seen.append(lengths)
    return {"rounds": len(lengths_seen),
            "gain": (tokens_scheduled / ms_scheduled) / (tokens_fixed / ms_fixed) - 1,
            "tokens_per_round": {"fixed": tokens_fixed / len(lengths_seen), "scheduled": tokens_scheduled / len(lengths_seen)},
            "verified_per_row": float(np.concatenate(lengths_seen).mean())}


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--protocol", type=Path, default=ROOT / "benchmarks" / "protocol-v2.3.json")
    parser.add_argument("--cost", type=Path, required=True, help="tools/profile_verify_cost.py's output")
    parser.add_argument("--logs", type=Path, nargs="+", required=True, help="the calibration runs' .confidences.json")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    protocol_text = args.protocol.read_text()
    protocol = json.loads(protocol_text)
    spec = protocol["verify_schedule"]
    cost, logs = json.loads(args.cost.read_text()), [json.loads(path.read_text()) for path in args.logs]
    protocol_sha256 = hashlib.sha256(protocol_text.encode()).hexdigest()
    assert cost["schema"] == COST_SCHEMA and cost["protocol_sha256"] == protocol_sha256
    wanted = spec["calibration"]
    assert all(log["schema"] == LOG_SCHEMA and (log["line"], log["panel"], log["concurrency"]) ==
               (wanted["line"], wanted["panel"], wanted["concurrency"]) for log in logs)
    assert sorted(log["seed"] for log in logs) == sorted(protocol["seeds"][wanted["seeds"]])
    rounds = [entry for log in logs for entry in log["rounds"]]
    logits = np.array([row for entry in rounds for row in entry["logits"]], dtype=np.float64)
    accepted = np.array([count for entry in rounds for count in entry["accepted"]])
    grid = np.geomspace(wanted["grid"]["low"], wanted["grid"]["high"], wanted["grid"]["points"])
    temperatures = fit_temperatures(logits, accepted, grid)
    concurrency = spec["replay"]["concurrency"]
    schedule = VerifySchedule(cost["config"]["batch_sizes"], cost["round_ms"], temperatures.tolist(), concurrency)
    predicted = replay(rounds, schedule, concurrency)
    stop = bool(predicted["gain"] < spec["replay"]["stop_if_gain_below"])
    table = {"schema": SCHEMA, "batch_sizes": cost["config"]["batch_sizes"], "round_ms": cost["round_ms"],
             "confidence_temperatures": temperatures.tolist(),
             "provenance": {"protocol_sha256": protocol_sha256,
                            "cost": {"sha256": sha256_of(args.cost), "node": cost["manifest"]["node"],
                                     "power_limit": cost["gpu_state_start"]["power.limit"],
                                     "commit": cost["manifest"]["commit"]},
                            "logs": [{"seed": log["seed"], "sha256": sha256_of(path), "commit": log["commit"]}
                                     for log, path in zip(logs, args.logs)]},
             "calibration": {"rows": len(logits), "raw": calibration(logits, accepted, np.ones(logits.shape[1])),
                             "sts": calibration(logits, accepted, temperatures)},
             "replay": {**predicted, "stop": stop}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(table, indent=1) + "\n")
    VerifySchedule.load(str(args.out), logits.shape[1], concurrency)       # it loads the way the engine loads it
    print(json.dumps({"out": str(args.out), "sha256": sha256_of(args.out), "temperatures": table["confidence_temperatures"],
                      "ece_pooled": [table["calibration"][key]["ece_pooled"] for key in ("raw", "sts")],
                      "replay_gain": predicted["gain"], "stop": stop}))
    return 1 if stop else 0


if __name__ == "__main__":
    sys.exit(main())
