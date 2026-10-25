"""G5: end-to-end task accuracy on GSM8K's 1319 test questions, a speculative line against k0 (bench-v2 design §3).

  run      one protocol line and one seed through the closed-loop driver at the gate's concurrency with the
           protocol's rec sampling; writes each question's prediction and whether it is right.
  compare  pairs a line's runs with k0's (question x seed, seeds matched) and applies the gate: the 95% interval of
           the accuracy difference must stay above -3 points. Invalid instead of failed when more than 2% of either
           side's answers cannot be parsed, or k0 scores below 80%.

  python tools/gate_task_accuracy.py run --line LINE [--protocol FILE] --seed S --data FILE --models-root DIR \
      --commit C --out FILE
  python tools/gate_task_accuracy.py compare --baseline FILE [FILE ...] --candidate FILE [FILE ...] --out FILE
"""
import argparse
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "benchmarks" / "protocol-v2.json"
PROMPT_SUFFIX = "\nPlease reason step by step, and put your final answer after ####."
NUMBER = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?")
CI_LOW_MIN, MAX_UNPARSEABLE, MIN_K0_ACCURACY = -0.03, 0.02, 0.80    # protocol gates.G5


def load_gsm8k(path: Path) -> list[dict]:
    """[{"question", "truth"}]; truth is the number after the answer's last ####, without commas."""
    rows = []
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        rows.append({"question": row["question"], "truth": row["answer"].rsplit("####", 1)[1].strip().replace(",", "")})
    return rows


def build_prompt(tokenizer, question: str) -> list[int]:
    return tokenizer.apply_chat_template([{"role": "user", "content": question + PROMPT_SUFFIX}], tokenize=True,
                                         return_dict=False, add_generation_prompt=True, enable_thinking=False)


def extract_final_number(text: str) -> str | None:
    """With ####: the first number after the last ####. Without: the last number of the text. No , or $."""
    after_marker = "####" in text
    numbers = NUMBER.findall(text.rsplit("####", 1)[1] if after_marker else text)
    if not numbers:
        return None
    return (numbers[0] if after_marker else numbers[-1]).replace(",", "").replace("$", "")


def is_correct(prediction: str | None, truth: str) -> bool:
    return prediction is not None and float(prediction) == float(truth.replace(",", ""))


def paired_delta(correct_k0: list[bool], correct_k3: list[bool]) -> tuple[float, float, float]:
    """b = k0 wrong, k3 right; c = k0 right, k3 wrong. Delta = (b - c) / n, SE = sqrt(b + c - (b - c)^2 / n) / n,
    95% interval Delta +- 1.96 SE."""
    n = len(correct_k0)
    b = sum(not first and second for first, second in zip(correct_k0, correct_k3))
    c = sum(first and not second for first, second in zip(correct_k0, correct_k3))
    delta = (b - c) / n
    error = math.sqrt(b + c - (b - c) ** 2 / n) / n
    return delta, delta - 1.96 * error, delta + 1.96 * error


def gate_verdict(accuracy_k0: float, unparseable_k0: float, unparseable_k3: float, ci_low: float) -> str:
    if max(unparseable_k0, unparseable_k3) > MAX_UNPARSEABLE or accuracy_k0 < MIN_K0_ACCURACY:
        return "INVALID"
    return "PASS" if ci_low > CI_LOW_MIN else "FAIL"


def run(args: argparse.Namespace) -> dict:
    from transformers import AutoTokenizer

    sys.path.insert(0, str(ROOT))
    from bench_spec import engine_kwargs, run_closed_loop
    from nanovllm import LLM, SamplingParams

    protocol = json.loads(args.protocol.read_text())
    questions = load_gsm8k(args.data)
    tokenizer = AutoTokenizer.from_pretrained(f"{args.models_root}/{protocol['models']['target']}")
    panel = [{"id": f"q{index}", "category": "gsm8k", "prompt_token_ids": build_prompt(tokenizer, row["question"])}
             for index, row in enumerate(questions)]
    concurrency = protocol["gates"]["G5"]["concurrency"]
    target, _, kwargs = engine_kwargs(protocol, protocol["lines"][args.line], panel, concurrency, args.seed,
                                      args.models_root)
    llm = LLM(target, **kwargs)
    sampling = protocol["sampling"]
    params = SamplingParams(temperature=sampling["temperature"], top_k=sampling["top_k"], top_p=sampling["top_p"],
                            max_tokens=sampling["max_tokens"], ignore_eos=sampling["ignore_eos"])
    by_id = {record["id"]: record for record in run_closed_loop(llm, panel, params, concurrency)["requests"]}
    llm.exit()
    texts = [tokenizer.decode(by_id[entry["id"]]["token_ids"], skip_special_tokens=True) for entry in panel]
    predictions = [extract_final_number(text) for text in texts]
    correct = [is_correct(prediction, row["truth"]) for prediction, row in zip(predictions, questions)]
    return {"gate": "G5 run", "engine_commit": args.commit, "line": args.line, "seed": args.seed, "n": len(questions),
            "accuracy": sum(correct) / len(correct),
            "unparseable": sum(prediction is None for prediction in predictions) / len(predictions),
            "correct": correct, "predictions": predictions, "texts": texts}


def compare(args: argparse.Namespace) -> dict:
    baseline = {run["seed"]: run for run in (json.loads(Path(path).read_text()) for path in args.baseline)}
    candidate = {run["seed"]: run for run in (json.loads(Path(path).read_text()) for path in args.candidate)}
    seeds = sorted(baseline)
    assert seeds == sorted(candidate), "baseline and candidate need the same seeds"
    correct_k0 = [value for seed in seeds for value in baseline[seed]["correct"]]
    correct_line = [value for seed in seeds for value in candidate[seed]["correct"]]
    delta, ci_low, ci_high = paired_delta(correct_k0, correct_line)
    accuracy_k0 = sum(correct_k0) / len(correct_k0)
    unparseable_k0 = sum(baseline[seed]["unparseable"] for seed in seeds) / len(seeds)
    unparseable_line = sum(candidate[seed]["unparseable"] for seed in seeds) / len(seeds)
    verdict = gate_verdict(accuracy_k0, unparseable_k0, unparseable_line, ci_low)
    line = candidate[seeds[0]]["line"]
    return {"gate": "G5", "engine_commit": candidate[seeds[0]]["engine_commit"], "passed": verdict == "PASS",
            "statistic": ci_low, "threshold": CI_LOW_MIN,
            "threshold_source": "bench-v2 design §3: lower end of the 95% interval of the paired accuracy difference",
            "n": len(correct_k0),
            "details": {"line": line, "seeds": seeds, "verdict": verdict, "delta": delta, "interval": [ci_low, ci_high],
                        "accuracy_k0": accuracy_k0, "accuracy_line": sum(correct_line) / len(correct_line),
                        "unparseable_k0": unparseable_k0, "unparseable_line": unparseable_line}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["run", "compare"])
    parser.add_argument("--line")
    parser.add_argument("--protocol", type=Path, default=PROTOCOL, help="run")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--models-root", default="/projects/nanovllm/models")
    parser.add_argument("--commit")
    parser.add_argument("--baseline", type=Path, nargs="+", help="k0's run results (compare)")
    parser.add_argument("--candidate", type=Path, nargs="+", help="the line's run results (compare)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(args) if args.mode == "run" else compare(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    if args.mode == "run":
        print(json.dumps({key: result[key] for key in ("line", "seed", "n", "accuracy", "unparseable")}))
        return 0
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
