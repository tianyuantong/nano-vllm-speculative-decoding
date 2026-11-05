"""G2-real: on one real target/draft distribution pair, the production sampler and acceptance rule reproduce the
analytic acceptance rate and emit the target distribution (bench-v2 design §3). Also re-runs the frozen reference
cases (exact distributions, total variation < 0.01).

The pair: prompt by prompt (D48), round by round through the engine, the first verified position whose accept mass
sum(min(p, q)) lies in [0.2, 0.9] and whose target p and draft q are each positive on at least 2 tokens; only that
rule picks it. Sliced to the tokens where p or q is positive (no other
token can be drawn), k=1 rounds through sample() and accept_random(): (a) 16384 acceptance draws within 0.02 of the
accept mass; (b) 100000 first emitted tokens, every frequency within the Hoeffding bound sqrt(ln(2M/delta) / 2N) of
p, M the support size, delta 0.001.

  python tools/gate_sampler_distribution.py --line LINE [--protocol FILE] --models-root DIR --commit C --out FILE
"""
import argparse
import json
import math
import sys
from pathlib import Path
from typing import TypedDict

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from random_sampling_cases import CASES  # noqa: E402

from bench_spec import engine_kwargs  # noqa: E402
from nanovllm import LLM, SamplingParams  # noqa: E402
from nanovllm.layers.spec_sampler import accept_random, probs_from_logits, sample  # noqa: E402

PROTOCOL = ROOT / "benchmarks" / "protocol-v2.json"
NUM_DRAWS = 200_000
TOLERANCE = 0.01


class RealPosition(TypedDict):
    prompt_index: int
    round: int
    accept_mass: float        # sum(min(p, q))
    support: list[int]        # the tokens where p or q is positive
    p: list[float]            # on the support only
    q: list[float]


def reference_cases(device) -> list[dict]:
    generator = torch.Generator(device=device).manual_seed(1234)
    results = []
    for case in (case for case in CASES if "p" in case):
        p_row = torch.tensor(case["p"], dtype=torch.float32, device=device)
        q_row = torch.tensor(case["q"], dtype=torch.float32, device=device)
        vocab = p_row.numel()
        drafts = sample(q_row.expand(NUM_DRAWS, vocab), generator).unsqueeze(1)
        n_accept, tail = accept_random(p_row.expand(NUM_DRAWS, 2, vocab), q_row.expand(NUM_DRAWS, 1, vocab), drafts, generator)
        emitted = torch.where(n_accept == 1, drafts[:, 0], tail)
        frequencies = torch.bincount(emitted, minlength=vocab).float() / NUM_DRAWS
        results.append({"name": case["name"], "total_variation": 0.5 * (frequencies - p_row).abs().sum().item()})
    return results


def pick_real_position(llm, prompts: list[list[int]], sampling: SamplingParams, low: float, high: float,
                       max_rounds: int, min_draft_support: int, min_target_support: int) -> RealPosition:
    """Prompt by prompt, round by round, look at each round's first verified position; return the first whose
    sum(min(p, q)) lies in [low, high], whose q is positive on at least min_draft_support tokens and whose p on at
    least min_target_support. Only this rule picks the position; the checks never feed back into it."""
    decoder = llm.speculative
    propose, verify = decoder.drafter.propose, decoder.verify_logits
    seen = {}

    def propose_and_keep(seqs, sampling_tensors):
        drafts, draft_probs = propose(seqs, sampling_tensors)
        seen["q"], seen["sampling"] = draft_probs[0, 0], sampling_tensors
        return drafts, draft_probs

    def verify_and_keep(seqs, drafts, aux_out=None):
        logits = verify(seqs, drafts, aux_out)
        seen["p"] = probs_from_logits(logits, seen["sampling"])[0, 0]
        return logits

    decoder.drafter.propose, decoder.verify_logits = propose_and_keep, verify_and_keep
    rounds = 0
    try:
        for prompt_index, prompt in enumerate(prompts):
            llm.add_request(prompt, sampling)
            round_index = 0
            while not llm.is_finished() and rounds < max_rounds:
                seen.clear()
                llm.step()
                if "p" not in seen:                    # a prefill step
                    continue
                p, q = seen["p"], seen["q"]
                mass = torch.minimum(p, q).sum().item()
                if (low <= mass <= high and int((q > 0).sum()) >= min_draft_support
                        and int((p > 0).sum()) >= min_target_support):
                    support = ((p > 0) | (q > 0)).nonzero().squeeze(1)
                    return RealPosition(prompt_index=prompt_index, round=round_index, accept_mass=mass,
                                        support=support.tolist(), p=p[support].tolist(), q=q[support].tolist())
                rounds += 1
                round_index += 1
    finally:
        decoder.drafter.propose, decoder.verify_logits = propose, verify
    raise RuntimeError(f"no verified position with accept mass in [{low}, {high}], a draft support of at least "
                       f"{min_draft_support} and a target support of at least {min_target_support} within {max_rounds} rounds")


def check_real_position(position: RealPosition, accept_trials: int, emit_trials: int, delta: float,
                        device: torch.device) -> dict:
    """k=1 rounds on the support. The bonus row repeats p: it is drawn from only after an acceptance, so it never
    becomes the first emitted token."""
    p = torch.tensor(position["p"], dtype=torch.float32, device=device)
    q = torch.tensor(position["q"], dtype=torch.float32, device=device)
    size = p.numel()
    generator = torch.Generator(device=device).manual_seed(17011)

    def first_tokens(trials: int) -> tuple[torch.Tensor, torch.Tensor]:
        drafts = sample(q.expand(trials, size), generator).unsqueeze(1)
        n_accept, tail = accept_random(torch.stack([p, p]).expand(trials, 2, size), q.expand(trials, 1, size), drafts,
                                       generator)
        return n_accept, torch.where(n_accept == 1, drafts[:, 0], tail)

    n_accept, _ = first_tokens(accept_trials)
    accept_rate = n_accept.float().mean().item()
    _, emitted = first_tokens(emit_trials)
    frequencies = torch.bincount(emitted, minlength=size).float() / emit_trials
    return {"support_size": size, "accept_rate": accept_rate, "accept_error": abs(accept_rate - position["accept_mass"]),
            "worst_frequency_gap": (frequencies - p).abs().max().item(),
            "frequency_bound": math.sqrt(math.log(2 * size / delta) / (2 * emit_trials))}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--line", required=True, help="a line with a drafter (v2: model_k3, dflash_k7; v2.1: dspark_k7)")
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--models-root", default="/projects/nanovllm/models")
    parser.add_argument("--commit", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    protocol = json.loads(args.protocol.read_text())
    gate, rec = protocol["gates"]["G2_real"], protocol["sampling"]
    panel = json.loads((ROOT / protocol["panels"]["D48"]["file"]).read_text())
    target, _, kwargs = engine_kwargs(protocol, protocol["lines"][args.line], panel, 1, protocol["seeds"]["dev"][0],
                                      args.models_root)
    llm = LLM(target, **kwargs)
    sampling = SamplingParams(temperature=rec["temperature"], top_k=rec["top_k"], top_p=rec["top_p"],
                              max_tokens=rec["max_tokens"], ignore_eos=rec["ignore_eos"])
    low, high = gate["accept_mass"]
    position = pick_real_position(llm, [entry["prompt_token_ids"] for entry in panel], sampling, low, high,
                                  gate["scan_rounds"], gate["min_draft_support"], gate["min_target_support"])
    llm.exit()
    device = torch.device("cuda")
    real = check_real_position(position, gate["accept_trials"], gate["emit_trials"], gate["delta"], device)
    references = reference_cases(device)
    passed = (real["accept_error"] < gate["accept_tolerance"] and real["worst_frequency_gap"] <= real["frequency_bound"]
              and all(case["total_variation"] < TOLERANCE for case in references))
    report = {"gate": "G2-real", "engine_commit": args.commit, "passed": passed,
              "statistic": {"accept_error": real["accept_error"], "worst_frequency_gap": real["worst_frequency_gap"]},
              "threshold": {"accept_error": gate["accept_tolerance"], "worst_frequency_gap": real["frequency_bound"]},
              "threshold_source": f"Hoeffding delta={gate['delta']}, M={real['support_size']}, N={gate['emit_trials']}; "
                                  f"acceptance over {gate['accept_trials']} draws (std <= 0.004)",
              "n": {"accept_trials": gate["accept_trials"], "emit_trials": gate["emit_trials"]},
              "details": {"line": args.line, "position": position, "real": real, "reference_cases": references}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("gate", "passed", "statistic", "threshold")}))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
