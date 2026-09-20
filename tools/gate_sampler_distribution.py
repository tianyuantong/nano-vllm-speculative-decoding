"""G2: the batched sampler on GPU reproduces (a) the exact reference distributions and
(b) the analytic acceptance rate sum(min(p, q)) on one real target/draft distribution pair."""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from random_sampling_cases import CASES  # noqa: E402

from nanovllm import LLM, SamplingParams  # noqa: E402
from nanovllm.layers.spec_sampler import accept_random, probs_from_logits, sample, sampling_tensors  # noqa: E402

GIB = 1 << 30
NUM_DRAWS = 200_000
TOLERANCE = 0.01
REAL_CHUNK_ROWS = 4096
REAL_CHUNKS = 4                 # 16384 Bernoulli draws: std <= 0.004, tolerance 0.02 is 5 sigma
REAL_TOLERANCE = 0.02


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


def real_distribution_case(args) -> dict:
    llm = LLM(args.target, draft_model=args.draft, num_speculative_tokens=args.k, max_model_len=4096,
              max_num_seqs=1, enable_prefix_cache=False, kv_cache_memory_bytes=4 * GIB,
              draft_kv_cache_memory_bytes=2 * GIB, seed=0)
    llm.add_request(args.prompt, SamplingParams(temperature=0.7, top_k=20, top_p=0.8, max_tokens=8))
    llm.step()                                                     # prefill
    (seq,) = list(llm.scheduler.running)
    llm.scheduler.schedule()                                       # reserves the round's blocks
    decoder = llm.speculative
    sampling = sampling_tensors([seq], decoder.device)
    drafts, draft_probs = decoder._propose([seq], sampling)
    target_probs = probs_from_logits(decoder.verify_logits([seq], drafts), sampling)
    p0, q0 = target_probs[0, 0], draft_probs[0, 0]
    expected_accept = torch.minimum(p0, q0).sum().item()
    generator = torch.Generator(device=decoder.device).manual_seed(7)
    vocab = p0.numel()
    accepted = 0
    for _ in range(REAL_CHUNKS):                                   # full-vocabulary rows: chunked to bound memory
        trial_drafts = torch.multinomial(q0, REAL_CHUNK_ROWS, replacement=True, generator=generator).unsqueeze(1)
        n_accept, _ = accept_random(target_probs[0, :2].expand(REAL_CHUNK_ROWS, 2, vocab),
                                    q0.expand(REAL_CHUNK_ROWS, 1, vocab), trial_drafts, generator)
        accepted += int((n_accept == 1).sum())
    observed_accept = accepted / (REAL_CHUNKS * REAL_CHUNK_ROWS)
    llm.exit()
    return {"expected_accept_position_0": expected_accept, "observed_accept_position_0": observed_accept}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--prompt", default="Write a short paragraph about mountain weather.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = {"reference_cases": reference_cases(torch.device("cuda")), "real_distribution": real_distribution_case(args)}
    with open(args.output, "w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
    assert all(entry["total_variation"] < TOLERANCE for entry in report["reference_cases"])
    real = report["real_distribution"]
    assert abs(real["expected_accept_position_0"] - real["observed_accept_position_0"]) < REAL_TOLERANCE


if __name__ == "__main__":
    main()
