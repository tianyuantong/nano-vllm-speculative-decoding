"""G1-wide: verification logits against step-by-step decode logits on the same prefix (bench-v2 design §3).

Per prompt, 1 + max(B) copies are prefilled greedily, so all share the first token. The first copy decodes k+1
tokens through the ordinary decode path and records each step's logits; the other copies are verified with those
tokens as drafts, B at a time, through the production path (graphs, padding rows). Eight prompts from D48: the first
of each category, the two longest, and those two cut to 2046 and 1022 tokens, so the verified window crosses a
256-token block boundary. Pass: every |delta| <= 0.5, and the argmax agrees wherever the reference's top-2 margin
exceeds 0.5.

  python tools/gate_verify_logits.py --line LINE --models-root DIR --commit C --out FILE
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench_spec import engine_kwargs  # noqa: E402
from nanovllm import LLM, SamplingParams  # noqa: E402
from nanovllm.utils.context import reset_context  # noqa: E402

PROTOCOL = ROOT / "benchmarks" / "protocol-v2.json"


def decode_step(llm, seq) -> torch.Tensor:
    """One ordinary decode step for `seq` outside the scheduler; returns the fp32 logits row."""
    runner = llm.model_runner
    llm.scheduler.block_manager.reserve(seq, len(seq))
    input_ids, positions = runner.prepare_decode([seq])
    logits = runner.run_model(input_ids, positions, False)
    reset_context()
    seq.target_kv.num_cached_tokens = len(seq)
    seq.append_token(int(logits.argmax(dim=-1)))
    return logits[0].float()


def wide_prompts(panel: list[dict], crossing_lengths: list[int]) -> list[list[int]]:
    """The first prompt of each category, the two longest, and those two cut to the block-crossing lengths
    (the longest to the larger length)."""
    firsts = {}
    for entry in panel:
        firsts.setdefault(entry["category"], entry["prompt_token_ids"])
    longest = sorted((entry["prompt_token_ids"] for entry in panel), key=len, reverse=True)[:2]
    cut = [prompt[:length] for prompt, length in zip(longest, sorted(crossing_lengths, reverse=True))]
    return [*firsts.values(), *longest, *cut]


def verify_vs_decode(llm, prompt_ids: list[int], k: int, batch_sizes: tuple[int, ...]) -> list[dict]:
    """Per batch size B, B copies verified together against the decode reference. Returns per (B, position)
    {max_abs_delta, argmax_agrees, top2_margin_reference}, taken over the B rows."""
    params = SamplingParams(temperature=0, max_tokens=k + 2)
    for _ in range(max(batch_sizes) + 1):
        llm.add_request(prompt_ids, params)
    while llm.scheduler.waiting:
        llm.step()                                             # prefill; greedy, so every copy gets the same first token
    reference_seq, *copies = list(llm.scheduler.running)
    assert all(seq.last_token == reference_seq.last_token for seq in copies)
    reference = torch.stack([decode_step(llm, reference_seq) for _ in range(k + 1)])         # [k+1, V]
    drafts = torch.tensor([reference_seq.completion_token_ids[1:k + 1]], device=llm.speculative.device)
    for seq in copies:
        llm.scheduler.block_manager.reserve(seq, len(seq) + k)
    rows = []
    for batch_size in batch_sizes:
        logits = llm.speculative.verify_logits(copies[:batch_size], drafts.expand(batch_size, -1)).float()  # [B, k+1, V]
        for position in range(k + 1):
            top2 = reference[position].topk(2).values
            rows.append({"batch": batch_size, "position": position,
                         "max_abs_delta": (logits[:, position] - reference[position]).abs().max().item(),
                         "argmax_agrees": bool((logits[:, position].argmax(-1) == reference[position].argmax()).all()),
                         "top2_margin_reference": float(top2[0] - top2[1])})
    for seq in list(llm.scheduler.running):
        llm.scheduler._finish(seq)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--line", required=True, help="a protocol line with a drafter (model_k3, dflash_k7)")
    parser.add_argument("--models-root", default="/projects/nanovllm/models")
    parser.add_argument("--commit", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    protocol = json.loads(PROTOCOL.read_text())
    gate = protocol["gates"]["G1_wide"]
    panel = json.loads((ROOT / protocol["panels"]["D48"]["file"]).read_text())
    line, batch_sizes = protocol["lines"][args.line], tuple(gate["batch"])
    target, _, kwargs = engine_kwargs(protocol, line, panel, max(batch_sizes) + 1, protocol["seeds"]["dev"][0],
                                      args.models_root)
    llm = LLM(target, **kwargs)
    prompts = wide_prompts(panel, gate["block_crossing_lengths"])
    assert len(prompts) == gate["prompts"]
    results = [{"prompt_len": len(prompt), "rows": verify_vs_decode(llm, prompt, line["k"], batch_sizes)}
               for prompt in prompts]
    llm.exit()
    rows = [row for result in results for row in result["rows"]]
    worst = max(row["max_abs_delta"] for row in rows)
    argmax_ok = all(row["argmax_agrees"] for row in rows if row["top2_margin_reference"] > gate["margin"])
    report = {"gate": "G1-wide", "engine_commit": args.commit, "passed": worst <= gate["max_abs_delta"] and argmax_ok,
              "statistic": worst, "threshold": gate["max_abs_delta"],
              "threshold_source": "bench-v2 design §3: bf16 kernels differ by 1e-2..1e-1; the argmax must agree where "
                                  "the reference's top-2 margin exceeds 0.5",
              "n": len(rows), "details": {"line": args.line, "k": line["k"], "batch": list(batch_sizes),
                                          "argmax_agrees_where_decided": argmax_ok, "prompts": results}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("gate", "passed", "statistic", "threshold", "n")}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
