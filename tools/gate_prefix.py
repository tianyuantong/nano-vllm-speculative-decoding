"""G-prefix (protocol v2.5 gates.G_prefix): a prefix-cache hit against a prefill of the whole prompt.

A hit reuses K/V another request's prefill wrote: the target's for the shared blocks, and the draft's context K/V,
which mirrors them; the private part then attends to them through the paged prefill path. On the line's engine with
the prefix cache on, greedy, per panel:
  reference  hashing off: each test prompt prefilled alone: the prefill's last-position logits, the draft's first
             block forward over it (k rows), then k+1 target decode steps, each appending its own argmax.
  floor      hashing off: the test prompts prefilled 4 at a time, the same draft forward and decode steps 4 rows at a
             time, every row forced to its reference's first token and decode tokens: how far the logits move when
             only the batch changes.
  hit        hashing on: the donor prefilled and kept, then each test prompt prefilled alone, which must hit at least
             the panel's shared tokens; the same draft forward and decode steps, forced to the reference's tokens.
Pass when, target and draft each, the largest |hit - reference| is at most floor_multiple x the largest
|floor - reference|, and the argmax agrees wherever the reference's top-2 margin exceeds margin. A wrong block, a stale
block or a draft context with a gap moves the logits far past the floor.

  python tools/gate_prefix.py --line LINE --protocol FILE --models-root DIR --commit C --out FILE
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

FLOOR_BATCH = 4


class CapturingSampler:
    """The target's sampler, keeping each call's fp32 logits [B, V]."""

    def __init__(self, sampler):
        self.sampler, self.logits = sampler, []

    def __call__(self, logits, sampling, generator=None):
        self.logits.append(logits.float().clone())
        return self.sampler(logits, sampling, generator)


def test_prompts(panel: list[dict], per_category: int) -> list[list[int]]:
    """The first per_category prompts of each category, in panel order."""
    taken, prompts = {}, []
    for entry in panel:
        if taken.get(entry["category"], 0) < per_category:
            taken[entry["category"]] = taken.get(entry["category"], 0) + 1
            prompts.append(entry["prompt_token_ids"])
    return prompts


def prefill(llm, prompts: list[list[int]], k: int):
    """One prefill step of exactly these prompts; their sequences and the step's last-position logits [B, V]."""
    capture = llm.model_runner.sampler
    capture.logits.clear()
    seqs = [llm.add_request(prompt, SamplingParams(temperature=0, max_tokens=k + 2, ignore_eos=True)) for prompt in prompts]
    llm.step()
    assert not llm.scheduler.waiting and len(capture.logits) == 1 and all(seq.num_completion_tokens == 1 for seq in seqs)
    return seqs, capture.logits[0]


def force_first(seqs, tokens: list[int]) -> None:
    for seq, token in zip(seqs, tokens):
        seq.token_ids[-1] = token
        seq.last_token = token


def draft_logits(llm, seqs) -> torch.Tensor | None:
    """The drafter's first block forward after the prefill, through the graphs as a round runs it: [B, k, V]."""
    decoder = llm.speculative
    if decoder is None:
        return None
    for seq in seqs:
        llm.scheduler.block_manager.reserve(seq, len(seq) + decoder.num_drafts)
    drafter = decoder.drafter
    hidden = drafter._forward_graph(seqs) if drafter.graphs else drafter._forward_eager(seqs)
    return drafter.model.compute_logits(hidden).float().view(len(seqs), decoder.num_drafts, -1).clone()


def decode(llm, seqs, steps: int, forced: list[list[int]] | None = None) -> tuple[torch.Tensor, list[list[int]]]:
    """`steps` ordinary decode steps of seqs as one batch: [B, steps, V] and the tokens appended, each row's own argmax
    or forced[row][step]."""
    runner, logits, appended = llm.model_runner, [], [[] for _ in seqs]
    for step in range(steps):
        for seq in seqs:
            llm.scheduler.block_manager.reserve(seq, len(seq))
        input_ids, positions = runner.prepare_decode(seqs)
        out = runner.run_model(input_ids, positions, False).float()
        reset_context()
        logits.append(out)
        for row, seq in enumerate(seqs):
            token = forced[row][step] if forced is not None else int(out[row].argmax())
            seq.target_kv.num_cached_tokens = len(seq)
            seq.append_token(token)
            appended[row].append(token)
    return torch.stack(logits, dim=1), appended


def finish(llm, seqs) -> None:
    for seq in seqs:
        llm.scheduler._finish(seq)


def observe(llm, prompts: list[list[int]], k: int, reference: list[dict] | None) -> list[dict]:
    """Prefill the prompts as one step, then the draft forward and k+1 decode steps; forced to the reference's tokens
    when one is given."""
    seqs, prefill_logits = prefill(llm, prompts, k)
    if reference is not None:
        force_first(seqs, [ref["first"] for ref in reference])
    first = [seq.last_token for seq in seqs]
    hits = [seq.prefix_hit_tokens for seq in seqs]
    drafts = draft_logits(llm, seqs)
    steps, tokens = decode(llm, seqs, k + 1, [ref["tokens"] for ref in reference] if reference is not None else None)
    finish(llm, seqs)
    return [{"first": first[row], "tokens": tokens[row], "hit_tokens": hits[row],
             "target": torch.cat([prefill_logits[row:row + 1], steps[row]]),
             "draft": drafts[row] if drafts is not None else None} for row in range(len(seqs))]


def worst(observed: list[dict], reference: list[dict], key: str) -> float:
    return max((ob[key] - ref[key]).abs().max().item() for ob, ref in zip(observed, reference))


def argmax_disagreements(observed: list[dict], reference: list[dict], key: str, margin: float) -> list[dict]:
    """Rows where the reference's top-2 margin exceeds margin and the argmax differs."""
    found = []
    for prompt, (ob, ref) in enumerate(zip(observed, reference)):
        top2 = ref[key].topk(2, dim=-1).values
        decided = (top2[:, 0] - top2[:, 1]) > margin
        differ = ob[key].argmax(-1) != ref[key].argmax(-1)
        found += [{"prompt": prompt, "position": int(position)} for position in torch.nonzero(decided & differ).flatten()]
    return found


@torch.inference_mode()
def gate_panel(llm, panel: list[dict], donor: list[int], shared_tokens: int, k: int, gate: dict) -> dict:
    manager = llm.scheduler.block_manager
    prompts = test_prompts(panel, gate["per_category"])
    manager.enable_prefix_cache = False
    reference = [observe(llm, [prompt], k, None)[0] for prompt in prompts]
    floor = []
    for start in range(0, len(prompts), FLOOR_BATCH):
        floor += observe(llm, prompts[start:start + FLOOR_BATCH], k, reference[start:start + FLOOR_BATCH])
    manager.enable_prefix_cache = True
    donor_seqs, _ = prefill(llm, [donor], k)
    hit = [observe(llm, [prompt], k, [ref])[0] for prompt, ref in zip(prompts, reference)]
    finish(llm, donor_seqs)
    manager.enable_prefix_cache = False
    keys = ["target"] + (["draft"] if reference[0]["draft"] is not None else [])
    result = {"prompts": len(prompts), "shared_tokens": shared_tokens, "hit_tokens": [ob["hit_tokens"] for ob in hit],
              "hits_as_designed": all(ob["hit_tokens"] >= shared_tokens for ob in hit),
              "reference_hit_tokens": [ob["hit_tokens"] for ob in reference + floor]}
    passed = result["hits_as_designed"] and not any(result["reference_hit_tokens"])
    for key in keys:
        bound = gate["floor_multiple"] * worst(floor, reference, key)
        disagree = argmax_disagreements(hit, reference, key, gate["margin"])
        result[key] = {"noise_floor": worst(floor, reference, key), "statistic": worst(hit, reference, key),
                       "threshold": bound, "argmax_disagreements": disagree}
        passed &= result[key]["statistic"] <= bound and not disagree
    result["passed"] = passed
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--line", required=True, help="a drafting line of the protocol's gates.G_prefix")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--models-root", default="/content/models")
    parser.add_argument("--commit", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    protocol = json.loads(args.protocol.read_text())
    gate = protocol["gates"]["G_prefix"]
    line = protocol["lines"][args.line]
    panels = {name: json.loads((ROOT / protocol["panels"][name]["file"]).read_text()) for name in gate["panels"]}
    donors = {name: json.loads((ROOT / protocol["panels"][protocol["panels"][name]["warmup"]]["file"]).read_text())[0]
              for name in gate["panels"]}
    budget_panel = [entry for name in gate["panels"] for entry in panels[name] + [donors[name]]]
    target, _, kwargs = engine_kwargs(protocol, line, budget_panel, 1 + FLOOR_BATCH, protocol["seeds"]["dev"][0],
                                      args.models_root)
    kwargs.update(enable_prefix_cache=True)
    llm = LLM(target, **kwargs)
    llm.model_runner.sampler = CapturingSampler(llm.model_runner.sampler)
    results = {name: gate_panel(llm, panels[name], donors[name]["prompt_token_ids"],
                                protocol["panels"][name]["shared_tokens"], line["k"], gate)
               for name in gate["panels"]}
    llm.exit()
    passed = all(result["passed"] for result in results.values())
    report = {"gate": "G-prefix", "engine_commit": args.commit, "line": args.line, "passed": passed,
              "rule": gate["rule"], "panels": results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"gate": "G-prefix", "line": args.line, "passed": passed,
                      **{name: {key: {field: result[key][field] for field in ("statistic", "threshold")}
                                for key in ("target", "draft") if key in result} for name, result in results.items()}}))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
