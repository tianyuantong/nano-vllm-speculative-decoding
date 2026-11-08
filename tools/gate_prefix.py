"""G-prefix (protocol v2.5 gates.G_prefix): a prefix-cache hit against prefills of the whole prompt.

A hit reuses K/V another request's prefill wrote: the target's for the shared blocks, and the draft's context K/V,
which mirrors them; the private part then attends to them through the paged prefill path. On the line's engine with
the prefix cache on, greedy, per panel and test prompt, four arms:
  reference  hashing off, the prompt prefilled alone in one step; then the line's own greedy rounds (a drafting line)
             or decode steps (k0). Each round: the draft's block forward, the target's verification, after_verify,
             the commit.
  floor      hashing off, the prompts prefilled `floor_batch` at a time and run as one batch: how far the logits move
             when only the batch changes (a batch short of a graph size pads).
  chunked    hashing off, the prompt prefilled alone in two steps split at the shared tokens, the second through the
             paged path over the prompt's own first chunk: the path of a hit without another request's K/V.
  hit        hashing on, the donor prefilled and kept, then the prompt alone, which must hit at least the shared tokens
             and at most its bound.
Every arm but the reference is forced to the reference's first token, drafts, verified lengths and commits, so each
round sees the same tokens. Compared per arm against the reference: the prefill's last-position logits and every
verification's (or decode step's) logits, the target's; every round's draft logits, before and, for DSpark, after the
Markov bias of the forced previous token, the draft's. Pass when chunked and hit each stay within floor_multiple x the
floor, target and draft, and agree on the argmax wherever the reference's top-2 margin exceeds margin.

Test prompts: the first per_category of each category, and the first two cut to end two tokens short of a block, so
the first rounds verify across a block boundary. A variable-length line verifies (prompt + round) mod (k+1) + 1
positions per round.

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
from nanovllm.layers.spec_sampler import sampling_tensors  # noqa: E402
from nanovllm.utils.context import reset_context  # noqa: E402


class CapturingSampler:
    """The target's sampler, keeping each call's fp32 logits [B, V]."""

    def __init__(self, sampler):
        self.sampler, self.logits = sampler, []

    def __call__(self, logits, sampling, generator=None):
        self.logits.append(logits.float().clone())
        return self.sampler(logits, sampling, generator)


def test_prompts(panel: list[dict], per_category: int, shared_tokens: int, block: int) -> list[dict]:
    """The first per_category entries of each category; two more entries cut so their last block is just full and
    so the first round crosses a block (length 0 and 254 mod 256); and the first entry's private part alone, which
    hits nothing. Each with the tokens it must hit and where its chunked prefill splits."""
    taken, picked, rest = {}, [], []
    for entry in panel:
        if taken.get(entry["category"], 0) < per_category:
            taken[entry["category"]] = taken.get(entry["category"], 0) + 1
            picked.append(entry)
        else:
            rest.append(entry)
    prompts = [{"id": entry["id"], "prompt_token_ids": entry["prompt_token_ids"], "shared": shared_tokens}
               for entry in picked]
    for entry, remainder in zip(rest[:2], (0, block - 2)):
        tokens = entry["prompt_token_ids"]
        cut = (len(tokens) - remainder) // block * block + remainder
        assert shared_tokens + block <= cut <= len(tokens)
        prompts.append({"id": f"{entry['id']}-cut{cut}", "prompt_token_ids": tokens[:cut], "shared": shared_tokens})
    private = picked[0]["prompt_token_ids"][shared_tokens:]
    prompts.append({"id": f"{picked[0]['id']}-private", "prompt_token_ids": private, "shared": 0})
    for prompt in prompts:
        prompt["split"] = prompt["shared"] or block
    return prompts


def prefill(llm, prompts: list[list[int]], k: int, split: int | None = None):
    """Prefill these prompts as one step, or one prompt in two steps split at `split`; their sequences and the last
    step's last-position logits [B, V]."""
    scheduler, capture = llm.scheduler, llm.model_runner.sampler
    capture.logits.clear()
    seqs = [llm.add_request(prompt, SamplingParams(temperature=0, max_tokens=64, ignore_eos=True)) for prompt in prompts]
    if split is not None:
        assert len(seqs) == 1
        budget, scheduler.max_num_batched_tokens = scheduler.max_num_batched_tokens, split
        llm.step()
        scheduler.max_num_batched_tokens = budget
        assert seqs[0].num_cached_tokens == split and seqs[0].num_completion_tokens == 0
    llm.step()
    assert not scheduler.waiting and all(seq.num_completion_tokens == 1 for seq in seqs)
    return seqs, capture.logits[-1]


def force_first(seqs, tokens: list[int]) -> None:
    for seq, token in zip(seqs, tokens):
        seq.token_ids[-1] = token
        seq.last_token = token


def commit(seqs, appended: list[list[int]]) -> None:
    """Scheduler.postprocess_speculative without the stop checks."""
    for seq, tokens in zip(seqs, appended):
        for token in tokens:
            seq.append_token(token)
        seq.target_kv.num_cached_tokens = len(seq) - 1


def speculative_round(llm, seqs, lengths: list[int] | None, forced: list[dict] | None) -> list[dict]:
    """One round of seqs as one batch, as SpeculativeDecoder.run_round runs it, greedy; drafts and commits each row's
    own unless forced. Per row: draft logits [k, V] (and after the Markov bias of the previous token for DSpark),
    the verification's logits [length, V], the drafts and what was committed."""
    decoder = llm.speculative
    drafter, k = decoder.drafter, decoder.num_drafts
    for seq in seqs:
        llm.scheduler.block_manager.reserve(seq, len(seq) + k)
    sampling = sampling_tensors(seqs, decoder.device)
    hidden = drafter._forward_graph(seqs) if drafter.graphs else drafter._forward_eager(seqs)
    base = drafter.model.compute_logits(hidden).float().view(len(seqs), k, -1).clone()
    if forced is None:
        drafts, _ = drafter._draw(seqs, hidden, sampling)
        drafts = drafts.clone()
    else:
        drafts = torch.tensor([row["drafts"] for row in forced], dtype=torch.int64, device=decoder.device)
    confidence = (drafter._confidence_logits(seqs, hidden, drafts).float().clone()
                  if decoder.variable_length else None)
    biased = None
    if drafter.model.markov_head is not None:
        anchors = torch.tensor([[seq.last_token] for seq in seqs], dtype=torch.int64, device=decoder.device)
        previous = torch.cat([anchors, drafts[:, :-1]], dim=1)
        biased = base + drafter.model.markov_head(previous).float()
    rows = [k + 1] * len(seqs) if lengths is None else lengths
    aux = decoder._aux_buffer(sum(rows))
    target = decoder.verify_logits(seqs, drafts, aux, lengths).float().clone()
    target = target.view(-1, target.shape[-1])
    drafter.after_verify(seqs, aux, lengths)
    if forced is None:
        n_accept, tail = decoder._accept(target if lengths is not None else target.view(len(seqs), k + 1, -1),
                                         drafts, None, sampling, lengths)
        appended = [drafts[row, :count].tolist() + [token]
                    for row, (count, token) in enumerate(zip(n_accept.tolist(), tail.tolist()))]
    else:
        appended = [row["appended"] for row in forced]
    commit(seqs, appended)
    starts = [sum(rows[:row]) for row in range(len(seqs))]
    return [{"draft": base[row], "draft_biased": None if biased is None else biased[row],
             "confidence": None if confidence is None else confidence[row:row + 1],
             "target": target[starts[row]:starts[row] + rows[row]], "drafts": drafts[row].tolist(),
             "appended": appended[row]} for row in range(len(seqs))]


def decode_step(llm, seqs, forced: list[int] | None) -> list[dict]:
    """One ordinary decode step of seqs as one batch; each row appends its own argmax unless forced."""
    runner = llm.model_runner
    for seq in seqs:
        llm.scheduler.block_manager.reserve(seq, len(seq))
    input_ids, positions = runner.prepare_decode(seqs)
    logits = runner.run_model(input_ids, positions, False).float()
    reset_context()
    tokens = forced if forced is not None else [int(row.argmax()) for row in logits]
    commit(seqs, [[token] for token in tokens])
    return [{"target": logits[row:row + 1].clone(), "appended": [tokens[row]]} for row in range(len(seqs))]


def observe(llm, prompts: list[dict], indices: list[int], k: int, gate: dict, reference: list[dict] | None,
            chunked: bool = False) -> list[dict]:
    """Prefill the prompts as one step (or one prompt split at its split), then gate.rounds rounds or decode steps;
    forced to the reference's tokens when one is given. Per prompt: its hit, first token, and every logit tensor in
    order, target and draft."""
    split = prompts[0]["split"] if chunked else None
    seqs, prefill_logits = prefill(llm, [prompt["prompt_token_ids"] for prompt in prompts], k, split)
    if reference is not None:
        force_first(seqs, [ref["first"] for ref in reference])
    out = [{"hit_tokens": seq.prefix_hit_tokens, "first": seq.last_token, "target": [prefill_logits[row:row + 1]],
            "draft": [], "confidence": [], "steps": []} for row, seq in enumerate(seqs)]
    for step in range(gate["rounds"]):
        forced = [ref["steps"][step] for ref in reference] if reference is not None else None
        if llm.speculative is None:
            rows = decode_step(llm, seqs, [row["appended"][0] for row in forced] if forced else None)
        else:
            lengths = ([(index + step) % (k + 1) + 1 for index in indices]
                       if llm.config.variable_length else None)
            rows = speculative_round(llm, seqs, lengths, forced)
        for record, row in zip(out, rows):
            record["target"].append(row["target"])
            record["draft"] += [tensor for tensor in (row.get("draft"), row.get("draft_biased")) if tensor is not None]
            record["confidence"] += [row["confidence"]] if row.get("confidence") is not None else []
            record["steps"].append({key: row[key] for key in ("drafts", "appended") if key in row})
    for seq in seqs:
        llm.scheduler._finish(seq)
    return out


def deltas(observed: list[dict], reference: list[dict], key: str) -> float:
    return max(((a - b).abs().max().item() for ob, ref in zip(observed, reference) for a, b in zip(ob[key], ref[key])),
               default=0.0)


def argmax_disagreements(observed: list[dict], reference: list[dict], key: str, margin: float) -> list[dict]:
    """Logit rows where the reference's top-2 margin exceeds margin and the argmax differs."""
    found = []
    for prompt, (ob, ref) in enumerate(zip(observed, reference)):
        for index, (a, b) in enumerate(zip(ob[key], ref[key])):
            top2 = b.topk(2, dim=-1).values
            differ = ((top2[:, 0] - top2[:, 1]) > margin) & (a.argmax(-1) != b.argmax(-1))
            found += [{"prompt": prompt, "tensor": index, "row": int(row)} for row in torch.nonzero(differ).flatten()]
    return found


def with_donor(llm, donor: list[int], k: int, run):
    """`run()` on a hash table that holds only a fresh donor's blocks, the donor prefilled and run 2 rounds first and
    kept until `run` returns: the shared blocks come from another request that went on decoding."""
    manager = llm.scheduler.block_manager
    manager.enable_prefix_cache = True
    manager.hash_to_block_id.clear()
    donor_seqs, _ = prefill(llm, [donor], k)
    for _ in range(2):
        if llm.speculative is None:
            decode_step(llm, donor_seqs, None)
        else:
            speculative_round(llm, donor_seqs, [k + 1] if llm.config.variable_length else None, None)
    try:
        return run()
    finally:
        for seq in donor_seqs:
            llm.scheduler._finish(seq)
        manager.hash_to_block_id.clear()
        manager.enable_prefix_cache = False


def batches(items: list, size: int) -> list[slice]:
    return [slice(start, start + size) for start in range(0, len(items), size)]


@torch.inference_mode()
def gate_panel(llm, panel: list[dict], donor: list[int], shared_tokens: int, k: int, gate: dict) -> dict:
    manager = llm.scheduler.block_manager
    prompts = test_prompts(panel, gate["per_category"], shared_tokens, manager.block_size)
    indices = list(range(len(prompts)))
    size = gate["floor_batch"]
    manager.enable_prefix_cache = False
    reference = [observe(llm, [prompt], [index], k, gate, None)[0] for index, prompt in enumerate(prompts)]
    arms = {"floor": [], "chunked": [], "hit": [], "hit_batch": []}
    for batch in batches(prompts, size):
        arms["floor"] += observe(llm, prompts[batch], indices[batch], k, gate, reference[batch])
    arms["chunked"] = [observe(llm, [prompt], [index], k, gate, [ref], chunked=True)[0]
                       for index, (prompt, ref) in enumerate(zip(prompts, reference))]
    arms["hit"] = with_donor(llm, donor, k, lambda: [observe(llm, [prompt], [index], k, gate, [ref])[0]
                                                    for index, (prompt, ref) in enumerate(zip(prompts, reference))])
    arms["hit_batch"] = with_donor(llm, donor, k, lambda: [row for batch in batches(prompts, size)
                                                          for row in observe(llm, prompts[batch], indices[batch], k,
                                                                             gate, reference[batch])])
    hashing_off = reference + arms["floor"] + arms["chunked"]
    result = {"prompts": [prompt["id"] for prompt in prompts], "shared_tokens": shared_tokens,
              "hits": {arm: [ob["hit_tokens"] for ob in arms[arm]] for arm in ("hit", "hit_batch")},
              "hits_as_designed": all(ob["hit_tokens"] == prompt["shared"] for arm in ("hit", "hit_batch")
                                      for ob, prompt in zip(arms[arm], prompts)),
              "hashing_off_hits": sum(ob["hit_tokens"] for ob in hashing_off)}
    passed = result["hits_as_designed"] and result["hashing_off_hits"] == 0
    for key in ("target", "draft", "confidence"):    # confidences are 7 binary logits: no argmax over them
        if not reference[0][key]:
            continue
        noise = deltas(arms["floor"], reference, key)
        entry = {"noise_floor": noise, "threshold": gate["floor_multiple"] * noise,
                 "hit_vs_chunked": deltas(arms["hit"], arms["chunked"], key)}
        for name in ("chunked", "hit", "hit_batch"):
            entry[name] = {"statistic": deltas(arms[name], reference, key),
                           "argmax_disagreements": (argmax_disagreements(arms[name], reference, key, gate["margin"])
                                                    if key != "confidence" else [])}
            passed &= entry[name]["statistic"] <= entry["threshold"] and not entry[name]["argmax_disagreements"]
        result[key] = entry
    result["passed"] = passed
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--line", required=True, help="a line of the protocol's gates.G_prefix")
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
    target, _, kwargs = engine_kwargs(protocol, line, budget_panel, 1 + gate["floor_batch"], protocol["seeds"]["dev"][0],
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
                      **{name: {key: {"floor": result[key]["noise_floor"],
                                      **{arm: result[key][arm]["statistic"] for arm in ("chunked", "hit", "hit_batch")}}
                                for key in ("target", "draft", "confidence") if key in result}
                         for name, result in results.items()}}))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
