"""vLLM baselines for protocol v2: every speculative line on the D48 panel, run on the PRO 6000.

  probe   every line once on 4 short prompts: did it run, and how many tokens did a round emit
  formal  every line at each concurrency, the panel run `repeats` times in one engine, one JSON per (line, concurrency)
  supplement  formal for the protocol's supplement: its lines and concurrencies, sampled with its settings
  run     one (line, concurrency) in this process; probe and formal call it in a child process,
          because vLLM reads VLLM_USE_V2_MODEL_RUNNER when it is imported. With --seed the engine samples from that seed:
          a same-machine comparison runs each seed in a new process, since repeats in one engine only advance one
          generator (protocol v2.4 samehost). With --panel-name the prompts are a protocol panel's, taken as
          bench_spec.py takes them, and the warm-up is that panel's own warm-up panel if it names one; --prefix-cache
          turns vLLM's prefix caching on; the result then records each request's prompt length, cached tokens and
          tokens (protocol v2.7 samehost_prefix)

vLLM imports only inside run_line, so the CPU tests can import this module.
"""
import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import NamedTuple

ROOT = Path(__file__).resolve().parents[1]


class Line(NamedTuple):
    runner: str                             # vLLM model runner, "v1" or "v2"
    method: str | None = None               # speculative method; None is the k=0 baseline
    draft: str | None = None                # key in protocol models.drafts
    k: int = 0
    adaptive: bool = False                  # DSpark's adaptive verification
    attention_backend: str | None = None    # None keeps vLLM's default for the GPU


# draft_model runs only on the v1 runner and dspark only on v2, so each runner gets its own k=0 baseline.
# DeepSpec's DFlash checkpoint is a Qwen3DSparkModel without the Markov head, so it runs through the dspark method:
# vLLM's dflash method puts one more position in the block and predicts nothing at the first one.
# Adaptive verification needs attention whose CUDA graphs take query lengths decided on the GPU; on sm_120 only
# TRITON_ATTN has them, so the adaptive line is compared with dspark_k7_triton.
LINES = {
    "k0_v1": Line("v1"),
    "draft_model_k3": Line("v1", "draft_model", "model_0p6b", 3),
    "k0_v2": Line("v2"),
    "dflash_k7": Line("v2", "dspark", "dflash", 7),
    "dspark_k7": Line("v2", "dspark", "dspark", 7),
    "dspark_k7_triton": Line("v2", "dspark", "dspark", 7, attention_backend="TRITON_ATTN"),
    "dspark_k7_adaptive": Line("v2", "dspark", "dspark", 7, adaptive=True, attention_backend="TRITON_ATTN"),
}
SPEC_COUNTERS = ("num_drafts", "num_draft_tokens", "num_accepted_tokens")
PROBE_PROMPTS, PROBE_TOKENS, PROBE_CONCURRENCY = 4, 128, 4


def speculative_config(line: str, protocol: dict, models_root: str) -> dict | None:
    spec = LINES[line]
    if spec.method is None:
        return None
    config = {"method": spec.method, "model": f"{models_root}/{protocol['models']['drafts'][spec.draft]}",
              "num_speculative_tokens": spec.k, "rejection_sample_method": "standard",
              "draft_sample_method": protocol["external"]["vllm"]["draft_sample_method"]}
    if spec.adaptive:
        config["enable_adaptive_verification"] = True
    return config


def engine_kwargs(line: str, concurrency: int, protocol: dict, models_root: str, seed: int | None = None,
                  prefix_cache: bool = False) -> dict:
    vllm_settings = protocol["external"]["vllm"]
    target = f"{models_root}/{protocol['models']['target']}"
    # vLLM's dummy runs give each slot max_num_batched_tokens / max_num_seqs tokens, which must fit max_model_len: at C=1
    # with 16384 on a Colab G4 the engine dies at startup with an illegal memory access. No step of a run reaches the cap.
    max_num_batched_tokens = min(protocol["max_num_batched_tokens"], concurrency * protocol["max_model_len"])
    kwargs = dict(model=target, tokenizer=target, dtype=protocol["models"]["dtype"], tensor_parallel_size=1,
                  max_model_len=protocol["max_model_len"], max_num_seqs=concurrency,
                  max_num_batched_tokens=max_num_batched_tokens, enable_prefix_caching=prefix_cache,
                  enable_chunked_prefill=True, async_scheduling=False, enforce_eager=False,
                  kv_cache_memory_bytes=vllm_settings["kv_cache_bytes"], generation_config="vllm",
                  seed=protocol["seeds"]["external"][0] if seed is None else seed, disable_log_stats=False,
                  per_request_spec_decode_metrics="none")
    spec = speculative_config(line, protocol, models_root)
    if spec is not None:
        kwargs["speculative_config"] = spec
    if LINES[line].attention_backend is not None:
        kwargs["attention_backend"] = LINES[line].attention_backend
    return kwargs


def sampling_kwargs(protocol: dict, max_tokens: int, supplement: bool = False) -> dict:
    """The recommended sampling, or the supplement's, with no per-request seed: vLLM moves requests that carry one
    to slower samplers."""
    rec = protocol["sampling"]
    settings = (protocol["external"]["vllm"]["supplement"]["sampling"] if supplement
                else dict(temperature=rec["temperature"], top_k=rec["top_k"], top_p=rec["top_p"]))
    # v2.7 turns detokenizing off, so both sides end where the tokens are back; EOS is a token check in the engine
    detokenize = {} if "detokenize" not in protocol["external"]["vllm"] else {"detokenize": protocol["external"]["vllm"]["detokenize"]}
    return dict(settings, max_tokens=max_tokens, ignore_eos=rec["ignore_eos"], **detokenize)


def panel_prompts(panel: list[dict], concurrency: int) -> list[list[int]]:
    """Whole cycles of the panel, at least 4C requests (bench-v2 §2.2), so the batch stays full."""
    cycles = math.ceil(max(len(panel), 4 * concurrency) / len(panel))
    return [entry["prompt_token_ids"] for entry in panel] * cycles


def protocol_panel_prompts(protocol: dict, panel_name: str, concurrency: int,
                           prefix_cache: bool) -> tuple[list[list[int]], list[list[int]]]:
    """The measured prompts and the warm-up prompts of a protocol panel, as bench_spec.py's load_panel and warmup_entries
    take them: "full_cycles" whole cycles of at least 4C, "prefix" the first max(48, 4C), a number that many, cycling;
    the warm-up the first C of the panel's own warm-up panel if it names one, else the first C of the measured prompts,
    cycled. With the prefix cache on the panel must not cycle, or a request would hit its own earlier copy."""
    spec = protocol["panels"][panel_name]
    entries = json.loads((ROOT / spec["file"]).read_text())
    take = spec["take"]
    count = take if isinstance(take, int) else {
        "full_cycles": len(entries) * math.ceil(max(len(entries), 4 * concurrency) / len(entries)),
        "prefix": max(48, 4 * concurrency)}[take]
    if prefix_cache and count > len(entries):
        raise ValueError(f"{count} requests from a panel of {len(entries)}: with the prefix cache on the panel must not cycle")
    measured = [entries[index % len(entries)]["prompt_token_ids"] for index in range(count)]
    if "warmup" not in spec:
        return measured, [measured[index % len(measured)] for index in range(concurrency)]
    warmup = json.loads((ROOT / protocol["panels"][spec["warmup"]]["file"]).read_text())
    if len(warmup) < concurrency:
        raise ValueError(f"the warm-up panel {spec['warmup']} has {len(warmup)} prompts, C is {concurrency}")
    return measured, [entry["prompt_token_ids"] for entry in warmup[:concurrency]]


def counter_totals(metrics) -> dict:
    """Sum every counter or gauge over its label sets; vector and histogram metrics are skipped."""
    totals = {}
    for metric in metrics:
        value = getattr(metric, "value", None)
        if isinstance(value, (int, float)):
            totals[metric.name] = totals.get(metric.name, 0) + value
    return totals


def histogram_totals(metrics) -> dict:
    """Each histogram's count and sum over its label sets."""
    totals = {}
    for metric in metrics:
        if hasattr(metric, "buckets"):
            count, total = totals.get(metric.name, (0, 0.0))
            totals[metric.name] = (count + metric.count, total + metric.sum)
    return totals


def engine_cache(llm) -> dict:
    """The cache settings the engine resolved: the engine core sends its block size and KV size back to this process
    once it is up (vllm/v1/engine/core_client.py, _apply_ready_response)."""
    config = llm.llm_engine.vllm_config
    cache, speculative = config.cache_config, config.speculative_config
    return dict(block_size=cache.block_size, num_gpu_blocks=cache.num_gpu_blocks,
                kv_cache_size_tokens=getattr(cache, "kv_cache_size_tokens", None),
                enable_prefix_caching=cache.enable_prefix_caching, hash_algo=cache.prefix_caching_hash_algo,
                eagle_block_drop=speculative.use_eagle_block_drop() if speculative is not None else None)


def gpu_memory_used_mib() -> int:
    """The GPU's memory in use, every process's: the engine runs in a process of its own."""
    return int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, check=True).stdout.splitlines()[0])


def vllm_manifest(commit: str | None) -> dict:
    """bench-v2's manifest fields for this venv; vLLM ships its own flash-attention."""
    import platform
    import torch
    import triton
    import vllm
    driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                            capture_output=True, text=True).stdout.strip()
    return {"gpu": torch.cuda.get_device_name(0), "driver": driver, "torch": torch.__version__, "cuda": torch.version.cuda,
            "vllm": vllm.__version__, "triton": triton.__version__, "python": platform.python_version(), "commit": commit}


def run_line(line: str, concurrency: int, repeats: int, prompts: list[list[int]], protocol: dict,
             models_root: str, max_tokens: int, warmup: bool, supplement: bool = False, seed: int | None = None,
             warmup_prompts: list[list[int]] | None = None, prefix_cache: bool = False, commit: str | None = None) -> dict:
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1" if LINES[line].runner == "v2" else "0"
    os.environ["VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN"] = str(
        protocol["external"]["vllm"]["adaptive_profile_context_len"])
    os.environ["HF_HUB_OFFLINE"] = "1"
    sys.path.insert(0, str(ROOT))
    from bench_spec import gpu_state, installed_packages    # subprocess and importlib only, no nanovllm
    import torch
    import vllm
    from vllm import LLM, SamplingParams

    state_start = gpu_state()
    kwargs = engine_kwargs(line, concurrency, protocol, models_root, seed, prefix_cache)
    llm = LLM(**kwargs)
    requests = [dict(prompt_token_ids=prompt) for prompt in prompts]
    if warmup:
        warm = requests[:concurrency] if warmup_prompts is None else [dict(prompt_token_ids=prompt) for prompt in warmup_prompts]
        llm.generate(warm, SamplingParams(**sampling_kwargs(protocol, protocol["warmup"]["max_tokens"], supplement)),
                     use_tqdm=False)
    # The engine sends a step's outputs before its post-step work, and this process cannot synchronize the engine's GPU:
    # v2.7 lets the warm-up's last milliseconds finish before the clock starts
    time.sleep(protocol["external"]["vllm"].get("settle_seconds", 0))
    params = SamplingParams(**sampling_kwargs(protocol, max_tokens, supplement))
    runs = []
    for repeat in range(repeats):
        metrics = llm.get_metrics()
        before, histograms_before = counter_totals(metrics), histogram_totals(metrics)
        start = time.perf_counter()
        outputs = llm.generate(requests, params, use_tqdm=False)
        seconds = time.perf_counter() - start
        memory_used = gpu_memory_used_mib()
        metrics = llm.get_metrics()
        after, histograms_after = counter_totals(metrics), histogram_totals(metrics)
        delta = {name: after[name] - before.get(name, 0) for name in after}
        histograms = {name: dict(count=count - histograms_before.get(name, (0, 0.0))[0],
                                 sum=total - histograms_before.get(name, (0, 0.0))[1])
                      for name, (count, total) in histograms_after.items()}
        spec = {name: delta.get(f"vllm:spec_decode_{name}", 0) for name in SPEC_COUNTERS}
        tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
        records = [dict(prompt_len=len(request["prompt_token_ids"]), cached_tokens=output.num_cached_tokens,
                        token_ids=list(output.outputs[0].token_ids), finish_reason=output.outputs[0].finish_reason)
                   for request, output in zip(requests, outputs)]
        runs.append(dict(repeat=repeat, requests=len(requests), seconds=seconds, output_tokens=tokens,
                         tokens_per_second=tokens / seconds, preemptions=delta.get("vllm:num_preemptions"),
                         tokens_per_round=1 + spec["num_accepted_tokens"] / spec["num_drafts"] if spec["num_drafts"] else None,
                         gpu_memory_used_mib=memory_used, counters=delta, histograms=histograms,
                         request_records=records, **spec))
    packages, packages_sha256 = installed_packages()
    return dict(line=line, concurrency=concurrency, runner=LINES[line].runner, prefix_cache=prefix_cache,
                supplement=supplement, sampling_kwargs=sampling_kwargs(protocol, max_tokens, supplement), engine=kwargs,
                engine_cache=engine_cache(llm),
                env={name: os.environ[name] for name in ("VLLM_USE_V2_MODEL_RUNNER",
                                                         "VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN")},
                vllm=vllm.__version__, torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
                node=os.uname().nodename, source_id=os.environ.get("SOURCE_ID"),
                manifest={**vllm_manifest(commit), "node": state_start["node"], "packages_sha256": packages_sha256},
                gpu_state_start=state_start, gpu_state_end=gpu_state(), runs=runs)


def child(out: Path, args: list[str]) -> bool:
    """Run `run` in a fresh interpreter; its log goes next to the JSON it writes."""
    with out.with_suffix(".log").open("w") as log:
        code = subprocess.run([sys.executable, __file__, "run", "--out", str(out), *args],
                              stdout=log, stderr=subprocess.STDOUT).returncode
    print(f"{out.stem}: {'ok' if code == 0 else f'FAILED (exit {code})'}", flush=True)
    return code == 0


def probe(out_dir: Path, protocol: dict, common: list[str]) -> int:
    """Every line once on a few short prompts; probe.json maps each line to its tokens per round, or "failed"."""
    probe_args = ["--concurrency", str(PROBE_CONCURRENCY), "--repeats", "1", "--max-tokens", str(PROBE_TOKENS),
                  "--prompts", str(PROBE_PROMPTS), *common]
    summary = {}
    for line in LINES:
        out = out_dir / f"probe-{line}.json"
        ran = child(out, ["--line", line, *probe_args])
        summary[line] = json.loads(out.read_text())["runs"][0]["tokens_per_round"] if ran else "failed"
    (out_dir / "probe.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return 1 if "failed" in summary.values() else 0


def formal(out_dir: Path, protocol: dict, common: list[str], supplement: bool = False) -> int:
    vllm_settings = protocol["external"]["vllm"]
    plan = vllm_settings["supplement"] if supplement else {"lines": list(LINES), "concurrency": vllm_settings["concurrency"]}
    flags = ["--supplement"] if supplement else []
    failed = [f"{line}-C{concurrency}"
              for concurrency in plan["concurrency"]
              for line in plan["lines"]
              if not child(out_dir / f"{line}-C{concurrency}.json",
                           ["--line", line, "--concurrency", str(concurrency), "--repeats", str(vllm_settings["repeats"]),
                            "--max-tokens", str(protocol["sampling"]["max_tokens"]), "--warmup", *flags, *common])]
    print("FAILED:", failed if failed else "none", flush=True)
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["probe", "formal", "supplement", "run"])
    parser.add_argument("--protocol", type=Path, default=ROOT / "benchmarks" / "protocol-v2.json")
    parser.add_argument("--panel", type=Path, default=ROOT / "benchmarks" / "inputs" / "panel-48.json")
    parser.add_argument("--models-root", default="/projects/nanovllm/models")
    parser.add_argument("--out", type=Path, required=True, help="directory for probe/formal, JSON file for run")
    parser.add_argument("--line", choices=list(LINES))
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--prompts", type=int, help="use only the first N prompts (probe)")
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--supplement", action="store_true", help="sample with the protocol's supplement settings")
    parser.add_argument("--seed", type=int, help="the engine's seed (run); default the protocol's external seed")
    parser.add_argument("--panel-name", help="a panel of the protocol, in place of --panel (run): its take rule and its "
                                             "own warm-up panel, as bench_spec.py")
    parser.add_argument("--prefix-cache", action="store_true", help="turn vLLM's prefix caching on (run)")
    parser.add_argument("--commit", help="the source tree's short hash, recorded with the result (run)")
    parser.add_argument("--group-repeat", type=int, default=1, help="2 for a group's second pass, the timing repeat (run)")
    args = parser.parse_args(argv)
    protocol_bytes = args.protocol.read_bytes()
    protocol = json.loads(protocol_bytes)
    common = ["--protocol", str(args.protocol), "--panel", str(args.panel), "--models-root", args.models_root]

    if args.command == "run":
        warmup_prompts, panel_record = None, {}
        if args.panel_name is None:
            prompts = panel_prompts(json.loads(args.panel.read_text()), args.concurrency)[:args.prompts]
        else:
            prompts, warmup_prompts = protocol_panel_prompts(protocol, args.panel_name, args.concurrency, args.prefix_cache)
            prompts = prompts[:args.prompts]
            spec = protocol["panels"][args.panel_name]
            warmup_file = protocol["panels"][spec["warmup"]]["file"] if "warmup" in spec else None
            panel_record = dict(panel=args.panel_name,
                                panel_sha256=hashlib.sha256((ROOT / spec["file"]).read_bytes()).hexdigest(),
                                warmup_panel_sha256=(hashlib.sha256((ROOT / warmup_file).read_bytes()).hexdigest()
                                                     if warmup_file else None))
        result = run_line(args.line, args.concurrency, args.repeats, prompts, protocol, args.models_root,
                          args.max_tokens, args.warmup, args.supplement, args.seed, warmup_prompts, args.prefix_cache,
                          args.commit)
        result.update(panel_record, protocol_sha256=hashlib.sha256(protocol_bytes).hexdigest(), commit=args.commit,
                      group_repeat=args.group_repeat)
        args.out.write_text(json.dumps(result) + "\n")    # compact: a run now carries every request's tokens
        return 0
    args.out.mkdir(parents=True, exist_ok=True)
    if args.command == "probe":
        return probe(args.out, protocol, common)
    return formal(args.out, protocol, common, supplement=args.command == "supplement")


if __name__ == "__main__":
    sys.exit(main())
