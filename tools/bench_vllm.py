"""vLLM baselines for protocol v2: every speculative line on the D48 panel, run on the PRO 6000.

  probe   every line once on 4 short prompts: did it run, and how many tokens did a round emit
  formal  every line at each concurrency, the panel run `repeats` times in one engine, one JSON per (line, concurrency)
  run     one (line, concurrency) in this process; probe and formal call it in a child process,
          because vLLM reads VLLM_USE_V2_MODEL_RUNNER when it is imported

vLLM imports only inside run_line, so the CPU tests can import this module.
"""
import argparse
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


def engine_kwargs(line: str, concurrency: int, protocol: dict, models_root: str) -> dict:
    vllm_settings = protocol["external"]["vllm"]
    target = f"{models_root}/{protocol['models']['target']}"
    kwargs = dict(model=target, tokenizer=target, dtype=protocol["models"]["dtype"], tensor_parallel_size=1,
                  max_model_len=protocol["max_model_len"], max_num_seqs=concurrency,
                  max_num_batched_tokens=protocol["max_num_batched_tokens"], enable_prefix_caching=False,
                  enable_chunked_prefill=True, async_scheduling=False, enforce_eager=False,
                  kv_cache_memory_bytes=vllm_settings["kv_cache_bytes"], generation_config="vllm",
                  seed=protocol["seeds"]["external"][0], disable_log_stats=False,
                  per_request_spec_decode_metrics="none")
    spec = speculative_config(line, protocol, models_root)
    if spec is not None:
        kwargs["speculative_config"] = spec
    if LINES[line].attention_backend is not None:
        kwargs["attention_backend"] = LINES[line].attention_backend
    return kwargs


def sampling_kwargs(protocol: dict, max_tokens: int) -> dict:
    """The recommended sampling, with no per-request seed: vLLM moves requests that carry one to slower samplers."""
    rec = protocol["sampling"]
    return dict(temperature=rec["temperature"], top_k=rec["top_k"], top_p=rec["top_p"], max_tokens=max_tokens,
                ignore_eos=rec["ignore_eos"])


def panel_prompts(panel: list[dict], concurrency: int) -> list[list[int]]:
    """Whole cycles of the panel, at least 4C requests (bench-v2 §2.2), so the batch stays full."""
    cycles = math.ceil(max(len(panel), 4 * concurrency) / len(panel))
    return [entry["prompt_token_ids"] for entry in panel] * cycles


def counter_totals(metrics) -> dict:
    """Sum every counter or gauge over its label sets; vector and histogram metrics are skipped."""
    totals = {}
    for metric in metrics:
        value = getattr(metric, "value", None)
        if isinstance(value, (int, float)):
            totals[metric.name] = totals.get(metric.name, 0) + value
    return totals


def run_line(line: str, concurrency: int, repeats: int, prompts: list[list[int]], protocol: dict,
             models_root: str, max_tokens: int, warmup: bool) -> dict:
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1" if LINES[line].runner == "v2" else "0"
    os.environ["VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN"] = str(
        protocol["external"]["vllm"]["adaptive_profile_context_len"])
    os.environ["HF_HUB_OFFLINE"] = "1"
    import torch
    import vllm
    from vllm import LLM, SamplingParams

    kwargs = engine_kwargs(line, concurrency, protocol, models_root)
    llm = LLM(**kwargs)
    requests = [dict(prompt_token_ids=prompt) for prompt in prompts]
    if warmup:
        llm.generate(requests[:concurrency], SamplingParams(**sampling_kwargs(protocol, protocol["warmup"]["max_tokens"])),
                     use_tqdm=False)
    runs = []
    for repeat in range(repeats):
        before = counter_totals(llm.get_metrics())
        start = time.perf_counter()
        outputs = llm.generate(requests, SamplingParams(**sampling_kwargs(protocol, max_tokens)), use_tqdm=False)
        seconds = time.perf_counter() - start
        after = counter_totals(llm.get_metrics())
        delta = {name: after[name] - before.get(name, 0) for name in after}
        spec = {name: delta.get(f"vllm:spec_decode_{name}", 0) for name in SPEC_COUNTERS}
        tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
        runs.append(dict(repeat=repeat, requests=len(requests), seconds=seconds, output_tokens=tokens,
                         tokens_per_second=tokens / seconds, preemptions=delta.get("vllm:num_preemptions", 0),
                         tokens_per_round=1 + spec["num_accepted_tokens"] / spec["num_drafts"] if spec["num_drafts"] else None,
                         **spec))
    return dict(line=line, concurrency=concurrency, runner=LINES[line].runner,
                sampling_kwargs=sampling_kwargs(protocol, max_tokens), engine=kwargs,
                env={name: os.environ[name] for name in ("VLLM_USE_V2_MODEL_RUNNER",
                                                         "VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN")},
                vllm=vllm.__version__, torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
                node=os.uname().nodename, source_id=os.environ.get("SOURCE_ID"), runs=runs)


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


def formal(out_dir: Path, protocol: dict, common: list[str]) -> int:
    vllm_settings = protocol["external"]["vllm"]
    failed = [f"{line}-C{concurrency}"
              for concurrency in vllm_settings["concurrency"]
              for line in LINES
              if not child(out_dir / f"{line}-C{concurrency}.json",
                           ["--line", line, "--concurrency", str(concurrency), "--repeats", str(vllm_settings["repeats"]),
                            "--max-tokens", str(protocol["sampling"]["max_tokens"]), "--warmup", *common])]
    print("FAILED:", failed if failed else "none", flush=True)
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["probe", "formal", "run"])
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
    args = parser.parse_args(argv)
    protocol = json.loads(args.protocol.read_text())
    common = ["--protocol", str(args.protocol), "--panel", str(args.panel), "--models-root", args.models_root]

    if args.command == "run":
        prompts = panel_prompts(json.loads(args.panel.read_text()), args.concurrency)[:args.prompts]
        result = run_line(args.line, args.concurrency, args.repeats, prompts, protocol, args.models_root,
                          args.max_tokens, args.warmup)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
        return 0
    args.out.mkdir(parents=True, exist_ok=True)
    if args.command == "probe":
        return probe(args.out, protocol, common)
    return formal(args.out, protocol, common)


if __name__ == "__main__":
    sys.exit(main())
