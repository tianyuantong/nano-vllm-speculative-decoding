"""Speculative decoding benchmarks on nano-vllm.

  --mode closed-loop  protocol v2: keep C requests in the engine, adding one as each finishes, until the panel is
                      used up (bench-v2 §1.1). Sampling, KV budgets, warm-up and the line's draft come from the protocol.
  --mode open-loop    protocol v2.4: requests arrive on a Poisson plan fixed before the run, whatever the engine does,
                      at most C in the engine, the rest queued. Latencies run from the planned arrival to the host's
                      delivery. tools/slo_report.py turns results into SLO capacity.

The v1 offline mode, one generate() over the 48-prompt panel with the 0.6B draft, is at tag results-batched-v1.

One invocation = one configuration = one JSON result with a manifest, per-step timings, per-request
records and acceptance statistics. tools/summarize_bench.py turns results into metrics.
"""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import subprocess
import time
from pathlib import Path
from typing import Callable, TypedDict

import torch

ROOT = Path(__file__).resolve().parent
SCHEMA = "bench-v2"
OPEN_LOOP_SCHEMA = "open-loop-v1"
CONFIDENCE_SCHEMA = "confidence-log-v1"
GC_FULL_OUT_OF_REACH = 10**9    # a third-generation threshold no run reaches (ctx_store diagnosis §5)
RESULT_KEYS = ("schema", "protocol_sha256", "manifest", "gpu_state_start", "gpu_state_end", "config",
               "wall_seconds", "steps", "phases", "requests", "speculative_stats", "preemptions")
QUERY_FIELDS = ("name", "power.limit", "clocks.max.sm", "clocks.max.memory",
                "ecc.mode.current", "temperature.gpu", "clocks_event_reasons.active")


class PanelEntry(TypedDict):
    id: str
    category: str
    prompt_token_ids: list[int]


class RequestRecord(TypedDict):
    id: str
    category: str
    prompt_len: int
    submit_step: int                  # engine.num_steps when add_request ran
    deliveries: list[list[int]]       # [[step, n_tokens], ...] in step order
    finish_step: int                  # the step that finished the request
    token_ids: list[int]


class OpenLoopRequestRecord(RequestRecord):
    arrival_seconds: float            # the plan's arrival; deliveries gain a third field, the host's seconds on that clock
    submit_seconds: float             # when add_request ran, between two steps


class ClosedLoopRecord(TypedDict):
    requests: list[RequestRecord]     # in panel order


class OpenLoopRecord(TypedDict):
    requests: list[OpenLoopRequestRecord]    # in arrival order


def load_panel(path: Path, concurrency: int, take: str | int) -> list[PanelEntry]:
    """take "full_cycles": L·⌈max(L, 4C)/L⌉ entries, whole cycles of the L-entry panel; "prefix": max(48, 4C), cycling;
    a number: that many entries, cycling."""
    panel = json.loads(Path(path).read_text())
    if isinstance(take, int):
        count = take
    else:
        count = {"full_cycles": len(panel) * math.ceil(max(len(panel), 4 * concurrency) / len(panel)),
                 "prefix": max(48, 4 * concurrency)}[take]
    return [panel[index % len(panel)] for index in range(count)]


def kv_budget_bytes(panel: list[PanelEntry], concurrency: int, max_tokens: int, k: int,
                    block_bytes: int, block_size: int = 256) -> int:
    """C × ⌈(longest prompt + max_tokens + k) / block_size⌉ blocks: with C requests in flight nothing is preempted."""
    longest = max(len(entry["prompt_token_ids"]) for entry in panel)
    return concurrency * math.ceil((longest + max_tokens + k) / block_size) * block_bytes


def kv_block_bytes(model_dir: str, block_size: int = 256) -> int:
    """One KV block of a bf16 model: K and V of every layer for block_size tokens, as ModelRunner.allocate_kv_cache sizes it."""
    config = json.loads((Path(model_dir) / "config.json").read_text())
    head_dim = config.get("head_dim") or config["hidden_size"] // config["num_attention_heads"]
    return 2 * config["num_hidden_layers"] * block_size * config["num_key_value_heads"] * head_dim * 2


def engine_kwargs(protocol: dict, line: dict, panel: list[PanelEntry], concurrency: int, seed: int,
                  models_root: str) -> tuple[str, str | None, dict]:
    """Target directory, draft directory and LLM keyword arguments of one protocol line. Every line gets the target
    KV budget of the largest k; a block draft's KV mirrors the target's blocks. The protocol names block drafts by
    their draft_type (dflash, dspark); a variable-length line (v2.2) verifies by the protocol's frozen schedule."""
    max_tokens = protocol["sampling"]["max_tokens"]
    budget_k = max(spec["k"] for spec in protocol["lines"].values())
    target = f"{models_root}/{protocol['models']['target']}"
    kwargs = dict(max_model_len=protocol["max_model_len"], max_num_seqs=concurrency,
                  max_num_batched_tokens=protocol["max_num_batched_tokens"], enable_prefix_cache=False,
                  kv_cache_memory_bytes=kv_budget_bytes(panel, concurrency, max_tokens, budget_k, kv_block_bytes(target)),
                  seed=seed, record_step_timings=True)
    draft = None
    if line["drafter"] is not None:
        draft = f"{models_root}/{protocol['models']['drafts'][line['drafter']]}"
        kwargs.update(draft_model=draft, num_speculative_tokens=line["k"], draft_type=line["drafter"])
        if line.get("variable_length"):
            kwargs.update(variable_length=True, verify_schedule=str(ROOT / protocol["verify_schedule"]["file"]))
    if line.get("round_robin_prefill"):
        kwargs["round_robin_prefill"] = True
    return target, draft, kwargs


def record_deliveries(in_flight: list[list], step: int, *stamp: float) -> list[list]:
    """Each in-flight request's new tokens as one delivery [step, n, *stamp]; returns the requests still unfinished."""
    for item in in_flight:
        record, seq, delivered = item
        count = seq.num_completion_tokens    # an int; completion_token_ids would copy a list per request per step
        if count > delivered:
            record["deliveries"].append([step, count - delivered, *stamp])
            item[2] = count
        if seq.is_finished:
            record["finish_step"] = step
            record["token_ids"] = seq.completion_token_ids
    return [item for item in in_flight if not item[1].is_finished]


def run_closed_loop(llm, panel: list[PanelEntry], params, concurrency: int) -> ClosedLoopRecord:
    """Before every step() top the engine up to C requests; after it, record each in-flight request's new tokens
    as one delivery. llm only needs add_request / step / num_steps, so the tests drive a fake engine."""
    records, in_flight = [], []    # in_flight items: [record, sequence, tokens delivered so far]
    next_entry = 0
    while in_flight or next_entry < len(panel):
        while len(in_flight) < concurrency and next_entry < len(panel):
            entry = panel[next_entry]
            record = RequestRecord(id=entry["id"], category=entry["category"], prompt_len=len(entry["prompt_token_ids"]),
                                   submit_step=llm.num_steps, deliveries=[], finish_step=-1, token_ids=[])
            in_flight.append([record, llm.add_request(entry["prompt_token_ids"], params), 0])
            records.append(record)
            next_entry += 1
        step = llm.num_steps
        llm.step()
        in_flight = record_deliveries(in_flight, step)
    return ClosedLoopRecord(requests=records)


def poisson_arrivals(count: int, rate: float, seed: int) -> list[float]:
    """`count` arrival times in seconds at `rate` per second, from a generator of their own: every line of a (rate,
    seed) gets the same plan, and the engine's sampling never touches it."""
    generator, now, arrivals = random.Random(seed), 0.0, []
    for _ in range(count):
        now += generator.expovariate(rate)
        arrivals.append(now)
    return arrivals


def run_open_loop(llm, panel: list[PanelEntry], params, arrivals: list[float], drain_seconds: float,
                  clock: Callable[[], float] = time.perf_counter, sleep: Callable[[float], None] = time.sleep) -> OpenLoopRecord:
    """Before every step add every request that is due, however full the engine is: a busy engine makes requests wait
    and never moves the plan. After it, stamp each in-flight request's new tokens with the host's clock, which is when
    the API would hand them over. A request unfinished drain_seconds after the last arrival keeps finish_step -1 and
    the tokens it has. llm only needs add_request / step / num_steps, so the tests drive a fake engine."""
    records, in_flight = [], []
    next_entry = 0
    origin = clock()
    deadline = arrivals[-1] + drain_seconds
    while (in_flight or next_entry < len(panel)) and clock() - origin < deadline:
        now = clock() - origin
        while next_entry < len(panel) and arrivals[next_entry] <= now:
            entry = panel[next_entry]
            record = OpenLoopRequestRecord(id=entry["id"], category=entry["category"], prompt_len=len(entry["prompt_token_ids"]),
                                           submit_step=llm.num_steps, deliveries=[], finish_step=-1, token_ids=[],
                                           arrival_seconds=arrivals[next_entry], submit_seconds=now)
            in_flight.append([record, llm.add_request(entry["prompt_token_ids"], params), 0])
            records.append(record)
            next_entry += 1
        if not in_flight:
            sleep(arrivals[next_entry] - now)
            continue
        step = llm.num_steps
        llm.step()
        in_flight = record_deliveries(in_flight, step, clock() - origin)
    for record, seq, _ in in_flight:
        record["token_ids"] = seq.completion_token_ids
    return OpenLoopRecord(requests=records)


def gpu_state(run=subprocess.run) -> dict:
    """QUERY_FIELDS of the first GPU and the node name; recorded before and after a run."""
    query = run(["nvidia-smi", f"--query-gpu={','.join(QUERY_FIELDS)}", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=True).stdout.splitlines()[0]
    node = run(["hostname"], capture_output=True, text=True, check=True).stdout.strip()
    return {**dict(zip(QUERY_FIELDS, (value.strip() for value in query.split(",")))), "node": node}


def assert_gpu(state: dict, expected_name: str) -> None:
    if state["name"] != expected_name:
        print(f"wrong GPU: {state['name']!r}; the protocol expects {expected_name!r}")
        raise SystemExit(3)


def installed_packages(distributions=importlib.metadata.distributions) -> tuple[str, str]:
    """Sorted `name==version` lines of every installed distribution, and their SHA256. The envs have no pip."""
    text = "".join(sorted(f"{dist.metadata['Name']}=={dist.version}\n" for dist in distributions()))
    return text, hashlib.sha256(text.encode()).hexdigest()


def build_result(schema: str = SCHEMA, **parts) -> dict:
    return {"schema": schema, **{key: parts[key] for key in RESULT_KEYS[1:]}}


def manifest(commit: str) -> dict:
    import flash_attn
    import triton
    driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                            capture_output=True, text=True).stdout.strip()
    return {"gpu": torch.cuda.get_device_name(), "driver": driver, "torch": torch.__version__, "cuda": torch.version.cuda,
            "flash_attn": flash_attn.__version__, "triton": triton.__version__, "python": platform.python_version(),
            "commit": commit}


def parse_args(argv: list[str], mode: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--protocol", type=Path, default=ROOT / "benchmarks" / "protocol-v2.json")
    parser.add_argument("--panel", required=True, help="a key of the protocol's panels (D48, K128)")
    parser.add_argument("--line", required=True, help="a key of the protocol's lines (k0, model_k3, ...)")
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--models-root", default="/projects/nanovllm/models")
    parser.add_argument("--commit", required=True, help="git commit of the synced tree (the sync step refuses a dirty tree)")
    parser.add_argument("--output", type=Path, required=True)
    if mode == "closed-loop":
        parser.add_argument("--repeat", type=int, default=1, help="2 for the timing-noise repeat")
        parser.add_argument("--max-model-len", type=int, help="in place of the protocol's: the long-context runs")
        parser.add_argument("--record-confidences", action="store_true",
                            help="keep every round's DSpark confidence logits and accepted counts in "
                                 "<output>.confidences.json (job A's calibration; the extra copy each round makes it "
                                 "no timing run)")
    else:
        parser.add_argument("--rate", type=float, required=True, help="planned arrivals per second")
        parser.add_argument("--requests", type=int, required=True, help="planned arrivals, the panel cycled")
    return parser.parse_args(argv)


def sampling_kwargs(protocol: dict) -> dict:
    sampling = protocol["sampling"]
    return dict(temperature=sampling["temperature"], top_k=sampling["top_k"], top_p=sampling["top_p"],
                ignore_eos=sampling["ignore_eos"])


def warmed_up_engine(protocol: dict, line: dict, panel: list[PanelEntry], args: argparse.Namespace, **overrides):
    """The line's engine after the protocol's warm-up, with its target and draft directories and LLM keyword arguments."""
    from nanovllm import LLM, SamplingParams

    target, draft, kwargs = engine_kwargs(protocol, line, panel, args.concurrency, args.seed, args.models_root)
    kwargs.update(overrides)
    llm = LLM(target, **kwargs)
    warmup = [panel[index % len(panel)] for index in range(args.concurrency)]
    run_closed_loop(llm, warmup, SamplingParams(max_tokens=protocol["warmup"]["max_tokens"], **sampling_kwargs(protocol)),
                    args.concurrency)
    return llm, target, draft, kwargs


def base_config(args: argparse.Namespace, mode: str, protocol: dict, line: dict, num_requests: int, target: str,
                draft: str | None, kwargs: dict) -> dict:
    return {"mode": mode, "panel": args.panel, "line": args.line, "drafter": line["drafter"],
            "num_speculative_tokens": line["k"], "concurrency": args.concurrency, "seed": args.seed,
            "num_requests": num_requests,
            "sampling": {**sampling_kwargs(protocol), "max_tokens": protocol["sampling"]["max_tokens"]},
            "target": target, "draft": draft,
            **{key: kwargs.get(key) for key in ("max_model_len", "max_num_batched_tokens", "kv_cache_memory_bytes",
                                                "draft_type", "variable_length", "record_confidences")}}


def write_result(args: argparse.Namespace, llm, schema: str, protocol_text: str, panel_path: Path, kwargs: dict, config: dict,
                 state_start: dict, wall_seconds: float, record: ClosedLoopRecord | OpenLoopRecord, **extra) -> None:
    steps, phases = llm.step_timings(), llm.phase_timings()
    speculative_stats = dict(llm.speculative.stats) if llm.speculative is not None else None
    state_end = gpu_state()
    packages, packages_sha256 = installed_packages()
    schedule_sha256 = (hashlib.sha256(Path(kwargs["verify_schedule"]).read_bytes()).hexdigest()
                       if "verify_schedule" in kwargs else None)
    result = build_result(
        schema=schema,
        protocol_sha256=hashlib.sha256(protocol_text.encode()).hexdigest(),
        manifest={**manifest(args.commit), "node": state_start["node"], "packages_sha256": packages_sha256,
                  "panel_sha256": hashlib.sha256(panel_path.read_bytes()).hexdigest(),
                  "verify_schedule_sha256": schedule_sha256},
        gpu_state_start=state_start, gpu_state_end=state_end, config=config, wall_seconds=wall_seconds,
        steps=steps, phases=phases, requests=record["requests"], speculative_stats=speculative_stats,
        preemptions=llm.scheduler.num_preemptions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({**result, **extra}))
    args.output.with_name("packages.txt").write_text(packages)


def closed_loop(args: argparse.Namespace) -> None:
    from nanovllm import SamplingParams

    protocol_text = args.protocol.read_text()
    protocol = json.loads(protocol_text)
    state_start = gpu_state()
    assert_gpu(state_start, protocol["gpu"]["name"])
    panel_spec = protocol["panels"][args.panel]
    panel_path = ROOT / panel_spec["file"]
    panel = load_panel(panel_path, args.concurrency, panel_spec["take"])
    line = protocol["lines"][args.line]
    overrides = {"record_confidences": True} if args.record_confidences else {}
    if args.max_model_len is not None:
        overrides["max_model_len"] = args.max_model_len
    llm, target, draft, kwargs = warmed_up_engine(protocol, line, panel, args, **overrides)
    params = SamplingParams(max_tokens=protocol["sampling"]["max_tokens"], **sampling_kwargs(protocol))
    llm.start_measurement(initial_group=args.concurrency)
    wall_start = time.perf_counter()
    record = run_closed_loop(llm, panel, params, args.concurrency)
    torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - wall_start
    config = {**base_config(args, "closed-loop", protocol, line, len(panel), target, draft, kwargs), "repeat": args.repeat}
    write_result(args, llm, SCHEMA, protocol_text, panel_path, kwargs, config, state_start, wall_seconds, record)
    if args.record_confidences:
        log = {"schema": CONFIDENCE_SCHEMA, "line": args.line, "panel": args.panel, "concurrency": args.concurrency,
               "seed": args.seed, "commit": args.commit, "rounds": llm.speculative.confidence_log}
        args.output.with_suffix(".confidences.json").write_text(json.dumps(log))
    tokens = sum(len(request["token_ids"]) for request in record["requests"])
    print(f"{args.output}: {tokens} tokens in {wall_seconds:.1f}s wall")
    llm.exit()


def open_loop(args: argparse.Namespace) -> None:
    """Long runs keep per-step events but no per-phase ones, and run without a full garbage collection: the run's
    bookkeeping would otherwise stop the process for 177-559 ms, which is the tail this mode measures. Only this mode
    does (ctx_store diagnosis §5; gc.freeze() leaves the bookkeeping growing)."""
    from nanovllm import SamplingParams

    protocol_text = args.protocol.read_text()
    protocol = json.loads(protocol_text)
    state_start = gpu_state()
    assert_gpu(state_start, protocol["gpu"]["name"])
    panel_path = ROOT / protocol["panels"][args.panel]["file"]
    entries = json.loads(panel_path.read_text())
    panel = [entries[index % len(entries)] for index in range(args.requests)]
    line = protocol["lines"][args.line]
    llm, target, draft, kwargs = warmed_up_engine(protocol, line, panel, args, record_phase_timings=False)
    params = SamplingParams(max_tokens=protocol["sampling"]["max_tokens"], **sampling_kwargs(protocol))
    arrivals = poisson_arrivals(args.requests, args.rate, args.seed)
    gc.collect()
    thresholds = gc.get_threshold()
    gc.set_threshold(thresholds[0], thresholds[1], GC_FULL_OUT_OF_REACH)
    full_collections = gc.get_stats()[2]["collections"]
    llm.start_measurement()
    wall_start = time.perf_counter()
    record = run_open_loop(llm, panel, params, arrivals, protocol["open_loop"]["drain_seconds"])
    torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - wall_start
    full_collections = gc.get_stats()[2]["collections"] - full_collections
    gc.set_threshold(*thresholds)
    config = {**base_config(args, "open-loop", protocol, line, len(panel), target, draft, kwargs), "rate": args.rate}
    write_result(args, llm, OPEN_LOOP_SCHEMA, protocol_text, panel_path, kwargs, config, state_start, wall_seconds, record,
                 full_collections=full_collections)
    finished = sum(request["finish_step"] >= 0 for request in record["requests"])
    print(f"{args.output}: {finished}/{args.requests} finished in {wall_seconds:.1f}s wall")
    llm.exit()


def main(argv: list[str] | None = None) -> None:
    mode_parser = argparse.ArgumentParser(add_help=False)
    mode_parser.add_argument("--mode", choices=["closed-loop", "open-loop"], default="closed-loop")
    known, rest = mode_parser.parse_known_args(argv)
    (closed_loop if known.mode == "closed-loop" else open_loop)(parse_args(rest, known.mode))


if __name__ == "__main__":
    main()
