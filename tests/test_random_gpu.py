"""CUDA frequency/edge checks of the shared engine functions, without Qwen.

After source review, run in the existing allocated GPU environment:
  python tests/test_random_gpu.py --result <new-output.json>
No CUDA/torch means NOT_RUN and exit 2, never a CPU fallback or a passing test.
The current patch tests primitives, not the still-missing LLMEngine integration.
"""

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

from random_sampling_cases import CASES, expected_distribution


DRAWS = 100_000
SEED = 1729
CHUNK = 4096
ALPHA = 0.001
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "nanovllm/layers/random_sampler.py"
CASES_SOURCE = Path(__file__).with_name("random_sampling_cases.py")


def load_primitives():
    # Load the actual source without importing optional model/transformers code.
    spec = importlib.util.spec_from_file_location("actual_random_sampler", SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def edge_checks(torch, sampler, device):
    def fp(rows):
        return torch.tensor(rows, dtype=torch.float32, device=device)

    # Fixed same-logits comparison; Python FP64 is only the independent oracle.
    logits_rows = [[-2.0, 0.0, 2.0, 4.0], [0.0, -math.inf, 0.0, -math.inf],
                   [80.0, 0.0, -80.0, -math.inf]]
    actual_probs = sampler.from_logits(fp(logits_rows), fp([1.0] * 3))
    sampler.require_valid_boundary(actual_probs.invalid)
    effective = (actual_probs.values.double() / actual_probs.mass[:, None]).tolist()
    max_abs, max_tv = 0.0, 0.0
    for logits_row, actual_row in zip(logits_rows, effective):
        weights = [math.exp(x - max(logits_row)) for x in logits_row]
        expected_row = [x / math.fsum(weights) for x in weights]
        errors = [abs(x - y) for x, y in zip(actual_row, expected_row)]
        max_abs, max_tv = max(max_abs, max(errors)), max(max_tv, sum(errors) / 2)
        assert all(actual == 0 for actual, expected in zip(actual_row, expected_row) if expected == 0)
    assert max_abs <= 1e-6 and max_tv <= 1e-6

    p = sampler.from_probs(fp([[0.0, 1.0], [0.4, 0.6], [0.7, 0.3], [1.0, 0.0]]))
    q = sampler.from_probs(fp([[1.0, 0.0], [0.4, 0.6], [0.4, 0.6], [1.0, 0.0]]))
    ids = torch.tensor([0, 1, 1, 1], dtype=torch.int64, device=device)
    u = torch.tensor([0.0, 0.999999, 0.75, 0.5], dtype=torch.float64, device=device)
    decision = sampler.accept(p, q, ids, u)
    assert decision.accepted.tolist() == [False, True, False, False]
    assert decision.invalid.tolist() == [False, False, False, True]
    # The q(x)=0 row must never silently become a valid correction.
    try:
        sampler.require_valid_boundary(decision.invalid)
    except FloatingPointError:
        pass
    else:
        raise AssertionError("invalid proposal escaped the publication boundary")

    same = sampler.from_probs(fp([[0.4, 0.6]]))
    inactive = sampler.residual(same, same, torch.tensor([False], device=device))
    impossible = sampler.residual(same, same, torch.tensor([True], device=device))
    assert inactive.invalid.tolist() == [False]
    assert impossible.invalid.tolist() == [True]
    for bad_id in (-1, 99):
        invalid_ids = sampler.accept(same, same, torch.tensor([bad_id], device=device),
                                     torch.tensor([0.0], dtype=torch.float64, device=device))
        assert invalid_ids.invalid.tolist() == [True]  # gather remained in bounds
        assert invalid_ids.accepted.tolist() == [False]

    for bad_uniform in (float("nan"), -0.1, 1.0, float("inf")):
        invalid_uniform = sampler.accept(
            same, same, torch.tensor([0], device=device),
            torch.tensor([bad_uniform], dtype=torch.float64, device=device))
        assert invalid_uniform.invalid.tolist() == [True]
        assert invalid_uniform.accepted.tolist() == [False]

    bad_temperatures = torch.tensor([0.0, -1.0, float("nan"), float("inf")],
                                    dtype=torch.float32, device=device)
    invalid_temperatures = sampler.from_logits(fp([[0.0, 1.0]] * 4), bad_temperatures)
    assert invalid_temperatures.invalid.tolist() == [True] * 4
    try:
        sampler.require_valid_boundary(invalid_temperatures.invalid)
    except FloatingPointError:
        pass
    else:
        raise AssertionError("invalid temperatures escaped the publication boundary")

    generator = torch.Generator(device=device).manual_seed(123)
    original = same.values.clone()
    first = sampler.draw(same, generator=generator)
    sampler.require_valid_boundary(first.invalid)
    assert torch.equal(same.values, original), "q was overwritten with random scores"
    generator.manual_seed(123)
    second = sampler.draw(same, generator=generator)
    assert torch.equal(first.token_ids, second.token_ids)

    # Effective row masses matter even when their error fits the input tolerance.
    p_scaled = sampler.from_probs(fp([[0.7000035047531128, 0.30000150203704834]]))
    q_scaled = sampler.from_probs(fp([[0.4000000059604645, 0.6000000238418579]]))
    midpoint = torch.tensor([0.500001], dtype=torch.float64, device=device)
    near = sampler.accept(p_scaled, q_scaled, torch.tensor([1], device=device), midpoint)
    sampler.require_valid_boundary(near.invalid)
    assert near.accepted.tolist() == [False], "used raw p/q rather than effective probabilities"

    bad = sampler.from_probs(fp([[float("nan"), 1.0], [-0.1, 1.1], [0.0, 0.0],
                                [float("inf"), 0.0], [0.2, 0.2]]))
    safe_sample = sampler.draw(bad, generator=generator)
    assert safe_sample.invalid.tolist() == [True] * 5
    assert all(0 <= x < 2 for x in safe_sample.token_ids.tolist())
    try:
        sampler.require_valid_boundary(safe_sample.invalid)
    except FloatingPointError:
        pass
    else:
        raise AssertionError("invalid probabilities were not rejected at boundary")
    return {"status": "PASS", "scope": "same logits, q ownership, effective mass, zero probability, masks, safe indices, boundary, replay",
            "same_logits": {"max_abs": max_abs, "max_tv": max_tv, "limit": 1e-6, "vocab": 4}}


def frequency_case(torch, sampler, device, case, index, bound):
    single = "p" in case
    p_rows = [case["p"]] if single else case["p_rows"]
    q_rows = [case["q"]] if single else case["q_rows"]
    cap = 1 if single else case["max_tokens"]
    eos = None if single else case["eos_token_id"]
    # log(0)=-inf is intentional and must produce exact zero support in softmax.
    p_logits = torch.tensor(p_rows, dtype=torch.float32, device=device).log()
    q_logits = torch.tensor(q_rows, dtype=torch.float32, device=device).log()
    # Once per case, then continuously advance; never reseed each sample/chunk.
    seeds = [SEED + 10 * index + role for role in range(3)]
    generators = [torch.Generator(device=device).manual_seed(seed) for seed in seeds]
    counts = Counter()
    for start in range(0, DRAWS, CHUNK):
        batch = min(CHUNK, DRAWS - start)
        state = torch.zeros(batch, dtype=torch.int64, device=device)
        active = torch.ones(batch, dtype=torch.bool, device=device)
        invalid = torch.zeros(batch, dtype=torch.bool, device=device)
        output = torch.full((batch, cap), -1, dtype=torch.int64, device=device)
        temperatures = torch.ones(batch, dtype=torch.float32, device=device)
        for position in range(cap):
            # Test glue selects the Markov prefix. All actual random decisions
            # come from the shared engine functions, not a reference sampler.
            p = sampler.from_logits(p_logits[state], temperatures)
            q = sampler.from_logits(q_logits[state], temperatures)
            proposal = sampler.draw(q, generator=generators[0])
            uniforms = torch.rand(batch, dtype=torch.float64, device=device, generator=generators[1])
            decision = sampler.accept(p, q, proposal.token_ids, uniforms)
            recovery_probs = sampler.residual(p, q, active & ~decision.accepted)
            recovery = sampler.draw(recovery_probs, generator=generators[2])
            invalid = invalid | proposal.invalid | decision.invalid | recovery.invalid
            token = torch.where(decision.accepted, proposal.token_ids, recovery.token_ids)
            output[:, position] = torch.where(active, token, output[:, position])
            if not single:
                state = torch.where(active, token + 1, state)
            if eos is not None:
                active = active & (token != eos)
        # Deliberate diagnostic boundary reads, not the timed S0/S1 path.
        sampler.require_valid_boundary(invalid)
        counts.update(tuple(token for token in row if token >= 0) for row in output.tolist())

    expected = expected_distribution(case)  # target-only CPU FP64 enumeration
    unexpected = [list(key) for key in counts if key not in expected]
    errors = {key: abs(counts[key] / DRAWS - p) for key, p in expected.items()}
    zero_violations = [list(key) for key, p in expected.items() if p == 0 and counts[key]]
    passed = not unexpected and not zero_violations and max(errors.values()) <= bound
    return {
        "name": case["name"], "status": "PASS" if passed else "FAIL", "draws": DRAWS,
        "seeds_by_role": dict(zip(["draft", "accept", "correction"], seeds)),
        "max_frequency_error": max(errors.values()), "bound": bound,
        "unexpected": unexpected, "zero_probability_violations": zero_violations,
        "categories": [{"tokens": list(key), "expected": p, "count": counts[key]} for key, p in expected.items()],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--compact", action="store_true", help="exercise the R1 production draw scratch path")
    parser.add_argument("--r2", action="store_true", help="exercise approved R2 GPU primitives")
    args = parser.parse_args()
    args.result.parent.mkdir(parents=True, exist_ok=True)
    # Refuse to replace any previous attempt, including a failed one.
    with args.result.open("x") as stream:
        stream.write("{}\n")
    result = {
        "status": "NOT_RUN", "scope": "shared GPU primitives only; no Qwen/KV/LLMEngine integration",
        "source": str(SOURCE), "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cases_source_sha256": hashlib.sha256(CASES_SOURCE.read_bytes()).hexdigest(),
        "case_definitions_sha256": hashlib.sha256(json.dumps(CASES, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "functions": ["from_logits", "from_probs", "draw", "accept", "residual", "require_valid_boundary"],
        "backend_required": "CUDA", "cpu_fallback": False,
        "draw_compact": args.compact, "r2_simplify": args.r2,
        "compile_mode": "eager torch tensor kernels, no torch.compile or CUDA Graph",
        "dtypes": {"p_q": "FP32", "row_mass_accept_uniform_residual": "FP64"},
        "qwen_loaded": False, "sampler_integration_covered": False,
        "frequency_composition": "single-token rejection repeated on fixed Markov rows",
        "not_covered": ["k>1 VERIFY scheduling", "initialization and bonus RNG integration",
                        "real KV lifecycle", "S0/S1 integration"],
    }
    code = 2
    try:
        import torch
        result["torch_version"] = torch.__version__
        result["cuda_version"] = torch.version.cuda
        if not torch.cuda.is_available():
            result["reason"] = "CUDA unavailable; CPU fallback prohibited"
        elif args.preflight_only:
            result["reason"] = "preflight only; no tensor tests executed"
        else:
            device = torch.device("cuda:0")
            result["device"] = torch.cuda.get_device_name(device)
            result["status"] = "RUNNING"
            sampler = load_primitives()
            if args.compact:
                from functools import partial
                sampler.draw = partial(sampler.draw, compact=True)
            if args.r2:
                from functools import partial
                for name in ("draw", "from_logits", "residual"):
                    setattr(sampler, name, partial(getattr(sampler, name), simplify=True))
            total_categories = sum(len(expected_distribution(case)) for case in CASES)
            bound = math.sqrt(math.log(2 * total_categories / ALPHA) / (2 * DRAWS))
            result["family"] = {"cases": len(CASES), "categories": total_categories, "alpha": ALPHA, "bound": bound}
            with torch.inference_mode():
                result["edge_checks"] = edge_checks(torch, sampler, device)
                result["cases"] = [frequency_case(torch, sampler, device, case, i, bound) for i, case in enumerate(CASES)]
            passed = all(case["status"] == "PASS" for case in result["cases"])
            result["status"] = "PASS" if passed else "FAIL"
            code = 0 if passed else 1
    except ModuleNotFoundError as error:
        result["reason"] = f"required module unavailable: {error.name}"
        if result["status"] == "RUNNING":
            result["status"] = "FAIL"
            code = 1
    except Exception as error:
        result["status"] = "FAIL"
        result["reason"] = f"{type(error).__name__}: {error}"
        code = 1
    finally:
        args.result.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": result["status"], "reason": result.get("reason"), "result": str(args.result)}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
