"""Build the K128 confirmation panel (bench-v2 plan Task 12) from SPEED-Bench throughput_1k.

Candidates are the pool's samples in the protocol's four categories (sub-categories mapped as D48 maps them),
ranked inside each category by sha256(salt + question_id). D48's question ids and (source, src_id) pairs are
excluded. A first turn that still says the benchmark data must be fetched is rebuilt with the upstream recipe
(NVIDIA Model-Optimizer specdec_bench, the SPEEDBench methods in upstream_speed.py): stackselect for NIAH, BAMBOO
meeting prediction for creative writing, padded or cut to 1000 o200k_base tokens. Prompts are Qwen3 chat-formatted
in non-thinking mode with D48's tokenizer files and template; a candidate fits when it is at most
max_formatted_tokens long. Each category takes its protocol quota; the panel interleaves the categories in the
protocol's order, so every prefix keeps their proportions.

Runs outside the engine's venv; SPEED_DIR holds the files listed in --speed-dir's help:
  uv venv ~/.venvs/k128 -p 3.12 && uv pip install -p ~/.venvs/k128 duckdb==1.4.3 tiktoken==0.11.0 tokenizers jinja2
  ~/.venvs/k128/bin/python tools/build_panel_k128.py --speed-dir SPEED_DIR \
      --exclude-panel benchmarks/inputs/panel-48.json --protocol benchmarks/protocol-v2.json \
      --output benchmarks/inputs/panel-K128.json
"""
import argparse
import ast
import hashlib
import json
import random
import re
import sys
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_panels import normalize_category  # noqa: E402

FETCH_MARKER = "FULL BENCHMARK DATA SHOULD BE FETCHED"
RECIPE_TOKENS = 1000
RECIPES = ("_generate_stackselect_prompt", "_pad_or_truncate_prompt", "_generate_writing_prompt")


def rank_key(salt: str, question_id: str) -> str:
    return hashlib.sha256((salt + question_id).encode()).hexdigest()


def select_candidates(candidates: list[dict], excluded_ids: set[str], excluded_sources: set[tuple[str, str]],
                      salt: str, quotas: dict[str, int], fits: Callable[[dict], bool]) -> dict[str, list[dict]]:
    """Per category in rank_key order, skipping excluded ids, excluded (source, src_id) pairs and candidates that do
    not fit, until the quota is full; ValueError when a category runs out."""
    chosen = {}
    for category, quota in quotas.items():
        ranked = sorted((c for c in candidates if c["category"] == category),
                        key=lambda c: rank_key(salt, c["question_id"]))
        picked = []
        for candidate in ranked:
            if len(picked) == quota:
                break
            if (candidate["question_id"] in excluded_ids or (candidate["source"], candidate["src_id"]) in excluded_sources
                    or not fits(candidate)):
                continue
            picked.append(candidate)
        if len(picked) < quota:
            raise ValueError(f"{category}: only {len(picked)} candidates fit, the quota is {quota}")
        chosen[category] = picked
    return chosen


def interleave(by_category: dict[str, list[dict]], order: list[str]) -> list[dict]:
    """Round robin over the categories in order until every list is used up."""
    queues = [list(by_category[category]) for category in order]
    panel = []
    while any(queues):
        for queue in queues:
            if queue:
                panel.append(queue.pop(0))
    return panel


def load_recipes(upstream: Path) -> dict:
    """The three SPEEDBench static methods of upstream_speed.py as plain functions (they need random and tiktoken)."""
    import tiktoken

    tree = ast.parse(upstream.read_text())
    bench = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SPEEDBench")
    functions = [node for node in bench.body if isinstance(node, ast.FunctionDef) and node.name in RECIPES]
    assert {function.name for function in functions} == set(RECIPES)
    for function in functions:
        function.decorator_list = []
    namespace = {"random": random, "tiktoken": tiktoken}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(upstream), "exec"), namespace)
    return namespace


def first_turn(row: dict, recipes: dict, stackselect: dict, bamboo: list[dict]) -> tuple[str, str]:
    """The prompt text of a candidate and how it was obtained."""
    prompt = row["turns"][0]
    if FETCH_MARKER not in prompt:
        return prompt, "embedded_first_turn"
    if row["sub_category"] == "NIAH":
        source = stackselect[row["src_id"]]
        prompt = recipes["_generate_stackselect_prompt"](source["question"], source["all_answers"], source["answer"],
                                                         RECIPE_TOKENS)
        return recipes["_pad_or_truncate_prompt"](prompt, RECIPE_TOKENS), "stackselect_upstream_1000"
    if "/RUCAIBox/BAMBOO/" in row["source"]:
        contents = [bamboo[int(index)]["content"] for index in re.findall(r"_(\d+)", row["src_id"])]
        prompt = recipes["_generate_writing_prompt"](contents)
        return recipes["_pad_or_truncate_prompt"](prompt, RECIPE_TOKENS), "bamboo_upstream_1000"
    raise ValueError(f"no recipe for {row['question_id']} from {row['source']}")


def chat_encoder(tokenizer_dir: Path) -> Callable[[str], list[int]]:
    """D48's formatting: the Qwen3 chat template, one user turn, non-thinking mode, no added special tokens."""
    import tokenizers
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    config = json.loads((tokenizer_dir / "tokenizer_config.json").read_text())
    tokenizer = tokenizers.Tokenizer.from_file(str(tokenizer_dir / "tokenizer.json"))
    template = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True).from_string(config["chat_template"])

    def encode(prompt: str) -> list[int]:
        text = template.render(messages=[{"role": "user", "content": prompt}], tools=None, enable_thinking=False,
                               add_generation_prompt=True)
        return tokenizer.encode(text, add_special_tokens=False).ids
    return encode


def main() -> None:
    import duckdb

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--speed-dir", type=Path, required=True,
                        help="throughput_1k.parquet, upstream_speed.py, stackselect_1k.json, meetingpred_4k.jsonl, qwen-tokenizer/")
    parser.add_argument("--exclude-panel", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.protocol.read_text())["panels"]["K128"]
    quotas, salt, limit = spec["quotas"], spec["salt"], spec["max_formatted_tokens"]
    directory = args.speed_dir
    connection = duckdb.connect()
    pool = connection.sql(f"select question_id, sub_category, source, src_id, turns "
                          f"from read_parquet('{directory / 'throughput_1k.parquet'}')").fetchall()
    rows = [dict(zip(("question_id", "sub_category", "source", "src_id", "turns"), row)) for row in pool]
    candidates = [{**row, "category": normalize_category(row["sub_category"])} for row in rows]
    excluded = json.loads(args.exclude_panel.read_text())
    excluded_ids = {entry["id"] for entry in excluded}
    excluded_sources = {(row["source"], row["src_id"]) for row in rows if row["question_id"] in excluded_ids}
    recipes = load_recipes(directory / "upstream_speed.py")
    stackselect = {}
    for row in json.loads((directory / "stackselect_1k.json").read_text()):
        stackselect.setdefault(str(row["question_id"]), row)            # upstream keeps the first match
    bamboo = [json.loads(line) for line in (directory / "meetingpred_4k.jsonl").read_text().splitlines() if line.strip()]
    encode = chat_encoder(directory / "qwen-tokenizer")
    encoded = {}

    def fits(candidate: dict) -> bool:
        prompt, recipe = first_turn(candidate, recipes, stackselect, bamboo)
        encoded[candidate["question_id"]] = (recipe, encode(prompt))
        return len(encoded[candidate["question_id"]][1]) <= limit

    chosen = select_candidates(candidates, excluded_ids, excluded_sources, salt, quotas, fits)
    panel = [{"id": c["question_id"], "category": c["category"], "source": c["source"], "src_id": c["src_id"],
              "recipe": encoded[c["question_id"]][0], "prompt_token_ids": encoded[c["question_id"]][1]}
             for c in interleave(chosen, list(quotas))]
    assert not {tuple(e["prompt_token_ids"]) for e in panel} & {tuple(e["prompt_token_ids"]) for e in excluded}
    args.output.write_text(json.dumps(panel) + "\n")
    inputs = [directory / name for name in ("throughput_1k.parquet", "upstream_speed.py", "stackselect_1k.json",
                                            "meetingpred_4k.jsonl", "qwen-tokenizer/tokenizer.json",
                                            "qwen-tokenizer/tokenizer_config.json")] + [args.exclude_panel]
    provenance = {"output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                  "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "inputs": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs},
                  "salt": salt, "quotas": quotas, "max_formatted_tokens": limit}
    args.output.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
