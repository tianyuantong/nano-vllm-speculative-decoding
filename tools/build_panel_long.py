"""Build the long-context panels of protocol v2.4 (long_context): L4k, L8k, L16k and L32k, by final Qwen-formatted length.

  L8k, L16k, L32k   SPEED-Bench's throughput_8k, 16k and 32k: code_completion and open_books, whose first turn the
                    dataset embeds, so nothing is rebuilt.
  L4k               SPEED-Bench has no such config: the upstream recipes on Ada-LEval's stackselect_4k (NIAH) and
                    BAMBOO's meetingpred_4k (creative writing), padded or cut to 4000 o200k_base tokens as upstream does.
                    The recipes seed their shuffle, so a question's answers and their order are fixed.

Candidates are ranked inside each category by sha256(salt + id), as K128's are. One fits when its formatted length lies
within window_fraction of the bucket. Each category takes its quota; the prefix of a panel keeps each share within one
prompt. Every entry records its length and recipe. These prompts are only reported on: length and content change
together, so a change of R is not all length's.

Runs outside the engine's venv, like build_panel_k128.py; SPEED_DIR holds the files of --speed-dir's help:
  ~/.venvs/k128/bin/python tools/build_panel_long.py --speed-dir SPEED_DIR --protocol benchmarks/protocol-v2.4.json \
      --output-dir benchmarks/inputs
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_panel_k128 import chat_encoder, interleave_proportional, load_recipes, select_candidates  # noqa: E402
from fetch_panels import normalize_category  # noqa: E402

RECIPE_TOKENS_4K = 4000


def stackselect_candidates(rows: list[dict]) -> list[dict]:
    return [{"question_id": f"stackselect_4k_{row['question_id']}", "category": "NIAH", "source": "stackselect_4k.json",
             "src_id": str(row["question_id"]), "row": row} for row in rows]


def bamboo_candidates(rows: list[dict]) -> list[dict]:
    return [{"question_id": f"meetingpred_4k_{index}", "category": "creative_writing", "source": "meetingpred_4k.jsonl",
             "src_id": str(index), "row": row} for index, row in enumerate(rows)]


def prompt_4k(candidate: dict, recipes: dict) -> tuple[str, str]:
    """The prompt of an L4k candidate and how it was obtained."""
    row = candidate["row"]
    if candidate["category"] == "NIAH":
        prompt = recipes["_generate_stackselect_prompt"](row["question"], row["all_answers"], row["answer"], RECIPE_TOKENS_4K)
        return recipes["_pad_or_truncate_prompt"](prompt, RECIPE_TOKENS_4K), "stackselect_upstream_4000"
    prompt = recipes["_generate_writing_prompt"]([row["content"]])
    return recipes["_pad_or_truncate_prompt"](prompt, RECIPE_TOKENS_4K), "bamboo_upstream_4000"


def embedded_candidates(rows: list[dict], categories: set[str]) -> list[dict]:
    """A long config's rows in the panel's categories; a row of theirs always embeds its first turn."""
    candidates = [{**row, "category": normalize_category(row["sub_category"])} for row in rows]
    return [candidate for candidate in candidates if candidate["category"] in categories]


def build_panel(candidates: list[dict], bucket: int, spec: dict, prompt_of: Callable[[dict], tuple[str, str]],
                encode: Callable[[str], list[int]]) -> list[dict]:
    """Each category's quota of candidates whose formatted length is within window_fraction of bucket, proportionally
    interleaved, with their lengths and recipes."""
    low, high = bucket * (1 - spec["window_fraction"]), bucket * (1 + spec["window_fraction"])
    encoded = {}

    def fits(candidate: dict) -> bool:
        prompt, recipe = prompt_of(candidate)
        token_ids = encode(prompt)
        encoded[candidate["question_id"]] = (recipe, token_ids)
        return low <= len(token_ids) <= high

    chosen = select_candidates(candidates, set(), set(), spec["salt"], spec["quotas"], fits)
    return [{"id": c["question_id"], "category": c["category"], "source": c["source"], "src_id": c["src_id"],
             "recipe": encoded[c["question_id"]][0], "formatted_tokens": len(encoded[c["question_id"]][1]),
             "prompt_token_ids": encoded[c["question_id"]][1]}
            for c in interleave_proportional(chosen, list(spec["quotas"]))]


def write_panel(panel: list[dict], path: Path, inputs: dict[str, Path], spec: dict) -> dict:
    path.write_text(json.dumps(panel) + "\n")
    provenance = {"panel": path.stem, "output_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "inputs": {name: hashlib.sha256(source.read_bytes()).hexdigest() for name, source in inputs.items()},
                  "formatted_tokens": [entry["formatted_tokens"] for entry in panel], **spec}
    path.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance


def main() -> None:
    import duckdb

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--speed-dir", type=Path, required=True,
                        help="throughput_8k.parquet, throughput_16k.parquet, throughput_32k.parquet, stackselect_4k.json, "
                             "meetingpred_4k.jsonl, upstream_speed.py, qwen-tokenizer/")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    directory = args.speed_dir
    specs = json.loads(args.protocol.read_text())["long_context"]
    encode = chat_encoder(directory / "qwen-tokenizer")
    tokenizer_files = {f"qwen-tokenizer/{name}": directory / "qwen-tokenizer" / name
                       for name in ("tokenizer.json", "tokenizer_config.json")}
    connection = duckdb.connect()
    for bucket in specs["buckets"]:
        name, tokens = bucket["name"], bucket["tokens"]
        spec = {key: specs[key] for key in ("salt", "window_fraction")} | {"quotas": bucket["quotas"]}
        if bucket["source"] == "recipes":
            recipes = load_recipes(directory / "upstream_speed.py")
            bamboo = [json.loads(line) for line in (directory / "meetingpred_4k.jsonl").read_text().splitlines() if line.strip()]
            candidates = stackselect_candidates(json.loads((directory / "stackselect_4k.json").read_text())) + bamboo_candidates(bamboo)

            def prompt_of(candidate: dict) -> tuple[str, str]:
                return prompt_4k(candidate, recipes)

            inputs = {file: directory / file for file in ("stackselect_4k.json", "meetingpred_4k.jsonl", "upstream_speed.py")}
        else:
            parquet = directory / f"{bucket['source']}.parquet"
            pool = connection.sql(f"select question_id, sub_category, source, src_id, turns from read_parquet('{parquet}')").fetchall()
            rows = [dict(zip(("question_id", "sub_category", "source", "src_id", "turns"), row)) for row in pool]
            candidates = embedded_candidates(rows, set(bucket["quotas"]))

            def prompt_of(candidate: dict) -> tuple[str, str]:
                return candidate["turns"][0], "embedded_first_turn"

            inputs = {parquet.name: parquet}
        panel = build_panel(candidates, tokens, spec, prompt_of, encode)
        provenance = write_panel(panel, args.output_dir / f"panel-{name}.json", inputs | tokenizer_files, spec)
        lengths = sorted(provenance["formatted_tokens"])
        print(name, provenance["output_sha256"], len(panel), lengths[0], lengths[len(lengths) // 2], lengths[-1])


if __name__ == "__main__":
    main()
