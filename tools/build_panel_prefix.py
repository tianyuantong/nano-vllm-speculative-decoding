"""Build the shared-prefix panels of protocol v2.5 (prefix): S0, S512, S1024, S2048 and their warm-up panels.

Private parts: candidates of SPEED-Bench throughput_1k picked as K128b's are (build_panel_k128.py: salt, quotas,
proportional interleave, D48's formatting, the upstream recipes for first turns that must be fetched), excluding the
question ids, (source, src_id) pairs and prompts of the panels the protocol lists. Each first turn starts with the
protocol's tag of its question id, so no two private parts share a whole block: NIAH's recipe template would otherwise
share its first one. The first `measured` of them are measured, the rest only warm up.

Shared part: P tokens in front of every private part, a system turn of Qwen3's template,
  <|im_start|>system\\n  intro + a fixed document, cut to fit  <|im_end|>\\n
followed by the private part unchanged, since a user turn after a system turn is token for token the user turn alone.
P is a multiple of the block size, so the shared part fills P / 256 whole blocks. P = 0 is the private part alone.
The document is the first turn of one throughput_8k row after the protocol's marker, line ends as \\n.

Each panel records, per entry, its shared and private lengths, and how many whole blocks of its private part equal
another private part's, which must be none, so with the cache on a request hits exactly P tokens.

Runs outside the engine's venv, like build_panel_k128.py:
  ~/.venvs/k128/bin/python tools/build_panel_prefix.py --speed-dir SPEED_DIR --long-dir LONG_DIR \\
      --exclude-panel D48=panel-48.json K128=panel-K128.json K128b=panel-K128b.json \\
      --protocol benchmarks/protocol-v2.5.json --output-dir OUT
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_panel_k128 import (chat_encoder, first_turn, interleave_proportional, load_recipes,  # noqa: E402
                              select_candidates)
from fetch_panels import normalize_category  # noqa: E402

BLOCK_SIZE = 256


def system_block(head: list[int], body: list[int], tail: list[int], length: int) -> list[int]:
    """Exactly `length` tokens: head, the first tokens of body, tail. Empty for length 0."""
    if length == 0:
        return []
    room = length - len(head) - len(tail)
    if not 0 < room <= len(body):
        raise ValueError(f"a shared part of {length} tokens needs 1..{len(body)} body tokens, not {room}")
    return head + body[:room] + tail


def whole_blocks(token_ids: list[int]) -> list[tuple[int, ...]]:
    return [tuple(token_ids[start:start + BLOCK_SIZE]) for start in range(0, len(token_ids) - BLOCK_SIZE + 1, BLOCK_SIZE)]


def shared_private_blocks(private: list[list[int]]) -> list[int]:
    """Per private part, the most leading whole blocks it has in common with another one: the engine hashes a chain,
    so block j only hits if blocks 0..j all match. Capped at its own blocks less the last, which is never a hit."""
    chains = [whole_blocks(tokens) for tokens in private]
    shared = []
    for index, chain in enumerate(chains):
        cap = (len(private[index]) + BLOCK_SIZE - 1) // BLOCK_SIZE - 1
        best = 0
        for other, other_chain in enumerate(chains):
            if other != index:
                common = 0
                for mine, theirs in zip(chain, other_chain):
                    if mine != theirs:
                        break
                    common += 1
                best = max(best, common)
        shared.append(min(best, cap))
    return shared


def document_tokens(long_dir: Path, spec: dict, encode_text) -> list[int]:
    import duckdb

    parquet = long_dir / f"{spec['source']}.parquet"
    connection = duckdb.connect()
    rows = connection.execute(f"select turns from read_parquet('{parquet}') where question_id = ?",
                              [spec["question_id"]]).fetchall()
    if len(rows) != 1:
        raise ValueError(f"{spec['question_id']}: {len(rows)} rows in {parquet.name}")
    text = rows[0][0][0]
    if spec["after"] not in text:
        raise ValueError(f"{spec['question_id']}: no marker {spec['after']!r}")
    return encode_text(text.split(spec["after"], 1)[1].replace("\r\n", "\n"))


def main() -> None:
    import duckdb
    import tokenizers

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--speed-dir", type=Path, required=True,
                        help="throughput_1k.parquet, upstream_speed.py, stackselect_1k.json, meetingpred_4k.jsonl, qwen-tokenizer/")
    parser.add_argument("--long-dir", type=Path, required=True, help="the document's parquet, e.g. throughput_8k.parquet")
    parser.add_argument("--exclude-panel", nargs="+", required=True, help="NAME=PATH of each panel the protocol excludes")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.protocol.read_text())["prefix"]
    excluded_paths = dict(item.split("=", 1) for item in args.exclude_panel)
    if sorted(excluded_paths) != sorted(spec["exclude"]):
        raise SystemExit(f"--exclude-panel must name {spec['exclude']}, not {sorted(excluded_paths)}")
    directory = args.speed_dir
    connection = duckdb.connect()
    pool = connection.sql(f"select question_id, sub_category, source, src_id, turns "
                          f"from read_parquet('{directory / 'throughput_1k.parquet'}')").fetchall()
    rows = [dict(zip(("question_id", "sub_category", "source", "src_id", "turns"), row)) for row in pool]
    candidates = [{**row, "category": normalize_category(row["sub_category"])} for row in rows]
    excluded = [entry for path in excluded_paths.values() for entry in json.loads(Path(path).read_text())]
    excluded_ids = {entry["id"] for entry in excluded}
    excluded_sources = {(row["source"], row["src_id"]) for row in rows if row["question_id"] in excluded_ids}
    excluded_prompts = {tuple(entry["prompt_token_ids"]) for entry in excluded}
    recipes = load_recipes(directory / "upstream_speed.py")
    stackselect = {}
    for row in json.loads((directory / "stackselect_1k.json").read_text()):
        stackselect.setdefault(str(row["question_id"]), row)            # upstream keeps the first match
    bamboo = [json.loads(line) for line in (directory / "meetingpred_4k.jsonl").read_text().splitlines() if line.strip()]
    encode = chat_encoder(directory / "qwen-tokenizer")
    encoded, bodies = {}, {}    # bodies: each untagged prompt, the first candidate that has it

    def fits(candidate: dict) -> bool:
        """At most max_formatted_tokens long with its tag; untagged, not an excluded panel's prompt and not one a
        candidate picked earlier already has (two question ids can carry the same text)."""
        prompt, recipe = first_turn(candidate, recipes, stackselect, bamboo)
        body = tuple(encode(prompt))
        encoded[candidate["question_id"]] = (recipe, encode(spec["tag"].format(id=candidate["question_id"]) + prompt))
        if len(encoded[candidate["question_id"]][1]) > spec["max_formatted_tokens"] or body in excluded_prompts:
            return False
        if bodies.setdefault(body, candidate["question_id"]) != candidate["question_id"]:
            return False
        return True

    chosen = select_candidates(candidates, excluded_ids, excluded_sources, spec["salt"], spec["quotas"], fits)
    picked = interleave_proportional(chosen, list(spec["quotas"]))
    private = [encoded[c["question_id"]][1] for c in picked]
    if len({tuple(tokens) for tokens in private}) != len(private):
        raise SystemExit("two private parts are the same prompt")
    shared_blocks = shared_private_blocks(private)
    if any(shared_blocks):
        raise SystemExit(f"{sum(map(bool, shared_blocks))} private parts share a whole block with another one")

    tokenizer = tokenizers.Tokenizer.from_file(str(directory / "qwen-tokenizer" / "tokenizer.json"))

    def encode_text(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False).ids

    head, tail = encode_text("<|im_start|>system\n"), encode_text("<|im_end|>\n")
    body = encode_text(spec["system_intro"]) + document_tokens(args.long_dir, spec["document"], encode_text)
    inputs = {name: directory / name for name in ("throughput_1k.parquet", "upstream_speed.py", "stackselect_1k.json",
                                                   "meetingpred_4k.jsonl", "qwen-tokenizer/tokenizer.json",
                                                   "qwen-tokenizer/tokenizer_config.json")}
    inputs |= {f"{spec['document']['source']}.parquet": args.long_dir / f"{spec['document']['source']}.parquet"}
    inputs |= {Path(path).name: Path(path) for path in excluded_paths.values()}
    input_sha256 = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in inputs.items()}
    measured = spec["measured"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for length in spec["lengths"]:
        if length % BLOCK_SIZE:
            raise SystemExit(f"a shared part of {length} tokens does not fill whole blocks")
        shared = system_block(head, body, tail, length)
        entries = [{"id": c["question_id"], "category": c["category"], "source": c["source"], "src_id": c["src_id"],
                    "recipe": encoded[c["question_id"]][0], "shared_tokens": length, "private_tokens": len(tokens),
                    "private_shared_blocks": blocks, "prompt_token_ids": shared + tokens}
                   for c, tokens, blocks in zip(picked, private, shared_blocks)]
        for name, part in ((f"S{length}", entries[:measured]), (f"S{length}w", entries[measured:])):
            path = args.output_dir / f"panel-{name}.json"
            path.write_text(json.dumps(part) + "\n")
            provenance = {"panel": name, "output_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                          "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "inputs": input_sha256,
                          "shared_tokens": length, "shared_sha256": hashlib.sha256(json.dumps(shared).encode()).hexdigest(),
                          "entries": len(part), "private_tokens": [entry["private_tokens"] for entry in part],
                          "private_shared_blocks": [entry["private_shared_blocks"] for entry in part],
                          **{key: spec[key] for key in ("salt", "quotas", "max_formatted_tokens", "document",
                                                        "system_intro")}}
            path.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
            lengths = sorted(len(entry["prompt_token_ids"]) for entry in part)
            print(name, provenance["output_sha256"], len(part), lengths[0], lengths[len(lengths) // 2], lengths[-1],
                  "private blocks shared:", sum(entry["private_shared_blocks"] for entry in part))


if __name__ == "__main__":
    main()
