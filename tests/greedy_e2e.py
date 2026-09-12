import argparse
import json
from pathlib import Path

from nanovllm import LLM, SamplingParams


parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--mode", choices=("eager", "graph"), required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

llm = LLM(
    args.model,
    enforce_eager=args.mode == "eager",
    tensor_parallel_size=1,
)

outputs = llm.generate(
    [
        "Write one sentence explaining CUDA.",
        "List three benefits of batching.",
    ],
    SamplingParams(
        temperature=0,
        max_tokens=32,
        ignore_eos=True,
    ),
    use_tqdm=False,
)

payload = {
    "mode": args.mode,
    "token_ids": [output["token_ids"] for output in outputs],
}
Path(args.output).write_text(
    json.dumps(payload, indent=2),
    encoding="utf-8",
)

print(json.dumps(payload))
