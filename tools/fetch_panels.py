"""Download the A1/A2 request panels of the expanded-k3-results release into benchmarks/inputs/panel-48.json.

The 48 prompts are the ones docs/PERFORMANCE.md measured: A1 groups 2, 6, 7, 8, 10, 11 (four
requests each) and A2 groups 0-5. Prompts are Qwen3 chat-formatted in non-thinking mode.
"""
import argparse
import io
import json
import urllib.request
import zipfile
from pathlib import Path

RELEASE_ASSET = ("https://github.com/tianyuantong/nano-vllm-speculative-decoding/releases/download/"
                 "expanded-k3-results/expanded-k3-evidence.zip")
A1_GROUPS = (2, 6, 7, 8, 10, 11)
A2_GROUPS = (0, 1, 2, 3, 4, 5)
GROUP_SIZE = 4
DEFAULT_OUTPUT = Path("benchmarks/inputs/panel-48.json")


def select_panel(a1_samples: list[dict], a2_samples: list[dict]) -> list[dict]:
    chosen = []
    for panel, samples, groups in (("A1", a1_samples, A1_GROUPS), ("A2", a2_samples, A2_GROUPS)):
        for group in groups:
            for sample in samples[group * GROUP_SIZE:(group + 1) * GROUP_SIZE]:
                chosen.append({"id": sample["question_id"], "panel": panel, "group": group,
                               "prompt_token_ids": sample["prompt_token_ids"]})
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    with urllib.request.urlopen(RELEASE_ASSET) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    a1 = json.loads(archive.read("evidence/A1-inputs.json"))["samples"]
    a2 = json.loads(archive.read("evidence/A2-inputs.json"))["samples"]
    panel = select_panel(a1, a2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(panel))
    lengths = [len(entry["prompt_token_ids"]) for entry in panel]
    print(f"wrote {len(panel)} prompts to {args.output}; prompt tokens min {min(lengths)} max {max(lengths)}")


if __name__ == "__main__":
    main()
