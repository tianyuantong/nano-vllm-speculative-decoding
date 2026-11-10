"""Speculative decoding with a DSpark block draft: Qwen3-8B verifies the tokens that dspark_qwen3_8b_block7 proposes."""
from transformers import AutoTokenizer

from nanovllm import LLM, SamplingParams

TARGET = "/models/Qwen3-8B"
DRAFT = "/models/dspark_qwen3_8b_block7"    # deepseek-ai/dspark_qwen3_8b_block7


def main():
    tokenizer = AutoTokenizer.from_pretrained(TARGET)
    llm = LLM(TARGET, draft_model=DRAFT, draft_type="dspark", max_num_seqs=8, enable_prefix_cache=False,
              kv_cache_memory_bytes=10 << 30, seed=0)
    params = SamplingParams(temperature=0.7, top_k=20, top_p=0.8, max_tokens=256)
    questions = ["introduce yourself", "list all prime numbers within 100"]
    prompts = [tokenizer.apply_chat_template([{"role": "user", "content": question}], tokenize=False,
                                             add_generation_prompt=True) for question in questions]
    try:
        outputs = llm.generate(prompts, params)
    finally:
        llm.exit()
    for prompt, output in zip(prompts, outputs):
        print(f"\nPrompt: {prompt!r}\nCompletion: {output['text']!r}")


if __name__ == "__main__":
    main()
