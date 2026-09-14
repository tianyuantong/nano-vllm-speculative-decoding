"""Explicit, offline random B/S0 entrypoint; not a replacement for LLMEngine.

prepare_requests creates RNG outside timing; generate contains request KV setup,
prefill, sampling, draft/VERIFY, catch-up, commit, and final synchronization.
Optional exact-shape VERIFY Graphs do not change scheduling or sampling.
"""

from transformers import AutoTokenizer

from nanovllm.engine.dual_model import DualModelRunner
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.random_backend import CUDARandomBackend, RequestRNG
from nanovllm.engine.random_decode import RandomDecode, Request


class RandomLLM:
    def __init__(self, target_config, draft_config=None, *, k=4, gpu_draft_tokens=False,
                 ngram=False, performance_mode=True, verify_graphs=True,
                 verify_graph_reserve_budget_bytes=2 << 30,
                 verify_graph_min_free_bytes=1 << 30):
        if type(performance_mode) is not bool or type(verify_graphs) is not bool:
            raise ValueError("performance_mode and verify_graphs must be bool")
        self.performance_mode = performance_mode
        if ngram and (draft_config is not None or gpu_draft_tokens or not k):
            raise ValueError("N requires target only, k>0 and no device draft continuation")
        if draft_config is None and not ngram:
            k = 0
        for config in [target_config] + ([draft_config] if draft_config is not None else []):
            if config.tensor_parallel_size != 1 or config.enable_prefix_cache or config.kv_cache_memory_bytes is None:
                raise ValueError("offline random path requires TP1, no prefix cache and explicit KV bytes")
        if type(k) is not int or not 0 <= k <= 4 or (draft_config is not None and k == 0):
            raise ValueError("draft config requires k=1..4; ordinary B has no draft")
        self.tokenizer = AutoTokenizer.from_pretrained(target_config.model, use_fast=True)
        if draft_config is not None:
            other = AutoTokenizer.from_pretrained(draft_config.model, use_fast=True)
            if (self.tokenizer.backend_tokenizer.to_str() != other.backend_tokenizer.to_str()
                    or self.tokenizer.special_tokens_map != other.special_tokens_map
                    or self.tokenizer.eos_token_id != other.eos_token_id
                    or target_config.hf_config.vocab_size != draft_config.hf_config.vocab_size):
                raise ValueError("target/draft tokenizer, special tokens or model vocab differ")
        self.owner = None
        self.closed = False
        try:
            if draft_config is None:
                self.owner = ModelRunner(target_config, 0, [])
                target, draft = self.owner, None
            else:
                self.owner = DualModelRunner(target_config, draft_config)
                target, draft = self.owner.target, self.owner.draft
            # TP1: validate the actual sampler/embedding dimensions before
            # any internal GPU proposal can become another model input.
            vocab = target_config.hf_config.vocab_size
            for runner in [target] + ([draft] if draft is not None else []):
                if (runner.model.lm_head.weight.shape[0] != vocab
                        or runner.model.model.embed_tokens.weight.shape[0] < vocab):
                    raise ValueError("sampler vocabulary exceeds consumer embedding rows")
            if performance_mode and verify_graphs and k and not target.enforce_eager:
                target.enable_verify_graphs(
                    reserve_budget_bytes=verify_graph_reserve_budget_bytes,
                    min_free_bytes=verify_graph_min_free_bytes)
            self.backend = CUDARandomBackend(target, draft, performance_mode=performance_mode)
            self.decoder = RandomDecode(self.backend, target_blocks=target_config.num_kvcache_blocks,
                                        draft_blocks=draft_config.num_kvcache_blocks if draft else 0,
                                        block_size=target_config.kvcache_block_size,
                                        max_model_len=target_config.max_model_len,
                                        max_num_seqs=target_config.max_num_seqs,
                                        vocab_size=target_config.hf_config.vocab_size,
                                        eos=self.tokenizer.eos_token_id, k=k, gpu_draft_tokens=gpu_draft_tokens,
                                        ngram=ngram, performance_mode=performance_mode)
        except BaseException as error:
            try:
                self.close()
            except BaseException as cleanup_error:
                raise error from cleanup_error
            raise

    def freeze_performance_caches(self):
        """Call after natural warmup, outside timed generate; misses fall back."""
        cache = self.backend.runners["target"].verify_graph_cache
        if cache is not None:
            cache.freeze()

    def performance_metadata(self):
        cache = self.backend.runners["target"].verify_graph_cache
        return {"performance_mode": self.performance_mode,
                "rng_version": "nano-request-sha256-v1", "rng_consumption_changed": False,
                "verify_graph": cache.statistics() if cache is not None else None}

    def prepare_requests(self, prompts, *, request_ids, seed, max_tokens, temperature=1.0, ignore_eos=False):
        if self.closed or len(prompts) != len(request_ids) or len(set(request_ids)) != len(request_ids):
            raise ValueError("open engine and one distinct stable ID per prompt required")
        return [Request(rid, self.tokenizer.encode(p) if isinstance(p, str) else p,
                        max_tokens, RequestRNG(seed, rid), temperature, ignore_eos)
                for p, rid in zip(prompts, request_ids, strict=True)]

    def generate(self, requests):
        if self.closed:
            raise RuntimeError("engine closed")
        if any(not isinstance(r.rng, RequestRNG) or r.rng.used or r.rng.request_id != r.request_id for r in requests):
            raise ValueError("fresh matching request RNG required for each generate")
        for r in requests:
            r.rng.used = True
        self.backend.invalid.clear()
        try:
            outputs = self.decoder.generate(requests)
            for out in outputs:
                out["text"] = self.tokenizer.decode(out["token_ids"])
            return outputs
        except BaseException as error:
            try:
                self.close()  # fail closed; no partially consumed RNG/CUDA retry
            except BaseException as cleanup_error:
                raise error from cleanup_error
            raise

    def close(self):
        if not self.closed:
            self.closed = True
            if self.owner is not None:
                self.owner.exit()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.close()
        except BaseException as cleanup_error:
            if exc_value is not None:
                raise exc_value from cleanup_error
            raise
