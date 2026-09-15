"""Explicit, offline random B/S0 entrypoint; not a replacement for LLMEngine.

prepare_requests creates RNG outside timing; generate contains request KV setup,
prefill, sampling, draft/VERIFY, catch-up, commit, and final synchronization.
Optional exact-shape VERIFY Graphs do not change scheduling or sampling.
"""

import math

from transformers import AutoTokenizer

from nanovllm.engine.dual_model import DualModelRunner
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.random_backend import CUDARandomBackend, RequestRNG
from nanovllm.engine.random_decode import RandomDecode, Request


class RandomLLM:
    def __init__(self, target_config, draft_config=None, *, k=4, gpu_draft_tokens=False,
                 ngram=False, performance_mode=True, verify_graphs=True,
                 verify_graph_reserve_budget_bytes=2 << 30,
                 verify_graph_min_free_bytes=1 << 30, r2_options=(), sampling_mode="random", verify_sampling="off"):
        if sampling_mode not in ("random", "greedy") or (ngram and sampling_mode != "random"):
            raise ValueError("known sampling mode required; N is random only")
        self.sampling_mode = sampling_mode
        if verify_sampling not in ("off", "eager", "graph"):
            raise ValueError("unknown deterministic verification execution mode")
        self.verify_sampling_mode = verify_sampling
        self.sampling_graph = None
        if type(performance_mode) is not bool or type(verify_graphs) is not bool:
            raise ValueError("performance_mode and verify_graphs must be bool")
        r2_options = frozenset(r2_options)
        if r2_options - {"draw", "softmax", "residual", "pack", "views"}:
            raise ValueError("unknown R2 experiment option")
        self.r2_options = r2_options
        if verify_sampling != "off" and (sampling_mode != "random" or not performance_mode
                or draft_config is None or k != 3 or not {"draw", "softmax", "residual"} <= r2_options):
            raise ValueError("post-p/q execution requires random k3 and the frozen three R2 paths")
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
            backend_type = CUDARandomBackend
            if sampling_mode == "greedy":
                from nanovllm.engine.greedy_backend import CUDAGreedyBackend
                backend_type = CUDAGreedyBackend
            self.backend = backend_type(target, draft, performance_mode=performance_mode)
            self.backend.r2_options = r2_options
            if verify_sampling != "off":
                from nanovllm.engine.verify_sampling import post_pq
                self.backend.post_pq_verifier = post_pq
            for runner in [target] + ([draft] if draft is not None else []):
                runner.r2_pack_decode = "pack" in r2_options
            self.decoder = RandomDecode(self.backend, target_blocks=target_config.num_kvcache_blocks,
                                        draft_blocks=draft_config.num_kvcache_blocks if draft else 0,
                                        block_size=target_config.kvcache_block_size,
                                        max_model_len=target_config.max_model_len,
                                        max_num_seqs=target_config.max_num_seqs,
                                        vocab_size=target_config.hf_config.vocab_size,
                                        eos=self.tokenizer.eos_token_id, k=k, gpu_draft_tokens=gpu_draft_tokens,
                                        ngram=ngram, performance_mode=performance_mode, sampling_mode=sampling_mode)
            self.decoder.r2_views = "views" in r2_options
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
        if self.verify_sampling_mode == "graph" and self.sampling_graph is None:
            from nanovllm.engine.verify_sampling import VerifySamplingGraph
            self.sampling_graph = VerifySamplingGraph(
                device=self.backend.runners["target"].model.lm_head.weight.device,
                vocab_size=self.decoder.vocab_size)
            self.backend.post_pq_verifier = self.sampling_graph

    def performance_metadata(self):
        cache = self.backend.runners["target"].verify_graph_cache
        return {"performance_mode": self.performance_mode, "r2_options": sorted(self.r2_options),
                "sampling_mode": self.sampling_mode,
                "rng_version": "nano-request-sha256-v1" if self.sampling_mode == "random" else None,
                "rng_consumption_changed": False if self.sampling_mode == "random" else None,
                "verify_graph": cache.statistics() if cache is not None else None,
                "verify_sampling": self.verify_sampling_mode,
                "sampling_graph": self.sampling_graph.statistics() if self.sampling_graph is not None else None}

    def prepare_requests(self, prompts, *, request_ids, seed, max_tokens, temperature=1.0, ignore_eos=False):
        if self.closed or len(prompts) != len(request_ids) or len(set(request_ids)) != len(request_ids):
            raise ValueError("open engine and one distinct stable ID per prompt required")
        if not math.isfinite(temperature) or (temperature <= 0 if self.sampling_mode == "random" else temperature != 0):
            raise ValueError("temperature must match the engine sampling mode")
        return [Request(rid, self.tokenizer.encode(p) if isinstance(p, str) else p,
                        max_tokens, RequestRNG(seed, rid) if self.sampling_mode == "random" else None,
                        temperature, ignore_eos)
                for p, rid in zip(prompts, request_ids, strict=True)]

    def generate(self, requests):
        if self.closed:
            raise RuntimeError("engine closed")
        self.decoder.validate_requests(requests)
        if any(r.used for r in requests):
            raise ValueError("fresh requests required for each generate")
        if self.sampling_mode == "random" and any(
                not isinstance(r.rng, RequestRNG) or r.rng.used or r.rng.request_id != r.request_id for r in requests):
            raise ValueError("fresh matching request RNG required for each generate")
        if self.sampling_mode == "greedy" and any(r.rng is not None for r in requests):
            raise ValueError("greedy requests must not carry sampling RNG")
        for r in requests:
            r.used = True
            if r.rng is not None:
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
            if self.sampling_graph is not None:
                self.backend.post_pq_verifier = None
                self.sampling_graph.close()
                self.sampling_graph = None
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
