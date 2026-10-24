"""DeepSpec's Qwen3DSparkModel on nano-vllm's layers: the DFlash draft (markov_rank 0) and DSpark, which adds a
Markov head and a confidence head.

Context: for each committed position t, c_t = hidden_norm(fc(aux_t)), aux_t being the target's outputs of the
aux_layer_ids layers, concatenated. Draft layer i keeps K = RoPE_t(k_norm(W_k c_t)) and V = W_v c_t at t; c_t
skips the layers' input_layernorm.
Block: the anchor (the last committed token, position P) and block_size - 1 mask tokens at P .. P+block_size-1
go through ordinary Qwen3 layers. They see the context below P and the whole block (no causal mask); row j
predicts the token at P+1+j.
DSpark's Markov head adds to row j's logits a bias from the token sampled at row j-1 (the anchor for row 0), so its
rows are sampled left to right. Its confidence head is loaded but unused: it serves variable-length verification.
"""
import torch
import torch.nn.functional as F
from torch import nn

from nanovllm.layers import attention
from nanovllm.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import ReplicatedLinear
from nanovllm.models.qwen3 import Qwen3DecoderLayer, Qwen3ForCausalLM


class MarkovHead(nn.Module):
    """DSpark's vanilla Markov head: markov_w1 embeds the previous token in rank r, markov_w2 projects it to a bias
    on the vocabulary."""

    def __init__(self, vocab_size: int, rank: int) -> None:
        super().__init__()
        self.markov_w1 = VocabParallelEmbedding(vocab_size, rank)
        self.markov_w2 = ReplicatedLinear(rank, vocab_size, bias=False)

    def forward(self, prev_token_ids: torch.Tensor) -> torch.Tensor:
        """[B] previous tokens -> [B, V] bias for the next row's logits."""
        return self.markov_w2(self.markov_w1(prev_token_ids))


class ConfidenceHead(nn.Module):
    """DSpark's acceptance predictor over [hidden, Markov embedding of the previous token]."""

    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.proj = ReplicatedLinear(input_size, 1, bias=True)


class DFlashDraftModel(nn.Module):
    packed_modules_mapping = Qwen3ForCausalLM.packed_modules_mapping

    def __init__(self, config) -> None:
        super().__init__()
        self.block_size = config.block_size
        self.mask_token_id = config.mask_token_id
        self.aux_layer_ids = tuple(config.target_layer_ids)
        # Module names follow the checkpoint, which has no "model." prefix.
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        self.fc = ReplicatedLinear(len(self.aux_layer_ids) * config.hidden_size, config.hidden_size, bias=False)
        self.hidden_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        markov_rank = config.markov_rank
        self.markov_head = None
        if markov_rank > 0:
            assert config.markov_head_type == "vanilla", "only DSpark's vanilla Markov head is implemented"
            self.markov_head = MarkovHead(config.vocab_size, markov_rank)
        self.confidence_head = None
        if config.enable_confidence_head:
            markov_width = markov_rank if config.confidence_head_with_markov else 0
            self.confidence_head = ConfidenceHead(config.hidden_size + markov_width)

    def context_features(self, aux: torch.Tensor) -> torch.Tensor:
        """[T, len(aux_layer_ids) * H] -> [T, H]."""
        return self.hidden_norm(self.fc(aux))

    def store_context_kv(self, features: torch.Tensor, positions: torch.Tensor, slot_mapping: torch.Tensor) -> None:
        """Write every layer's context K/V at slot_mapping. W_k, W_v are the K and V rows of the packed qkv_proj."""
        for layer in self.layers:
            attn = layer.self_attn
            k, v = F.linear(features, attn.qkv_proj.weight[attn.q_size:]).split([attn.kv_size, attn.kv_size], dim=-1)
            k = attn.k_norm(k.view(-1, attn.num_kv_heads, attn.head_dim))
            k = attn.rotary_emb.apply_rotary_emb_k(positions, k)
            attention.store_kvcache(k, v.view(-1, attn.num_kv_heads, attn.head_dim),
                                    attn.attn.k_cache, attn.attn.v_cache, slot_mapping)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Block rows, sequence-major ([B * block_size]) -> hidden states after the final norm."""
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states, all_logits=True)
