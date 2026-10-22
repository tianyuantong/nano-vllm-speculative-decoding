"""DeepSpec's DFlash draft (a Qwen3DSparkModel without its Markov and confidence heads) on nano-vllm's layers.

Context: for each committed position t, c_t = hidden_norm(fc(aux_t)), aux_t being the target's outputs of the
aux_layer_ids layers, concatenated. Draft layer i keeps K = RoPE_t(k_norm(W_k c_t)) and V = W_v c_t at t; c_t
skips the layers' input_layernorm.
Block: the anchor (the last committed token, position P) and block_size - 1 mask tokens at P .. P+block_size-1
go through ordinary Qwen3 layers. They see the context below P and the whole block (no causal mask); row j
predicts the token at P+1+j.
"""
import torch
import torch.nn.functional as F
from torch import nn

from nanovllm.layers import attention
from nanovllm.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import ReplicatedLinear
from nanovllm.models.qwen3 import Qwen3DecoderLayer, Qwen3ForCausalLM


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
