import pytest
import torch
import torch.distributed as dist
from torch import nn
from transformers import Qwen3Config

import gpu_import_stubs  # noqa: F401  (before the engine imports)
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.models.qwen3 import Qwen3Model

TOKENS, WIDTH, LAYERS = 3, 4, 36
AUX_LAYER_IDS = (1, 9, 17, 25, 33)


@pytest.fixture(scope="module", autouse=True)
def single_process_group(tmp_path_factory):
    if not dist.is_initialized():
        store = tmp_path_factory.mktemp("dist") / "store"
        dist.init_process_group("gloo", init_method=f"file://{store}", rank=0, world_size=1)


class ConstantLayer(nn.Module):
    """Layer l returns hidden_states = l and residual = 1000·l + 0.5, so a missing residual or a neighbouring layer shows."""

    def __init__(self, index):
        super().__init__()
        self.index = index

    def forward(self, positions, hidden_states, residual):
        return torch.full((TOKENS, WIDTH), float(self.index)), torch.full((TOKENS, WIDTH), 1000.0 * self.index + 0.5)


class ZeroEmbedding(nn.Module):
    def forward(self, input_ids):
        return torch.zeros(TOKENS, WIDTH)


class PassThroughNorm(nn.Module):
    def forward(self, hidden_states, residual):
        return hidden_states, residual


def output_of(layer_id: int) -> float:
    return layer_id + 1000.0 * layer_id + 0.5


def constant_model() -> Qwen3Model:
    config = Qwen3Config(vocab_size=8, hidden_size=WIDTH, intermediate_size=8, num_hidden_layers=LAYERS,
                         num_attention_heads=1, num_key_value_heads=1, head_dim=WIDTH, max_position_embeddings=16)
    with torch.device("meta"):
        model = Qwen3Model(config)
    get_rope.cache_clear()    # the cached RotaryEmbedding is on the meta device; see tests/test_dflash_model.py
    model.embed_tokens, model.norm = ZeroEmbedding(), PassThroughNorm()
    model.layers = nn.ModuleList(ConstantLayer(index) for index in range(LAYERS))
    model.aux_layer_ids = AUX_LAYER_IDS
    return model


def run(model, aux_out=None):
    ids = torch.zeros(TOKENS, dtype=torch.long)
    return model(ids, ids, aux_out)


def test_aux_holds_each_listed_layers_output_in_order():
    aux = torch.full((TOKENS, len(AUX_LAYER_IDS) * WIDTH), -1.0)
    run(constant_model(), aux)
    for segment, layer_id in enumerate(AUX_LAYER_IDS):
        values = aux[:, segment * WIDTH:(segment + 1) * WIDTH]
        assert torch.all(values == output_of(layer_id))
        assert not torch.any(values == output_of(layer_id - 1)) and not torch.any(values == output_of(layer_id + 1))


def test_no_aux_leaves_the_output_unchanged():
    assert torch.equal(run(constant_model()), torch.full((TOKENS, WIDTH), float(LAYERS - 1)))
