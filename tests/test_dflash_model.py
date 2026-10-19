"""nano-vllm's DFlashDraftModel against the checkpoint layout and against DeepSpec's Qwen3DSparkModel.

The tests marked local_weights load real drafts and DeepSpec's source; they run with four variables set:
DFLASH_4B_DIR, DFLASH_8B_DIR, DEEPSPEC_DIR and TORCHDYNAMO_DISABLE=1 (nano's RMSNorm and RoPE use torch.compile).
"""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from transformers import Qwen3Config

import fake_paged_attention
import gpu_import_stubs  # noqa: F401  (before the engine imports)
from nanovllm.layers import attention
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.models import qwen3
from nanovllm.models.dflash import DFlashDraftModel
from nanovllm.utils.context import reset_context, set_context
from nanovllm.utils.loader import load_model

DATA = Path(__file__).resolve().parent / "data"
ENV = ("DFLASH_4B_DIR", "DFLASH_8B_DIR", "DEEPSPEC_DIR", "TORCHDYNAMO_DISABLE")
local_weights = pytest.mark.skipif(not all(os.environ.get(name) for name in ENV), reason="needs " + ", ".join(ENV))
MASK, ANCHOR, CONTEXT_LEN = 151669, 9707, 37


@pytest.fixture(scope="module", autouse=True)
def single_process_group(tmp_path_factory):
    if not dist.is_initialized():
        store = tmp_path_factory.mktemp("dist") / "store"
        dist.init_process_group("gloo", init_method=f"file://{store}", rank=0, world_size=1)


def data_config(name: str) -> Qwen3Config:
    return Qwen3Config(**json.loads((DATA / f"{name}.config.json").read_text()))


def test_parameters_match_the_checkpoint():
    with torch.device("meta"):
        model = DFlashDraftModel(data_config("dflash_qwen3_8b_block7"))
    expected = defaultdict(lambda: [0, None])            # parameter -> [rows, row shape]
    for name, shape in json.loads((DATA / "dflash_qwen3_8b_block7.tensors.json").read_text()).items():
        packed = next((key for key in model.packed_modules_mapping if key in name), None)
        if packed is not None:
            name = name.replace(packed, model.packed_modules_mapping[packed][0])
        expected[name][0] += shape[0]
        expected[name][1] = shape[1:]
    actual = {name: list(param.shape) for name, param in model.named_parameters()}
    assert actual == {name: [rows, *rest] for name, (rows, rest) in expected.items()}


@pytest.mark.parametrize("name", ["dflash_qwen3_4b_block7", "dflash_qwen3_8b_block7"])
def test_config_is_read_as_deepspec_reads_it(name, monkeypatch):
    bases = []

    def recording_get_rope(head_size, rotary_dim, max_position, base):
        bases.append(base)
        return get_rope(head_size, rotary_dim, max_position, base)

    monkeypatch.setattr(qwen3, "get_rope", recording_get_rope)
    with torch.device("meta"):
        model = DFlashDraftModel(data_config(name))
    assert bases == [1e6] * 5    # the base in effect: the config has rope_parameters, not rope_theta
    assert model.layers[0].self_attn.head_dim == 128
    assert (model.block_size, model.mask_token_id, model.aux_layer_ids) == (7, MASK, (1, 9, 17, 25, 33))


# ---------------------------------------------------------------- against DeepSpec, with real weights
@pytest.fixture
def paged(monkeypatch):
    monkeypatch.setattr(attention, "store_kvcache", fake_paged_attention.store_kvcache)
    monkeypatch.setattr(attention.Attention, "forward", fake_paged_attention.attention_forward)
    yield
    reset_context()


def load_both(directory: str, dtype: torch.dtype):
    """nano's model with one 256-slot KV block per layer, and DeepSpec's, from the same checkpoint."""
    sys.path.insert(0, os.environ["DEEPSPEC_DIR"])
    from deepspec.modeling.dspark.qwen3 import Qwen3DSparkModel

    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        model = DFlashDraftModel(Qwen3Config.from_pretrained(directory))
        load_model(model, directory)
    finally:
        torch.set_default_dtype(default_dtype)
    for layer in model.layers:
        attn = layer.self_attn
        layer.self_attn.attn.k_cache = torch.zeros(1, 256, attn.num_kv_heads, attn.head_dim, dtype=dtype)
        layer.self_attn.attn.v_cache = torch.zeros(1, 256, attn.num_kv_heads, attn.head_dim, dtype=dtype)
    reference = Qwen3DSparkModel.from_pretrained(directory, dtype=dtype, attn_implementation="sdpa").eval()
    return model, reference


def random_aux(model, length: int, dtype: torch.dtype) -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    width = len(model.aux_layer_ids) * model.norm.weight.shape[0]
    return torch.randn(length, width, generator=generator).to(dtype)


@torch.inference_mode()
def block_logits(model, aux: torch.Tensor, causal: bool = False) -> torch.Tensor:
    """nano: write the context K/V of positions 0 .. len(aux)-1, then run the block [anchor, mask x 6] after them."""
    length = aux.shape[0]
    context = torch.arange(length)
    model.store_context_kv(model.context_features(aux), context, context.to(torch.int32))
    block = torch.arange(length, length + model.block_size)
    set_context(False, slot_mapping=block.to(torch.int32), context_lens=torch.tensor([length + model.block_size],
                dtype=torch.int32), block_tables=torch.zeros(1, 1, dtype=torch.int32),
                query_len_per_seq=model.block_size, causal=causal)
    input_ids = torch.tensor([ANCHOR] + [MASK] * (model.block_size - 1))
    return model.compute_logits(model(input_ids, block))


@torch.inference_mode()
def reference_block_logits(reference, aux: torch.Tensor) -> torch.Tensor:
    from transformers import DynamicCache
    from deepspec.eval.dspark.draft_ops import forward_dspark_draft_block

    length = aux.shape[0]
    block_ids = torch.tensor([[ANCHOR] + [MASK] * (reference.block_size - 1)])
    hidden = forward_dspark_draft_block(reference, draft_input_ids=block_ids,
                                        position_ids=torch.arange(length + reference.block_size)[None],
                                        past_key_values_draft=DynamicCache(), target_hidden_states=aux[None],
                                        start=length, block_size=reference.block_size)
    return reference.compute_logits(hidden)[0]


@pytest.fixture
def pair_4b():
    return load_both(os.environ.get("DFLASH_4B_DIR", ""), torch.float32) if all(map(os.environ.get, ENV)) else None


@local_weights
@torch.inference_mode()
def test_context_features(pair_4b):
    model, reference = pair_4b
    aux = random_aux(model, 11, torch.float32)
    expected = reference.hidden_norm(reference.fc(aux))
    assert (model.context_features(aux) - expected).abs().max() <= 1e-4


@local_weights
@torch.inference_mode()
def test_context_kv(pair_4b, paged):
    from deepspec.modeling.dspark.qwen3.modeling import apply_rotary_pos_emb

    model, reference = pair_4b
    aux = random_aux(model, 11, torch.float32)
    positions = torch.arange(11)
    model.store_context_kv(model.context_features(aux), positions, positions.to(torch.int32))
    features = reference.hidden_norm(reference.fc(aux))[None]                       # [1, 11, H]
    cos, sin = reference.rotary_emb(features, positions[None])
    for layer, reference_layer in zip(model.layers, reference.layers):
        attn, reference_attn = layer.self_attn, reference_layer.self_attn
        k = reference_attn.k_norm(reference_attn.k_proj(features).view(1, 11, attn.num_kv_heads, attn.head_dim))
        _, k = apply_rotary_pos_emb(k.transpose(1, 2), k.transpose(1, 2), cos, sin)
        v = reference_attn.v_proj(features).view(11, attn.num_kv_heads, attn.head_dim)
        assert (attn.attn.k_cache[0, :11] - k[0].transpose(0, 1)).abs().max() <= 1e-4
        assert (attn.attn.v_cache[0, :11] - v).abs().max() <= 1e-4


@local_weights
def test_block_logits_match_deepspec(pair_4b, paged):
    model, reference = pair_4b
    aux = random_aux(model, CONTEXT_LEN, torch.float32)
    logits, expected = block_logits(model, aux), reference_block_logits(reference, aux)
    assert (logits - expected).abs().max() <= 1e-3
    assert torch.equal(logits.argmax(-1), expected.argmax(-1))


@local_weights
def test_a_causal_block_is_caught(pair_4b, paged):
    model, reference = pair_4b
    aux = random_aux(model, CONTEXT_LEN, torch.float32)
    assert (block_logits(model, aux, causal=True) - reference_block_logits(reference, aux)).abs().max() > 1e-2


@local_weights
def test_8b_bf16_smoke(paged):
    model, reference = load_both(os.environ["DFLASH_8B_DIR"], torch.bfloat16)
    aux = random_aux(model, CONTEXT_LEN, torch.bfloat16)
    logits, expected = block_logits(model, aux).float(), reference_block_logits(reference, aux).float()
    top2 = expected.topk(2, dim=-1).values
    decided = top2[:, 0] - top2[:, 1] > 0.5    # G1-wide's rule: in bf16, rows with near-tied top-2 may flip
    assert torch.equal(logits.argmax(-1)[decided], expected.argmax(-1)[decided])
    assert (logits - expected).abs().max() / expected.abs().max() <= 2e-2
