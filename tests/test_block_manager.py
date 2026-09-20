import pytest

from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence

BLOCK_SIZE = 4


@pytest.fixture(autouse=True)
def small_blocks():
    previous = Sequence.block_size
    Sequence.block_size = BLOCK_SIZE
    yield
    Sequence.block_size = previous


def test_allocate_and_deallocate_use_the_manager_role():
    manager = BlockManager(num_blocks=4, block_size=BLOCK_SIZE, enable_prefix_cache=False, role="draft")
    seq = Sequence(list(range(5)))
    manager.allocate(seq, 0)
    assert seq.draft_kv.block_table == [0, 1]
    assert seq.target_kv.block_table == []
    manager.deallocate(seq)
    assert seq.draft_kv.block_table == [] and len(manager.free_block_ids) == 4


def test_reserve_grows_the_table_to_cover_the_tokens():
    manager = BlockManager(num_blocks=4, block_size=BLOCK_SIZE, enable_prefix_cache=False)
    seq = Sequence(list(range(5)))
    manager.allocate(seq, 0)                 # 2 blocks cover 8 tokens
    manager.reserve(seq, 8)
    assert len(seq.block_table) == 2
    manager.reserve(seq, 9)
    assert len(seq.block_table) == 3


def test_can_reserve_reports_free_capacity():
    manager = BlockManager(num_blocks=2, block_size=BLOCK_SIZE, enable_prefix_cache=False)
    seq = Sequence(list(range(5)))
    manager.allocate(seq, 0)
    assert manager.can_reserve(seq, 8)
    assert not manager.can_reserve(seq, 9)


def test_reserve_len_matches_upstream_may_append_rule():
    manager = BlockManager(num_blocks=16, block_size=BLOCK_SIZE, enable_prefix_cache=False)
    seq = Sequence([0])
    manager.allocate(seq, 0)
    for _ in range(30):
        seq.append_token(0)
        expected_blocks = (len(seq) + BLOCK_SIZE - 1) // BLOCK_SIZE     # upstream allocates when len % bs == 1
        manager.reserve(seq, len(seq))
        assert len(seq.block_table) == expected_blocks
