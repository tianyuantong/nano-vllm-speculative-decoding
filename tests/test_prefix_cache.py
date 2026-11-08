import unittest
from types import SimpleNamespace

from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence


class TestPrefixCacheSwitch(unittest.TestCase):

    @staticmethod
    def make_sequence():
        return Sequence(list(range(512)))

    def test_enabled_reuses_completed_prefix_block(self):
        manager = BlockManager(
            num_blocks=8,
            block_size=256,
            enable_prefix_cache=True,
        )
        first = self.make_sequence()
        self.assertEqual(manager.can_allocate(first), 0)
        manager.allocate(first, 0)
        first.num_scheduled_tokens = 512
        manager.hash_blocks(first)

        second = self.make_sequence()
        self.assertEqual(manager.can_allocate(second), 1)

    def test_disabled_skips_hash_registration_and_reuse(self):
        manager = BlockManager(
            num_blocks=8,
            block_size=256,
            enable_prefix_cache=False,
        )
        first = self.make_sequence()
        self.assertEqual(manager.can_allocate(first), 0)
        manager.allocate(first, 0)
        original_block_table = list(first.block_table)
        first.num_scheduled_tokens = 512
        manager.hash_blocks(first)

        self.assertEqual(first.block_table, original_block_table)
        self.assertEqual(manager.hash_to_block_id, {})
        self.assertTrue(
            all(manager.blocks[block_id].hash == -1 for block_id in first.block_table)
        )

        second = self.make_sequence()
        self.assertEqual(manager.can_allocate(second), 0)

    def test_disabled_requires_exclusive_free_blocks(self):
        manager = BlockManager(
            num_blocks=2,
            block_size=256,
            enable_prefix_cache=False,
        )
        first = self.make_sequence()
        manager.allocate(first, manager.can_allocate(first))

        second = self.make_sequence()
        self.assertEqual(manager.can_allocate(second), -1)

    def test_scheduler_forwards_switch(self):
        config = SimpleNamespace(
            max_num_seqs=1,
            max_num_batched_tokens=256,
            eos=-1,
            kvcache_block_size=256,
            num_kvcache_blocks=8,
            enable_prefix_cache=False,
            num_speculative_tokens=0,
            round_robin_prefill=False,
        )

        scheduler = Scheduler(config)


def small_scheduler(enable_prefix_cache: bool, num_speculative_tokens: int = 0, num_blocks: int = 64) -> Scheduler:
    config = SimpleNamespace(max_num_seqs=4, max_num_batched_tokens=64, eos=-1, kvcache_block_size=4,
                             num_kvcache_blocks=num_blocks, enable_prefix_cache=enable_prefix_cache,
                             num_speculative_tokens=num_speculative_tokens, round_robin_prefill=False)
    return Scheduler(config)


def drive(scheduler: Scheduler, prompts: list[list[int]], max_tokens: int, accepted: int) -> list[list[tuple]]:
    """A closed loop over the prompts: every step's scheduled block tables in order, the k0 decode appending one token
    and a speculative round 1 + accepted."""
    from nanovllm.sampling_params import SamplingParams
    seqs = [Sequence(prompt, SamplingParams(max_tokens=max_tokens, ignore_eos=True)) for prompt in prompts]
    for seq in seqs:
        scheduler.add(seq)
    steps = []
    while not scheduler.is_finished():
        batch, is_prefill = scheduler.schedule()
        steps.append([tuple(seq.block_table) for seq in batch])
        if is_prefill:
            scheduler.postprocess(batch, [1] * len(batch), True)
        elif scheduler.num_speculative_tokens:
            scheduler.postprocess_speculative(batch, [[1] * (1 + accepted) for _ in batch])
        else:
            scheduler.postprocess(batch, [1] * len(batch), False)
    return steps


def test_without_a_whole_shared_block_the_cache_changes_no_block_table():
    Sequence.block_size = 4
    try:
        prompts = [[100 * index + position for position in range(5 + 3 * index)] for index in range(6)]
        for k, accepted in ((0, 0), (3, 2)):
            off = drive(small_scheduler(False, k), prompts, max_tokens=9, accepted=accepted)
            on_scheduler = small_scheduler(True, k)
            on = drive(on_scheduler, prompts, max_tokens=9, accepted=accepted)
            assert on == off
            assert on_scheduler.block_manager.hash_to_block_id    # the prefills did register their whole blocks
    finally:
        Sequence.block_size = 256


def test_only_a_prefill_hashes_its_blocks_and_an_allocation_records_its_hit():
    Sequence.block_size = 4
    try:
        scheduler = small_scheduler(True)
        drive(scheduler, [list(range(5))], max_tokens=7, accepted=0)          # positions 0..10: blocks 0, 1, 2
        manager = scheduler.block_manager
        assert len(manager.hash_to_block_id) == 1                            # the prompt's whole block, not decode's
        second = Sequence(list(range(9)))
        assert manager.can_allocate(second) == 1
        manager.allocate(second, 1)
        assert second.prefix_hit_tokens == 4 and second.num_cached_tokens == 4
    finally:
        Sequence.block_size = 256
