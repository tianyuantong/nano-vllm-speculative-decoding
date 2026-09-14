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
        )

        scheduler = Scheduler(config)
