"""KV metadata only. Updating a counter never computes a missing KV tensor."""

from dataclasses import dataclass, field


@dataclass
class KVState:
    num_cached_tokens: int = 0
    block_table: list[int] = field(default_factory=list)


def blocks_for_budget(budget_bytes: int, block_bytes: int) -> int:
    if (type(budget_bytes) is not int or type(block_bytes) is not int
            or budget_bytes <= 0 or block_bytes <= 0):
        raise ValueError("KV budget and block bytes must be positive integers")
    blocks = budget_bytes // block_bytes
    if blocks == 0:
        raise ValueError("KV budget cannot hold one complete block")
    return blocks
