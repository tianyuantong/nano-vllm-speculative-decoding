"""Variable-length verification's schedule (M4): how many positions each row of a DSpark round verifies.

DSpark's Algorithm 1 (arXiv 2607.05147, §3.2.2): rank every draft position of the batch by its survival probability,
the product of its row's confidences up to it, and admit them in that order while the estimated throughput, expected
tokens over the round's time, keeps rising. Two changes: the time comes from a frozen table of measured round times
interpolated in the token count, so one more token costs one token's worth rather than a whole graph step; and once
the walk stops, the verification is rounded up to a graph's token count and the graph filled with the next positions
in the same order. Whether a row's draft i is verified then depends only on positions ranked no later than it, never
on draft i or the drafts after it, which keeps the acceptance lossless.
"""
import json

import numpy as np

SCHEMA = "verify-schedule-v1"


def verify_token_counts(max_batch_size: int, query_len: int) -> list[int]:
    """Token counts of the variable-length verification graphs: 1, 2, 4 below query_len, then every multiple of
    query_len up to max_batch_size * query_len. B rows of at most query_len tokens each fit the graph of
    B * query_len tokens, which is always one of them."""
    return [size for size in (1, 2, 4) if size < query_len] + list(range(query_len, max_batch_size * query_len + 1,
                                                                         query_len))


def survival_probabilities(confidence_logits: np.ndarray, temperatures: np.ndarray) -> np.ndarray:
    """[B, k] confidence logits -> [B, k] predicted probabilities that a row's drafts up to each position are all
    accepted: the running product of sigmoid(logit / temperature)."""
    return np.cumprod(1.0 / (1.0 + np.exp(-confidence_logits / temperatures)), axis=1)


def verify_lengths(survival: np.ndarray, round_ms: np.ndarray, token_counts: list[int]) -> list[int]:
    """Each row's verified length, 1..k+1, from the survival probabilities [B, k], the round's estimated time by
    token count (round_ms[T] for T = B..B * (k+1)) and the graphs' token counts.

    Positions rank by survival, ties by position then row, so a row's admitted positions are a prefix. The walk
    admits them in that order while expected tokens / round_ms keeps rising; then the smallest graph holding the
    admitted ones is filled with the next positions in the same order, zero-survival ones included."""
    batch_size = survival.shape[0]
    rows, steps = np.indices(survival.shape)
    order = np.lexsort((rows.ravel(), steps.ravel(), -survival.ravel()))
    expected = batch_size + np.concatenate(([0.0], np.cumsum(survival.ravel()[order])))
    throughput = expected / round_ms[batch_size:batch_size + len(expected)]
    drops = np.flatnonzero(throughput[1:] <= throughput[:-1])
    admitted = int(drops[0]) if drops.size else len(order)
    graph_tokens = next(size for size in token_counts if size >= batch_size + admitted)
    taken = rows.ravel()[order[:graph_tokens - batch_size]]
    return (1 + np.bincount(taken, minlength=batch_size)).tolist()


class VerifySchedule:
    """A frozen verify schedule: measured round times by batch size and verified length, and the confidence head's
    temperatures. A round of B rows reads the table row of the smallest batch size at least B, interpolated in the
    token count after making it non-decreasing in the length, so that verifying more never looks cheaper."""

    def __init__(self, batch_sizes: list[int], round_ms: list[list[float]], temperatures: list[float],
                 max_num_seqs: int) -> None:
        query_len = len(temperatures) + 1
        self.batch_sizes = batch_sizes
        self.temperatures = np.asarray(temperatures, dtype=np.float64)
        self.token_counts = verify_token_counts(max_num_seqs, query_len)
        self._round_ms_by_batch_size = {
            batch_size: np.interp(np.arange(batch_size * query_len + 1), batch_size * np.arange(1, query_len + 1),
                                  np.maximum.accumulate(times))
            for batch_size, times in zip(batch_sizes, round_ms)
        }

    @classmethod
    def load(cls, path: str, num_drafts: int, max_num_seqs: int) -> "VerifySchedule":
        """Read a frozen verify-schedule JSON: schema, batch_sizes, round_ms [batch size][length 1..k+1] and
        confidence_temperatures [k]. Raises ValueError when it does not fit this engine."""
        with open(path) as file:
            table = json.load(file)
        if table.get("schema") != SCHEMA:
            raise ValueError(f"{path} is not a {SCHEMA} file")
        batch_sizes, round_ms, temperatures = table["batch_sizes"], table["round_ms"], table["confidence_temperatures"]
        if len(temperatures) != num_drafts or len(round_ms) != len(batch_sizes) \
                or any(len(times) != num_drafts + 1 for times in round_ms):
            raise ValueError(f"{path} does not hold {num_drafts} temperatures and {num_drafts + 1} lengths per batch size")
        if batch_sizes != sorted(set(batch_sizes)) or min(temperatures) <= 0:
            raise ValueError(f"{path} needs increasing batch sizes and positive temperatures")
        if max_num_seqs not in batch_sizes:
            raise ValueError(f"max_num_seqs {max_num_seqs} is not one of the schedule's batch sizes {batch_sizes}")
        return cls(batch_sizes, round_ms, temperatures, max_num_seqs)

    def round_ms(self, batch_size: int) -> np.ndarray:
        """A round of batch_size rows' estimated time by token count, indexed 0..table batch size * (k+1)."""
        return self._round_ms_by_batch_size[next(size for size in self.batch_sizes if size >= batch_size)]

    def verify_lengths(self, confidence_logits: np.ndarray) -> list[int]:
        """[B, k] confidence logits of one round -> each row's verified length."""
        assert np.isfinite(confidence_logits).all(), "non-finite confidence logits"
        survival = survival_probabilities(confidence_logits.astype(np.float64), self.temperatures)
        return verify_lengths(survival, self.round_ms(confidence_logits.shape[0]), self.token_counts)
