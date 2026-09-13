"""Deterministic history proposals. No model, RNG, or persistent index."""


def history_proposal(tokens, budget, *, min_match=1, max_match=4):
    if type(budget) is not int or budget < 0:
        raise ValueError("nonnegative proposal budget required")
    if not 1 <= min_match <= max_match:
        raise ValueError("positive ordered match bounds required")
    if not budget:
        return []
    # Longest suffix first, then earliest previous occurrence. A match may
    # overlap the current suffix, but its continuation never reads future IDs.
    for size in range(min(max_match, len(tokens) - 1), min_match - 1, -1):
        suffix = tokens[-size:]
        for start in range(len(tokens) - size):
            if tokens[start:start + size] == suffix:
                end = start + size
                return list(tokens[end:end + budget])
    return []
