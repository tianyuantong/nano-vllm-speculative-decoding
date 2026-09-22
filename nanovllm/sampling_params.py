from dataclasses import dataclass


@dataclass(slots=True)
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False
    top_k: int = 0        # 0 disables top-k
    top_p: float = 1.0    # 1.0 disables top-p

    def __post_init__(self):
        if not (self.temperature == 0 or self.temperature > 1e-10):
            raise ValueError("temperature must be exactly 0 (greedy) or > 1e-10")
        if self.top_k < 0:
            raise ValueError("top_k must be >= 0 (0 disables it)")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.top_p < 1 and self.top_k == 0:
            raise ValueError("top_p requires top_k > 0: the nucleus is computed over the top-k candidates")
