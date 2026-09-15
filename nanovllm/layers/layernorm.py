import torch
from torch import nn


class RMSNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        *,
        compile_rms: bool = True,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.compile_rms = compile_rms
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def eager_rms_forward(self, x: torch.Tensor) -> torch.Tensor:
        # Keep the BF16/FP16 rounding boundary before multiplying the weight.
        # Called outside torch.compile; CUDA Graph may capture these eager ops.
        x32 = x.float()
        variance = x32.pow(2).mean(dim=-1, keepdim=True)
        normalized = (x32 * torch.rsqrt(variance + self.eps)).to(x.dtype)
        return normalized * self.weight

    @torch.compile
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x

    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        x = x.float().add_(residual.float())
        residual = x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            if not self.compile_rms:
                return self.eager_rms_forward(x)
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)
