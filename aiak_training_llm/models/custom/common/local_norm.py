"""megatron local norm — works with or without Apex / TE"""

import torch
import torch.nn as nn

from megatron.core.transformer.transformer_config import TransformerConfig

# ---- Apex ---------------------------------------------------------------
try:
    from apex.normalization.fused_layer_norm import FusedRMSNorm as ApexFusedRMSNorm
    HAVE_APEX_RMS = True
except Exception:
    HAVE_APEX_RMS = False

try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as ApexFusedLayerNorm
    HAVE_APEX_LN = True
except Exception:
    HAVE_APEX_LN = False

# ---- Megatron FusedLayerNorm (needs Apex or persist_layer_norm) ----------
try:
    from megatron.core.fusions.fused_layer_norm import FusedLayerNorm as MegatronFusedLayerNorm
    _HAVE_MEGATRON_FUSED_LN = True
except Exception:
    _HAVE_MEGATRON_FUSED_LN = False

# ---- TE TENorm -----------------------------------------------------------
try:
    from megatron.core.extensions.transformer_engine import TENorm
    _HAVE_TE_NORM = True
except Exception:
    _HAVE_TE_NORM = False


# ---- Plain PyTorch RMSNorm (torch >= 2.4 has it built-in) ---------------
class _TorchRMSNorm(nn.Module):
    """Plain PyTorch RMSNorm, sequence-parallel aware."""
    def __init__(self, config: TransformerConfig, hidden_size: int,
                 eps: float = 1e-5, elementwise_affine: bool = True):
        super().__init__()
        self.eps = eps
        self.hidden_size = hidden_size
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        else:
            self.register_parameter('weight', None)
        self.config = config
        if elementwise_affine:
            setattr(self.weight, 'sequence_parallel', config.sequence_parallel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        norm = norm.to(dtype)
        if self.weight is not None:
            norm = norm * self.weight
        return norm


class _TorchLayerNorm(nn.LayerNorm):
    """Plain PyTorch LayerNorm, sequence-parallel aware."""
    def __init__(self, config: TransformerConfig, hidden_size: int,
                 eps: float = 1e-5, elementwise_affine: bool = True):
        super().__init__(hidden_size, eps=eps, elementwise_affine=elementwise_affine)
        self.config = config
        self.sequence_parallel = config.sequence_parallel
        if elementwise_affine:
            setattr(self.weight, 'sequence_parallel', self.sequence_parallel)
            setattr(self.bias,   'sequence_parallel', self.sequence_parallel)


# ---- FusedRMSNorm --------------------------------------------------------
if HAVE_APEX_RMS:
    class FusedRMSNorm(ApexFusedRMSNorm):
        """Fused RMS Norm backed by Apex."""
        def __init__(self, config: TransformerConfig, hidden_size: int,
                     eps: float = 1e-5, elementwise_affine: bool = True):
            super().__init__(hidden_size, eps=eps, elementwise_affine=elementwise_affine)
            self.config = config
            self.sequence_parallel = config.sequence_parallel
            setattr(self.weight, 'sequence_parallel', self.sequence_parallel)
else:
    FusedRMSNorm = _TorchRMSNorm


# ---- LocalNorm -----------------------------------------------------------
class LocalNorm:
    """
    Returns the right LayerNorm / RMSNorm instance based on config.normalization.
    Priority: Apex > MegatronFused > TENorm > plain PyTorch.
    """

    def __new__(cls, config: TransformerConfig, hidden_size: int,
                eps: float = 1e-5, elementwise_affine: bool = True):

        if config.normalization == "LayerNorm":
            # 1. Try Megatron FusedLayerNorm (needs Apex or persist_layer_norm kernel)
            if _HAVE_MEGATRON_FUSED_LN:
                try:
                    return MegatronFusedLayerNorm(config=config, hidden_size=hidden_size, eps=eps)
                except Exception:
                    pass  # falls through to next option
            # 2. Try TE TENorm
            if _HAVE_TE_NORM:
                try:
                    return TENorm(config=config, hidden_size=hidden_size, eps=eps)
                except Exception:
                    pass
            # 3. Plain PyTorch LayerNorm
            return _TorchLayerNorm(config=config, hidden_size=hidden_size,
                                   eps=eps, elementwise_affine=elementwise_affine)

        elif config.normalization == "RMSNorm":
            return FusedRMSNorm(config=config, hidden_size=hidden_size,
                                eps=eps, elementwise_affine=elementwise_affine)

        else:
            raise ValueError(f"Unsupported normalization: {config.normalization}")
