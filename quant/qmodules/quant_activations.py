"""QAT activation modules used by the causal-TCN hardware path."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .quantizers import DISCARD_LSB_SIGNED_FLOOR, INT_Quantizer


class INT_Hardswish(torch.nn.Module):
    """HardSwish preceded by an explicit signed power-of-two quantizer."""

    def __init__(
        self,
        bits: int,
        rounding: str = DISCARD_LSB_SIGNED_FLOOR,
    ) -> None:
        super().__init__()
        self.input_quantizer = INT_Quantizer(
            bits=bits,
            all_positive=False,
            rounding=rounding,
        )
        self.input_quantizer.init_act_params()
        minimum_scale = 2.0 ** math.ceil(
            math.log2(3.0 / self.input_quantizer.Qp)
        )
        with torch.no_grad():
            self.input_quantizer.scale.fill_(minimum_scale)
        self.register_buffer("n_bits_a", torch.tensor([int(bits)]))

    def forward(self, x):
        return F.hardswish(self.input_quantizer(x))
