"""QAT activation modules used by the causal-TCN hardware path."""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F

from .quantizers import DISCARD_LSB_SIGNED_FLOOR, INT_Quantizer


class INT_Hardswish(torch.nn.Module):
    """HardSwish with an explicit signed integer input boundary.

    The convolution still accumulates at full precision.  Immediately before
    HardSwish, ``input_quantizer`` maps that accumulator value onto a signed
    ``bits``-wide power-of-two grid.  The default rounding policy implements
    literal two's-complement LSB discard (floor for negative values as well).
    """

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
        # Preserve both +/-3 region boundaries even before data calibration.
        minimum_scale = 2.0 ** math.ceil(
            math.log2(3.0 / self.input_quantizer.Qp)
        )
        with torch.no_grad():
            self.input_quantizer.scale.fill_(minimum_scale)
        self.register_buffer("n_bits_a", torch.tensor([int(bits)]))

    def forward(self, x):
        return F.hardswish(self.input_quantizer(x))
