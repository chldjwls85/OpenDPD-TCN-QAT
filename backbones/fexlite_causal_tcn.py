"""RTL-oriented causal TCN for digital predistortion.

Modified work for the OpenDPD-TCN-QAT fork, licensed under Apache-2.0.
"""

from __future__ import annotations

import torch
from torch import nn


class Chomp1d(nn.Module):
    """Remove right padding while preserving causal sequence length."""

    def __init__(self, chomp_size: int):
        super().__init__()
        if chomp_size < 0:
            raise ValueError("chomp_size must be non-negative")
        self.chomp_size = int(chomp_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class FExLiteCausalTCN(nn.Module):
    """Parameterizable FExLite causal TCN with an I/Q residual.

    The feature order is ``I, Q, p, p^2, I*p, Q*p`` with
    ``p = I^2 + Q^2``.  A pointwise input projection is followed by
    ``num_layers`` depthwise temporal layers whose dilations grow as powers
    of two.  This topology maps directly to the TCN-Compiler manifest contract.
    """

    def __init__(
        self,
        hidden_channels: int,
        num_layers: int = 4,
        kernel_size: int = 5,
        dilation_base: int = 2,
    ):
        super().__init__()
        if hidden_channels < 1:
            raise ValueError("hidden_channels must be positive")
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        if kernel_size < 1:
            raise ValueError("kernel_size must be positive")
        if dilation_base < 1:
            raise ValueError("dilation_base must be positive")

        self.in_channels = 6
        self.hidden_channels = int(hidden_channels)
        self.out_channels = 2
        self.kernel_size = int(kernel_size)
        self.num_layers = int(num_layers)
        self.dilations = tuple(
            int(dilation_base**index) for index in range(self.num_layers)
        )
        self.register_buffer(
            "_rtl_spec",
            torch.tensor(
                [1, self.num_layers, self.kernel_size, int(dilation_base)],
                dtype=torch.int32,
            ),
        )

        modules: list[nn.Module] = [
            nn.Conv1d(self.in_channels, self.hidden_channels, kernel_size=1),
            nn.Hardswish(),
        ]
        for dilation in self.dilations:
            padding = (self.kernel_size - 1) * dilation
            modules.extend(
                [
                    nn.Conv1d(
                        self.hidden_channels,
                        self.hidden_channels,
                        self.kernel_size,
                        padding=padding,
                        dilation=dilation,
                        groups=self.hidden_channels,
                        bias=False,
                    ),
                    Chomp1d(padding),
                    nn.Hardswish(),
                ]
            )
        modules.append(
            nn.Conv1d(
                self.hidden_channels,
                self.out_channels,
                kernel_size=1,
                bias=False,
            )
        )
        self.network = nn.Sequential(*modules)

    @property
    def receptive_field_samples(self) -> int:
        return 1 + sum(
            (self.kernel_size - 1) * dilation for dilation in self.dilations
        )

    def count_flops(self, input_shape) -> int:
        del input_shape
        feature_ops = 8
        projection_macs = self.in_channels * self.hidden_channels
        temporal_macs = (
            self.num_layers * self.kernel_size * self.hidden_channels
        )
        hardswish_ops = (
            (self.num_layers + 1) * self.hidden_channels * 4
        )
        output_macs = self.hidden_channels * self.out_channels
        residual_ops = self.out_channels
        return (
            feature_ops
            + projection_macs
            + temporal_macs
            + hardswish_ops
            + output_macs
            + residual_ops
        )

    def forward(
        self, x: torch.Tensor, h_0: torch.Tensor | None = None
    ) -> torch.Tensor:
        del h_0
        i_x = x[..., 0:1]
        q_x = x[..., 1:2]
        power = i_x.square() + q_x.square()
        features = torch.cat(
            (i_x, q_x, power, power.square(), i_x * power, q_x * power),
            dim=-1,
        )
        correction = self.network(features.transpose(1, 2)).transpose(1, 2)
        return correction + torch.cat((i_x, q_x), dim=-1)
