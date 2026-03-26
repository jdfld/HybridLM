from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from fla.modules import ShortConvolution


class ShortConv(nn.Module):
    """
    A standalone sequence mixer that applies a single short 1D depthwise convolution
    over the full d_model dimension.

    Args:
        d_model (int): Model/hidden dimension size.
        conv_size (int): Kernel size of the convolution. Default: 4.
        activation (str): Activation applied inside ShortConvolution. Default: 'silu'.
        layer_idx (int, optional): Layer index, used for cache look-up. Default: None.
    """

    def __init__(
        self,
        d_model: int,
        conv_size: int = 4,
        activation: str = 'silu',
        layer_idx: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.conv_size = conv_size
        self.layer_idx = layer_idx

        self.conv = ShortConvolution(d_model, conv_size, activation=activation)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values=None,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> torch.Tensor:
        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        conv_state = last_state['conv_state'] if last_state is not None else None
        conv_mask = attention_mask[:, -hidden_states.shape[1]:] if attention_mask is not None else None

        o, conv_state = self.conv(
            x=hidden_states,
            mask=conv_mask,
            cache=conv_state,
            output_final_state=use_cache,
        )

        if past_key_values is not None:
            past_key_values.update(
                conv_state=conv_state,
                layer_idx=self.layer_idx,
                offset=hidden_states.shape[1],
            )

        return self.o_proj(o)
