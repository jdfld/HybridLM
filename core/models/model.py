import math
from typing import Optional

import torch
from torch import nn
from torch.nn import RMSNorm

from .mlps import MLP, SwiGLU, MLPReluSquared, ConvSwiGLU
from .model_config import ModelConfig
from .layers import (
    Attention, LinearAttention, GatedLinearAttention,
    DeltaNet, GatedDeltaNet, ShortConv
)
# 


MLP_CLASSES = {
    'mlp': MLP,
    'swiglu': SwiGLU,
    'mlp_relu_sq': MLPReluSquared,
    'conv_glu': ConvSwiGLU,
}

SEQUENCE_MIXERS = {
    'attn':  Attention,
    'la':    LinearAttention,
    'gla':   GatedLinearAttention,
    'dn':    DeltaNet,
    'gdn':   GatedDeltaNet,
    'conv':  ShortConv,
}


def _build_mixer(mixer_key: str, layer_id: int, cfg: ModelConfig, extra_kwargs: dict) -> nn.Module:
    """
    Instantiate a sequence mixer by key.

    All FLA layer constructors accept **kwargs, so we can safely pass both
    `d_model` and `hidden_size`; each layer uses whichever name it declares
    and silently drops the other via **kwargs.
    `layer_idx` is forwarded so FLA layers can key their KV-cache correctly.
    """
    if mixer_key not in SEQUENCE_MIXERS:
        raise ValueError(
            f"Unknown sequence mixer '{mixer_key}'. "
            f"Available: {list(SEQUENCE_MIXERS.keys())}"
        )
    cls = SEQUENCE_MIXERS[mixer_key]
    kwargs = {
        'd_model':     cfg.dim,   
        'num_heads':   cfg.n_heads,
        'layer_idx':   layer_id,
        **extra_kwargs,
    }
    return cls(**kwargs)


class Block(nn.Module):
    """
    A single transformer block: pre-norm sequence mixer + pre-norm state mixer (MLP).

    Handles the variable return signatures of FLA layers transparently:
      - Raw tensor          (LinearAttention, GatedLinearAttention, DeltaNet, GatedDeltaNet)
      - (o, attn, cache)   (Attention, MLA, KimiDeltaAttention)
    """

    def __init__(self, layer_id: int, cfg: ModelConfig):
        super().__init__()
        self.layer_id = layer_id

        use_secondary = (
            cfg.secondary_sequence_mixer is not None
            and cfg.secondary_every is not None
            and (layer_id + 1) % cfg.secondary_every == 0
        )
        if use_secondary:
            self.mixer = _build_mixer(
                cfg.secondary_sequence_mixer, layer_id, cfg,
                cfg.secondary_sequence_mixer_config,
            )
        else:
            self.mixer = _build_mixer(
                cfg.sequence_mixer, layer_id, cfg,
                cfg.sequence_mixer_config,
            )

        self.mixer_norm = RMSNorm(cfg.dim, cfg.rmsnorm_eps)
        self.mlp = MLP_CLASSES[cfg.state_mixer](cfg.dim, **cfg.state_mixer_config)
        self.mlp_norm = RMSNorm(cfg.dim, cfg.rmsnorm_eps)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        v_first: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
    ):
        # Run sequence mixer; pass v_first and cu_seqlens via keyword so each
        # layer can pick up what it needs through **kwargs.
        mixer_out = self.mixer(
            self.mixer_norm(x),
            attention_mask=attn_mask,
            v_first=v_first,
            cu_seqlens=cu_seqlens,
        )

        # Unpack variable-length return tuples from FLA layers.
        if isinstance(mixer_out, tuple):
            h = mixer_out[0]
            # RWKV7Attention returns (o, None, past_key_values, v_first)
            v_first = mixer_out[3] if len(mixer_out) >= 4 else v_first
        else:
            h = mixer_out

        x = x + h
        x = x + self.mlp(self.mlp_norm(x))
        return x, v_first



class Model(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.n_layers = cfg.n_layers

        if cfg.dim % cfg.n_heads != 0:
            raise ValueError(
                f"dim ({cfg.dim}) must be divisible by n_heads ({cfg.n_heads})"
            )

        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleList([Block(idx, cfg) for idx in range(cfg.n_layers)])
        self.out_norm = RMSNorm(cfg.dim, cfg.rmsnorm_eps)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

        self.apply(self._init_weights)
        self._scale_residual_branches()

        if cfg.tie_embeddings:
            self.tie_weights()

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
    ):
        # x: (bsz, seqlen)
        x = self.embed_tokens(x)   # (bsz, seqlen, dim)
        v_first = None              # RWKV7 inter-layer state; None for all other mixers
        for layer in self.layers:
            x, v_first = layer(x, attn_mask=attn_mask, v_first=v_first, cu_seqlens=cu_seqlens)
        return self.lm_head(self.out_norm(x))   # (bsz, seqlen, vocab_size)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _scale_residual_branches(self):
        """Apply GPT-2-style residual scaling to output projections."""
        std = 0.02 / math.sqrt(2 * self.n_layers)
        for n, p in self.named_parameters():
            if n.endswith('fc2.weight'):      # MLP / GLU output projection
                torch.nn.init.normal_(p, mean=0.0, std=std)
            if n.endswith('o_proj.weight'):   # FLA mixer output projection
                torch.nn.init.normal_(p, mean=0.0, std=std)

    def tie_weights(self):
        self.lm_head.weight = self.embed_tokens.weight

    def count_params(self, non_embedding: bool = True) -> int:
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.embed_tokens.weight.numel()
            if self.lm_head.weight is not self.embed_tokens.weight:
                n_params -= self.lm_head.weight.numel()
        return n_params
