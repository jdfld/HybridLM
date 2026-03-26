from fractions import Fraction

import wandb

from .model import Model
from .model_config import ModelConfig


def construct_model(cfg):
    """Initialise a model from a training config namedtuple. Counts and prints parameters."""

    if getattr(cfg, 'hf', None) is not None:
        # ------------------------------------------------------------------ #
        # HuggingFace model path                                               #
        # ------------------------------------------------------------------ #
        from transformers import AutoConfig, AutoModelForCausalLM

        if getattr(cfg, 'hf_init', False):
            # Architecture from a pretrained config, weights randomly initialised.
            hf_cfg = AutoConfig.from_pretrained(cfg.hf_name)
            model = AutoModelForCausalLM.from_config(hf_cfg)
            model.init_weights()
        else:
            # Load pretrained weights directly.
            model = AutoModelForCausalLM.from_pretrained(cfg.hf_name)

        model_cfg = model.config

    else:
        # ------------------------------------------------------------------ #
        # Custom plainLM Model                                                 #
        # ------------------------------------------------------------------ #
        hidden_dim = int(Fraction(cfg.expand) * cfg.d_model)

        model_cfg = ModelConfig(
            vocab_size=cfg.vocab_size,
            dim=cfg.d_model,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            seq_len=cfg.seq_len,
            state_mixer=cfg.mlp_class,
            # hidden_dim is the only required key; user extra keys are merged on top.
            state_mixer_config={
                'hidden_dim': hidden_dim,
                **_as_dict(getattr(cfg, 'state_mixer_config', None)),
            },
            sequence_mixer=getattr(cfg, 'sequence_mixer', 'gdn'),
            sequence_mixer_config=_as_dict(
                getattr(cfg, 'sequence_mixer_config', None)
            ),
            secondary_sequence_mixer=getattr(cfg, 'secondary_sequence_mixer', None),
            secondary_sequence_mixer_config=_as_dict(
                getattr(cfg, 'secondary_sequence_mixer_config', None)
            ),
            secondary_every=getattr(cfg, 'secondary_every', None),
            tie_embeddings=cfg.tie_embeddings,
            rmsnorm_eps=getattr(cfg, 'rmsnorm_eps', 1e-6),
            use_rope=getattr(cfg, 'use_rope', False),
        )
        model = Model(model_cfg)

    if hasattr(model, 'count_params'):
        n_params = model.count_params(non_embedding=False)
        n_params_no_embed = model.count_params(non_embedding=True)
        print(f'Number of parameters: {n_params:_}')
        print(f'Number of non-embedding parameters: {n_params_no_embed:_}')
        if wandb.run is not None:
            wandb.log({'n_params': n_params, 'n_params_no_embed': n_params_no_embed})

    return model, model_cfg


def _as_dict(value) -> dict:
    """Return value as-is if it is already a dict, otherwise return {}."""
    return value if isinstance(value, dict) else {}


def get_param_groups(model, weight_decay):
    """Create param groups with and without weight decay."""

    named_param_dict = {n: p for n, p in model.named_parameters() if p.requires_grad}

    # Exclude parameters that opt out of weight decay (e.g. Mamba's A_log, D)
    decay_params_names = [
        n for n, p in model.named_parameters()
        if not getattr(p, '_no_weight_decay', False)
    ]
    decay_params_names = [n for n in decay_params_names if 'bias' not in n]
    decay_params_names = [n for n in decay_params_names if 'norm' not in n]

    decay_params   = [p for n, p in named_param_dict.items() if     n in decay_params_names]
    no_decay_params = [p for n, p in named_param_dict.items() if n not in decay_params_names]

    return [
        {'params': decay_params,    'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ]
