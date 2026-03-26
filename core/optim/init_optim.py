"""Intialize optimizer and scheduler."""

import torch
from .lr_schedule import WarmupCosine, WSD, WarmupConstant, LinearCooldown


def intialize_optimizer(param_groups, cfg):
  """
  Intialize an optimizer.
  NOTE: we pass weight_decay to optim, but it gets overwritten by the weight_decay in param_groups!
  """

  if cfg.optim == 'adamw':
    optimizer = torch.optim.AdamW(
      param_groups,
      lr=cfg.lr,
      betas=[cfg.beta1, cfg.beta2],
      weight_decay=cfg.weight_decay,
      fused=cfg.fused_optim,
      eps=getattr(cfg, 'eps', 1e-8),
    )

  elif cfg.optim == 'nadamw':
    optimizer = torch.optim.NAdam(
      param_groups,
      lr=cfg.lr,
      betas=[cfg.beta1, cfg.beta2],
      weight_decay=cfg.weight_decay,
      decoupled_weight_decay=True,
      fused=cfg.fused_optim,
      eps=getattr(cfg, 'eps', 1e-8),
    )

  elif cfg.optim == 'sgd':
    optimizer = torch.optim.SGD(
      param_groups,
      lr=cfg.lr,
      momentum=cfg.beta1,
      dampening=cfg.dampening,
      weight_decay=cfg.weight_decay,
    )

  elif cfg.optim == 'signSGD':
    from .signSGD import signSGD

    optimizer = signSGD(
      param_groups,
      lr=cfg.lr,
      momentum=cfg.beta1,
      dampening=cfg.dampening,
      weight_decay=cfg.weight_decay,
    )

  elif cfg.optim == 'sfo_adamw':
    import schedulefree

    # warmup steps for schedulefree must be specified here
    warmup_steps = cfg.warmup_steps if isinstance(cfg.warmup_steps, int) else int(cfg.warmup_steps * cfg.steps_budget)
    optimizer = schedulefree.AdamWScheduleFree(
      param_groups,
      lr=cfg.lr,
      warmup_steps=warmup_steps,
      betas=[cfg.beta1, cfg.beta2],
      weight_decay=cfg.weight_decay,
    )

  elif cfg.optim == 'muon':
    optimizer = MuonWithAdamW(
      param_groups,
      lr=cfg.lr,
      beta1=cfg.beta1,
      beta2=cfg.beta2,
      eps=getattr(cfg, 'eps', 1e-8),
      muon_momentum=getattr(cfg, 'muon_momentum', 0.95),
      muon_nesterov=getattr(cfg, 'muon_nesterov', True),
      muon_ns_steps=getattr(cfg, 'muon_ns_steps', 5),
    )

  else:
    raise NotImplementedError(f'Not implemented optim: {cfg.optim}.')

  return optimizer


def initialize_scheduler(optimizer, cfg):
  if cfg.scheduler is None:
    return None

  ## Number of warmup steps
  # either specified directly (int) or as a fraction of steps_budget (float)
  if getattr(cfg, 'warmup_steps', None) is not None:
    warmup_steps = cfg.warmup_steps if isinstance(cfg.warmup_steps, int) else int(cfg.warmup_steps * cfg.steps_budget)

  ## Number of cooldown steps
  # either specified directly (int) or as a fraction of steps_budget (float)
  if getattr(cfg, 'cooldown_steps', None) is not None:
    cooldown_steps = (
      cfg.cooldown_steps if isinstance(cfg.cooldown_steps, int) else int(cfg.cooldown_steps * cfg.steps_budget)
    )

  ##Final LR of the schedule
  # either specified directly via `lr_end` or as a fraction of top lr via `lr_end_pct`
  if getattr(cfg, 'lr_end', None) is not None or getattr(cfg, 'lr_end_pct', None) is not None:
    lr_end = cfg.lr_end if (cfg.lr_end is not None) else (cfg.lr_end_pct * cfg.lr)

  if cfg.scheduler == 'warmup_cosine':
    scheduler = WarmupCosine(
      optimizer,
      lr_start=cfg.lr_start,
      lr_max=cfg.lr,
      lr_end=lr_end,
      warmup_steps=warmup_steps,
      T=cfg.steps_budget,
    )

  elif cfg.scheduler == 'wsd':
    cooldown_start_step = cfg.steps_budget - cooldown_steps
    scheduler = WSD(
      optimizer,
      lr_start=cfg.lr_start,
      lr_max=cfg.lr,
      lr_end=lr_end,
      warmup_steps=warmup_steps,
      cooldown_start_step=cooldown_start_step,
      cooldown_steps=cooldown_steps,
    )

  elif cfg.scheduler == 'warmup_constant':
    scheduler = WarmupConstant(
      optimizer,
      lr_start=cfg.lr_start,
      lr_max=cfg.lr,
      warmup_steps=warmup_steps,
    )

  elif cfg.scheduler == 'linear_cooldown':
    cooldown_start_step = cfg.resume_step
    scheduler = LinearCooldown(
      optimizer,
      lr_max=cfg.lr,
      lr_end=lr_end,
      cooldown_start_step=cooldown_start_step,
      cooldown_steps=cooldown_steps,
    )

  else:
    raise NotImplementedError(f'Not implemented scheduler: {cfg.scheduler}.')

  return scheduler


class MuonWithAdamW:
  """
  Combined optimiser: torch.optim.Muon for 2-D parameters, AdamW for all others.

  Parameters with ndim == 2 (weight matrices) are updated with Muon;
  everything else (biases, norms, embeddings, scalars) is updated with AdamW.

  Muon is constructed with adjust_lr_fn="match_rms_adamw", which scales each
  update by 0.2 * sqrt(max(rows, cols)).  This normalises the per-element RMS
  of the orthogonalised update to match AdamW's, so both optimisers can share
  the same lr without additional tuning.

  The param_groups property returns the concatenated groups of both internal
  optimisers.  Because Python dicts are mutable and schedulers write
  group['lr'] in-place, a single LR schedule automatically controls both
  optimisers with no extra wiring.

  Args:
      param_groups: List of parameter-group dicts (same format as any
                    PyTorch optimiser). Each dict must contain 'params';
                    'weight_decay' is forwarded to both sub-optimisers.
      lr:           Learning rate shared by Muon and AdamW.
      beta1:        AdamW beta1 (default: 0.9).
      beta2:        AdamW beta2 (default: 0.95).
      eps:          AdamW / Muon numerical stability epsilon (default: 1e-8).
      muon_momentum:  Muon momentum coefficient (default: 0.95).
      muon_nesterov:  Use Nesterov momentum in Muon (default: True).
      muon_ns_steps:  Newton-Schulz iterations (default: 5).
  """

  def __init__(
    self,
    param_groups,
    lr: float,
    beta1: float = 0.9,
    beta2: float = 0.95,
    eps: float = 1e-8,
    muon_momentum: float = 0.95,
    muon_nesterov: bool = True,
    muon_ns_steps: int = 5,
  ):
    muon_groups = []
    adam_groups = []

    for group in param_groups:
      if isinstance(group, dict):
        params = group['params']
        extra  = {k: v for k, v in group.items() if k != 'params'}
      else:
        params = list(group)
        extra  = {}

      # torch.optim.Muon only accepts strictly 2-D tensors.
      muon_ps = [p for p in params if p.ndim == 2]
      adam_ps  = [p for p in params if p.ndim != 2]

      if muon_ps:
        muon_groups.append({'params': muon_ps, **extra})
      if adam_ps:
        adam_groups.append({'params': adam_ps,  **extra})

    self.muon = torch.optim.Muon(
      muon_groups,
      lr=lr,
      momentum=muon_momentum,
      nesterov=muon_nesterov,
      ns_steps=muon_ns_steps,
      eps=eps,
      adjust_lr_fn='match_rms_adamw',
    )
    self.adamw = torch.optim.AdamW(
      adam_groups,
      lr=lr,
      betas=(beta1, beta2),
      eps=eps,
    )

  @property
  def param_groups(self):
    return self.muon.param_groups + self.adamw.param_groups

  def zero_grad(self, set_to_none: bool = True):
    self.muon.zero_grad(set_to_none=set_to_none)
    self.adamw.zero_grad(set_to_none=set_to_none)

  def step(self, closure=None):
    self.muon.step(closure)
    self.adamw.step(closure)

  def state_dict(self):
    return {'muon': self.muon.state_dict(), 'adamw': self.adamw.state_dict()}

  def load_state_dict(self, state_dict):
    self.muon.load_state_dict(state_dict['muon'])
    self.adamw.load_state_dict(state_dict['adamw'])