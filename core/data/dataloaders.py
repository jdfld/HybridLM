import torch

from datasets import Dataset, load_from_disk, load_dataset
from torch import distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from transformers import AutoTokenizer

from data.datasamplers import StatefulSequentialSampler, StatefulRandomSampler, StatefulDistributedSampler
from data.streaming_tokenized_dataset import TokenizedDataset


def get_dataloaders(cfg):
  """Load trainset and perhaps validset. Returns correspondent DataLoaders.

  Dispatches to one of two modes based on cfg.streaming_tokenization:
    - False (default): loads pre-tokenized HuggingFace datasets from disk.
    - True:            loads a raw-text HuggingFace dataset and tokenizes
                       on the fly using OnTheFlyTokenizedDataset.
  """
  if getattr(cfg, 'streaming_tokenization', False):
    return _get_streaming_dataloaders(cfg)
  return _get_pretokenized_dataloaders(cfg)


# ---------------------------------------------------------------------------
# Pre-tokenized path (original behaviour)
# ---------------------------------------------------------------------------

def _get_pretokenized_dataloaders(cfg):
  """Load pre-tokenized, chunked datasets from disk."""

  train_set = load_from_disk(cfg.trainset_path)
  if not isinstance(train_set, Dataset):
    raise ValueError('dataset should be a datasets.Dataset')

  train_sampler = _get_sampler(train_set, cfg)

  # only used with intra-document masking
  def collate_fn(batch):
    return {
      'input_ids': torch.stack([x['input_ids'] for x in batch], dim=0),
      'docs_lengths': [x['docs_lengths'].tolist() for x in batch],
    }

  trainloader = DataLoader(
    train_set,
    sampler=train_sampler,
    batch_size=cfg.micro_batch_size,
    num_workers=cfg.num_workers,
    pin_memory=True,
    prefetch_factor=2 if cfg.num_workers > 0 else None,
    persistent_workers=True if cfg.num_workers > 0 else False,
    collate_fn=collate_fn if 'docs_lengths' in train_set.column_names else None,
  )

  if not cfg.validset_path:
    validloader = None
  else:
    valid_set = load_from_disk(cfg.validset_path)
    if not isinstance(valid_set, Dataset):
      raise ValueError("'dataset' should be a datasets.Dataset")

    if getattr(cfg, 'valid_tokens', False):  # subsample validation set
      valid_rows = cfg.valid_tokens // (cfg.seq_len + 1)
      valid_set = valid_set.take(valid_rows)

    if dist.is_initialized():
      valid_sampler = DistributedSampler(valid_set, drop_last=True)
    else:
      valid_sampler = SequentialSampler(valid_set)

    validloader = DataLoader(
      valid_set,
      batch_size=cfg.micro_batch_size,
      drop_last=True,  # makes eval with DDP easier
      shuffle=False,
      sampler=valid_sampler,
      num_workers=cfg.num_workers,
      pin_memory=True,
      prefetch_factor=2 if cfg.num_workers > 0 else None,
      persistent_workers=False,
      collate_fn=collate_fn if 'docs_lengths' in valid_set.column_names else None,
    )

  return trainloader, validloader


# ---------------------------------------------------------------------------
# On-the-fly tokenization path
# ---------------------------------------------------------------------------

def _get_streaming_dataloaders(cfg):
  """Build DataLoaders that tokenize raw HuggingFace text datasets on the fly."""

  hf_dataset_path = getattr(cfg, 'hf_dataset_path', None)
  if hf_dataset_path is None:
    raise ValueError(
      'cfg.hf_dataset_path must be set when cfg.streaming_tokenization=True.'
    )

  tokenizer_path = getattr(cfg, 'tokenizer_path', None)
  if tokenizer_path is None:
    raise ValueError(
      'cfg.tokenizer_path must be set when cfg.streaming_tokenization=True.'
    )

  hf_dataset_name = getattr(cfg, 'hf_dataset_name', None)
  hf_dataset_split = getattr(cfg, 'hf_dataset_split', 'train')
  hf_streaming = getattr(cfg, 'hf_streaming', True)
  text_column = getattr(cfg, 'text_column', 'text')
  intra_doc_masking = getattr(cfg, 'intra_doc_masking', False)
  add_bos = getattr(cfg, 'add_bos', True)
  add_eos = getattr(cfg, 'add_eos', True)

  # Tokenizer
  tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
  # Disable truncation — we handle chunking ourselves
  tokenizer.model_max_length = int(1e30)

  # DDP info
  ddp = dist.is_initialized()
  rank = dist.get_rank() if ddp else 0
  world_size = dist.get_world_size() if ddp else 1

  # Training dataset
  train_hf_ds = load_dataset(
    hf_dataset_path,
    name=hf_dataset_name,
    split=hf_dataset_split,
    streaming=hf_streaming,
  )

  train_ds = TokenizedDataset(
    dataset=train_hf_ds,
    tokenizer=tokenizer,
    seq_len=cfg.seq_len,
    text_column=text_column,
    intra_doc_masking=intra_doc_masking,
    rank=rank,
    world_size=world_size,
    add_bos=add_bos,
    add_eos=add_eos,
  )

  trainloader = DataLoader(
    train_ds,
    batch_size=cfg.micro_batch_size,
    num_workers=cfg.num_workers,
    pin_memory=True,
    prefetch_factor=2 if cfg.num_workers > 0 else None,
    persistent_workers=True if cfg.num_workers > 0 else False,
    drop_last=True,  # ensures all DDP ranks see the same number of batches
  )

  # Validation loader — falls back to the pre-tokenized path when
  # cfg.validset_path is provided; otherwise tries hf_valid_dataset_split.
  validloader = None
  validset_path = getattr(cfg, 'validset_path', None)
  hf_valid_split = getattr(cfg, 'hf_valid_dataset_split', None)

  if validset_path:
    # Use pre-tokenized validation set from disk
    valid_set = load_from_disk(validset_path)
    if not isinstance(valid_set, Dataset):
      raise ValueError("'validset_path' should point to a datasets.Dataset")

    if getattr(cfg, 'valid_tokens', False):
      valid_rows = cfg.valid_tokens // (cfg.seq_len + 1)
      valid_set = valid_set.take(valid_rows)

    valid_sampler = DistributedSampler(valid_set, drop_last=True) if ddp else SequentialSampler(valid_set)

    def _pretok_collate(batch):
      result = {'input_ids': torch.stack([x['input_ids'] for x in batch], dim=0)}
      if 'docs_lengths' in valid_set.column_names:
        result['docs_lengths'] = [x['docs_lengths'].tolist() for x in batch]
      return result

    validloader = DataLoader(
      valid_set,
      batch_size=cfg.micro_batch_size,
      drop_last=True,
      shuffle=False,
      sampler=valid_sampler,
      num_workers=cfg.num_workers,
      pin_memory=True,
      prefetch_factor=2 if cfg.num_workers > 0 else None,
      persistent_workers=False,
      collate_fn=_pretok_collate,
    )

  elif hf_valid_split:
    # Stream and tokenize a separate validation split on the fly
    valid_hf_ds = load_dataset(
      hf_dataset_path,
      name=hf_dataset_name,
      split=hf_valid_split,
      streaming=hf_streaming,
    )

    valid_ds = TokenizedDataset(
      dataset=valid_hf_ds,
      tokenizer=tokenizer,
      seq_len=cfg.seq_len,
      text_column=text_column,
      intra_doc_masking=intra_doc_masking,
      rank=rank,
      world_size=world_size,
      add_bos=add_bos,
      add_eos=add_eos,
    )

    validloader = DataLoader(
      valid_ds,
      batch_size=cfg.micro_batch_size,
      num_workers=cfg.num_workers,
      pin_memory=True,
      prefetch_factor=2 if cfg.num_workers > 0 else None,
      persistent_workers=False,
      drop_last=True
    )

  return trainloader, validloader


# ---------------------------------------------------------------------------
# Sampler factory (used by pre-tokenized path only)
# ---------------------------------------------------------------------------

def _get_sampler(train_set, cfg):
  """Initializes a sampler for a torch.Dataloader.
  Options:
    - random sampler
    - sequential sampler
    - stateful random sampler
    - stateful sequential sampler
  We implement "stateful" sequential samplers for resuming training from a specified step.
  """
  ddp = dist.is_initialized()

  if cfg.sampler == 'random':
    if ddp:
      sampler = DistributedSampler(train_set, shuffle=True, seed=cfg.sampler_seed, drop_last=True)
    else:
      sampler = RandomSampler(
        train_set, generator=torch.Generator().manual_seed(cfg.sampler_seed) if cfg.sampler_seed else None
      )

  elif cfg.sampler == 'sequential':
    if ddp:
      sampler = DistributedSampler(train_set, shuffle=False, drop_last=True)
    else:
      sampler = SequentialSampler(train_set)

  elif cfg.sampler == 'stateful_random':
    micro_step_start = cfg.resume_step * cfg.grad_accumulation_steps if cfg.resume else 0
    if ddp:
      # TODO: allow support for drop_last=True!
      sampler = StatefulDistributedSampler(
        train_set, batch_size=cfg.micro_batch_size, seed=cfg.sampler_seed, start_iter=micro_step_start
      )
    else:
      sampler = StatefulRandomSampler(
        train_set, batch_size=cfg.micro_batch_size, shuffle=True, seed=cfg.sampler_seed, start_idx=micro_step_start
      )

  elif cfg.sampler == 'stateful_sequential':
    micro_step_start = cfg.resume_step * cfg.grad_accumulation_steps if cfg.resume else 0
    if ddp:
      raise NotImplementedError('StatefulDistributedSampler currently needs a seed.')
    else:
      sampler = StatefulSequentialSampler(train_set, batch_size=cfg.micro_batch_size, start_idx=micro_step_start)

  else:
    raise NotImplementedError(f'Sampler {cfg.sampler} is not implemented.')

  return sampler
