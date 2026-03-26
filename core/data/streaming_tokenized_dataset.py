"""
Simple per-document tokenization dataset for fine-tuning or evaluation.

Loads a HuggingFace dataset from disk (Arrow format), tokenizes each document
independently using the provided tokenizer (with padding and truncation), and
yields batches containing input_ids and attention_masks.

No document packing or merging is performed. Each sample is one document,
padded or truncated to the tokenizer's model_max_length.

Usage:
    from data.streaming_tokenized_dataset import get_dataloader

    dataloader = get_dataloader(
        dataset_path="/path/to/hf_dataset",
        tokenizer=tokenizer,
        batch_size=32,
        text_column="text",
    )
    for batch in dataloader:
        input_ids = batch["input_ids"]       # (batch_size, seq_len)
        attention_mask = batch["attention_mask"]  # (batch_size, seq_len)
"""

from __future__ import annotations

from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset

from datasets import load_from_disk
from transformers import PreTrainedTokenizerBase


class TokenizedDataset(Dataset):
    """
    Map-style dataset that loads a HuggingFace Arrow dataset from disk and
    tokenizes each document on access.

    Each item returned is a dict with 'input_ids' and 'attention_mask',
    both of shape (seq_len,), where seq_len is the tokenizer's model_max_length.

    Args:
        dataset_path:   Path to the HuggingFace dataset directory (Arrow files).
        tokenizer:      HuggingFace tokenizer; controls padding, truncation, and
                        special tokens. model_max_length sets the sequence length.
        text_column:    Name of the column containing the raw text (default: 'text').
        max_length:     Override for the tokenizer's model_max_length. If None,
                        uses tokenizer.model_max_length.
        split:          Dataset split to load (e.g. 'train', 'validation').
                        If None, loads the entire dataset.
    """

    def __init__(
        self,
        dataset_path: str,
        tokenizer: PreTrainedTokenizerBase,
        text_column: str = "text",
        max_length: Optional[int] = None,
        split: Optional[str] = None,
    ) -> None:
        super().__init__()

        hf_dataset = load_from_disk(dataset_path)
        if split is not None:
            hf_dataset = hf_dataset[split]
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.max_length = max_length or tokenizer.model_max_length

        if tokenizer.pad_token_id is None:
            raise ValueError(
                f"Tokenizer '{tokenizer.__class__.__name__}' has no pad_token. "
                "Set tokenizer.pad_token = tokenizer.eos_token (or another token) "
                "before constructing this dataset."
            )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        text = self.dataset[idx][self.text_column]

        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),         # (seq_len,)
            "attention_mask": encoded["attention_mask"].squeeze(0),  # (seq_len,)
        }


def get_dataloader(
    dataset_path: str,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
    text_column: str = "text",
    max_length: Optional[int] = None,
    split: Optional[str] = None,
    shuffle: bool = False,
    num_workers: int = 0,
    **dataloader_kwargs,
) -> DataLoader:
    """
    Build a DataLoader over a HuggingFace Arrow dataset stored on disk.

    Each batch is a dict with:
        'input_ids':       LongTensor of shape (batch_size, seq_len)
        'attention_mask':  LongTensor of shape (batch_size, seq_len)

    Args:
        dataset_path:   Path to the HuggingFace dataset directory.
        tokenizer:      HuggingFace tokenizer.
        batch_size:     Number of documents per batch.
        text_column:    Column with raw text (default: 'text').
        max_length:     Sequence length override; defaults to tokenizer.model_max_length.
        split:          Dataset split to load (e.g. 'train'). None loads everything.
        shuffle:        Whether to shuffle the dataset (default: False).
        num_workers:    DataLoader worker processes (default: 0).
        **dataloader_kwargs: Passed through to torch.utils.data.DataLoader.

    Returns:
        A DataLoader that yields dicts of input_ids and attention_mask tensors.
    """
    dataset = TokenizedDataset(
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        text_column=text_column,
        max_length=max_length,
        split=split,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        **dataloader_kwargs,
    )
