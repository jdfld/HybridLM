#!/bin/bash

# This script will download and preprocess FineWebEdu-100BT.
# Expect some token loss by batched concat_chunk.


# Expect some token loss by batched concat_chunk.
TOKENIZER="Llama-2"
TOKENIZER_PATH="meta-llama/LLama-2-7b-hf"
SEQ_LEN=2048
BASE_DIR="/fast/jlindqvist"
DATASET="fineweb-edu"
DATASET_PATH="HuggingFaceFW/${DATASET}"
TMP_DIR="${BASE_DIR}/plainLM_tmp"
OUT_DIR="${BASE_DIR}/data/lm/${DATASET}/${TOKENIZER}"
SUBSET="sample-100BT"

mkdir -p $TMP_DIR

cd ~/MRNN_Memory/plainLM


export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

if ! command -v uv &> /dev/null; then
    echo "uv not found in PATH. Attempting automatic install..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Source the new environment settings for this session
    source $HOME/.local/bin/env 2>/dev/null || export PATH="$HOME/.local/bin:$PATH"
fi


export UV_LINK_MODE=copy
export PYTHONPATH=$PYTHONPATH:.


uv sync
uv run python data/datasets/prepare.py \
  --out_path=$OUT_DIR \
  --cache_path=$TMP_DIR \
  --chunk \
  --save_tokenized --save_tokenizer \
  --dataset_path=$DATASET_PATH \
  --dataset_split="train" \
  --dataset_name=$SUBSET \
  --tokenizer=$TOKENIZER_PATH \
  --seq_length=$SEQ_LEN \
  --split_train_valid \
  --n_tokens_valid=10000000
