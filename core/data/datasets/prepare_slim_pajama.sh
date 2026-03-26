#!/bin/bash

# This script will download and preprocess SlimPajama-627B.
# Expect some token loss by batched concat_chunk.

TOKENIZER = "EleutherAI/gpt-neox-20b"
SEQ_LEN = 2048
BASE_DIR = /fast/jlindqvist
DATASET = "cerebras/SlimPajama-627B"
TMP_DIR = $BASE_DIR/plainlm_tmp
OUT_DIR = $BASE_DIR/data/lm/


mkdir -p tmp_dir

cd ~/plainLM

# TRAIN SET
PYTHONPATH=. python data/datasets/prepare.py \
  --out_path=$OUT_DIR \
  --cache_path=$TMP_DIR \
  --download --tokenize --chunk \
  --save_tokenized --save_tokenizer \
  --dataset_path=$DATASET \
  --dataset_split="train" \
  --dataset_name="sample-100BT" \
  --tokenizer=$TOKENIZER \
  --seq_length=$SEQ_LEN

# VALID SET
PYTHONPATH=. python data/datasets/prepare.py \
  --out_path=$OUT_DIR \
  --cache_path=$TMP_DIR \
  --download --tokenize --chunk \
  --save_tokenized --save_tokenizer \
  --dataset_path=$DATASET \
  --dataset_split="validation" \
  --dataset_name="sample-100BT" \
  --tokenizer=$TOKENIZER \
  --seq_length=$SEQ_LEN
