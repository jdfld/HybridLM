#!/bin/bash

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

if ! command -v uv &> /dev/null; then
    echo "uv not found in PATH. Attempting automatic install..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Source the new environment settings for this session
    source $HOME/.local/bin/env 2>/dev/null || export PATH="$HOME/.local/bin:$PATH"
fi

source /etc/profile.d/modules.sh
module load cuda/12.4

export PYTHONPATH=$PYTHONPATH:.

# Job specific vars
config=$1
job_idx=$2 # CONDOR job arrays range from 0 to n-1

# Check number of gpus before starting...
NUM_GPUS=$(uv run python -c "import torch; print(torch.cuda.device_count())")

uv run python -c "import fla"

if [ "$NUM_GPUS" -le 1 ]; then # if only a single gpu run python
  uv run python -u plainLM/train.py --config=$config --job_idx=$job_idx
else # start the torchrun distributed mode
# Build redirects string: "1:0,2:0,...,N-1:0"
  REDIRECTS=$(python -c "n=$NUM_GPUS; print(','.join(f'{i}:0' for i in range(1,n)))")
  uv run torchrun \
    --redirects $REDIRECTS \
    --standalone --nnodes=1 \
    --nproc_per_node=$NUM_GPUS \
    plainLM/train.py --config=$config --job_idx=$job_idx
fi