#!/bin/bash

#SBATCH --job-name=baseline_buddy_test
#SBATCH --output=baseline_buddy_test.%j.out
#SBATCH --error=baseline_buddy_test.%j.err
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --constraint=v100|a100

if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "hyper-enz" ]; then
  source ~/anaconda3/etc/profile.d/conda.sh
  conda activate hyper-enz
fi

PROJECT_ROOT="/home/songt/hyper-enz"
cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

INIT_CKPT_DIR="$1"
if [ -z "$INIT_CKPT_DIR" ]; then
  echo "Usage: sbatch scripts/test_buddy.sh <INIT_CKPT_DIR> [valid|test]"
  exit 2
fi
SPLIT="${2:-test}"

python -u baselines/test_buddy.py \
  --init "$INIT_CKPT_DIR" \
  --hidden_dim 128 \
  --dropout 0.4 \
  --layers 2 \
  --k_hop 2 \
  --max_nodes 800 \
  --cuda \
  --split "$SPLIT"

EXIT_CODE=$?
echo "Done with code: $EXIT_CODE"
exit $EXIT_CODE
