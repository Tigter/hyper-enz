#!/bin/bash

#SBATCH --job-name=baseline_buddy
#SBATCH --output=baseline_buddy.%j.out
#SBATCH --error=baseline_buddy.%j.err
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --constraint=v100|a100

if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "hyper-enz" ]; then
  source ~/anaconda3/etc/profile.d/conda.sh
  conda activate hyper-enz
fi

PROJECT_ROOT="/home/songt/hyper-enz"
cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

SAVE_PATH="baselines/buddy_run1"

python -u baselines/train_buddy.py \
  --save_path "$SAVE_PATH" \
  --batch_size 256 \
  --eval_batch_size 128 \
  --hidden_dim 256 \
  --dropout 0.3 \
  --lr 1e-3 \
  --epochs 10 \
  --k_hop 2 \
  --cuda

EXIT_CODE=$?
echo "Done with code: $EXIT_CODE"
exit $EXIT_CODE
