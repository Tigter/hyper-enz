#!/bin/bash

#SBATCH --job-name=baseline_lightgcn
#SBATCH --output=baseline_lightgcn.%j.out
#SBATCH --error=baseline_lightgcn.%j.err
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

SAVE_PATH="baselines/lightgcn_run1"

python -u baselines/train_lightgcn.py \
  --save_path "$SAVE_PATH" \
  --embed_dim 256 \
  --layers 2 \
  --dropout 0.3 \
  --lr 1e-3 \
  --epochs 10 \
  --batch_size 2048 \
  --cuda

EXIT_CODE=$?
echo "Done with code: $EXIT_CODE"
exit $EXIT_CODE
