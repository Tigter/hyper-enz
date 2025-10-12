#!/bin/bash

#SBATCH --job-name=hlgnn_relation
#SBATCH --output=hlgnn_relation.%j.out
#SBATCH --error=hlgnn_relation.%j.err
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --constraint=v100|a100

if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "hyper-enz" ]; then
  source ~/anaconda3/etc/profile.d/conda.sh
  conda activate hyper-enz
fi

PROJECT_ROOT="/home/songt/hyper-enz"
cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

SAVE_PATH="hlgnn_relation_run1"

python -u baselines/train_hlgnn_relation.py \
  --save_path "$SAVE_PATH" \
  --embed_dim 256 \
  --layers 15 \
  --mlp_hidden 256 \
  --dropout 0.3 \
  --alpha 0.5 \
  --init KI \
  --lr 1e-3 \
  --epochs 30 \
  --batch_size 512 \
  --agg mean \
  --cuda

EXIT_CODE=$?
echo "Done with code: $EXIT_CODE"
exit $EXIT_CODE

