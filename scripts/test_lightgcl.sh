#!/bin/bash

#SBATCH --job-name=baseline_lightgcl_test
#SBATCH --output=baseline_lightgcl_test.%j.out
#SBATCH --error=baseline_lightgcl_test.%j.err
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00

if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "hyper-enz" ]; then
  source ~/anaconda3/etc/profile.d/conda.sh
  conda activate hyper-enz
fi

PROJECT_ROOT="/home/songt/hyper-enz"
cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

INIT_CKPT_DIR="$1"
if [ -z "$INIT_CKPT_DIR" ]; then
  echo "Usage: sbatch scripts/test_lightgcl.sh <INIT_CKPT_DIR> [valid|test]"
  exit 2
fi
SPLIT="${2:-test}"

python -u baselines/test_lightgcl.py \
  --init "$INIT_CKPT_DIR" \
  --embed_dim 256 \
  --layers 2 \
  --proj_dim 128 \
  --drop_prob 0.2 \
  --agg mean \
  --cuda \
  --split "$SPLIT"

EXIT_CODE=$?
echo "Done with code: $EXIT_CODE"
exit $EXIT_CODE
