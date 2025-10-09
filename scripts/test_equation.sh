#!/bin/bash

#SBATCH --job-name=hyperenz_brenda_test
#SBATCH --output=hyperenz_brenda_test.%j.out
#SBATCH --error=hyperenz_brenda_test.%j.err
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=0:30:00
#SBATCH --constraint=v100|a100
#SBATCH --constraint=v100|a100

echo "Date: $(date)"
echo "GPU Information:"
nvidia-smi || true
echo ""

# Activate conda environment
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "hyper-enz" ]; then
  echo "Activating conda environment: hyper-enz"
  source ~/anaconda3/etc/profile.d/conda.sh
  conda activate hyper-enz
fi
echo "Active environment: ${CONDA_DEFAULT_ENV}"

echo "Package versions:"
python -c "import torch; print(f'PyTorch: {torch.__version__}')" || true
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" || true
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')" || true

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONHASHSEED=42

PROJECT_ROOT="/home/songt/hyper-enz"
cd "$PROJECT_ROOT" || exit 1

# Usage:
#   sbatch scripts/test_equation.sh /home/songt/hyper-enz/models/brenda/run1/MRR
# Example INIT_CKPT_DIR: /home/songt/hyper-enz/models/brenda/run1/hit10

INIT_CKPT_INPUT="$1"
CONFIG_NAME_INPUT="$2"
SAVE_PATH_INPUT="$3"

# Defaults (match training script if not provided)
CONFIG_NAME_DEFAULT="hyper_graph_noise_07"
SAVE_PATH_DEFAULT="tests/equation_eval"

CONFIG_NAME="${CONFIG_NAME_INPUT:-$CONFIG_NAME_DEFAULT}"
SAVE_PATH="${SAVE_PATH_INPUT:-$SAVE_PATH_DEFAULT}"
INIT_CKPT="${INIT_CKPT_INPUT}"

if [ -z "$INIT_CKPT" ]; then
  echo "ERROR: INIT_CKPT (first arg) is required."
  echo "Usage: sbatch scripts/test_equation.sh <INIT_CKPT_DIR> [CONFIG_NAME] [SAVE_PATH]"
  exit 2
fi

CKPT_FILE="${INIT_CKPT}/checkpoint"
if [ ! -f "$CKPT_FILE" ]; then
  echo "ERROR: checkpoint not found at: $CKPT_FILE"
  exit 3
fi

# Basic data presence check (same files used in training)
GRAPH_INFO="${PROJECT_ROOT}/pre_handle_data/all/brenda_bigger_lhf_no_e_add_edge_type_add_ne_reaction_graph_info.pkl"
TRAIN_INFO="${PROJECT_ROOT}/pre_handle_data/all/brenda_bigger_lhf_no_e_add_edge_type_add_ne_reaction_train_info.pkl"
if [ ! -f "$GRAPH_INFO" ] || [ ! -f "$TRAIN_INFO" ]; then
  echo "Missing preprocessed data files under pre_handle_data/all/. Aborting."
  exit 1
fi

echo "Starting evaluation..."
set -e
python ${PROJECT_ROOT}/hyper_graph_brenda.py \
  --cuda \
  --test \
  --save_path "$SAVE_PATH" \
  --configName "$CONFIG_NAME" \
  --init "$INIT_CKPT"

EXIT_CODE=${PIPESTATUS[0]}
echo "=================================="
if [ $EXIT_CODE -eq 0 ]; then
  echo "Evaluation completed successfully."
  echo "Logs written to: ${PROJECT_ROOT}/models/${SAVE_PATH}/log/test.log"
else
  echo "Evaluation failed with exit code: $EXIT_CODE"
fi

echo "End time: $(date)"
echo "Final GPU status:"
nvidia-smi || true

exit $EXIT_CODE

