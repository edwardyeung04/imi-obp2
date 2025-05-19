#!/bin/bash

# Script to run fair comparison of all imitation learning methods
# This will clean existing models/plots/results and run all methods with standardized parameters

# Set standardized parameters
EPOCHS=50
BATCH_SIZE=4096
EVAL_INTERVAL=5

# Make sure we're in the right directory
cd "$(dirname "$0")"

# Activate the correct conda environment
echo "===== Switching to imi-arm conda environment ====="
eval "$(conda shell.bash hook)"
conda deactivate
conda activate imi-arm

# Create directories if they don't exist
mkdir -p models plots results
mkdir -p results/logprob results/mog results/diffusion results/autoreg_disc

echo "===== Cleaning existing models, plots, and results ====="
# Back them up first
BACKUP_DIR="backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
cp -r models "$BACKUP_DIR/models_backup" 2>/dev/null
cp -r plots "$BACKUP_DIR/plots_backup" 2>/dev/null
cp -r results "$BACKUP_DIR/results_backup" 2>/dev/null
echo "Backed up existing files to $BACKUP_DIR"

# Clean directories
rm -f models/*.pt plots/*.png
rm -f results/logprob/*.json results/mog/*.json results/diffusion/*.json results/autoreg_disc/*.json

echo ""
echo "===== Running LogProb Method ====="
echo "python train.py --method logprob --epochs $EPOCHS --batch_size $BATCH_SIZE --eval_interval $EVAL_INTERVAL --lr 3e-4"
python train.py --method logprob --epochs $EPOCHS --batch_size $BATCH_SIZE --eval_interval $EVAL_INTERVAL --lr 3e-4
echo ""

echo "===== Running MoG Method ====="
echo "python train.py --method mog --epochs $EPOCHS --batch_size $BATCH_SIZE --eval_interval $EVAL_INTERVAL --lr 3e-4 --num_components 5"
python train.py --method mog --epochs $EPOCHS --batch_size $BATCH_SIZE --eval_interval $EVAL_INTERVAL --lr 3e-4 --num_components 5
echo ""

echo "===== Running Diffusion Method ====="
echo "python train.py --method diffusion --epochs $EPOCHS --batch_size $BATCH_SIZE --eval_interval $EVAL_INTERVAL --lr 1e-4 --n_timesteps 100"
python train.py --method diffusion --epochs $EPOCHS --batch_size $BATCH_SIZE --eval_interval $EVAL_INTERVAL --lr 1e-4 --n_timesteps 100
echo ""

echo "===== Running Autoregressive Method ====="
echo "python train.py --method autoreg --epochs $EPOCHS --batch_size $BATCH_SIZE --eval_interval $EVAL_INTERVAL --lr 1e-4 --num_bins 21"
python train.py --method autoreg --epochs $EPOCHS --batch_size $BATCH_SIZE --eval_interval $EVAL_INTERVAL --lr 1e-4 --num_bins 21
echo ""

echo "===== All methods completed! ====="
echo "Results saved in the following directories:"
echo "  - Models: ./models/"
echo "  - Plots: ./plots/"
echo "  - Result data: ./results/<method>/"

# Return to original conda environment
conda deactivate
conda activate py3 