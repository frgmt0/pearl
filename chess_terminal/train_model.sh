#!/bin/bash
set -e

# Create necessary directories
mkdir -p ../saved_models
mkdir -p ../logs

# Set model name (default or from argument)
MODEL_NAME="${1:-base_model.pt}"
LOG_FILE="../logs/training_$(date +%Y%m%d_%H%M%S).log"

echo "Starting chess model training using Hugging Face dataset: laion/strategic_game_chess"
echo "Model will be saved as: $MODEL_NAME"
echo "Training log will be saved to: $LOG_FILE"

# Train on the dataset from Hugging Face
python train.py \
  --model-name "$MODEL_NAME" \
  --batch-size 1024 \
  --epochs 10 \
  --lr 0.001 \
  --residual-blocks 19 \
  --channels 256 \
  --start-file 1 \
  --end-file 20 \
  --delete-after-processing \
  --checkpoint-interval 1 | tee -a "$LOG_FILE"

echo "Training complete. Model saved as: ../saved_models/$MODEL_NAME" 