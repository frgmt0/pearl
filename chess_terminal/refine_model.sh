#!/bin/bash
set -e

# Check if base model exists
BASE_MODEL="${1:-../saved_models/base_model.pt}"
REFINED_MODEL="${2:-refined_model.pt}"
LOG_FILE="../logs/refinement_$(date +%Y%m%d_%H%M%S).log"

if [ ! -f "$BASE_MODEL" ]; then
    echo "Error: Base model not found at $BASE_MODEL"
    echo "Usage: ./refine_model.sh [base_model_path] [refined_model_name]"
    echo "Please run train_model.sh first or specify the correct path to the base model."
    exit 1
fi

mkdir -p ../logs

echo "Starting self-play refinement of chess model..."
echo "Using base model: $BASE_MODEL"
echo "Refined model will be saved as: $REFINED_MODEL"
echo "Refinement log will be saved to: $LOG_FILE"

# Self-play refinement
python self_play.py \
  --games 1000 \
  --iterations 10 \
  --mcts-sims 800 \
  --batch-size 256 \
  --epochs 10 \
  --temperature 1.0 \
  --temperature-drop 20 \
  --exploration-constant 1.0 \
  --save-interval 1 \
  --model-name "$REFINED_MODEL" \
  --base-model "$BASE_MODEL" | tee -a "$LOG_FILE"

echo "Refinement complete. Model saved as: ../saved_models/$REFINED_MODEL" 