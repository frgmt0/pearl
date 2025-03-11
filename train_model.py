#!/usr/bin/env python3
"""
Script to train the Pearl chess engine neural network.
"""

import os
import sys
from src.modes.etrain import train_model

def main():
    """
    Main function to train the model.
    """
    # Default parameters
    dataset_path = "dataset/dataset.csv"
    output_path = None  # Will use default timestamp-based name
    epochs = 50
    batch_size = 64
    learning_rate = 0.001
    max_games = None  # Use all games
    patience = 5  # Early stopping patience
    
    # Parse command-line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description="Train the Pearl chess engine neural network")
    parser.add_argument("--dataset", "-d", default=dataset_path, help="Path to the dataset CSV file")
    parser.add_argument("--output", "-o", help="Path to save the trained model")
    parser.add_argument("--epochs", "-e", type=int, default=epochs, help="Number of training epochs")
    parser.add_argument("--batch-size", "-b", type=int, default=batch_size, help="Batch size for training")
    parser.add_argument("--learning-rate", "-lr", type=float, default=learning_rate, help="Learning rate")
    parser.add_argument("--max-games", "-m", type=int, help="Maximum number of games to use")
    parser.add_argument("--patience", "-p", type=int, default=patience, help="Early stopping patience (0 to disable)")
    
    args = parser.parse_args()
    
    # Train the model
    try:
        model = train_model(
            dataset_path=args.dataset,
            output_path=args.output,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_games=args.max_games,
            patience=args.patience
        )
        print("Training completed successfully!")
        return 0
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 