import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import sys
import random
import time
import chess
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.engine.nnue.trainer import ChessPositionDataset
from src.engine.nnue.network import NNUE, board_to_features
from src.engine.score import classical_evaluate

class PearlTrainer:
    """
    Trainer for the Pearl chess engine neural network.
    """
    def __init__(self, model=None, learning_rate=0.001, batch_size=64, device=None):
        """
        Initialize the trainer.
        
        Args:
            model: Pre-trained model (None to create a new one)
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            device: Torch device (None for auto-detection)
        """
        # Set device
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create or load model
        if model is None:
            self.model = NNUE()
            print("Created new NNUE neural network model")
        else:
            self.model = model
            print("Using provided neural network model")
            
        self.model = self.model.to(self.device)
        
        # Training hyperparameters
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'epochs': 0
        }
    
    def load_dataset(self, csv_path, max_games=None, val_split=0.1):
        """
        Load a dataset from a CSV file.
        
        Args:
            csv_path: Path to the CSV file
            max_games: Maximum number of games to load (None for all)
            val_split: Proportion of data for validation
            
        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        print(f"\033[1;36mLoading dataset from \033[1;33m{csv_path}\033[1;36m...\033[0m")
        
        # Load CSV file
        df = pd.read_csv(csv_path)
        
        if max_games is not None:
            # Limit to max_games
            df = df.sample(min(max_games, len(df)), random_state=42)
            
        print(f"\033[1;32mLoaded \033[1;33m{len(df)}\033[1;32m games from dataset\033[0m")
        
        # Process the data to create training examples
        position_data = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing games"):
            # Get move sequence and evaluations
            move_sequence = row['move_sequence'].split(',')
            evaluations = [float(eval_str) for eval_str in row['evaluations'].split(',')]
            result = row['result']
            
            # Convert result to score (-1, 0, 1)
            result_score = 0
            if result == "1-0":
                result_score = 1
            elif result == "0-1":
                result_score = -1
                
            # Create board and play through moves to create training samples
            board = chess.Board()
            
            # Process each move
            for i, move_uci in enumerate(move_sequence):
                # Skip if we don't have a matching evaluation
                if i >= len(evaluations):
                    break
                    
                # Get the position before the move
                position_fen = board.fen()
                
                # Get the move evaluation
                # Blend of immediate evaluation and final result
                # Weight the result more as the game progresses
                move_num = i // 2  # which move number we're on (0, 1, 2...)
                progress = min(1.0, move_num / 40)  # progress through game (0.0 to 1.0)
                
                # Blend immediate evaluation with final result
                blended_eval = (1 - progress) * evaluations[i] + progress * result_score * 100
                
                # Cap evaluation to reasonable values
                capped_eval = max(-600, min(600, blended_eval))
                
                # Add training example
                position_data.append((position_fen, capped_eval))
                
                # Make the move
                try:
                    move = chess.Move.from_uci(move_uci)
                    board.push(move)
                except Exception as e:
                    print(f"Error processing move {move_uci}: {e}")
                    break
        
        print(f"\033[1;32mCreated \033[1;33m{len(position_data)}\033[1;32m training examples\033[0m")
        
        # Split into training and validation sets
        random.shuffle(position_data)
        val_size = int(len(position_data) * val_split)
        train_data = position_data[val_size:]
        val_data = position_data[:val_size]
        
        print(f"\033[1;36mSplitting dataset:\033[0m")
        print(f"\033[1;32m - Training set: \033[1;33m{len(train_data)}\033[1;32m examples\033[0m")
        print(f"\033[1;32m - Validation set: \033[1;33m{len(val_data)}\033[1;32m examples\033[0m")
        
        # Create datasets - using 'standard' as the model type for our new network
        train_dataset = ChessPositionDataset(train_data, model_type="standard")
        val_dataset = ChessPositionDataset(val_data, model_type="standard")
        
        return train_dataset, val_dataset
    
    def train(self, train_dataset, val_dataset=None, epochs=10, patience=5, save_best=True):
        """
        Train the model on a dataset.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset (None to skip validation)
            epochs: Number of training epochs
            patience: Early stopping patience (0 to disable)
            save_best: Whether to save the best model
            
        Returns:
            History of training (loss)
        """
        print(f"\033[1;35m━━━━━━━━━━━━━━━━━━━ Training for {epochs} epochs ━━━━━━━━━━━━━━━━━━━\033[0m")
        
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True
        )
        
        if val_dataset:
            val_loader = torch.utils.data.DataLoader(
                val_dataset, 
                batch_size=self.batch_size, 
                shuffle=False
            )
        
        # Early stopping variables
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        # Track start time
        start_time = time.time()
        
        # Training loop
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            # Progress bar
            train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
            
            for features, targets in train_bar:
                # Move to device
                features = features.to(self.device)
                targets = targets.to(self.device)
                
                # Reset gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(features)
                
                # Calculate loss
                loss = self.criterion(outputs, targets)
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                
                # Update statistics
                train_loss += loss.item() * features.size(0)
                train_bar.set_postfix(loss=loss.item())
            
            # Calculate average training loss
            train_loss /= len(train_dataset)
            self.history['train_loss'].append(train_loss)
            
            # Validation phase
            if val_dataset:
                val_loss = self.evaluate(val_loader)
                self.history['val_loss'].append(val_loss)
                
                print(f"\033[1;36mEpoch {epoch+1}/{epochs}\033[0m - Train Loss: \033[1;33m{train_loss:.6f}\033[0m - Val Loss: \033[1;34m{val_loss:.6f}\033[0m")
                
                # Check for improvement for early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    
                    # Save best model state
                    if save_best:
                        best_model_state = self.model.state_dict().copy()
                        print(f"\033[1;32m✓ New best model saved (val_loss: {val_loss:.6f})\033[0m")
                else:
                    patience_counter += 1
                    if patience > 0 and patience_counter >= patience:
                        print(f"\033[1;33m⚠ Early stopping after {epoch+1} epochs (no improvement for {patience} epochs)\033[0m")
                        break
                    else:
                        print(f"\033[1;33m⚠ No improvement. Patience: {patience_counter}/{patience}\033[0m")
            else:
                print(f"\033[1;36mEpoch {epoch+1}/{epochs}\033[0m - Train Loss: \033[1;33m{train_loss:.6f}\033[0m")
        
        # Update epochs trained
        self.history['epochs'] += epochs
        
        # Calculate training time
        training_time = time.time() - start_time
        training_minutes = training_time / 60
        
        print(f"\033[1;32m✓ Training completed in {training_minutes:.2f} minutes ({training_time:.2f} seconds)\033[0m")
        
        # Load best model if using early stopping and best model is better
        if save_best and best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"\033[1;32m✓ Loaded best model (val_loss: {best_val_loss:.6f})\033[0m")
        
        return self.history
    
    def evaluate(self, data_loader):
        """
        Evaluate the model on a dataset.
        
        Args:
            data_loader: DataLoader with evaluation data
            
        Returns:
            Average loss
        """
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for features, targets in data_loader:
                # Move to device
                features = features.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(features)
                
                # Calculate loss
                loss = self.criterion(outputs, targets)
                
                # Update statistics
                total_loss += loss.item() * features.size(0)
        
        # Calculate average loss
        avg_loss = total_loss / len(data_loader.dataset)
        
        return avg_loss
    
    def save_model(self, path):
        """
        Save the model weights.
        
        Args:
            path: Path to save the model
            
        Returns:
            True if successful
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save the model
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
        
        return True
    
    def load_model(self, path):
        """
        Load model weights.
        
        Args:
            path: Path to the weights file
            
        Returns:
            True if successful
        """
        # Check if file exists
        if not os.path.exists(path):
            print(f"Model file not found: {path}")
            return False
        
        # Load the weights
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Model loaded from {path}")
        
        return True
    
    def plot_history(self, save_path=None):
        """
        Plot the training history.
        
        Args:
            save_path: Path to save the plot (None to display only)
            
        Returns:
            None
        """
        plt.figure(figsize=(10, 6))
        
        # Plot training loss
        plt.plot(self.history['train_loss'], label='Training Loss')
        
        # Plot validation loss if available
        if len(self.history['val_loss']) > 0:
            plt.plot(self.history['val_loss'], label='Validation Loss')
        
        plt.title('Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Save the plot
            plt.savefig(save_path)
            print(f"Training plot saved to {save_path}")
        else:
            plt.show()
            
    def run_evaluation(self, num_positions=100):
        """
        Run evaluation on random positions to check model performance.
        
        Args:
            num_positions: Number of positions to evaluate
            
        Returns:
            Dictionary with evaluation metrics
        """
        print(f"\033[1;36mEvaluating model on \033[1;33m{num_positions}\033[1;36m random positions...\033[0m")
        
        # Generate random positions
        positions = []
        for _ in range(num_positions):
            board = chess.Board()
            
            # Make 1-30 random moves
            num_moves = random.randint(1, 30)
            for _ in range(num_moves):
                if board.is_game_over():
                    break
                    
                legal_moves = list(board.legal_moves)
                if not legal_moves:
                    break
                    
                move = random.choice(legal_moves)
                board.push(move)
            
            positions.append(board)
        
        # Evaluate positions
        self.model.eval()
        
        neural_evals = []
        classical_evals = []
        evaluation_times = []
        
        with torch.no_grad():
            for board in tqdm(positions, desc="Evaluating"):
                # Get classical evaluation
                classical_eval = classical_evaluate(board)
                classical_evals.append(classical_eval)
                
                # Get neural evaluation
                start_time = time.time()
                features = board_to_features(board).to(self.device)
                neural_eval = self.model(features.unsqueeze(0)).item()  # NNUE model already outputs in centipawns
                end_time = time.time()
                
                neural_evals.append(neural_eval)
                evaluation_times.append(end_time - start_time)
        
        # Calculate statistics
        avg_time = sum(evaluation_times) / len(evaluation_times)
        evaluations_per_second = 1.0 / avg_time
        
        # Calculate correlation between classical and neural evals
        correlation = np.corrcoef(classical_evals, neural_evals)[0, 1]
        
        # Calculate mean absolute error
        mae = np.mean(np.abs(np.array(classical_evals) - np.array(neural_evals)))
        
        # Print results
        print(f"\033[1;35m━━━━━━━━━━━━━━━━━━━ Evaluation Results ━━━━━━━━━━━━━━━━━━━\033[0m")
        print(f"\033[1;37mAverage evaluation time:\033[0m \033[1;33m{avg_time*1000:.2f}\033[0m ms")
        print(f"\033[1;37mEvaluations per second:\033[0m \033[1;32m{evaluations_per_second:.1f}\033[0m")
        print(f"\033[1;37mCorrelation with classical eval:\033[0m \033[1;36m{correlation:.4f}\033[0m")
        print(f"\033[1;37mMean absolute error:\033[0m \033[1;33m{mae:.2f}\033[0m centipawns")
        
        return {
            'avg_time': avg_time,
            'evals_per_second': evaluations_per_second,
            'correlation': correlation,
            'mae': mae
        }

def train_model(dataset_path, output_path=None, epochs=50, batch_size=128, 
                learning_rate=0.001, max_games=None, patience=5):
    """
    Train a model on a dataset.
    
    Args:
        dataset_path: Path to the dataset CSV
        output_path: Path to save the model (None for default)
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        max_games: Maximum number of games to use (None for all)
        patience: Early stopping patience (0 to disable)
        
    Returns:
        Trained model
    """
    # Set default output path if not provided
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join("saved_models", f"nnue_weights_{timestamp}.pt")
    
    print(f"\033[1;35m━━━━━━━━━━━━━━━━━━━ Pearl Chess Engine Training ━━━━━━━━━━━━━━━━━━━\033[0m")
    print(f"\033[1;36mStarting training session at \033[1;33m{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\033[0m")
    print(f"\033[1;36mOutput model: \033[1;33m{output_path}\033[0m")
    print(f"\033[1;36mBatch size: \033[1;33m{batch_size}\033[0m | \033[1;36mLearning rate: \033[1;33m{learning_rate}\033[0m")
    
    # Create trainer
    trainer = PearlTrainer(learning_rate=learning_rate, batch_size=batch_size)
    
    # Load dataset
    train_dataset, val_dataset = trainer.load_dataset(dataset_path, max_games=max_games)
    
    # Train the model
    trainer.train(train_dataset, val_dataset, epochs=epochs, patience=patience)
    
    # Evaluate the model
    eval_metrics = trainer.run_evaluation(num_positions=100)
    
    # Plot training history
    plot_path = os.path.splitext(output_path)[0] + "_training.png"
    trainer.plot_history(save_path=plot_path)
    
    # Save the model
    trainer.save_model(output_path)
    
    # Save the standard model version (for engine compatibility)
    standard_path = os.path.join("saved_models", "default_weights.pt")
    trainer.save_model(standard_path)
    
    # Also save as base.pt to ensure it's recognized by the engine
    base_path = os.path.join("saved_models", "base.pt")
    trainer.save_model(base_path)
    
    print(f"\033[1;32m✓ Training complete!\033[0m")
    print(f"\033[1;35m━━━━━━━━━━━━━━━━━━━ Models Saved ━━━━━━━━━━━━━━━━━━━\033[0m")
    print(f"\033[1;36mTimestamp model: \033[1;33m{output_path}\033[0m")
    print(f"\033[1;36mDefault model: \033[1;33m{standard_path}\033[0m")
    print(f"\033[1;36mBase model: \033[1;33m{base_path}\033[0m")
    print(f"\033[1;35m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\033[0m")
    
    return trainer.model

if __name__ == "__main__":
    # Parse command-line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description="Train the Pearl chess engine neural network")
    parser.add_argument("dataset", help="Path to the dataset CSV file")
    parser.add_argument("--output", "-o", help="Path to save the trained model")
    parser.add_argument("--epochs", "-e", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", "-b", type=int, default=128, help="Batch size for training")
    parser.add_argument("--learning-rate", "-lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--max-games", "-m", type=int, help="Maximum number of games to use")
    parser.add_argument("--patience", "-p", type=int, default=5, help="Early stopping patience (0 to disable)")
    
    args = parser.parse_args()
    
    # Train the model
    train_model(
        dataset_path=args.dataset,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_games=args.max_games,
        patience=args.patience
    )