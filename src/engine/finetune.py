import chess
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from datetime import datetime
import os
import time
from collections import deque
import json

from src.engine.nnue.network import NNUE, board_to_features
from src.engine.nnue.trainer import NNUETrainer, ChessPositionDataset
from src.engine.nnue.weights import save_weights, load_weights, get_latest_weights

class SelfPlayData:
    """
    Class for collecting self-play data to fine-tune the NNUE model.
    """
    def __init__(self, max_positions=10000):
        """
        Initialize self-play data collector.
        
        Args:
            max_positions: Maximum number of positions to store
        """
        self.positions = deque(maxlen=max_positions)
        self.results = {"white_win": 0, "black_win": 0, "draw": 0}
        
    def add_game(self, game_history, result, evaluations=None):
        """
        Add a completed game to the dataset.
        
        Args:
            game_history: List of FEN strings from the game
            result: Game result ('white_win', 'black_win', 'draw')
            evaluations: Optional list of evaluations for each position
        """
        # Update results counter
        if result in self.results:
            self.results[result] += 1
        
        # Skip if no positions
        if not game_history:
            return
            
        # Determine position value based on result
        if result == "white_win":
            final_score = 1.0
        elif result == "black_win":
            final_score = -1.0
        else:  # draw
            final_score = 0.0
        
        # If evaluations are provided, use them
        if evaluations and len(evaluations) == len(game_history):
            for fen, eval_score in zip(game_history, evaluations):
                # Store position with evaluation
                self.positions.append((fen, eval_score))
        else:
            # Assign decaying values based on result
            # Positions closer to the end get stronger values
            game_length = len(game_history)
            decay_factor = 0.9  # How quickly the value decays as we go backward
            
            for i, fen in enumerate(game_history):
                # Calculate position value (decays as we go backward from end)
                # Scale by position in game and decay
                position_value = final_score * (decay_factor ** (game_length - i - 1))
                
                # Add scaling factor for more explainability
                # Convert to centipawn value (-500 to 500 range)
                scaled_value = position_value * 500
                
                # Store position with calculated value
                self.positions.append((fen, scaled_value))
    
    def create_dataset(self):
        """
        Create a PyTorch dataset from collected positions.
        
        Returns:
            ChessPositionDataset
        """
        return ChessPositionDataset(list(self.positions))
    
    def save_to_csv(self, filename=None):
        """
        Save the dataset to a CSV file.
        
        Args:
            filename: Output filename (None for auto-generated)
            
        Returns:
            Path to the saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"selfplay_data_{timestamp}.csv"
        
        # Create directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)
        
        # Create full path
        path = os.path.join("logs", filename)
        
        # Create DataFrame
        df = pd.DataFrame(self.positions, columns=["fen", "eval"])
        
        # Save to CSV
        df.to_csv(path, index=False)
        print(f"Self-play data saved to {path}")
        
        return path
    
    def load_from_csv(self, path):
        """
        Load the dataset from a CSV file.
        
        Args:
            path: Path to CSV file
            
        Returns:
            Number of positions loaded
        """
        try:
            df = pd.read_csv(path)
            
            # Check if the file has the required columns
            if "fen" not in df.columns or "eval" not in df.columns:
                print(f"Error: CSV file must have 'fen' and 'eval' columns")
                return 0
            
            # Convert to positions
            positions = [(row['fen'], row['eval']) for _, row in df.iterrows()]
            
            # Add to existing positions
            for pos in positions:
                if len(self.positions) < self.positions.maxlen:
                    self.positions.append(pos)
                else:
                    break
                    
            print(f"Loaded {len(positions)} positions from {path}")
            return len(positions)
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return 0
    
    def get_statistics(self):
        """
        Get statistics about the collected data.
        
        Returns:
            Dictionary with statistics
        """
        return {
            "positions": len(self.positions),
            "games": sum(self.results.values()),
            "results": self.results
        }

class RealtimeFinetuner:
    """
    Class for fine-tuning the NNUE model in real-time based on move quality feedback.
    """
    def __init__(self, model=None, learning_rate=0.0001, weights_path=None):
        """
        Initialize a real-time fine-tuner.
        
        Args:
            model: NNUE model to fine-tune (None to load or create new)
            learning_rate: Learning rate for fine-tuning
            weights_path: Path to initial weights file (None for default)
        """
        # Initialize model
        if model is None:
            self.model = NNUE()
            
            # Try to load weights
            if weights_path:
                try:
                    self.model = load_weights(self.model, weights_path)
                    print(f"Loaded weights from {weights_path}")
                except Exception as e:
                    print(f"Error loading weights: {e}, initializing new model")
            else:
                # Try to load latest weights
                latest_weights = get_latest_weights()
                if latest_weights:
                    try:
                        self.model = load_weights(self.model, latest_weights)
                        print(f"Loaded latest weights from {latest_weights}")
                    except Exception as e:
                        print(f"Error loading latest weights: {e}, initializing new model")
                else:
                    print("No weights found, initializing new model")
        else:
            self.model = model
        
        # Set model to training mode
        self.model.train()
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Track recent positions for batch updates
        self.recent_positions = []
        self.max_recent_positions = 50
        
        # Create default weights file if it doesn't exist
        self._ensure_default_weights()
    
    def _ensure_default_weights(self):
        """Ensure default weights file exists."""
        default_weights_path = os.path.join("saved_models", "default_weights.pt")
        
        if not os.path.exists(default_weights_path):
            # Create directory if it doesn't exist
            os.makedirs("saved_models", exist_ok=True)
            
            # Save current model as default
            torch.save(self.model.state_dict(), default_weights_path)
            print(f"Created default weights at {default_weights_path}")
    
    def adjust_for_move(self, board, move, quality_value):
        """
        Adjust model weights based on move quality feedback.
        
        Args:
            board: Chess board position before the move
            move: The move that was made
            quality_value: Quality value (-3 to 3, negative for bad moves)
            
        Returns:
            True if weights were adjusted
        """
        # Convert board to features
        features = board_to_features(board)
        
        # Make the move on a copy of the board
        board_copy = chess.Board(board.fen())
        board_copy.push(move)
        
        # Get current evaluation
        with torch.no_grad():
            current_eval = self.model(features)
        
        # Calculate target evaluation based on quality
        # For bad moves (negative quality), we want to adjust in the opposite direction
        # For good moves (positive quality), we want to reinforce the evaluation
        adjustment_factor = abs(quality_value) * 50  # Scale factor
        
        if quality_value < 0:
            # Bad move - adjust in opposite direction
            target_eval = -current_eval * (abs(quality_value) / 3.0)  # Scale by severity
        else:
            # Good move - reinforce evaluation
            target_eval = current_eval * (1.0 + quality_value / 3.0)  # Boost by quality
        
        # Clamp target to reasonable range
        target_eval = torch.clamp(target_eval, -600, 600)
        
        # Store position for batch update
        self.recent_positions.append((features, target_eval))
        
        # Keep only the most recent positions
        if len(self.recent_positions) > self.max_recent_positions:
            self.recent_positions.pop(0)
        
        # Perform mini-batch update if we have enough positions
        if len(self.recent_positions) >= 10:
            return self._update_batch()
        
        return False
    
    def _update_batch(self):
        """
        Update model weights using a mini-batch of recent positions.
        
        Returns:
            True if update was successful
        """
        # Prepare batch
        features_batch = []
        targets_batch = []
        
        for features, target in self.recent_positions:
            features_batch.append(features)
            targets_batch.append(target)
        
        features_tensor = torch.stack(features_batch)
        targets_tensor = torch.stack(targets_batch)
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Forward pass
        outputs = self.model(features_tensor)
        
        # Calculate loss
        loss = self.criterion(outputs, targets_tensor)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        self.optimizer.step()
        
        print(f"Updated model weights with batch of {len(self.recent_positions)} positions, loss: {loss.item():.6f}")
        
        return True
    
    def save_model(self, name=None):
        """
        Save the current model weights.
        
        Args:
            name: Optional name for the weights file
            
        Returns:
            Path to the saved weights file
        """
        return save_weights(self.model, name)
    
    def reset_to_default(self):
        """
        Reset model to default weights.
        
        Returns:
            True if reset was successful
        """
        default_weights_path = os.path.join("saved_models", "default_weights.pt")
        
        if os.path.exists(default_weights_path):
            try:
                self.model = load_weights(self.model, default_weights_path)
                print(f"Reset to default weights from {default_weights_path}")
                return True
            except Exception as e:
                print(f"Error resetting to default weights: {e}")
                return False
        else:
            print("Default weights not found")
            return False

class FineTuner:
    """
    Class for fine-tuning the NNUE model based on self-play data.
    """
    def __init__(self, model=None, learning_rate=0.0001):
        """
        Initialize a fine-tuner.
        
        Args:
            model: NNUE model to fine-tune (None to load latest)
            learning_rate: Learning rate for fine-tuning
        """
        # Load model if none provided
        if model is None:
            self.model = NNUE()
            
            # Try to load latest weights
            weights_path = get_latest_weights()
            if weights_path:
                try:
                    self.model = load_weights(self.model, weights_path)
                    print(f"Loaded weights from {weights_path}")
                except Exception as e:
                    print(f"Error loading weights: {e}")
        else:
            self.model = model
            
        # Create trainer with lower learning rate for fine-tuning
        self.trainer = NNUETrainer(model=self.model, learning_rate=learning_rate)
    
    def finetune(self, data, epochs=10, batch_size=64, validation_split=0.1):
        """
        Fine-tune the model on self-play data.
        
        Args:
            data: SelfPlayData object
            epochs: Number of epochs to train
            batch_size: Batch size for training
            validation_split: Fraction of data to use for validation
            
        Returns:
            Tuple of (model, training_history)
        """
        # Create dataset
        dataset = data.create_dataset()
        
        # Split dataset for validation
        dataset_size = len(dataset)
        val_size = int(dataset_size * validation_split)
        train_size = dataset_size - val_size
        
        # Create random indices
        indices = np.random.permutation(dataset_size)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        # Create split datasets
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices) if val_size > 0 else None
        
        # Train the model
        history = self.trainer.train(
            train_dataset, 
            batch_size=batch_size, 
            epochs=epochs, 
            save_interval=5,
            validation_dataset=val_dataset
        )
        
        # Save the fine-tuned model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.trainer.save_model(f"nnue_weights_finetuned_{timestamp}")
        
        return self.model, history

def auto_finetune(engine, games=10, epochs=5, positions_per_game=30):
    """
    Automatically finetune NNUE model by playing games against itself.
    
    Args:
        engine: NNUEEngine instance
        games: Number of self-play games
        epochs: Number of fine-tuning epochs
        positions_per_game: Maximum positions to save per game
        
    Returns:
        Fine-tuned model
    """
    # Create data collector
    data = SelfPlayData(max_positions=games * positions_per_game)
    
    # Play games
    for i in range(games):
        print(f"Playing self-play game {i+1}/{games}")
        
        # Reset engine for new game
        engine.reset()
        
        # Play until game over
        move_count = 0
        game_positions = []
        evaluations = []
        
        while not engine.board.is_game_over() and move_count < 100:
            # Store position
            game_positions.append(engine.board.fen())
            
            # Evaluate position
            eval_score = engine.evaluate()
            evaluations.append(eval_score)
            
            # Make a move
            engine.play_move()
            move_count += 1
        
        # Add game to dataset
        result = engine.game_result or "draw"
        data.add_game(game_positions, result, evaluations)
        
        # Print game result
        print(f"Game {i+1} result: {result}, moves: {move_count}")
    
    # Create fine-tuner
    finetuner = FineTuner(model=engine.model)
    
    # Fine-tune the model
    model, _ = finetuner.finetune(data, epochs=epochs)
    
    # Save dataset
    data.save_to_csv()
    
    # Update engine model
    engine.model = model
    
    # Save model
    engine.save_model("nnue_weights_auto_finetuned")
    
    return model

def initialize_default_weights():
    """
    Initialize default weights file if it doesn't exist.
    
    Returns:
        Path to default weights file
    """
    default_weights_path = os.path.join("saved_models", "default_weights.pt")
    
    if not os.path.exists(default_weights_path):
        # Create directory if it doesn't exist
        os.makedirs("saved_models", exist_ok=True)
        
        # Create a new model
        model = NNUE()
        
        # Save as default weights
        torch.save(model.state_dict(), default_weights_path)
        print(f"Created default weights at {default_weights_path}")
    
    return default_weights_path
