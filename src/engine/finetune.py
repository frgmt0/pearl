import chess
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from datetime import datetime
import os
import time
import pickle
from collections import deque
import json

from src.engine.score import create_model, save_model, load_model
from src.engine.nnue.network import board_to_features, NNUE

class PositionMemory:
    """
    Long-term memory for chess positions and their evaluations.
    Persistently stores positions across multiple games for enhanced learning.
    """
    def __init__(self, max_positions=50000, persistent_file="saved_models/position_memory.pkl"):
        """
        Initialize the position memory.
        
        Args:
            max_positions: Maximum number of positions to store
            persistent_file: File to save/load the memory
        """
        self.positions = {}  # Dictionary of {fen: (eval, timestamp, importance)}
        self.max_positions = max_positions
        self.persistent_file = persistent_file
        
        # Try to load existing memory
        self.load()
    
    def add_position(self, fen, eval_score, importance=1.0):
        """
        Add a position to memory or update existing one.
        
        Args:
            fen: FEN string of the position
            eval_score: Evaluation score for the position
            importance: Importance factor (higher = more important to remember)
        """
        # Store with current timestamp
        timestamp = time.time()
        
        # If position exists, update with weighted average
        if fen in self.positions:
            old_eval, old_timestamp, old_importance = self.positions[fen]
            # Weight new evaluation higher if it's more important
            weighted_eval = (old_eval * old_importance + eval_score * importance) / (old_importance + importance)
            # Increase importance
            new_importance = min(old_importance + importance, 5.0)
            self.positions[fen] = (weighted_eval, timestamp, new_importance)
        else:
            # Add new position
            self.positions[fen] = (eval_score, timestamp, importance)
        
        # Trim memory if needed
        if len(self.positions) > self.max_positions:
            self._trim_memory()
    
    def _trim_memory(self):
        """Remove least important/oldest positions to maintain size limit."""
        # Calculate a score for each position based on age and importance
        position_scores = []
        current_time = time.time()
        
        for fen, (_, timestamp, importance) in self.positions.items():
            # Score formula: importance / age
            age = max(current_time - timestamp, 1)  # In seconds
            score = importance / age
            position_scores.append((fen, score))
        
        # Sort by score (lowest first)
        position_scores.sort(key=lambda x: x[1])
        
        # Remove 10% of positions with lowest scores
        positions_to_remove = int(len(position_scores) * 0.1)
        for i in range(positions_to_remove):
            fen = position_scores[i][0]
            del self.positions[fen]
    
    def get_training_batch(self, batch_size=1000):
        """
        Get a batch of positions for training, prioritizing important ones.
        
        Args:
            batch_size: Number of positions to return
            
        Returns:
            List of (fen, eval) tuples
        """
        if not self.positions:
            return []
            
        # Calculate selection probability based on importance
        position_items = list(self.positions.items())
        importances = [item[1][2] for item in position_items]
        total_importance = sum(importances)
        
        if total_importance == 0:
            # If all importances are 0, use uniform distribution
            probs = None
        else:
            # Convert to probabilities
            probs = [imp / total_importance for imp in importances]
        
        # Select positions with replacement, weighted by importance
        selected_indices = np.random.choice(
            len(position_items), 
            size=min(batch_size, len(position_items)),
            replace=True,
            p=probs
        )
        
        # Extract selected positions
        selected_positions = []
        for idx in selected_indices:
            fen, (eval_score, _, _) = position_items[idx]
            selected_positions.append((fen, eval_score))
            
        return selected_positions
    
    def save(self):
        """Save the memory to a file."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.persistent_file), exist_ok=True)
            
            with open(self.persistent_file, 'wb') as f:
                pickle.dump(self.positions, f)
            print(f"Position memory saved ({len(self.positions)} positions)")
            return True
        except Exception as e:
            print(f"Error saving position memory: {e}")
            return False
    
    def load(self):
        """Load the memory from a file."""
        try:
            if os.path.exists(self.persistent_file):
                with open(self.persistent_file, 'rb') as f:
                    self.positions = pickle.load(f)
                print(f"Position memory loaded ({len(self.positions)} positions)")
                return True
            return False
        except Exception as e:
            print(f"Error loading position memory: {e}")
            self.positions = {}
            return False

# Global instance of position memory
POSITION_MEMORY = PositionMemory()

class SelfPlayData:
    """
    Class for collecting self-play data to fine-tune the NNUE model.
    """
    def __init__(self, max_positions=10000, use_memory=True):
        """
        Initialize self-play data collector.
        
        Args:
            max_positions: Maximum number of positions to store
            use_memory: Whether to use the global position memory
        """
        self.positions = deque(maxlen=max_positions)
        self.results = {"white_win": 0, "black_win": 0, "draw": 0}
        self.use_memory = use_memory
        
    def add_game(self, game_history, result, evaluations=None, position_qualities=None):
        """
        Add a completed game to the dataset.
        
        Args:
            game_history: List of FEN strings from the game
            result: Game result ('white_win', 'black_win', 'draw')
            evaluations: Optional list of evaluations for each position
            position_qualities: Optional list of quality scores (0.0-2.0) for positions
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
        
        # Calculate result importance - decisive games are more important
        result_importance = 1.5 if result != "draw" else 1.0
        
        # If evaluations are provided, use them
        if evaluations and len(evaluations) == len(game_history):
            for i, (fen, eval_score) in enumerate(zip(game_history, evaluations)):
                # Calculate position importance based on game progression and result
                # End-game positions are more important
                if position_qualities and i < len(position_qualities):
                    # Use provided quality score if available
                    position_importance = position_qualities[i]
                else:
                    # Otherwise calculate based on position in game
                    position_importance = 1.0 + (i / len(game_history)) * 0.5
                
                # For decisive games, increase importance of positions aligned with final result
                if result != "draw":
                    # If evaluation matches result direction, boost importance
                    if (eval_score > 0 and result == "white_win") or (eval_score < 0 and result == "black_win"):
                        eval_bonus = min(1.0, abs(eval_score) / 300) * 0.5
                        position_importance += eval_bonus
                
                # Store position with evaluation
                self.positions.append((fen, eval_score))
                
                # Also add to global position memory if enabled
                if self.use_memory:
                    # Multiply importances for overall importance
                    importance = result_importance * position_importance
                    POSITION_MEMORY.add_position(fen, eval_score, importance)
        else:
            # Assign decaying values based on result
            # Positions closer to the end get stronger values
            game_length = len(game_history)
            decay_factor = 0.8  # Reduced to make end-game lessons stronger
            
            for i, fen in enumerate(game_history):
                # Calculate position value (decays as we go backward from end)
                # Scale by position in game and decay
                position_value = final_score * (decay_factor ** (game_length - i - 1))
                
                # Add scaling factor for more explainability
                # Convert to centipawn value (-500 to 500 range)
                scaled_value = position_value * 500
                
                # Store position with calculated value
                self.positions.append((fen, scaled_value))
                
                # Also add to global position memory if enabled
                if self.use_memory:
                    # Calculate position importance
                    if position_qualities and i < len(position_qualities):
                        # Use provided quality score if available
                        position_importance = position_qualities[i]
                    else:
                        # Otherwise calculate based on position in game
                        position_importance = 1.0 + (i / len(game_history)) * 0.5
                    
                    # Multiply importances for overall importance
                    importance = result_importance * position_importance
                    POSITION_MEMORY.add_position(fen, scaled_value, importance)
        
        # Save the position memory after adding a game
        if self.use_memory:
            POSITION_MEMORY.save()
    
    def create_dataset(self, include_memory=True, memory_ratio=0.5, memory_batch_size=1000):
        """
        Create a PyTorch dataset from collected positions.
        
        Args:
            include_memory: Whether to include positions from global memory
            memory_ratio: Ratio of memory positions to current game positions (0-1)
            memory_batch_size: Number of positions to sample from memory
            
        Returns:
            ChessPositionDataset
        """
        # Start with current game positions
        dataset_positions = list(self.positions)
        
        # Add positions from memory if enabled
        if include_memory and self.use_memory and len(POSITION_MEMORY.positions) > 0:
            # Determine how many memory positions to include (balancing with current positions)
            if len(dataset_positions) > 0:
                memory_size = int(len(dataset_positions) * memory_ratio / (1 - memory_ratio))
                memory_size = min(memory_size, memory_batch_size)
            else:
                memory_size = memory_batch_size
                
            # Sample positions from memory
            if memory_size > 0:
                memory_positions = POSITION_MEMORY.get_training_batch(memory_size)
                
                # Print statistics
                print(f"Using {len(dataset_positions)} positions from current game + " 
                      f"{len(memory_positions)} positions from memory")
                
                # Combine with current positions
                dataset_positions.extend(memory_positions)
        
        # Pre-process the positions to ensure we have consistent feature sizes
        # Import our feature extractor
        from src.engine.nnue.network import board_to_features
        
        # Convert positions to features
        processed_positions = []
        for fen, eval_score in dataset_positions:
            # Check if this is already a feature tensor
            if torch.is_tensor(fen):
                processed_positions.append((fen, eval_score))
                continue
            
            # Convert FEN to board
            if not isinstance(fen, str):
                fen = str(fen)
            board = chess.Board(fen)
            
            # Extract features
            features = board_to_features(board)
            processed_positions.append((features, eval_score))
        
        # Define a simple dataset class right here to avoid circular imports
        class ChessPositionDataset(torch.utils.data.Dataset):
            def __init__(self, positions):
                self.positions = positions

            def __len__(self):
                return len(self.positions)

            def __getitem__(self, idx):
                features, target = self.positions[idx]
                return features, torch.tensor([target], dtype=torch.float32)
                
        # Create dataset
        return ChessPositionDataset(processed_positions)
    
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
    def __init__(self, model=None, learning_rate=0.00002, weights_path=None):
        """
        Initialize a real-time fine-tuner.
        
        Args:
            model: NNUE model to fine-tune (None to load or create new)
            learning_rate: Learning rate for fine-tuning (very low for better retention)
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
            # More aggressive negative adjustment: flip the sign and scale by severity
            target_eval = -current_eval * (abs(quality_value) / 2.0)  # Stronger negative adjustment
        else:
            # Good move - reinforce evaluation
            target_eval = current_eval * (1.0 + quality_value / 2.0)  # Stronger positive reinforcement
        
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
        # Create directory if it doesn't exist
        os.makedirs("saved_models", exist_ok=True)
        
        # Generate filename
        if name:
            filename = f"saved_models/{name}.pt"
        else:
            filename = f"saved_models/finetune_weights.pt"
        
        # Save model weights
        torch.save(self.model.state_dict(), filename)
        print(f"Model weights saved to {filename}")
        
        return filename
    
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
    def __init__(self, model=None, learning_rate=0.00002):
        """
        Initialize a fine-tuner.
        
        Args:
            model: NNUE model to fine-tune (None to load latest)
            learning_rate: Learning rate for fine-tuning (very low for better retention)
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
    
    def finetune(self, data, epochs=50, batch_size=64, validation_split=0.1):
        """
        Fine-tune the model on self-play data.
        
        Args:
            data: SelfPlayData object
            epochs: Number of epochs to train (default: 50)
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

def auto_finetune(engine, games=10, epochs=50, positions_per_game=30):
    """
    Automatically finetune NNUE model by playing games against itself.
    
    Args:
        engine: NNUEEngine instance
        games: Number of self-play games
        epochs: Number of fine-tuning epochs (default: 50)
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

def finetune_from_pgn(pgn_file, epochs=50, batch_size=1, feedback=None, use_memory=True):
    """
    Finetune the NNUE model from a PGN file.
    
    Args:
        pgn_file: Path to the PGN file
        epochs: Number of training epochs (default: 50)
        batch_size: Batch size for training
        feedback: Optional dictionary with game outcome feedback. Examples:
                 - For win: {"result": "win", "emphasis": 1.5}
                 - For loss with inverse learning: {"result": "loss", "inverse_learning": True, "emphasis": 2.0}
                 - For selective learning: {"learn_from_winner": True, "engine_color": "white"}
        use_memory: Whether to use the position memory for additional training data
        
    Returns:
        Fine-tuned model
    """
    print(f"Loading model and PGN file: {pgn_file}")
    
    # Try to load model from default location
    model_path = "saved_models/default_weights.pt"
    if os.path.exists(model_path):
        try:
            model = load_model(model_path)
            print(f"Loaded model from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}, creating new model")
            model = create_model()
    else:
        print(f"No model found, creating new model")
        model = create_model()
    
    # Create trainer with much lower learning rate for better knowledge retention
    trainer = NNUETrainer(model=model, learning_rate=0.00002)
    
    # Load dataset from PGN with optional feedback
    dataset = trainer.load_dataset_from_pgn(pgn_file, feedback=feedback)
    
    if dataset is None or len(dataset) == 0:
        print("No valid positions found in PGN file")
        return model
    
    print(f"Loaded {len(dataset)} positions from PGN file")
    
    # Implement curriculum learning - gradually train with increasing data complexity
    print("\n📚 Implementing curriculum learning strategy:")
    
    # Phase 1: Train on just the current game
    print("\n📝 Phase 1: Learning from current game only")
    history1 = trainer.train(dataset, batch_size=batch_size, epochs=int(epochs * 0.4))
    
    # Phase 2: Add similar positions from memory
    if use_memory and len(POSITION_MEMORY.positions) > 0:
        print("\n🧠 Phase 2: Adding knowledge from memory (similar positions)")
        # Create a dataset that includes memory positions
        from src.engine.finetune import SelfPlayData
        memory_data = SelfPlayData(use_memory=True)
        
        # For each position in the current dataset, extract features directly
        current_features = []
        for i in range(len(dataset)):
            features, eval_score = dataset[i]
            # Store the features directly
            current_features.append((features, eval_score))
        
        # Add current features to memory_data
        memory_data.positions = current_features
        
        # Create an augmented dataset with memory
        memory_dataset = memory_data.create_dataset(include_memory=True, memory_ratio=0.3)
        
        # Train on the combined dataset with slightly lower learning rate
        trainer.optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
        history2 = trainer.train(memory_dataset, batch_size=batch_size, epochs=int(epochs * 0.6))
        
        # Combine training histories
        history = {
            'train_loss': history1['train_loss'] + history2['train_loss'],
            'val_loss': history1.get('val_loss', []) + history2.get('val_loss', [])
        }
    else:
        # Just continue training on the same dataset
        history = history1
    
    print("\n✨ Curriculum learning complete!")
    
    # Save the model to default_weights.pt
    weights_path = save_model(model, "default_weights")
    
    # Also save with descriptive name for record keeping
    pgn_basename = os.path.basename(pgn_file).replace('.pgn', '')
    result_tag = ""
    if feedback and feedback.get('result'):
        result_tag = f"_{feedback['result']}"
        
    # Create a timestamp-based filename
    backup_path = save_model(model, f"from_{pgn_basename}{result_tag}")
    
    print(f"Fine-tuned model saved to {weights_path}")
    print(f"Final training loss: {history['train_loss'][-1]:.6f}")
    
    return model

def finetune_from_recent_pgns(pgn_dir="pgns", num_files=5, epochs=50, batch_size=32, feedback=None):
    """
    Finetune the NNUE model using recent PGN files in the given directory.
    
    Args:
        pgn_dir: Directory containing PGN files
        num_files: Number of most recent files to use
        epochs: Number of training epochs (default: 50)
        batch_size: Batch size for training
        feedback: Optional dictionary with game outcome feedback
        
    Returns:
        Fine-tuned model
    """
    import os
    import glob
    
    # Get list of PGN files in the directory
    pgn_files = glob.glob(os.path.join(pgn_dir, "*.pgn"))
    
    if not pgn_files:
        print(f"No PGN files found in {pgn_dir}")
        return None
    
    # Sort by modification time (newest first)
    pgn_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    # Take only the specified number of files
    pgn_files = pgn_files[:num_files]
    
    if not pgn_files:
        print("No PGN files selected for finetuning")
        return None
    
    print(f"Finetuning from {len(pgn_files)} recent PGN files:")
    for idx, pgn_file in enumerate(pgn_files):
        print(f"{idx+1}. {os.path.basename(pgn_file)}")
    
    # Try to load model or create a new one
    model_path = "saved_models/default_weights.pt"
    if os.path.exists(model_path):
        try:
            model = load_model(model_path)
            print(f"Loaded model from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}, creating new model")
            model = create_model()
    else:
        print(f"No model found, creating new model")
        model = create_model()
    
    # Finetune on each PGN file
    for pgn_file in pgn_files:
        print(f"\nFinetuning on {os.path.basename(pgn_file)}...")
        model = finetune_from_pgn(
            pgn_file, 
            epochs=epochs, 
            batch_size=batch_size, 
            feedback=feedback
        )
    
    # Save final model as default weights
    weights_path = save_model(model, "default_weights")
    
    # Backup with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = save_model(model, f"finetuned_multi_{timestamp}")
    
    print(f"Final fine-tuned model saved to {weights_path}")
    
    return model

def initialize_default_weights():
    """
    Initialize default weights file if it doesn't exist.
    
    Returns:
        Path to default weights file
    """
    # Create default_weights.pt if it doesn't exist
    default_weights_path = os.path.join("saved_models", "default_weights.pt")
    if not os.path.exists(default_weights_path):
        # Create directory if needed
        os.makedirs("saved_models", exist_ok=True)
        
        # Create a new model
        model = create_model()
        
        # Save state dict directly for maximum compatibility
        torch.save(model.state_dict(), default_weights_path)
        print(f"Created default weights at {default_weights_path}")
    
    # Return default weights path
    return default_weights_path
