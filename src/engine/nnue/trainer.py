import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import chess
import random
import numpy as np
from tqdm import tqdm
import os
import pandas as pd

from src.engine.nnue.network import NNUE, board_to_features
from src.engine.nnue.weights import save_weights

class ChessPositionDataset(Dataset):
    """
    Dataset of chess positions with evaluations.
    Each sample contains a FEN string and an evaluation score in centipawns.
    """
    def __init__(self, data):
        """
        Args:
            data: List of (fen, evaluation) tuples or DataFrame with 'fen' and 'eval' columns
        """
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if isinstance(self.data, pd.DataFrame):
            fen = self.data.iloc[idx]['fen']
            eval_score = self.data.iloc[idx]['eval']
        else:
            fen, eval_score = self.data[idx]
        
        # Convert FEN to board
        board = chess.Board(fen)
        
        # Convert board to features
        features = board_to_features(board)
        
        # Normalize evaluation score to [-1, 1] range
        # Most engines use centipawn values where 100 cp = 1 pawn
        normalized_eval = torch.tensor([eval_score / 600], dtype=torch.float32)
        
        return features, normalized_eval

class NNUETrainer:
    """
    Class for training NNUE models.
    """
    def __init__(self, model=None, learning_rate=0.001):
        """
        Initialize a new trainer.
        
        Args:
            model: NNUE model (if None, a new one will be created)
            learning_rate: Learning rate for optimizer
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create model if none provided
        if model is None:
            self.model = NNUE()
        else:
            self.model = model
            
        self.model = self.model.to(self.device)
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
    def train(self, dataset, batch_size=64, epochs=10, save_interval=None, validation_dataset=None):
        """
        Train the NNUE model.
        
        Args:
            dataset: Dataset of chess positions with evaluations
            batch_size: Batch size for training
            epochs: Number of epochs to train
            save_interval: Save model weights every N epochs (None to disable)
            validation_dataset: Optional dataset for validation
            
        Returns:
            Dictionary with training history (loss by epoch)
        """
        # Create data loader
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Create validation loader if validation dataset provided
        val_loader = None
        if validation_dataset is not None:
            val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': []
        }
        
        # Training loop
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            # Progress bar
            progress = tqdm(enumerate(data_loader), total=len(data_loader), 
                           desc=f"Epoch {epoch+1}/{epochs}")
            
            # Batch loop
            for i, (features, targets) in progress:
                # Move data to device
                features = features.to(self.device)
                targets = targets.to(self.device)
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(features)
                
                # Calculate loss
                loss = self.criterion(outputs, targets)
                
                # Backward pass
                loss.backward()
                
                # Update weights
                self.optimizer.step()
                
                # Update loss
                train_loss += loss.item()
                
                # Update progress bar
                progress.set_postfix({'train_loss': train_loss / (i + 1)})
            
            # Average training loss for epoch
            avg_train_loss = train_loss / len(data_loader)
            history['train_loss'].append(avg_train_loss)
            
            # Validation phase
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                history['val_loss'].append(val_loss)
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f} - Val Loss: {val_loss:.6f}")
            else:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f}")
            
            # Save model weights
            if save_interval and (epoch + 1) % save_interval == 0:
                save_weights(self.model, f"nnue_weights_epoch_{epoch+1}")
        
        # Save final weights
        save_weights(self.model)
        
        return history
    
    def validate(self, data_loader):
        """
        Validate the model on a dataset.
        
        Args:
            data_loader: DataLoader for validation dataset
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for features, targets in data_loader:
                # Move data to device
                features = features.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(features)
                
                # Calculate loss
                loss = self.criterion(outputs, targets)
                
                # Update loss
                val_loss += loss.item()
        
        return val_loss / len(data_loader)
    
    def load_dataset_from_pgn(self, pgn_file, num_positions=10000, engine_depth=15):
        """
        Generate a dataset from PGN file using an external engine for evaluation.
        
        Args:
            pgn_file: Path to PGN file
            num_positions: Number of positions to extract
            engine_depth: Depth for engine analysis
            
        Returns:
            ChessPositionDataset
        """
        # This would require integration with an external engine like Stockfish
        # For now, this is a placeholder for future implementation
        raise NotImplementedError("PGN dataset loading is not implemented yet")
    
    def save_model(self, name=None):
        """
        Save the model weights.
        
        Args:
            name: Optional name for the weights file
            
        Returns:
            Path to the saved weights file
        """
        return save_weights(self.model, name)

def create_synthetic_dataset(size=1000):
    """
    Create a synthetic dataset for testing.
    
    Args:
        size: Number of positions to generate
        
    Returns:
        ChessPositionDataset with random positions
    """
    data = []
    
    for _ in range(size):
        # Create a random board with 1-30 random moves
        board = chess.Board()
        num_moves = random.randint(1, 30)
        
        for _ in range(num_moves):
            if board.is_game_over():
                break
                
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                break
                
            move = random.choice(legal_moves)
            board.push(move)
        
        # Generate a random evaluation score between -500 and 500 centipawns
        eval_score = random.uniform(-500, 500)
        
        # Add the position and evaluation to the dataset
        data.append((board.fen(), eval_score))
    
    return ChessPositionDataset(data)
