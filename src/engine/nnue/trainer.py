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

# Import directly from the network module
from src.engine.nnue.network import board_to_features

class ChessPositionDataset(Dataset):
    """
    Dataset of chess positions with evaluations.
    Each sample contains a FEN string and an evaluation score in centipawns.
    """
    def __init__(self, data, model_type="standard"):
        """
        Args:
            data: List of (fen, evaluation) tuples or DataFrame with 'fen' and 'eval' columns
            model_type: Type of model ("standard", "pearl", or "pearlxl")
        """
        self.data = data
        self.model_type = model_type.lower()
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if isinstance(self.data, pd.DataFrame):
            fen = self.data.iloc[idx]['fen']
            eval_score = self.data.iloc[idx]['eval']
        else:
            fen, eval_score = self.data[idx]
        
        # Handle different types of fen and eval_score
        # If we have a Tensor that's actually our features, short-circuit and return it
        if torch.is_tensor(fen) and fen.size(0) in [768, 15360]:
            # This is already a feature tensor, not a FEN string
            features = fen
            if torch.is_tensor(eval_score):
                if eval_score.numel() == 1:
                    normalized_eval = eval_score
                else:
                    normalized_eval = torch.tensor([eval_score.item() / 600], dtype=torch.float32)
            else:
                normalized_eval = torch.tensor([eval_score / 600], dtype=torch.float32)
            
            # Ensure features require gradients
            if not features.requires_grad:
                features = features.detach().clone().requires_grad_(True)
                
            return features, normalized_eval
            
        # Otherwise, convert to string if needed
        if not isinstance(fen, str):
            if hasattr(fen, 'item') and callable(getattr(fen, 'item')) and fen.numel() == 1:
                fen = fen.item()  # Convert single-item tensor to Python value
            fen = str(fen)  # Convert to string
        
        # Convert FEN to board
        board = chess.Board(fen)
        
        # Convert board to features
        features = board_to_features(board)
        
        # Ensure features require gradients
        if not features.requires_grad:
            features = features.detach().clone().requires_grad_(True)
        
        # Normalize evaluation score to [-1, 1] range
        # Most engines use centipawn values where 100 cp = 1 pawn
        if torch.is_tensor(eval_score):
            if eval_score.numel() == 1:
                normalized_eval = eval_score
            else:
                normalized_eval = torch.tensor([eval_score.item() / 600], dtype=torch.float32)
        else:
            normalized_eval = torch.tensor([eval_score / 600], dtype=torch.float32)
        
        return features, normalized_eval

class NNUETrainer:
    """
    Class for training NNUE models.
    """
    def __init__(self, model=None, learning_rate=0.00002, model_type="standard"):
        """
        Initialize a new trainer.
        
        Args:
            model: NNUE model (if None, a new one will be created)
            learning_rate: Learning rate for optimizer (default: 0.00002, very low for better retention)
            model_type: Type of model to create if model is None ("standard", "pearl", or "pearlxl")
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create model if none provided
        if model is None:
            from src.engine.nnue.network import NNUE
            self.model = NNUE()
        else:
            self.model = model
            
        # Extract model type from the model itself
        self.model_type = getattr(self.model, 'model_type', 'standard')
            
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
        # Function to handle batches with mixed feature sizes
        def custom_collate(batch):
            # Check if we have tensors of the same size
            first_size = batch[0][0].size(0)
            same_size = all(item[0].size(0) == first_size for item in batch)
            
            if same_size:
                # All feature tensors are the same size, use default collate
                return torch.utils.data.dataloader.default_collate(batch)
            else:
                # Handle mixed sizes by extracting features and targets
                features = torch.stack([item[0] for item in batch])
                targets = torch.stack([item[1] for item in batch])
                return features, targets
        
        # Create data loader with custom collate function
        try:
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                                    collate_fn=custom_collate)
            
            # Create validation loader if validation dataset provided
            val_loader = None
            if validation_dataset is not None:
                val_loader = DataLoader(validation_dataset, batch_size=batch_size, 
                                       shuffle=False, collate_fn=custom_collate)
        except Exception as e:
            print(f"Error creating data loader: {e}")
            print("Falling back to batch size of 1")
            # Fall back to batch size of 1 if we have issues
            batch_size = 1
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
                print(f"\033[1;34mEpoch {epoch+1}/{epochs} - Train Loss: \033[1;33m{avg_train_loss:.2f}\033[0m - Val Loss: \033[1;35m{val_loss:.2f}\033[0m")
            else:
                print(f"\033[1;34mEpoch {epoch+1}/{epochs} - Train Loss: \033[1;33m{avg_train_loss:.2f}\033[0m")
            
            # Save model weights
            if save_interval and (epoch + 1) % save_interval == 0:
                self.save_model(f"{self.model_type}_epoch_{epoch+1}")
        
        # Save final weights
        self.save_model()
        
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
    
    def save_model(self, name=None):
        """
        Save the model weights.
        
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
            filename = f"saved_models/{self.model_type}_weights.pt"
        
        # Save model weights
        torch.save(self.model.state_dict(), filename)
        print(f"Model weights saved to {filename}")
        
        return filename

def create_position_variants(board, score, num_variants=5):
    """
    Create variants of a position through legal moves and their mirror positions.
    
    Args:
        board: Original board position
        score: Evaluation score for the original position
        num_variants: Number of variants to generate
        
    Returns:
        List of (fen, score) tuples with variants
    """
    variants = [(board.fen(), score)]
    
    # Create mirror position along the vertical axis (A-H becomes H-A)
    mirrored_board = chess.Board()
    for square in range(64):
        # Calculate mirrored square (reflecting across the vertical axis)
        file = square % 8
        rank = square // 8
        mirrored_file = 7 - file
        mirrored_square = rank * 8 + mirrored_file
        
        # Copy piece from original to mirrored square
        piece = board.piece_at(square)
        if piece:
            mirrored_board.set_piece_at(mirrored_square, piece)
    
    # Set castling rights, en passant, etc.
    mirrored_board.turn = board.turn
    mirrored_board.castling_rights = board.castling_rights
    if board.ep_square:
        file = board.ep_square % 8
        rank = board.ep_square // 8
        mirrored_file = 7 - file
        mirrored_ep_square = rank * 8 + mirrored_file
        mirrored_board.ep_square = mirrored_ep_square
    
    # Add mirrored position with same score
    variants.append((mirrored_board.fen(), score))
    
    # Try to make a few neutral moves that shouldn't drastically change the evaluation
    # These provide similar but not identical positions
    variant_board = chess.Board(board.fen())
    for _ in range(min(3, num_variants)):
        legal_moves = list(variant_board.legal_moves)
        if not legal_moves:
            break
            
        # Choose a random move - in a real implementation we'd want to filter 
        # for "quiet" moves that don't drastically change the position
        move = random.choice(legal_moves)
        variant_board.push(move)
        
        # Slightly adjust score (±15 centipawns) as the position has changed slightly
        variant_score = score + random.uniform(-15, 15)
        
        # Add variant
        variants.append((variant_board.fen(), variant_score))
    
    return variants

def augment_dataset(dataset, augmentation_factor=5):
    """
    Augment a dataset with position variants.
    
    Args:
        dataset: List of (fen, score) tuples
        augmentation_factor: How many times to multiply the dataset
        
    Returns:
        Augmented list of (fen, score) tuples
    """
    augmented_data = []
    
    for fen, score in dataset:
        board = chess.Board(fen)
        variants = create_position_variants(board, score, augmentation_factor)
        augmented_data.extend(variants)
    
    return augmented_data

def create_synthetic_dataset(size=1000, include_book_positions=True):
    """
    Create a synthetic dataset for training or testing.
    
    Args:
        size: Number of positions to generate
        include_book_positions: Whether to include common opening positions
        
    Returns:
        ChessPositionDataset with positions
    """
    data = []
    
    # Add some common opening positions with established evaluations
    if include_book_positions:
        common_openings = [
            # Format: FEN, evaluation in centipawns (+ for white, - for black)
            ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 0),  # Starting position
            ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1", 10),  # After 1.e4
            ("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2", 0),  # After 1.e4 e5
            ("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2", 10),  # After 1.e4 d5
            ("rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1", 10),  # After 1.d4
            ("rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq d6 0 2", 0),  # After 1.d4 d5
            # Add more common openings with evaluations
        ]
        data.extend(common_openings)
        
        # Augment the opening positions
        data = augment_dataset(data, 3)
    
    # Generate random positions
    remaining = size - len(data)
    if remaining > 0:
        for _ in range(remaining):
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
            
            # Generate an evaluation score
            # For completely random positions, we'll use a more controlled distribution
            # Most positions should be close to equal
            eval_score = random.normalvariate(0, 100)  # Normal distribution, centered at 0, std=100
            
            # Add the position and evaluation to the dataset
            data.append((board.fen(), eval_score))
    
    return ChessPositionDataset(data, model_type="standard")
