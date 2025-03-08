import torch
import torch.nn as nn
import torch.nn.functional as F
import chess
import numpy as np

# Constants for the NNUE architecture
INPUT_SIZE = 768  # 12 piece types * 64 squares = 768
HIDDEN_SIZE = 256
FEATURE_TRANSFORMER_OUTPUT = 256
OUTPUT_SIZE = 1

class FeatureTransformer(nn.Module):
    """
    First layer of NNUE that transforms sparse piece-square features
    into a dense representation, allowing for efficient updates.
    """
    def __init__(self):
        super(FeatureTransformer, self).__init__()
        self.weight = nn.Parameter(torch.zeros(FEATURE_TRANSFORMER_OUTPUT, INPUT_SIZE))
        self.bias = nn.Parameter(torch.zeros(FEATURE_TRANSFORMER_OUTPUT))
        self.initialize_weights()
        
    def initialize_weights(self):
        # Initialize weights using Xavier/Glorot initialization
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
    
    def forward(self, x):
        # x is a sparse binary vector (0s and 1s)
        # We perform matrix multiplication and add bias
        output = F.linear(x, self.weight, self.bias)
        # Apply clipped ReLU activation (min(max(x, 0), 1))
        return torch.clamp(F.relu(output), 0, 1)
    
    def incremental_forward(self, accumulator, add_features, remove_features):
        """
        Efficiently update the accumulator when a move is made without
        recomputing the entire position.
        
        Args:
            accumulator: Current hidden state
            add_features: Features to add (new piece positions)
            remove_features: Features to remove (old piece positions)
        
        Returns:
            Updated accumulator
        """
        # Subtract weights for removed features
        for idx in remove_features:
            accumulator = accumulator - self.weight[:, idx]
            
        # Add weights for added features
        for idx in add_features:
            accumulator = accumulator + self.weight[:, idx]
            
        # Add bias and apply activation
        output = accumulator + self.bias
        return torch.clamp(F.relu(output), 0, 1)

class NNUE(nn.Module):
    """
    Neural Network for Chess Position Evaluation with Efficient Updates
    
    Architecture:
    1. Feature Transformer (sparse binary input -> dense representation)
    2. Hidden Layer with ReLU activation
    3. Output Layer with linear activation (scaled to centipawn value)
    """
    def __init__(self):
        super(NNUE, self).__init__()
        
        # Feature transformer (efficiently updatable first layer)
        self.feature_transformer = FeatureTransformer()
        
        # Network layers
        self.layers = nn.Sequential(
            nn.Linear(FEATURE_TRANSFORMER_OUTPUT, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)
        )
        
        self.initialize_weights()
        
    def initialize_weights(self):
        # Initialize the network weights
        for module in self.layers:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        Full forward pass through the network.
        
        Args:
            x: Binary feature vector representation of the board
            
        Returns:
            Evaluation score in centipawns
        """
        # Pass through feature transformer
        transformed = self.feature_transformer(x)
        
        # Pass through hidden layers
        output = self.layers(transformed)
        
        # Scale output to centipawn value
        # Typically multiplied by a scaling factor like 600-700
        # to match standard evaluation scales
        return 600 * torch.tanh(output)
    
    def incremental_forward(self, accumulator, add_features, remove_features):
        """
        Efficiently update evaluation when a move is made.
        
        Args:
            accumulator: Current transformer state
            add_features: Features to add (new piece positions)
            remove_features: Features to remove (old piece positions)
            
        Returns:
            Updated evaluation score
        """
        # Update the accumulator
        updated_accumulator = self.feature_transformer.incremental_forward(
            accumulator, add_features, remove_features
        )
        
        # Pass through hidden layers
        output = self.layers(updated_accumulator)
        
        # Scale output to centipawn value
        return 600 * torch.tanh(output), updated_accumulator

def create_feature_index():
    """
    Maps chess pieces and squares to feature indices.
    
    Returns:
        Dictionary mapping (piece, square) to feature index
    """
    feature_map = {}
    idx = 0
    piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, 
                   chess.ROOK, chess.QUEEN, chess.KING]
    
    # For each piece type and color
    for color in [chess.WHITE, chess.BLACK]:
        for piece_type in piece_types:
            # For each square on the board
            for square in range(64):
                # Create piece
                piece = chess.Piece(piece_type, color)
                # Map (piece, square) to index
                feature_map[(piece, square)] = idx
                idx += 1
    
    return feature_map

def board_to_features(board):
    """
    Convert a chess board to binary feature vector.
    
    Args:
        board: A chess.Board object
        
    Returns:
        Binary feature vector (torch tensor)
    """
    feature_map = create_feature_index()
    features = torch.zeros(INPUT_SIZE, dtype=torch.float32)
    
    # Set features for each piece on the board
    for square in range(64):
        piece = board.piece_at(square)
        if piece is not None:
            feature_idx = feature_map[(piece, square)]
            features[feature_idx] = 1.0
            
    return features

def get_feature_diff(board, move):
    """
    Calculate which features to add and remove when a move is made.
    
    Args:
        board: Chess board before the move
        move: The move to be made
        
    Returns:
        Tuple of (features_to_add, features_to_remove)
    """
    feature_map = create_feature_index()
    add_features = []
    remove_features = []
    
    # Get source and destination squares
    from_square = move.from_square
    to_square = move.to_square
    
    # Get the moving piece
    piece = board.piece_at(from_square)
    
    if piece is None:
        return add_features, remove_features
    
    # Remove the piece from the source square
    remove_features.append(feature_map[(piece, from_square)])
    
    # If a piece is captured, remove it
    captured = board.piece_at(to_square)
    if captured is not None:
        remove_features.append(feature_map[(captured, to_square)])
    
    # Add the piece to the destination square
    add_features.append(feature_map[(piece, to_square)])
    
    # Handle promotions
    if move.promotion:
        # Remove the pawn at the destination
        remove_features.append(feature_map[(piece, to_square)])
        # Add the promoted piece
        promoted_piece = chess.Piece(move.promotion, piece.color)
        add_features.append(feature_map[(promoted_piece, to_square)])
    
    # Handle castling
    if piece.piece_type == chess.KING:
        king_move_distance = to_square - from_square
        
        # Kingside castling
        if king_move_distance == 2:
            # Move the rook
            rook = chess.Piece(chess.ROOK, piece.color)
            old_rook_square = (from_square | 7)  # H1 for white, H8 for black
            new_rook_square = to_square - 1      # F1 for white, F8 for black
            
            remove_features.append(feature_map[(rook, old_rook_square)])
            add_features.append(feature_map[(rook, new_rook_square)])
            
        # Queenside castling
        elif king_move_distance == -2:
            # Move the rook
            rook = chess.Piece(chess.ROOK, piece.color)
            old_rook_square = (from_square & 56)  # A1 for white, A8 for black
            new_rook_square = to_square + 1       # D1 for white, D8 for black
            
            remove_features.append(feature_map[(rook, old_rook_square)])
            add_features.append(feature_map[(rook, new_rook_square)])
    
    # Handle en passant
    if piece.piece_type == chess.PAWN and to_square != from_square and not board.piece_at(to_square):
        if board.ep_square == to_square:
            # Determine the captured pawn's square
            pawn_dir = 8 if piece.color == chess.WHITE else -8
            captured_square = to_square - pawn_dir
            captured_pawn = chess.Piece(chess.PAWN, not piece.color)
            
            remove_features.append(feature_map[(captured_pawn, captured_square)])
    
    return add_features, remove_features
