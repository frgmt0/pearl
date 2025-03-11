"""
Chess position evaluation module.

This module provides functions for evaluating chess positions using:
1. Neural Network for Chess Position Evaluation (NNUE)
2. Classical evaluation (as fallback)
"""

import chess
import torch
import os
import numpy as np

# Import from our neural network implementation
from src.engine.nnue.network import NNUE, board_to_features, get_feature_diff

# Global model instance
nnue_model = None
# Accumulator for efficient updates
current_accumulator = None
# Last evaluated board FEN
last_board_fen = None

# Standard piece-square tables for fallback evaluation
pst = {
    'P': [  # Pawn
        [ 0,  0,  0,  0,  0,  0,  0,  0],
        [50, 50, 50, 50, 50, 50, 50, 50],
        [10, 10, 20, 30, 30, 20, 10, 10],
        [ 5,  5, 10, 25, 25, 10,  5,  5],
        [ 0,  0,  0, 20, 20,  0,  0,  0],
        [ 5, -5,-10,  0,  0,-10, -5,  5],
        [ 5, 10, 10,-20,-20, 10, 10,  5],
        [ 0,  0,  0,  0,  0,  0,  0,  0]
    ],
    'N': [  # Knight
        [-50,-40,-30,-30,-30,-30,-40,-50],
        [-40,-20,  0,  0,  0,  0,-20,-40],
        [-30,  0, 10, 15, 15, 10,  0,-30],
        [-30,  5, 15, 20, 20, 15,  5,-30],
        [-30,  0, 15, 20, 20, 15,  0,-30],
        [-30,  5, 10, 15, 15, 10,  5,-30],
        [-40,-20,  0,  5,  5,  0,-20,-40],
        [-50,-40,-30,-30,-30,-30,-40,-50]
    ],
    'B': [  # Bishop
        [-20,-10,-10,-10,-10,-10,-10,-20],
        [-10,  0,  0,  0,  0,  0,  0,-10],
        [-10,  0, 10, 10, 10, 10,  0,-10],
        [-10,  5,  5, 10, 10,  5,  5,-10],
        [-10,  0,  5, 10, 10,  5,  0,-10],
        [-10,  5,  5,  5,  5,  5,  5,-10],
        [-10,  0,  5,  0,  0,  5,  0,-10],
        [-20,-10,-10,-10,-10,-10,-10,-20]
    ],
    'R': [  # Rook
        [ 0,  0,  0,  0,  0,  0,  0,  0],
        [ 5, 10, 10, 10, 10, 10, 10,  5],
        [-5,  0,  0,  0,  0,  0,  0, -5],
        [-5,  0,  0,  0,  0,  0,  0, -5],
        [-5,  0,  0,  0,  0,  0,  0, -5],
        [-5,  0,  0,  0,  0,  0,  0, -5],
        [-5,  0,  0,  0,  0,  0,  0, -5],
        [ 0,  0,  0,  5,  5,  0,  0,  0]
    ],
    'Q': [  # Queen
        [-20,-10,-10, -5, -5,-10,-10,-20],
        [-10,  0,  0,  0,  0,  0,  0,-10],
        [-10,  0,  5,  5,  5,  5,  0,-10],
        [ -5,  0,  5,  5,  5,  5,  0, -5],
        [  0,  0,  5,  5,  5,  5,  0, -5],
        [-10,  5,  5,  5,  5,  5,  0,-10],
        [-10,  0,  5,  0,  0,  0,  0,-10],
        [-20,-10,-10, -5, -5,-10,-10,-20]
    ],
    'K': [  # King (Middlegame)
        [-30,-40,-40,-50,-50,-40,-40,-30],
        [-30,-40,-40,-50,-50,-40,-40,-30],
        [-30,-40,-40,-50,-50,-40,-40,-30],
        [-30,-40,-40,-50,-50,-40,-40,-30],
        [-20,-30,-30,-40,-40,-30,-30,-20],
        [-10,-20,-20,-20,-20,-20,-20,-10],
        [ 20, 20,  0,  0,  0,  0, 20, 20],
        [ 20, 30, 10,  0,  0, 10, 30, 20]
    ],
    'K_e': [  # King (Endgame)
        [-50,-40,-30,-20,-20,-30,-40,-50],
        [-30,-20,-10,  0,  0,-10,-20,-30],
        [-30,-10, 20, 30, 30, 20,-10,-30],
        [-30,-10, 30, 40, 40, 30,-10,-30],
        [-30,-10, 30, 40, 40, 30,-10,-30],
        [-30,-10, 20, 30, 30, 20,-10,-30],
        [-30,-30,  0,  0,  0,  0,-30,-30],
        [-50,-30,-30,-30,-30,-30,-30,-50]
    ]
}

# Material values for basic evaluation
piece_values = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000
}

def create_model():
    """
    Create a new NNUE model.
    
    Returns:
        New NNUE model
    """
    return NNUE()

def save_model(model, filename="default_weights.pt"):
    """
    Save model weights to a file.
    
    Args:
        model: NNUE model to save
        filename: Target filename
        
    Returns:
        Path to saved file
    """
    # Create directory if it doesn't exist
    os.makedirs("saved_models", exist_ok=True)
    
    # Full path to save
    path = os.path.join("saved_models", filename)
    
    # Save the model
    torch.save(model.state_dict(), path)
    print(f"Model weights saved to {path}")
    
    return path

def load_model(model_path="saved_models/default_weights.pt"):
    """
    Load model weights from a file.
    
    Args:
        model_path: Path to weights file
        
    Returns:
        NNUE model with loaded weights
    """
    # Create new model
    model = create_model()
    
    # Check if model weights exist
    if os.path.exists(model_path):
        try:
            # Load weights
            model.load_state_dict(torch.load(model_path))
            print(f"Loaded model from {model_path}")
        except Exception as e:
            print(f"Error loading weights: {e}")
            print("Creating new model instead")
            model = create_model()
    else:
        print(f"Model weights not found: {model_path}")
        print("Creating new model instead")
        
        # Save the new model for future use
        default_path = os.path.join("saved_models", "default_weights.pt")
        save_model(model, "default_weights.pt")
        print(f"Created default weights at {default_path}")
    
    return model

def initialize_nnue(model_path=None):
    """
    Initialize the NNUE model with saved weights or a new model.
    
    Args:
        model_path: Path to model weights file (None for default)
        
    Returns:
        Initialized NNUE model
    """
    global nnue_model, current_accumulator
    
    # Default model path if not specified
    if model_path is None:
        model_path = os.path.join("saved_models", "default_weights.pt")
    
    # Load model
    nnue_model = load_model(model_path)
    
    # Reset accumulator
    current_accumulator = None
    
    # Set model to evaluation mode
    nnue_model.eval()
    
    return nnue_model

def evaluate_position(board):
    """
    Evaluate a chess position using the NNUE model.
    If NNUE is not available, fall back to classical evaluation.
    
    Args:
        board: Chess board position
        
    Returns:
        Score for the position in centipawns from the current player's perspective
    """
    global nnue_model, current_accumulator, last_board_fen
    
    # Initialize NNUE model if not already done
    if nnue_model is None:
        initialize_nnue()
    
    try:
        # Try to use NNUE evaluation
        if last_board_fen and current_accumulator is not None:
            # Check if we can do an incremental update
            last_board = chess.Board(last_board_fen)
            
            # Get the last move
            moves = list(board.move_stack)
            if len(moves) > len(last_board.move_stack) and len(last_board.move_stack) > 0:
                last_move = moves[-1]
                
                # Calculate feature differences
                add_features = get_feature_diff(last_board, last_move)
                
                # Update incrementally
                with torch.no_grad():
                    score, current_accumulator = nnue_model.incremental_forward(
                        current_accumulator, 
                        add_features, 
                        None,
                        board=board,
                        move=None
                    )
            else:
                # Full evaluation if incremental is not possible
                features = board_to_features(board)
                
                with torch.no_grad():
                    # Full forward pass
                    score = nnue_model(features)
                    
                    # Update accumulator for next incremental update
                    _, current_accumulator = nnue_model.incremental_forward(
                        None, 
                        None, 
                        None, 
                        board=board
                    )
            
            # Store current board FEN for future incremental updates
            last_board_fen = board.fen()
            
            # Return the score from current player's perspective
            perspective = 1 if board.turn == chess.WHITE else -1
            return perspective * score.item()
            
        else:
            # First evaluation, do a full forward pass
            features = board_to_features(board)
            
            with torch.no_grad():
                # Full forward pass
                score = nnue_model(features)
                
                # Initialize accumulator for future incremental updates
                _, current_accumulator = nnue_model.incremental_forward(
                    None, 
                    None, 
                    None, 
                    board=board
                )
            
            # Store current board FEN for future incremental updates
            last_board_fen = board.fen()
            
            # Return the score from current player's perspective
            perspective = 1 if board.turn == chess.WHITE else -1
            return perspective * score.item()
            
    except Exception as e:
        # If NNUE fails, fall back to classical evaluation
        print(f"NNUE evaluation failed: {e}, falling back to classical evaluation")
        return classical_evaluate(board)

def get_attack_map(board, color):
    """
    Generate an attack map for a specific color.
    
    Args:
        board: Chess board position
        color: Color to generate attack map for (chess.WHITE or chess.BLACK)
        
    Returns:
        Dictionary mapping squares to the number of attackers
    """
    attack_map = {}
    
    # For each piece of the given color
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and piece.color == color:
            # Get all squares this piece attacks
            for attack_square in board.attacks(square):
                if attack_square not in attack_map:
                    attack_map[attack_square] = []
                attack_map[attack_square].append((piece.piece_type, square))
    
    return attack_map

def get_piece_value(piece_type):
    """Get the value of a piece type."""
    values = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 20000
    }
    return values.get(piece_type, 0)

# Early game piece-square tables
pst_early = {
    'P': [  # Pawns (early game)
        [  0,   0,   0,   0,   0,   0,   0,   0],
        [ 50,  50,  50,  50,  50,  50,  50,  50],
        [ 10,  10,  20,  30,  30,  20,  10,  10],
        [  5,   5,  10,  25,  25,  10,   5,   5],
        [  0,   0,   0,  20,  20,   0,   0,   0],
        [  5,  -5, -10,   0,   0, -10,  -5,   5],
        [  5,  10,  10, -20, -20,  10,  10,   5],
        [  0,   0,   0,   0,   0,   0,   0,   0]
    ],
    'N': [  # Knights (early game)
        [-50, -40, -30, -30, -30, -30, -40, -50],
        [-40, -20,   0,   0,   0,   0, -20, -40],
        [-30,   0,  10,  15,  15,  10,   0, -30],
        [-30,   5,  15,  20,  20,  15,   5, -30],
        [-30,   0,  15,  20,  20,  15,   0, -30],
        [-30,   5,  10,  15,  15,  10,   5, -30],
        [-40, -20,   0,   5,   5,   0, -20, -40],
        [-50, -40, -30, -30, -30, -30, -40, -50]
    ],
    'B': [  # Bishops (early game)
        [-20, -10, -10, -10, -10, -10, -10, -20],
        [-10,   0,   0,   0,   0,   0,   0, -10],
        [-10,   0,  10,  10,  10,  10,   0, -10],
        [-10,   5,   5,  10,  10,   5,   5, -10],
        [-10,   0,   5,  10,  10,   5,   0, -10],
        [-10,  10,  10,  10,  10,  10,  10, -10],
        [-10,   5,   0,   0,   0,   0,   5, -10],
        [-20, -10, -10, -10, -10, -10, -10, -20]
    ],
    'R': [  # Rooks (early game)
        [  0,   0,   0,   0,   0,   0,   0,   0],
        [  5,  10,  10,  10,  10,  10,  10,   5],
        [ -5,   0,   0,   0,   0,   0,   0,  -5],
        [ -5,   0,   0,   0,   0,   0,   0,  -5],
        [ -5,   0,   0,   0,   0,   0,   0,  -5],
        [ -5,   0,   0,   0,   0,   0,   0,  -5],
        [ -5,   0,   0,   0,   0,   0,   0,  -5],
        [  0,   0,   0,   5,   5,   0,   0,   0]
    ],
    'Q': [  # Queen (early game)
        [-20, -10, -10,  -5,  -5, -10, -10, -20],
        [-10,   0,   0,   0,   0,   0,   0, -10],
        [-10,   0,   5,   5,   5,   5,   0, -10],
        [ -5,   0,   5,   5,   5,   5,   0,  -5],
        [  0,   0,   5,   5,   5,   5,   0,  -5],
        [-10,   5,   5,   5,   5,   5,   0, -10],
        [-10,   0,   5,   0,   0,   0,   0, -10],
        [-20, -10, -10,  -5,  -5, -10, -10, -20]
    ],
    'K': [  # King (early game)
        [-30, -40, -40, -50, -50, -40, -40, -30],
        [-30, -40, -40, -50, -50, -40, -40, -30],
        [-30, -40, -40, -50, -50, -40, -40, -30],
        [-30, -40, -40, -50, -50, -40, -40, -30],
        [-20, -30, -30, -40, -40, -30, -30, -20],
        [-10, -20, -20, -20, -20, -20, -20, -10],
        [ 20,  20,   0,   0,   0,   0,  20,  20],
        [ 20,  30,  10,   0,   0,  10,  30,  20]
    ],
    'K_e': [  # King (endgame)
        [-50, -40, -30, -20, -20, -30, -40, -50],
        [-30, -20, -10,   0,   0, -10, -20, -30],
        [-30, -10,  20,  30,  30,  20, -10, -30],
        [-30, -10,  30,  40,  40,  30, -10, -30],
        [-30, -10,  30,  40,  40,  30, -10, -30],
        [-30, -10,  20,  30,  30,  20, -10, -30],
        [-30, -30,   0,   0,   0,   0, -30, -30],
        [-50, -30, -30, -30, -30, -30, -30, -50]
    ]
}

# End game piece-square tables
pst_end = {
    'P': [  # Pawns (end game)
        [  0,   0,   0,   0,   0,   0,   0,   0],
        [ 80,  80,  80,  80,  80,  80,  80,  80],
        [ 50,  50,  50,  50,  50,  50,  50,  50],
        [ 30,  30,  30,  40,  40,  30,  30,  30],
        [ 20,  20,  20,  30,  30,  20,  20,  20],
        [ 10,  10,  10,  10,  10,  10,  10,  10],
        [ 10,  10,  10,  10,  10,  10,  10,  10],
        [  0,   0,   0,   0,   0,   0,   0,   0]
    ],
    'N': [  # Knights (end game)
        [-50, -40, -30, -30, -30, -30, -40, -50],
        [-40, -20,   0,   5,   5,   0, -20, -40],
        [-30,   5,  15,  20,  20,  15,   5, -30],
        [-30,   0,  15,  20,  20,  15,   0, -30],
        [-30,   5,  15,  20,  20,  15,   5, -30],
        [-30,   0,  10,  15,  15,  10,   0, -30],
        [-40, -20,   0,   0,   0,   0, -20, -40],
        [-50, -40, -30, -30, -30, -30, -40, -50]
    ],
    'B': [  # Bishops (end game)
        [-20, -10, -10, -10, -10, -10, -10, -20],
        [-10,   0,   0,   0,   0,   0,   0, -10],
        [-10,   0,  10,  10,  10,  10,   0, -10],
        [-10,   5,   5,  10,  10,   5,   5, -10],
        [-10,   0,   5,  10,  10,   5,   0, -10],
        [-10,   5,   5,   5,   5,   5,   5, -10],
        [-10,   0,   0,   0,   0,   0,   0, -10],
        [-20, -10, -10, -10, -10, -10, -10, -20]
    ],
    'R': [  # Rooks (end game)
        [  0,   0,   0,   5,   5,   0,   0,   0],
        [ -5,   0,   0,   0,   0,   0,   0,  -5],
        [ -5,   0,   0,   0,   0,   0,   0,  -5],
        [ -5,   0,   0,   0,   0,   0,   0,  -5],
        [ -5,   0,   0,   0,   0,   0,   0,  -5],
        [ -5,   0,   0,   0,   0,   0,   0,  -5],
        [  5,  10,  10,  10,  10,  10,  10,   5],
        [  0,   0,   0,   0,   0,   0,   0,   0]
    ],
    'Q': [  # Queen (end game)
        [-20, -10, -10,  -5,  -5, -10, -10, -20],
        [-10,   0,   5,   0,   0,   0,   0, -10],
        [-10,   5,   5,   5,   5,   5,   0, -10],
        [  0,   0,   5,   5,   5,   5,   0,  -5],
        [ -5,   0,   5,   5,   5,   5,   0,  -5],
        [-10,   0,   5,   5,   5,   5,   0, -10],
        [-10,   0,   0,   0,   0,   0,   0, -10],
        [-20, -10, -10,  -5,  -5, -10, -10, -20]
    ],
    'K': [  # King (end game) - same as K_e in early game
        [-50, -40, -30, -20, -20, -30, -40, -50],
        [-30, -20, -10,   0,   0, -10, -20, -30],
        [-30, -10,  20,  30,  30,  20, -10, -30],
        [-30, -10,  30,  40,  40,  30, -10, -30],
        [-30, -10,  30,  40,  40,  30, -10, -30],
        [-30, -10,  20,  30,  30,  20, -10, -30],
        [-30, -30,   0,   0,   0,   0, -30, -30],
        [-50, -30, -30, -30, -30, -30, -30, -50]
    ],
    'K_e': [  # King (endgame) - same as K in end game
        [-50, -40, -30, -20, -20, -30, -40, -50],
        [-30, -20, -10,   0,   0, -10, -20, -30],
        [-30, -10,  20,  30,  30,  20, -10, -30],
        [-30, -10,  30,  40,  40,  30, -10, -30],
        [-30, -10,  30,  40,  40,  30, -10, -30],
        [-30, -10,  20,  30,  30,  20, -10, -30],
        [-30, -30,   0,   0,   0,   0, -30, -30],
        [-50, -30, -30, -30, -30, -30, -30, -50]
    ]
}

def classical_evaluate(board):
    """
    Classical evaluation function as a fallback.
    
    Args:
        board: Chess board position
        
    Returns:
        Score in centipawns from the current player's perspective
    """
    if board.is_checkmate():
        # Return a high negative score if in checkmate
        return -20000
    
    if board.is_stalemate() or board.is_insufficient_material():
        # Return draw score
        return 0
    
    # Material score
    material_score = 0
    
    # Piece-square tables score
    pst_score = 0
    
    # Mobility score (number of legal moves)
    mobility_score = len(list(board.legal_moves)) * 5
    
    # Check for endgame - simplified as when queens are off the board or
    # when both sides have <= 13 points in pieces
    is_endgame = (
        len(board.pieces(chess.QUEEN, chess.WHITE)) == 0 and 
        len(board.pieces(chess.QUEEN, chess.BLACK)) == 0
    ) or (
        get_non_pawn_material(board, chess.WHITE) <= 1300 and
        get_non_pawn_material(board, chess.BLACK) <= 1300
    )
    
    # Early game flag (for king movement penalties)
    move_count = len(board.move_stack)
    is_early_game = move_count < 20
    
    # Development score - bonus for developed pieces and center control
    development_score = 0
    
    # King safety score
    king_safety_score = 0
    
    # Get attack maps for both sides
    white_attack_map = get_attack_map(board, chess.WHITE)
    black_attack_map = get_attack_map(board, chess.BLACK)
    
    # Attack score - based on attack maps
    attack_score = 0
    
    # Select the appropriate piece-square tables based on game phase
    pst_tables = pst_early if not is_endgame else pst_end
    
    # Calculate material and piece-square scores
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue
            
        # Material score
        value = get_piece_value(piece.piece_type)
        if piece.color == chess.WHITE:
            material_score += value
        else:
            material_score -= value
        
        # Piece-square table score
        file_idx = chess.square_file(square)
        rank_idx = chess.square_rank(square)
        
        # Select the appropriate piece-square table
        if piece.piece_type == chess.KING and is_endgame:
            piece_symbol = 'K_e'  # King endgame table
        else:
            piece_symbol = piece.symbol().upper()  # Piece symbol ('P', 'N', etc.)
            
        if piece_symbol in pst_tables:
            # Add score with rank mirroring for black
            if piece.color == chess.WHITE:
                pst_score += pst_tables[piece_symbol][7 - rank_idx][file_idx]
            else:
                pst_score -= pst_tables[piece_symbol][rank_idx][file_idx]
        
        # Development score for early game
        if is_early_game:
            if piece.color == chess.WHITE:
                # Bonus for knights and bishops being developed
                if piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
                    if rank_idx > 0:  # Piece has moved from starting position
                        development_score += 10
                # Penalty for undeveloped pieces
                elif piece.piece_type == chess.QUEEN:
                    if rank_idx == 0 and file_idx == 3:  # Queen still on starting square
                        development_score -= 5
            else:
                # Same for black
                if piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
                    if rank_idx < 7:  # Piece has moved from starting position
                        development_score -= 10
                elif piece.piece_type == chess.QUEEN:
                    if rank_idx == 7 and file_idx == 3:  # Queen still on starting square
                        development_score += 5
    
    # Calculate attack score based on attack maps
    for square in chess.SQUARES:
        # Check if square is attacked by white
        white_attackers = white_attack_map.get(square, [])
        black_attackers = black_attack_map.get(square, [])
        
        # Get piece on the square (if any)
        piece = board.piece_at(square)
        
        if piece:
            # If white piece is attacked by black
            if piece.color == chess.WHITE and black_attackers:
                # Calculate the exchange value
                min_attacker_value = min([get_piece_value(attacker[0]) for attacker in black_attackers])
                piece_value = get_piece_value(piece.piece_type)
                
                # If piece is undefended or attacker is less valuable
                if not white_attackers or min_attacker_value < piece_value:
                    attack_score -= (piece_value - min_attacker_value) // 10
            
            # If black piece is attacked by white
            elif piece.color == chess.BLACK and white_attackers:
                # Calculate the exchange value
                min_attacker_value = min([get_piece_value(attacker[0]) for attacker in white_attackers])
                piece_value = get_piece_value(piece.piece_type)
                
                # If piece is undefended or attacker is less valuable
                if not black_attackers or min_attacker_value < piece_value:
                    attack_score += (piece_value - min_attacker_value) // 10
    
    # King safety evaluation
    white_king_square = board.king(chess.WHITE)
    black_king_square = board.king(chess.BLACK)
    
    if white_king_square:
        # Count attacks near the white king
        white_king_file = chess.square_file(white_king_square)
        white_king_rank = chess.square_rank(white_king_square)
        
        white_king_danger = 0
        for file_offset in range(-1, 2):
            for rank_offset in range(-1, 2):
                file = white_king_file + file_offset
                rank = white_king_rank + rank_offset
                
                if 0 <= file < 8 and 0 <= rank < 8:
                    square = chess.square(file, rank)
                    if square in black_attack_map:
                        white_king_danger += len(black_attack_map[square])
        
        # Penalize exposed white king
        king_safety_score -= white_king_danger * 5
    
    if black_king_square:
        # Count attacks near the black king
        black_king_file = chess.square_file(black_king_square)
        black_king_rank = chess.square_rank(black_king_square)
        
        black_king_danger = 0
        for file_offset in range(-1, 2):
            for rank_offset in range(-1, 2):
                file = black_king_file + file_offset
                rank = black_king_rank + rank_offset
                
                if 0 <= file < 8 and 0 <= rank < 8:
                    square = chess.square(file, rank)
                    if square in white_attack_map:
                        black_king_danger += len(white_attack_map[square])
        
        # Penalize exposed black king
        king_safety_score += black_king_danger * 5
    
    # Combine all evaluation components
    total_score = (
        material_score + 
        pst_score + 
        mobility_score + 
        development_score + 
        attack_score + 
        king_safety_score
    )
    
    # Return score from current player's perspective
    return total_score if board.turn == chess.WHITE else -total_score

def get_non_pawn_material(board, color):
    """
    Calculate the total value of non-pawn material for a side.
    
    Args:
        board: Chess board
        color: Color to calculate for (WHITE/BLACK)
        
    Returns:
        Total value of non-pawn pieces in centipawns
    """
    material = 0
    for piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
        material += len(board.pieces(piece_type, color)) * piece_values[piece_type]
    return material