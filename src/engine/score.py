import chess
import torch
import numpy as np
import os

from src.engine.nnue.network import NNUE, board_to_features, get_feature_diff
from src.engine.nnue.weights import load_weights, get_latest_weights

# Global NNUE model instance
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

def initialize_nnue():
    """
    Initialize the NNUE model with saved weights or a new model.
    """
    global nnue_model, current_accumulator
    
    # Create a new NNUE model
    nnue_model = NNUE()
    
    # Check for saved weights
    weights_path = get_latest_weights()
    if weights_path:
        try:
            nnue_model = load_weights(nnue_model, weights_path)
            print(f"Loaded NNUE weights from {weights_path}")
        except Exception as e:
            print(f"Error loading weights: {e}")
    else:
        print("No saved weights found, using initialized model")
    
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
                add_features, remove_features = get_feature_diff(last_board, last_move)
                
                # Update incrementally
                with torch.no_grad():
                    score, current_accumulator = nnue_model.incremental_forward(
                        current_accumulator, add_features, remove_features
                    )
            else:
                # Full evaluation if incremental is not possible
                features = board_to_features(board)
                with torch.no_grad():
                    # Get the transformed features to use as new accumulator
                    transformed = nnue_model.feature_transformer(features)
                    current_accumulator = transformed
                    
                    # Full forward pass
                    score = nnue_model(features)
            
            # Store current board FEN for future incremental updates
            last_board_fen = board.fen()
            
            # Return the score from current player's perspective
            perspective = 1 if board.turn == chess.WHITE else -1
            return perspective * score.item()
            
        else:
            # First evaluation, do a full forward pass
            features = board_to_features(board)
            with torch.no_grad():
                # Get the transformed features to use as accumulator for future incremental updates
                transformed = nnue_model.feature_transformer(features)
                current_accumulator = transformed
                
                # Full forward pass
                score = nnue_model(features)
            
            # Store current board FEN for future incremental updates
            last_board_fen = board.fen()
            
            # Return the score from current player's perspective
            perspective = 1 if board.turn == chess.WHITE else -1
            return perspective * score.item()
            
    except Exception as e:
        # If NNUE fails, fall back to classical evaluation
        print(f"NNUE evaluation failed: {e}, falling back to classical evaluation")
        return classical_evaluate(board)

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
    
    # Calculate material and piece-square scores
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue
            
        # Material score
        value = piece_values[piece.piece_type]
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
            
        if piece_symbol in pst:
            # Add score with rank mirroring for black
            if piece.color == chess.WHITE:
                pst_score += pst[piece_symbol][7 - rank_idx][file_idx]
            else:
                pst_score -= pst[piece_symbol][rank_idx][file_idx]
    
    # King safety (simple version)
    king_safety_score = 0
    for color in [chess.WHITE, chess.BLACK]:
        # Penalty for exposed king in middle game
        if not is_endgame:
            king_square = board.king(color)
            if king_square is not None:
                file_idx = chess.square_file(king_square)
                
                # Penalize central king in middlegame
                central_penalty = 0
                if 2 < file_idx < 5:
                    central_penalty = 20
                
                if color == chess.WHITE:
                    king_safety_score -= central_penalty
                else:
                    king_safety_score += central_penalty
    
    # Pawn structure (basic)
    pawn_structure_score = 0
    for color in [chess.WHITE, chess.BLACK]:
        # Doubled pawns (penalty)
        for file_idx in range(8):
            pawns_on_file = 0
            for rank_idx in range(8):
                square = chess.square(file_idx, rank_idx)
                piece = board.piece_at(square)
                if piece and piece.piece_type == chess.PAWN and piece.color == color:
                    pawns_on_file += 1
            
            # Penalty for doubled pawns
            if pawns_on_file > 1:
                if color == chess.WHITE:
                    pawn_structure_score -= (pawns_on_file - 1) * 20
                else:
                    pawn_structure_score += (pawns_on_file - 1) * 20
    
    # Combine all evaluation terms
    total_score = (
        material_score +
        pst_score +
        mobility_score +
        king_safety_score +
        pawn_structure_score
    )
    
    # Return score from perspective of current player
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
