import chess
import random
from collections import defaultdict

# Piece values for MVV-LVA (Most Valuable Victim - Least Valuable Aggressor)
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000
}

# Bonus scores for move ordering
HASH_MOVE_BONUS = 10000    # Transposition table move
CAPTURE_BONUS = 8000       # Base bonus for captures
PROMOTION_BONUS = 7000     # Base bonus for promotions
KILLER_MOVE_BONUS = 5000   # Killer move (first killer)
KILLER2_MOVE_BONUS = 4000  # Killer move (second killer)
CHECK_BONUS = 3000         # Move that gives check
CASTLE_BONUS = 2000        # Castling move

# Penalties for bad captures
BAD_CAPTURE_PENALTY = -100

def get_piece_value(piece_type):
    """Get the value of a piece type."""
    return PIECE_VALUES.get(piece_type, 0)

def score_move(board, move, hash_move, killer_moves, history_heuristic, ply):
    """
    Score a move for move ordering.
    
    Args:
        board: Chess board position
        move: Move to score
        hash_move: Hash move from transposition table
        killer_moves: Killer moves for the current ply
        history_heuristic: History heuristic scores
        ply: Current ply
        
    Returns:
        Score for the move (higher is better)
    """
    # If this is the hash move, give it the highest priority
    if hash_move and move == hash_move:
        return HASH_MOVE_BONUS
    
    score = 0
    from_square = move.from_square
    to_square = move.to_square
    moving_piece = board.piece_at(from_square)
    
    # If this is a capture, score it using MVV-LVA
    if board.is_capture(move):
        captured_piece = board.piece_at(to_square)
        
        # Handle en passant captures
        if board.is_en_passant(move):
            captured_piece = chess.Piece(chess.PAWN, not board.turn)
        
        # MVV-LVA: Most Valuable Victim - Least Valuable Aggressor
        if captured_piece:
            victim_value = get_piece_value(captured_piece.piece_type)
            aggressor_value = get_piece_value(moving_piece.piece_type)
            
            # Base capture score
            score = CAPTURE_BONUS + victim_value - (aggressor_value // 10)
            
            # Penalize bad captures (lower value piece capturing higher value piece)
            if victim_value < aggressor_value:
                score += BAD_CAPTURE_PENALTY
    
    # Promotions
    if move.promotion:
        promotion_value = get_piece_value(move.promotion)
        score = max(score, PROMOTION_BONUS + promotion_value)
    
    # Killer moves
    if killer_moves and ply < len(killer_moves):
        if move == killer_moves[ply][0]:
            score = max(score, KILLER_MOVE_BONUS)
        elif move == killer_moves[ply][1]:
            score = max(score, KILLER2_MOVE_BONUS)
    
    # Check if the move gives check
    board.push(move)
    gives_check = board.is_check()
    board.pop()
    
    if gives_check:
        score = max(score, CHECK_BONUS)
    
    # Castling moves
    if board.is_castling(move):
        score = max(score, CASTLE_BONUS)
    
    # History heuristic for quiet moves
    if not board.is_capture(move) and not move.promotion:
        history_score = history_heuristic.get((from_square, to_square), 0)
        score += history_score
    
    return score

def pick_move(board, moves, info, hash_move=None, ply=0):
    """
    Order moves for better pruning.
    
    Args:
        board: Chess board position
        moves: List of legal moves
        info: SearchInfo object containing history and killer moves
        hash_move: Hash move from transposition table
        ply: Current ply
        
    Returns:
        Ordered list of moves
    """
    # Score each move
    scored_moves = []
    for move in moves:
        score = score_move(
            board, 
            move, 
            hash_move, 
            info.killer_moves if hasattr(info, 'killer_moves') else None,
            info.history_heuristic if hasattr(info, 'history_heuristic') else {},
            ply
        )
        scored_moves.append((move, score))
    
    # Sort by score (descending)
    scored_moves.sort(key=lambda x: x[1], reverse=True)
    
    # Return ordered moves
    return [move for move, _ in scored_moves]

def get_best_capture(board, moves):
    """
    Get the best capture move from a list of moves.
    
    Args:
        board: Chess board position
        moves: List of legal moves
        
    Returns:
        Best capture move or None if no captures
    """
    captures = [move for move in moves if board.is_capture(move)]
    
    if not captures:
        return None
    
    # Score captures using MVV-LVA
    scored_captures = []
    for move in captures:
        from_square = move.from_square
        to_square = move.to_square
        
        moving_piece = board.piece_at(from_square)
        captured_piece = board.piece_at(to_square)
        
        # Handle en passant
        if board.is_en_passant(move):
            captured_piece = chess.Piece(chess.PAWN, not board.turn)
        
        if moving_piece and captured_piece:
            # MVV-LVA score
            score = get_piece_value(captured_piece.piece_type) - (get_piece_value(moving_piece.piece_type) // 10)
            scored_captures.append((move, score))
    
    # Sort by score (descending)
    scored_captures.sort(key=lambda x: x[1], reverse=True)
    
    # Return the best capture
    return scored_captures[0][0] if scored_captures else None

def is_quiet_position(board):
    """
    Check if a position is quiet (no captures or checks available).
    
    Args:
        board: Chess board position
        
    Returns:
        True if the position is quiet, False otherwise
    """
    for move in board.legal_moves:
        if board.is_capture(move):
            return False
        
        # Check if the move gives check
        board.push(move)
        gives_check = board.is_check()
        board.pop()
        
        if gives_check:
            return False
    
    return True
