import chess
import numpy as np

def pick_move(board, moves, info, hash_move=None, ply=0):
    """
    Order moves to improve alpha-beta pruning efficiency.
    
    The move ordering is:
    1. Hash move (from transposition table)
    2. Captures (ordered by Most Valuable Victim - Least Valuable Aggressor)
    3. Killer moves (quiet moves that caused beta cutoffs)
    4. History heuristic (quiet moves that were good in other positions)
    5. Remaining moves
    
    Args:
        board: Chess board position
        moves: List of legal moves to order
        info: SearchInfo object containing move history
        hash_move: Best move from transposition table (if any)
        ply: Current ply from root
        
    Returns:
        Ordered list of moves
    """
    if not moves:
        return []
    
    # Piece values for MVV-LVA ordering
    piece_values = {
        chess.PAWN: 1, 
        chess.KNIGHT: 3, 
        chess.BISHOP: 3, 
        chess.ROOK: 5, 
        chess.QUEEN: 9, 
        chess.KING: 20
    }
    
    move_scores = []
    
    for move in moves:
        score = 0
        
        # 1. Hash move from transposition table (highest priority)
        if hash_move is not None and move == hash_move:
            score = 10000000
        
        # 2. Captures with MVV-LVA
        elif board.is_capture(move):
            # Get the captured piece (victim)
            victim_square = move.to_square
            victim = board.piece_at(victim_square)
            victim_value = piece_values.get(victim.piece_type, 0) if victim else 0
            
            # Get the moving piece (aggressor)
            aggressor_square = move.from_square
            aggressor = board.piece_at(aggressor_square)
            aggressor_value = piece_values.get(aggressor.piece_type, 0) if aggressor else 0
            
            # MVV-LVA score: 1,000,000 + 10 * victim value - aggressor value
            # This prioritizes capturing high-value pieces with low-value pieces
            score = 1000000 + 10 * victim_value - aggressor_value
            
            # Promotions get a bonus
            if move.promotion:
                score += 900000 + piece_values.get(move.promotion, 0) * 10000
                
            # Check if this is an en passant capture
            if board.is_en_passant(move):
                score = 1000500  # Between regular capture and promotion
        
        # 3. Killer moves
        elif move == info.killer_moves[ply][0]:
            score = 900000
        elif move == info.killer_moves[ply][1]:
            score = 800000
        
        # 4. History heuristic for quiet moves
        else:
            from_sq = move.from_square
            to_sq = move.to_square
            score = info.history_heuristic.get((from_sq, to_sq), 0)
            
            # Promotions
            if move.promotion:
                score += 700000 + piece_values.get(move.promotion, 0) * 10000
                
            # Give a slight bonus to checks
            board.push(move)
            if board.is_check():
                score += 50000
            board.pop()
            
            # Castle moves
            if board.is_castling(move):
                score += 60000
                
            # Advanced pawns
            piece = board.piece_at(move.from_square)
            if piece and piece.piece_type == chess.PAWN:
                # Bonus for advancing pawns
                rank = chess.square_rank(move.to_square)
                if board.turn == chess.WHITE:
                    score += rank * 1000  # Higher ranks for white pawns
                else:
                    score += (7 - rank) * 1000  # Higher ranks for black pawns
        
        move_scores.append((move, score))
    
    # Sort by score (descending)
    move_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Return sorted moves
    return [move for move, _ in move_scores]
