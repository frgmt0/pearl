import chess
import random
import time
import numpy as np

def get_move_from_uci(board, uci_string):
    """
    Convert a UCI string to a chess.Move object.
    
    Args:
        board: Chess board
        uci_string: UCI string (e.g., 'e2e4')
        
    Returns:
        chess.Move object or None if invalid
    """
    try:
        move = chess.Move.from_uci(uci_string)
        if move in board.legal_moves:
            return move
    except ValueError:
        pass
    
    return None

def get_random_move(board):
    """
    Get a random legal move.
    
    Args:
        board: Chess board
        
    Returns:
        Random legal move
    """
    legal_moves = list(board.legal_moves)
    if legal_moves:
        return random.choice(legal_moves)
    return None

def get_material_balance(board):
    """
    Calculate material balance of the position.
    
    Args:
        board: Chess board
        
    Returns:
        Material balance in centipawns (positive for white advantage)
    """
    piece_values = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 0  # Kings are not counted in material balance
    }
    
    white_material = 0
    black_material = 0
    
    for piece_type in piece_values:
        white_material += len(board.pieces(piece_type, chess.WHITE)) * piece_values[piece_type]
        black_material += len(board.pieces(piece_type, chess.BLACK)) * piece_values[piece_type]
    
    return white_material - black_material

def is_endgame(board):
    """
    Check if the position is an endgame.
    
    Args:
        board: Chess board
        
    Returns:
        True if endgame, False otherwise
    """
    # Endgame is when both sides have no queens or
    # when every side which has a queen has additionally
    # no other pieces or only one minor piece
    queens_white = len(board.pieces(chess.QUEEN, chess.WHITE))
    queens_black = len(board.pieces(chess.QUEEN, chess.BLACK))
    
    # No queens on the board
    if queens_white == 0 and queens_black == 0:
        return True
    
    # White has a queen but few other pieces
    if queens_white > 0:
        minor_pieces_white = (
            len(board.pieces(chess.KNIGHT, chess.WHITE)) +
            len(board.pieces(chess.BISHOP, chess.WHITE))
        )
        rooks_white = len(board.pieces(chess.ROOK, chess.WHITE))
        if minor_pieces_white + rooks_white <= 1:
            return True
    
    # Black has a queen but few other pieces
    if queens_black > 0:
        minor_pieces_black = (
            len(board.pieces(chess.KNIGHT, chess.BLACK)) +
            len(board.pieces(chess.BISHOP, chess.BLACK))
        )
        rooks_black = len(board.pieces(chess.ROOK, chess.BLACK))
        if minor_pieces_black + rooks_black <= 1:
            return True
    
    # Count total material (excluding pawns and kings)
    material = (
        len(board.pieces(chess.KNIGHT, chess.WHITE)) +
        len(board.pieces(chess.BISHOP, chess.WHITE)) +
        len(board.pieces(chess.ROOK, chess.WHITE)) +
        len(board.pieces(chess.QUEEN, chess.WHITE)) +
        len(board.pieces(chess.KNIGHT, chess.BLACK)) +
        len(board.pieces(chess.BISHOP, chess.BLACK)) +
        len(board.pieces(chess.ROOK, chess.BLACK)) +
        len(board.pieces(chess.QUEEN, chess.BLACK))
    )
    
    # Few pieces means endgame
    return material <= 6

def has_insufficient_material(board):
    """
    Check if position has insufficient material for checkmate.
    
    Args:
        board: Chess board
        
    Returns:
        True if insufficient material, False otherwise
    """
    return board.is_insufficient_material()

def time_control_to_ms(time_control):
    """
    Convert time control string to milliseconds.
    
    Args:
        time_control: Time control string (e.g., '5+3', '3+2', '1+0')
        
    Returns:
        Time in milliseconds for the first move
    """
    try:
        if '+' in time_control:
            minutes, increment = time_control.split('+')
            minutes = float(minutes)
            increment = float(increment)
            
            # Start with 1/40th of main time
            move_time = (minutes * 60 * 1000) / 40
            
            # Add increment (holding some back)
            move_time += increment * 1000 * 0.8
            
            return int(move_time)
        else:
            # Assume fixed time per move
            return int(float(time_control) * 1000)
    except:
        # Default to 1 second per move
        return 1000

def format_time(ms):
    """
    Format time in milliseconds to human-readable string.
    
    Args:
        ms: Time in milliseconds
        
    Returns:
        Formatted time string
    """
    seconds = ms / 1000
    
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        seconds = seconds % 60
        return f"{minutes}m {seconds:.1f}s"
    else:
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        return f"{hours}h {minutes}m"

def get_zobrist_hash(board):
    """
    Calculate Zobrist hash for a position.
    This is a proxy for the built-in chess.Board.fen_hash().
    
    Args:
        board: Chess board
        
    Returns:
        Zobrist hash value
    """
    return hash(board.fen())

def display_board(board, perspective=chess.WHITE):
    """
    Convert a chess board to a string for display.
    
    Args:
        board: Chess board
        perspective: Perspective to view from (WHITE/BLACK)
        
    Returns:
        String representation of the board
    """
    board_str = []
    
    # Create header
    if perspective == chess.WHITE:
        board_str.append("  a b c d e f g h")
        ranks = range(7, -1, -1)
    else:
        board_str.append("  h g f e d c b a")
        ranks = range(0, 8)
    
    board_str.append("  ---------------")
    
    # Create board
    for rank in ranks:
        rank_str = f"{rank+1}|"
        
        if perspective == chess.WHITE:
            files = range(0, 8)
        else:
            files = range(7, -1, -1)
        
        for file in files:
            square = chess.square(file, rank)
            piece = board.piece_at(square)
            
            if piece is None:
                if (file + rank) % 2 == 0:
                    rank_str += "."
                else:
                    rank_str += " "
            else:
                rank_str += piece.symbol()
            
            rank_str += " "
        
        rank_str += f"|{rank+1}"
        board_str.append(rank_str)
    
    board_str.append("  ---------------")
    
    # Add footer
    if perspective == chess.WHITE:
        board_str.append("  a b c d e f g h")
    else:
        board_str.append("  h g f e d c b a")
    
    return "\n".join(board_str)

def centipawn_to_win_probability(cp_score):
    """
    Convert centipawn score to winning probability.
    
    Args:
        cp_score: Centipawn score
        
    Returns:
        Win probability (0 to 1)
    """
    # Based on logistic function, commonly used in Elo calculations
    return 1.0 / (1.0 + 10.0 ** (-cp_score / 400.0))

def evaluate_move_quality(prev_eval, post_eval, color):
    """
    Evaluate the quality of a move.
    
    Args:
        prev_eval: Evaluation before the move
        post_eval: Evaluation after the move
        color: Player color (WHITE/BLACK)
        
    Returns:
        Tuple of (quality, description)
        where quality is a number from -3 to 3
        and description is a text label
    """
    # Normalize evaluations to white's perspective
    if color == chess.BLACK:
        prev_eval = -prev_eval
        post_eval = -post_eval
    
    # Calculate the change in evaluation
    eval_change = post_eval - prev_eval
    
    # Determine move quality
    if eval_change < -200:
        return -3, "Blunder"
    elif eval_change < -100:
        return -2, "Mistake"
    elif eval_change < -50:
        return -1, "Inaccuracy"
    elif eval_change < 20:
        return 0, "Neutral"
    elif eval_change < 50:
        return 1, "Good"
    elif eval_change < 100:
        return 2, "Excellent"
    else:
        return 3, "Brilliant"
