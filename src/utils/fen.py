import chess
import re

def is_valid_fen(fen):
    """
    Check if a FEN string is valid.
    
    Args:
        fen: FEN string to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        # Try to create a board from the FEN
        chess.Board(fen)
        return True
    except ValueError:
        return False

def get_starting_fen():
    """
    Get the FEN string for the starting position.
    
    Returns:
        FEN string for the starting position
    """
    return chess.STARTING_FEN

def fen_to_board(fen):
    """
    Convert a FEN string to a chess board.
    
    Args:
        fen: FEN string
        
    Returns:
        chess.Board object or None if invalid FEN
    """
    try:
        return chess.Board(fen)
    except ValueError:
        return None

def board_to_fen(board):
    """
    Convert a chess board to a FEN string.
    
    Args:
        board: Chess board
        
    Returns:
        FEN string
    """
    return board.fen()

def get_piece_count(fen):
    """
    Count the number of pieces for each type in a position.
    
    Args:
        fen: FEN string
        
    Returns:
        Dictionary with piece counts
    """
    # Extract the piece placement part of the FEN
    placement = fen.split(' ')[0]
    
    # Count pieces
    counts = {
        'P': 0, 'N': 0, 'B': 0, 'R': 0, 'Q': 0, 'K': 0,
        'p': 0, 'n': 0, 'b': 0, 'r': 0, 'q': 0, 'k': 0
    }
    
    for char in placement:
        if char in counts:
            counts[char] += 1
    
    return counts

def get_material_difference(fen):
    """
    Calculate the material difference in a position.
    
    Args:
        fen: FEN string
        
    Returns:
        Material difference in centipawns (positive for white advantage)
    """
    # Piece values
    values = {
        'P': 100, 'N': 320, 'B': 330, 'R': 500, 'Q': 900, 'K': 0,
        'p': -100, 'n': -320, 'b': -330, 'r': -500, 'q': -900, 'k': 0
    }
    
    # Get piece counts
    counts = get_piece_count(fen)
    
    # Calculate material difference
    material_diff = 0
    for piece, count in counts.items():
        material_diff += values[piece] * count
    
    return material_diff

def get_current_player(fen):
    """
    Get the player to move in a position.
    
    Args:
        fen: FEN string
        
    Returns:
        'white' or 'black'
    """
    try:
        parts = fen.split(' ')
        if len(parts) >= 2:
            return 'white' if parts[1] == 'w' else 'black'
    except:
        pass
    
    # Default to white if there's an error
    return 'white'

def get_fullmove_number(fen):
    """
    Get the fullmove number from a FEN string.
    
    Args:
        fen: FEN string
        
    Returns:
        Fullmove number or None if not found
    """
    try:
        parts = fen.split(' ')
        if len(parts) >= 6:
            return int(parts[5])
    except:
        pass
    
    return None

def get_halfmove_clock(fen):
    """
    Get the halfmove clock from a FEN string.
    
    Args:
        fen: FEN string
        
    Returns:
        Halfmove clock or None if not found
    """
    try:
        parts = fen.split(' ')
        if len(parts) >= 5:
            return int(parts[4])
    except:
        pass
    
    return None

def get_castling_rights(fen):
    """
    Get the castling rights from a FEN string.
    
    Args:
        fen: FEN string
        
    Returns:
        Dictionary with castling rights
    """
    result = {
        'K': False,  # White kingside
        'Q': False,  # White queenside
        'k': False,  # Black kingside
        'q': False   # Black queenside
    }
    
    try:
        parts = fen.split(' ')
        if len(parts) >= 3:
            castling = parts[2]
            if castling != '-':
                for c in castling:
                    if c in result:
                        result[c] = True
    except:
        pass
    
    return result

def fen_to_readable(fen):
    """
    Convert a FEN string to a human-readable description.
    
    Args:
        fen: FEN string
        
    Returns:
        Human-readable description of the position
    """
    try:
        board = chess.Board(fen)
        
        # Get player to move
        player = "White" if board.turn == chess.WHITE else "Black"
        
        # Check for check or checkmate
        status = ""
        if board.is_check():
            if board.is_checkmate():
                status = "Checkmate"
            else:
                status = "Check"
        elif board.is_stalemate():
            status = "Stalemate"
        
        # Get move number
        move_number = board.fullmove_number
        
        # Get material difference
        material_diff = get_material_difference(fen)
        material_str = ""
        if material_diff > 0:
            material_str = f"White is ahead by {material_diff/100:.1f} pawns"
        elif material_diff < 0:
            material_str = f"Black is ahead by {-material_diff/100:.1f} pawns"
        else:
            material_str = "Material is equal"
        
        # Combine information
        result = f"Position after move {move_number}. {player} to move."
        if status:
            result += f" {status}!"
        result += f" {material_str}."
        
        return result
    except:
        return "Invalid FEN string"

def get_pgn_from_moves(moves, headers=None):
    """
    Create a PGN string from a list of moves.
    
    Args:
        moves: List of moves in UCI format
        headers: Optional dictionary with PGN headers
        
    Returns:
        PGN string
    """
    board = chess.Board()
    pgn = ""
    
    # Add headers
    if headers:
        for key, value in headers.items():
            pgn += f'[{key} "{value}"]\n'
        pgn += "\n"
    
    # Add moves
    move_num = 1
    for i, move_uci in enumerate(moves):
        try:
            move = chess.Move.from_uci(move_uci)
            
            # Add move number for white's moves
            if i % 2 == 0:
                pgn += f"{move_num}. "
                move_num += 1
            
            # Add move in algebraic notation
            pgn += board.san(move) + " "
            
            # Make the move
            board.push(move)
            
            # Add newline every 5 full moves
            if i % 10 == 9:
                pgn += "\n"
        except:
            # Skip invalid moves
            pass
    
    # Add result
    if board.is_checkmate():
        if board.turn == chess.WHITE:
            pgn += "0-1"
        else:
            pgn += "1-0"
    elif board.is_stalemate() or board.is_insufficient_material():
        pgn += "1/2-1/2"
    
    return pgn

def save_pgn(pgn, filename):
    """
    Save a PGN string to a file.
    
    Args:
        pgn: PGN string
        filename: Output filename
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with open(filename, 'w') as f:
            f.write(pgn)
        return True
    except:
        return False
