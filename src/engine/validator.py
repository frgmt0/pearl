import chess

def validate_move(board, move):
    """
    Validate that a move is legal in the current position.
    
    Args:
        board: Chess board
        move: Move to validate
        
    Returns:
        True if the move is legal, False otherwise
    """
    if not isinstance(move, chess.Move):
        try:
            move = chess.Move.from_uci(str(move))
        except ValueError:
            return False
    
    return move in board.legal_moves

def validate_board_state(board):
    """
    Validate that a board is in a legal state.
    
    Args:
        board: Chess board
        
    Returns:
        Dictionary with validation results
    """
    results = {
        'is_valid': True,
        'errors': [],
        'legal_moves_count': 0,
        'turn': 'white' if board.turn == chess.WHITE else 'black',
        'is_check': board.is_check(),
        'is_checkmate': board.is_checkmate(),
        'is_stalemate': board.is_stalemate(),
        'is_game_over': board.is_game_over()
    }
    
    # Check if the board has a valid number of kings
    white_kings = len(board.pieces(chess.KING, chess.WHITE))
    black_kings = len(board.pieces(chess.KING, chess.BLACK))
    
    if white_kings != 1:
        results['is_valid'] = False
        results['errors'].append(f"Invalid number of white kings: {white_kings}")
    
    if black_kings != 1:
        results['is_valid'] = False
        results['errors'].append(f"Invalid number of black kings: {black_kings}")
    
    # Count legal moves
    try:
        results['legal_moves_count'] = len(list(board.legal_moves))
    except Exception as e:
        results['is_valid'] = False
        results['errors'].append(f"Error generating legal moves: {e}")
    
    return results

def debug_board_state(board):
    """
    Get debug information about a board state.
    
    Args:
        board: Chess board
        
    Returns:
        Dictionary with debug information
    """
    debug_info = {
        'fen': board.fen(),
        'turn': 'white' if board.turn == chess.WHITE else 'black',
        'legal_moves_count': len(list(board.legal_moves)),
        'legal_moves': [move.uci() for move in board.legal_moves],
        'is_check': board.is_check(),
        'is_checkmate': board.is_checkmate(),
        'is_stalemate': board.is_stalemate(),
        'is_game_over': board.is_game_over(),
        'piece_count': {
            'white': {
                'pawn': len(board.pieces(chess.PAWN, chess.WHITE)),
                'knight': len(board.pieces(chess.KNIGHT, chess.WHITE)),
                'bishop': len(board.pieces(chess.BISHOP, chess.WHITE)),
                'rook': len(board.pieces(chess.ROOK, chess.WHITE)),
                'queen': len(board.pieces(chess.QUEEN, chess.WHITE)),
                'king': len(board.pieces(chess.KING, chess.WHITE))
            },
            'black': {
                'pawn': len(board.pieces(chess.PAWN, chess.BLACK)),
                'knight': len(board.pieces(chess.KNIGHT, chess.BLACK)),
                'bishop': len(board.pieces(chess.BISHOP, chess.BLACK)),
                'rook': len(board.pieces(chess.ROOK, chess.BLACK)),
                'queen': len(board.pieces(chess.QUEEN, chess.BLACK)),
                'king': len(board.pieces(chess.KING, chess.BLACK))
            }
        }
    }
    
    return debug_info

def compare_board_states(board1, board2):
    """
    Compare two board states and identify differences.
    
    Args:
        board1: First chess board
        board2: Second chess board
        
    Returns:
        Dictionary with differences
    """
    differences = {
        'are_identical': board1.fen() == board2.fen(),
        'differences': []
    }
    
    if board1.turn != board2.turn:
        differences['differences'].append(f"Turn differs: {board1.turn} vs {board2.turn}")
    
    if board1.castling_rights != board2.castling_rights:
        differences['differences'].append("Castling rights differ")
    
    if board1.ep_square != board2.ep_square:
        differences['differences'].append("En passant square differs")
    
    if board1.halfmove_clock != board2.halfmove_clock:
        differences['differences'].append(f"Halfmove clock differs: {board1.halfmove_clock} vs {board2.halfmove_clock}")
    
    if board1.fullmove_number != board2.fullmove_number:
        differences['differences'].append(f"Fullmove number differs: {board1.fullmove_number} vs {board2.fullmove_number}")
    
    # Check piece placement
    for square in chess.SQUARES:
        piece1 = board1.piece_at(square)
        piece2 = board2.piece_at(square)
        
        if piece1 != piece2:
            square_name = chess.square_name(square)
            piece1_str = piece1.symbol() if piece1 else "None"
            piece2_str = piece2.symbol() if piece2 else "None"
            differences['differences'].append(f"Piece at {square_name} differs: {piece1_str} vs {piece2_str}")
    
    return differences

def validate_board_consistency(board1, board2, move):
    """
    Validate that board2 is the result of applying move to board1.
    
    Args:
        board1: Chess board before the move
        board2: Chess board after the move
        move: Move that was applied
        
    Returns:
        True if consistent, False otherwise
    """
    # Create a copy of board1
    test_board = chess.Board(board1.fen())
    
    # Apply the move
    test_board.push(move)
    
    # Check if the resulting position matches board2
    return test_board.fen() == board2.fen()

def get_move_from_board_diff(board1, board2):
    """
    Try to determine the move that was made between board1 and board2.
    
    Args:
        board1: Chess board before the move
        board2: Chess board after the move
        
    Returns:
        Move object or None if it can't be determined
    """
    # Check if it's the same position
    if board1.fen() == board2.fen():
        return None
    
    # Check if it's the same player to move
    if board1.turn == board2.turn:
        return None
    
    # Try each legal move from board1
    for move in board1.legal_moves:
        test_board = chess.Board(board1.fen())
        test_board.push(move)
        
        # Check if this move leads to board2
        if test_board.board_fen() == board2.board_fen():
            return move
    
    return None

def find_legal_moves_for_piece(board, square):
    """
    Find all legal moves for a piece on a specific square.
    
    Args:
        board: Chess board
        square: Square to find moves for (0-63)
        
    Returns:
        List of legal moves for the piece on the square
    """
    # Check if there's a piece on the square
    piece = board.piece_at(square)
    if piece is None:
        return []
    
    # Find all legal moves for the piece
    legal_moves = []
    for move in board.legal_moves:
        if move.from_square == square:
            legal_moves.append(move)
    
    return legal_moves

def is_valid_position(fen):
    """
    Check if a FEN string represents a valid chess position.
    
    Args:
        fen: FEN string
        
    Returns:
        True if valid, False otherwise
    """
    try:
        chess.Board(fen)
        return True
    except ValueError:
        return False

def get_move_san(board, move):
    """
    Get the SAN notation for a move.
    
    Args:
        board: Chess board
        move: Move object
        
    Returns:
        SAN notation for the move or None if the move is illegal
    """
    if move in board.legal_moves:
        return board.san(move)
    return None 