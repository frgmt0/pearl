import chess
import time
import numpy as np

# Define piece values in centipawns
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000  # high value, though not used directly in material counting
}

# For game phase calculation
PIECE_PHASE_VALUES = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 1,
    chess.ROOK: 2,
    chess.QUEEN: 4,
    chess.KING: 0
}

TOTAL_PHASE = 24  # = 4*1 + 4*1 + 4*2 + 2*4 = 16 (all pieces except pawns and kings)

# Piece-square tables for middlegame/opening
# The tables are flipped for black, so we only need to define them for white
# Format is A1, B1, ..., H1, A2, ..., H8
PAWN_MG = np.array([
      0,   0,   0,   0,   0,   0,   0,   0,
     98, 134,  61,  95,  68, 126,  34, -11,
     -6,   7,  26,  31,  65,  56,  25, -20,
    -14,  13,   6,  21,  23,  12,  17, -23,
    -27, -2,  -5,  12,  17,   6,  10, -25,
    -26, -4,   3, -10,  -3, -7,  3, -27,
    -22,   0,  -1,  -1,  10, -12, 5, -28,
      0,   0,   0,   0,   0,   0,   0,   0
], dtype=np.int16)

KNIGHT_MG = np.array([
    -169, -50, -35, -29, -29, -35, -50, -169,
     -50, -25,  -10,  -5,  -5, -10, -25,  -50,
     -35, -10,   0,   5,   5,   0, -10,  -35,
     -29,  -5,   5,  10,  10,   5,  -5,  -29,
     -29,  -5,   5,  10,  10,   5,  -5,  -29,
     -35, -10,   0,   5,   5,   0, -10,  -35,
     -50, -25, -10,  -5,  -5, -10, -25,  -50,
    -169, -50, -35, -29, -29, -35, -50, -169
], dtype=np.int16)

BISHOP_MG = np.array([
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10,   5,   0,   0,   0,   0,   5, -10,
    -10,  10,  10,  10,  10,  10,  10, -10,
    -10,   0,  10,  10,  10,  10,   0, -10,
    -10,   5,   5,  10,  10,   5,   5, -10,
    -10,   0,   5,  10,  10,   5,   0, -10,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -20, -10, -10, -10, -10, -10, -10, -20
], dtype=np.int16)

ROOK_MG = np.array([
      0,  0,  0,  5,  5,  0,  0,  0,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     5, 10, 10, 10, 10, 10, 10,  5,
    0,  0,  0,  0,  0,  0,  0,  0
], dtype=np.int16)

QUEEN_MG = np.array([
    -20, -10, -10, -5, -5, -10, -10, -20,
    -10,   0,   0,  0,  0,   0,   0, -10,
    -10,   0,   5,  5,  5,   5,   0, -10,
     -5,   0,   5,  5,  5,   5,   0,  -5,
      0,   0,   5,  5,  5,   5,   0,  -5,
    -10,   5,   5,  5,  5,   5,   0, -10,
    -10,   0,   5,  0,  0,   0,   0, -10,
    -20, -10, -10, -5, -5, -10, -10, -20
], dtype=np.int16)

KING_MG = np.array([
     20,  30,  10,   0,   0,  10,  30,  20,
     20,  20,   0,   0,   0,   0,  20,  20,
    -10, -20, -20, -20, -20, -20, -20, -10,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30
], dtype=np.int16)

# Piece-square tables for endgame
PAWN_EG = np.array([
      0,   0,   0,   0,   0,   0,   0,   0,
    178, 173, 158, 134, 147, 132, 165, 187,
     94, 100,  85,  67,  56,  53,  82,  84,
     32,  24,  13,   5,  -2,   4,  17,  17,
     13,   9,  -3,  -7,  -7,  -8,   3,  -1,
      4,   7,  -6,   1,   0,  -5,  -1,  -8,
     13,   8,   8,  10,  13,   0,   2,  -7,
      0,   0,   0,   0,   0,   0,   0,   0
], dtype=np.int16)

KNIGHT_EG = np.array([
    -58, -38, -13, -28, -28, -13, -38, -58,
    -30, -13,   0,  -10,  -10,   0, -13, -30,
    -15,   8,  15,  15,  15,  15,   8, -15,
    -15,   0,  15,  15,  15,  15,   0, -15,
    -15,   0,  15,  15,  15,  15,   0, -15,
    -15,   8,  15,  20,  20,  15,   8, -15,
    -30, -13,   0,  -10,  -10,   0, -13, -30,
    -58, -38, -13, -28, -28, -13, -38, -58
], dtype=np.int16)

BISHOP_EG = np.array([
    -14, -21, -11,  -8,  -8, -11, -21, -14,
    -21,  -8,  -3,  -3,  -3,  -3,  -8, -21,
    -11,  -3,   8,   9,   9,   8,  -3, -11,
     -8,  -3,   9,  12,  12,   9,  -3,  -8,
     -8,  -3,   9,  12,  12,   9,  -3,  -8,
    -11,  -3,   8,   9,   9,   8,  -3, -11,
    -21,  -8,  -3,  -3,  -3,  -3,  -8, -21,
    -14, -21, -11,  -8,  -8, -11, -21, -14
], dtype=np.int16)

ROOK_EG = np.array([
    0,  0,  0,  0,  0,  0,  0,  0,
    0,  0,  0,  0,  0,  0,  0,  0,
    0,  0,  0,  0,  0,  0,  0,  0,
    0,  0,  0,  0,  0,  0,  0,  0,
    0,  0,  0,  0,  0,  0,  0,  0,
    0,  0,  0,  0,  0,  0,  0,  0,
    0,  0,  0,  0,  0,  0,  0,  0,
    0,  0,  0,  0,  0,  0,  0,  0
], dtype=np.int16)

QUEEN_EG = np.array([
    -9,  -9,  -9,  -9,  -9,  -9,  -9,  -9,
    -9,  -9,  -9,  -9,  -9,  -9,  -9,  -9,
    -9,  -9,  -9,  -9,  -9,  -9,  -9,  -9,
    -9,  -9,  -9,  -9,  -9,  -9,  -9,  -9,
    -9,  -9,  -9,  -9,  -9,  -9,  -9,  -9,
    -9,  -9,  -9,  -9,  -9,  -9,  -9,  -9,
    -9,  -9,  -9,  -9,  -9,  -9,  -9,  -9,
    -9,  -9,  -9,  -9,  -9,  -9,  -9,  -9
], dtype=np.int16)

KING_EG = np.array([
    -50, -30, -30, -30, -30, -30, -30, -50,
    -30, -20, -10,   0,   0, -10, -20, -30,
    -30, -10,  20,  30,  30,  20, -10, -30,
    -30, -10,  30,  40,  40,  30, -10, -30,
    -30, -10,  30,  40,  40,  30, -10, -30,
    -30, -10,  20,  30,  30,  20, -10, -30,
    -30, -30,   0,   0,   0,   0, -30, -30,
    -50, -30, -30, -30, -30, -30, -30, -50
], dtype=np.int16)

# Group the piece-square tables in dictionaries for easier access
PST_MG = {
    chess.PAWN: PAWN_MG,
    chess.KNIGHT: KNIGHT_MG,
    chess.BISHOP: BISHOP_MG,
    chess.ROOK: ROOK_MG,
    chess.QUEEN: QUEEN_MG,
    chess.KING: KING_MG
}

PST_EG = {
    chess.PAWN: PAWN_EG,
    chess.KNIGHT: KNIGHT_EG,
    chess.BISHOP: BISHOP_EG,
    chess.ROOK: ROOK_EG,
    chess.QUEEN: QUEEN_EG,
    chess.KING: KING_EG
}

# Flip index for black pieces
def _flip_vertical(square):
    return square ^ 56  # Flips rank

class Evaluator:
    """
    Position evaluator using classical chess heuristics.
    
    This evaluator uses material balance, piece-square tables, mobility,
    pawn structure, and other positional features to evaluate a position.
    """
    def __init__(self):
        # Initialize any evaluation parameters here
        pass
    
    def evaluate(self, board):
        """
        Evaluate the given chess position.
        
        Args:
            board: A chess.Board object representing the position to evaluate
            
        Returns:
            A score in centipawns (positive means white is better,
            negative means black is better)
        """
        start_time = time.time()
        
        # if the game is over, return a winning or drawing score
        if board.is_checkmate():
            if board.turn == chess.WHITE:
                return -20000  # Black wins
            else:
                return 20000   # White wins
        
        if board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves() or board.is_fivefold_repetition():
            return 0  # Draw
        
        # Calculate material and piece-square table scores
        mg_score = 0  # middlegame score
        eg_score = 0  # endgame score
        phase = 0     # game phase (0 = endgame, 24 = middlegame)
        
        # Calculate material + piece-square scores and phase
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is None:
                continue
                
            # Add to phase calculation
            phase += PIECE_PHASE_VALUES[piece.piece_type]
            
            # Material and PST calculations
            value = PIECE_VALUES[piece.piece_type]
            
            # Get square index (0-63) and adjust for black pieces
            sq_idx = square
            if piece.color == chess.BLACK:
                sq_idx = _flip_vertical(square)
            
            # Add material and piece-square table scores
            if piece.color == chess.WHITE:
                mg_score += value
                mg_score += PST_MG[piece.piece_type][sq_idx]
                eg_score += value
                eg_score += PST_EG[piece.piece_type][sq_idx]
            else:
                mg_score -= value
                mg_score -= PST_MG[piece.piece_type][sq_idx]
                eg_score -= value
                eg_score -= PST_EG[piece.piece_type][sq_idx]
        
        # Avoid negative phase (can happen when pieces are captured)
        phase = min(phase, TOTAL_PHASE)
        
        # Interpolate between middle and endgame scores based on phase
        phase_factor = phase / TOTAL_PHASE
        score = int(mg_score * phase_factor + eg_score * (1 - phase_factor))
        
        # Mobility evaluation
        score += self._evaluate_mobility(board)
        
        # Pawn structure evaluation
        score += self._evaluate_pawn_structure(board)
        
        # King safety evaluation
        score += self._evaluate_king_safety(board, phase_factor)
        
        # Bishop pair bonus
        score += self._evaluate_bishop_pair(board)
        
        # Rooks on open files
        score += self._evaluate_rooks(board)
        
        # End timing
        end_time = time.time()
        
        # Return final score, adjusted for side to move
        return score if board.turn == chess.WHITE else -score
    
    def _evaluate_mobility(self, board):
        """
        Evaluate piece mobility: more possible moves = better position.
        
        Args:
            board: A chess.Board object
            
        Returns:
            Mobility score in centipawns
        """
        # we need to make a copy and switch side to move to calc opponent mobility
        temp_board = board.copy()
        temp_board.turn = not board.turn
        
        # count legal moves for both sides (need to convert generators to lists so we can count)
        white_moves = len(list(move for move in board.legal_moves if board.turn == chess.WHITE))
        black_moves = len(list(move for move in temp_board.legal_moves if temp_board.turn == chess.BLACK))
        
        # maybe we need to be more fancy later but for now simpler is better and more performant
        # approx 5 centipawns per extra move
        return 5 * (white_moves - black_moves)
    
    def _evaluate_pawn_structure(self, board):
        """
        Evaluate pawn structure including doubled, isolated, and passed pawns.
        
        Args:
            board: A chess.Board object
            
        Returns:
            Pawn structure score in centipawns
        """
        score = 0
        
        # doubled pawns penalty
        for file_idx in range(8):
            file_mask = chess.BB_FILES[file_idx]
            # Convert SquareSet to integer using int() before bitwise operations
            white_pawns_on_file = bin(int(board.pieces(chess.PAWN, chess.WHITE)) & file_mask).count('1')
            black_pawns_on_file = bin(int(board.pieces(chess.PAWN, chess.BLACK)) & file_mask).count('1')
            
            if white_pawns_on_file > 1:
                score -= 10 * (white_pawns_on_file - 1)  # -10 per doubled pawn
            if black_pawns_on_file > 1:
                score += 10 * (black_pawns_on_file - 1)  # +10 for opponent's doubled pawns
        
        # isolated pawns penalty
        for file_idx in range(8):
            # Create a mask for the file and adjacent files
            adj_files_mask = 0
            if file_idx > 0:
                adj_files_mask |= chess.BB_FILES[file_idx - 1]
            if file_idx < 7:
                adj_files_mask |= chess.BB_FILES[file_idx + 1]
            
            file_mask = chess.BB_FILES[file_idx]
            
            # Convert SquareSet to integer using int() before bitwise operations
            white_pawns = int(board.pieces(chess.PAWN, chess.WHITE))
            black_pawns = int(board.pieces(chess.PAWN, chess.BLACK))
            
            # Check if there are white pawns on this file but none on adjacent files
            if (white_pawns & file_mask) and not (white_pawns & adj_files_mask):
                # Count how many white pawns are isolated on this file
                isolated_pawns = bin(white_pawns & file_mask).count('1')
                score -= 20 * isolated_pawns  # -20 per isolated pawn
            
            # Same for black pawns
            if (black_pawns & file_mask) and not (black_pawns & adj_files_mask):
                isolated_pawns = bin(black_pawns & file_mask).count('1')
                score += 20 * isolated_pawns  # +20 for opponent's isolated pawns
        
        # passed pawns bonus (simplified check)
        white_pawns = board.pieces(chess.PAWN, chess.WHITE)
        black_pawns = board.pieces(chess.PAWN, chess.BLACK)
        
        # Convert SquareSet to list of squares for iteration
        white_pawn_squares = white_pawns.tolist()
        black_pawn_squares = black_pawns.tolist()
        
        # this is a lazy way to check for passed pawns
        # a proper impl would check if there are no enemy pawns that can capture or block
        for square in white_pawn_squares:
            rank = chess.square_rank(square)
            file = chess.square_file(square)
            passed = True
            
            # Check if there are any black pawns that can block or capture this pawn
            for r in range(rank + 1, 8):
                # Check forward, and diagonally forward
                for f in [file - 1, file, file + 1]:
                    if 0 <= f < 8:
                        check_square = chess.square(f, r)
                        if board.piece_at(check_square) == chess.Piece(chess.PAWN, chess.BLACK):
                            passed = False
                            break
                if not passed:
                    break
            
            if passed:
                # Bonus increases as pawn advances
                score += 10 + 10 * rank  # Higher bonus for more advanced pawns
        
        # Same logic for black passed pawns
        for square in black_pawn_squares:
            rank = chess.square_rank(square)
            file = chess.square_file(square)
            passed = True
            
            for r in range(0, rank):
                for f in [file - 1, file, file + 1]:
                    if 0 <= f < 8:
                        check_square = chess.square(f, r)
                        if board.piece_at(check_square) == chess.Piece(chess.PAWN, chess.WHITE):
                            passed = False
                            break
                if not passed:
                    break
            
            if passed:
                # Bonus increases as pawn advances (from black's perspective)
                score -= 10 + 10 * (7 - rank)
        
        return score
    
    def _evaluate_king_safety(self, board, phase_factor):
        """
        Evaluate king safety based on pawn shield and piece attacks.
        
        Args:
            board: A chess.Board object
            phase_factor: Factor indicating middle vs endgame (1.0 = middlegame)
            
        Returns:
            King safety score in centipawns
        """
        # only important in middlegame, not endgame
        if phase_factor < 0.5:
            return 0
            
        score = 0
        
        # king safety is super complicated but just a simple version for now
        
        # Check pawn shield for white king (assuming kingside castling)
        white_king_square = board.king(chess.WHITE)
        if white_king_square is not None:
            white_king_file = chess.square_file(white_king_square)
            white_king_rank = chess.square_rank(white_king_square)
            
            # Bonus for having pawns in front of castled king
            if white_king_file >= 5 and white_king_rank == 0:  # Kingside castle
                shield_squares = [
                    chess.square(white_king_file - 1, 1),
                    chess.square(white_king_file, 1),
                    chess.square(white_king_file + 1, 1) if white_king_file < 7 else None
                ]
                for sq in shield_squares:
                    if sq is not None and board.piece_at(sq) == chess.Piece(chess.PAWN, chess.WHITE):
                        score += 15  # Bonus for each pawn shielding the king
            
            elif white_king_file <= 3 and white_king_rank == 0:  # Queenside castle
                shield_squares = [
                    chess.square(white_king_file - 1, 1) if white_king_file > 0 else None,
                    chess.square(white_king_file, 1),
                    chess.square(white_king_file + 1, 1)
                ]
                for sq in shield_squares:
                    if sq is not None and board.piece_at(sq) == chess.Piece(chess.PAWN, chess.WHITE):
                        score += 15  # Bonus for each pawn shielding the king
        
        # Same for black king
        black_king_square = board.king(chess.BLACK)
        if black_king_square is not None:
            black_king_file = chess.square_file(black_king_square)
            black_king_rank = chess.square_rank(black_king_square)
            
            # Bonus for having pawns in front of castled king
            if black_king_file >= 5 and black_king_rank == 7:  # Kingside castle
                shield_squares = [
                    chess.square(black_king_file - 1, 6),
                    chess.square(black_king_file, 6),
                    chess.square(black_king_file + 1, 6) if black_king_file < 7 else None
                ]
                for sq in shield_squares:
                    if sq is not None and board.piece_at(sq) == chess.Piece(chess.PAWN, chess.BLACK):
                        score -= 15  # Bonus for each pawn shielding the king
            
            elif black_king_file <= 3 and black_king_rank == 7:  # Queenside castle
                shield_squares = [
                    chess.square(black_king_file - 1, 6) if black_king_file > 0 else None,
                    chess.square(black_king_file, 6),
                    chess.square(black_king_file + 1, 6)
                ]
                for sq in shield_squares:
                    if sq is not None and board.piece_at(sq) == chess.Piece(chess.PAWN, chess.BLACK):
                        score -= 15  # Bonus for each pawn shielding the king
                        
        # Scale king safety by phase factor (more important in middlegame)
        return int(score * phase_factor)
    
    def _evaluate_bishop_pair(self, board):
        """
        Award a bonus for having the bishop pair (both bishops on different colored squares).
        
        Args:
            board: A chess.Board object
            
        Returns:
            Bishop pair bonus in centipawns
        """
        score = 0
        
        white_bishop_count = len(list(board.pieces(chess.BISHOP, chess.WHITE)))
        black_bishop_count = len(list(board.pieces(chess.BISHOP, chess.BLACK)))
        
        # Bishop pair bonus
        if white_bishop_count >= 2:
            # Check if bishops are on different colored squares
            white_bishops = list(board.pieces(chess.BISHOP, chess.WHITE))
            # If first bishop square is odd and second is even (or vice versa), they're on different colors
            if (white_bishops[0] + chess.square_rank(white_bishops[0])) % 2 != (white_bishops[1] + chess.square_rank(white_bishops[1])) % 2:
                score += 50  # Bishop pair bonus
        
        if black_bishop_count >= 2:
            black_bishops = list(board.pieces(chess.BISHOP, chess.BLACK))
            if (black_bishops[0] + chess.square_rank(black_bishops[0])) % 2 != (black_bishops[1] + chess.square_rank(black_bishops[1])) % 2:
                score -= 50  # Bishop pair bonus for black
        
        return score
    
    def _evaluate_rooks(self, board):
        """
        Evaluate rooks: bonus for rooks on open files and 7th/2nd rank.
        
        Args:
            board: A chess.Board object
            
        Returns:
            Rook evaluation in centipawns
        """
        score = 0
        
        # Bonus for rooks on open files
        for square in board.pieces(chess.ROOK, chess.WHITE):
            file_idx = chess.square_file(square)
            file_mask = chess.BB_FILES[file_idx]
            
            # Convert SquareSet to integer for bitwise operations
            white_pawns = int(board.pieces(chess.PAWN, chess.WHITE))
            black_pawns = int(board.pieces(chess.PAWN, chess.BLACK))
            
            # Check if file is completely open (no pawns)
            if not (white_pawns & file_mask) and not (black_pawns & file_mask):
                score += 25  # Completely open file
            # Check if file is semi-open (no friendly pawns)
            elif not (white_pawns & file_mask):
                score += 15  # Semi-open file
            
            # Bonus for rook on 7th rank (relative to opponent's king)
            if chess.square_rank(square) == 6:  # 7th rank
                score += 30
        
        # Same for black rooks
        for square in board.pieces(chess.ROOK, chess.BLACK):
            file_idx = chess.square_file(square)
            file_mask = chess.BB_FILES[file_idx]
            
            # Check if file is completely open (no pawns)
            if not (white_pawns & file_mask) and not (black_pawns & file_mask):
                score -= 25  # Completely open file
            # Check if file is semi-open (no friendly pawns)
            elif not (black_pawns & file_mask):
                score -= 15  # Semi-open file
            
            # Bonus for rook on 2nd rank (relative to opponent's king)
            if chess.square_rank(square) == 1:  # 2nd rank
                score -= 30
        
        return score 