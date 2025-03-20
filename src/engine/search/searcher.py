import chess
import time
import math
from collections import defaultdict

# Node types for transposition table
EXACT = 0
ALPHA = 1  # upper bound
BETA = 2   # lower bound

def get_board_hash(board):
    """
    Generate a hash value for the board position.
    
    Args:
        board: A chess.Board object
        
    Returns:
        A hash value for the board position
    """
    # Use the hash of the board's FEN string
    # but strip the halfmove clock and fullmove number to avoid unnecessary 
    # transposition table misses
    fen_parts = board.fen().split(' ')
    position_fen = ' '.join(fen_parts[:4])  # Keep only the position, side to move, castling, and en passant parts
    return hash(position_fen) & ((1 << 64) - 1)  # Ensure it fits in 64 bits

class Searcher:
    """
    Chess position searcher that implements alpha-beta search with various optimizations.
    """
    def __init__(self, evaluator, transposition_table):
        self.evaluator = evaluator
        self.tt = transposition_table
        
        # Search state
        self.nodes_searched = 0
        self.start_time = 0
        self.time_limit_ms = None
        self.stop_search = False
        
        # Move ordering helpers
        self.killer_moves = defaultdict(list)  # indexed by [depth][0 or 1]
        self.history_table = defaultdict(int)  # indexed by (piece_type, to_square)
        
        # PV tracking
        self.pv_table = {}
        self.pv_length = {}
        
    def search(self, board, depth=4, time_limit_ms=None):
        """
        Search for the best move using iterative deepening.
        
        Args:
            board: A chess.Board object
            depth: Maximum search depth
            time_limit_ms: Optional time limit in milliseconds
            
        Returns:
            Tuple (best_move, score, info)
        """
        self.nodes_searched = 0
        self.start_time = time.time()
        self.time_limit_ms = time_limit_ms
        self.stop_search = False
        
        # Reset PV tracking
        self.pv_table = {}
        self.pv_length = {}
        
        # If there's only one legal move, return it immediately
        legal_moves = list(board.legal_moves)
        if len(legal_moves) == 1:
            return legal_moves[0], 0, {
                'depth': 1,
                'nodes': 1,
                'time': 0,
                'pv': [legal_moves[0].uci()],
                'nps': 0
            }
        
        # Iterative deepening
        best_move = None
        best_score = -math.inf
        
        # might seem weird but doing depth/2 first helps us get better move ordering
        # when we start full search. this can speed up pruning by 2-3x
        for curr_depth in range(1, depth + 1):
            if self.stop_search:
                break
                
            # Reset killer moves for this iteration
            self.killer_moves = defaultdict(list)
            
            # Run alpha-beta search
            score = self._alpha_beta(board, curr_depth, -math.inf, math.inf, 0)
            
            # Check if search was stopped
            if self.stop_search:
                break
                
            # Get PV move
            position_hash = get_board_hash(board)
            if (0, position_hash) in self.pv_table:
                best_move = self.pv_table[(0, position_hash)]
                best_score = score
            
            # Check time
            elapsed = (time.time() - self.start_time) * 1000
            if self.time_limit_ms and elapsed >= self.time_limit_ms * 0.5:
                break
        
        # If no best move was found, use the first legal move
        if best_move is None and legal_moves:
            best_move = legal_moves[0]
        
        # Get principal variation
        pv = self._get_pv(board, depth)
        
        # Return info
        info = {
            'depth': depth,
            'nodes': self.nodes_searched,
            'time': int((time.time() - self.start_time) * 1000),
            'pv': pv,
            'nps': int(self.nodes_searched / (time.time() - self.start_time)) if time.time() > self.start_time else 0
        }
        
        return best_move, best_score, info
    
    def _alpha_beta(self, board, depth, alpha, beta, ply):
        """
        Alpha-beta search with various optimizations.
        
        Args:
            board: A chess.Board object
            depth: Remaining search depth
            alpha: Alpha bound
            beta: Beta bound
            ply: Current ply from root
            
        Returns:
            Score in centipawns
        """
        # Update nodes searched count
        self.nodes_searched += 1
        
        # Check if we need to stop the search
        if self.time_limit_ms and (time.time() - self.start_time) * 1000 > self.time_limit_ms:
            self.stop_search = True
            return 0
        
        # Initialize PV length
        self.pv_length[ply] = ply
        
        # Check for immediate draw
        if board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves() or board.is_fivefold_repetition():
            return 0
            
        # Check for checkmate
        if board.is_checkmate():
            return -20000 + ply  # Prefer shorter mates
            
        # Get board hash
        position_hash = get_board_hash(board)
            
        # Check transposition table
        tt_entry = self.tt.get(position_hash)
        if tt_entry and tt_entry['depth'] >= depth:
            tt_value, tt_type = tt_entry['value'], tt_entry['type']
            
            if tt_type == EXACT:
                self.pv_table[(ply, position_hash)] = tt_entry['best_move']
                # Make sure the next ply exists in pv_length before setting current ply's length
                if ply + 1 in self.pv_length:
                    self.pv_length[ply] = self.pv_length[ply + 1]
                return tt_value
            elif tt_type == ALPHA and tt_value <= alpha:
                return alpha
            elif tt_type == BETA and tt_value >= beta:
                return beta
        
        # At leaf nodes, perform quiescence search
        if depth <= 0:
            return self._quiescence_search(board, alpha, beta, ply)
            
        # Null move pruning (skip if in check or at low depths)
        if depth >= 3 and not board.is_check() and self._has_non_pawn_material(board, board.turn):
            board.push(chess.Move.null())
            null_score = -self._alpha_beta(board, depth - 3, -beta, -beta + 1, ply + 1)
            board.pop()
            
            if self.stop_search:
                return 0
                
            if null_score >= beta:
                return beta
        
        # Generate and order moves
        moves = self._order_moves(board, tt_entry['best_move'] if tt_entry else None, ply)
        
        moves_searched = 0
        best_score = -math.inf
        best_move = None
        
        # Late move reduction variables
        is_pv_node = beta > alpha + 1
        
        # Loop through ordered moves
        for move in moves:
            board.push(move)
            
            # Full-depth search for first move
            if moves_searched == 0:
                score = -self._alpha_beta(board, depth - 1, -beta, -alpha, ply + 1)
            else:
                # Late move reduction for non-tactical moves after trying a few moves
                # dont need a full depth on all these crazy moves, just check a few
                # plies ahead at first
                if moves_searched >= 4 and depth >= 3 and not is_pv_node and not board.is_check() and not self._is_tactical(board, move):
                    # Reduce depth for late non-tactical moves
                    score = -self._alpha_beta(board, depth - 2, -(alpha + 1), -alpha, ply + 1)
                else:
                    score = alpha + 1  # Ensure full-depth search
                    
                # If reduced search beats alpha, we need to do a full search 
                if score > alpha:
                    score = -self._alpha_beta(board, depth - 1, -(alpha + 1), -alpha, ply + 1)
                    
                    # If it might fail high, do a full window search
                    if score > alpha and score < beta:
                        score = -self._alpha_beta(board, depth - 1, -beta, -alpha, ply + 1)
            
            board.pop()
            
            if self.stop_search:
                return 0
                
            moves_searched += 1
            
            if score > best_score:
                best_score = score
                best_move = move
                
                # Update alpha
                if score > alpha:
                    alpha = score
                    
                    # Update PV table
                    self.pv_table[(ply, position_hash)] = move
                    
                    # Make sure next ply exists in pv_length before copying PV
                    if ply + 1 in self.pv_length:
                        # Copy PV from deeper plies
                        for next_ply in range(ply + 1, self.pv_length[ply + 1] + 1):
                            if (ply + 1, next_ply) in self.pv_table:
                                self.pv_table[(ply, next_ply)] = self.pv_table[(ply + 1, next_ply)]
                                
                        self.pv_length[ply] = self.pv_length[ply + 1]
                    
                    # Check for beta cutoff
                    if alpha >= beta:
                        # Store killer move
                        if not board.is_capture(move):
                            if len(self.killer_moves[ply]) < 2:
                                self.killer_moves[ply].append(move)
                            elif move not in self.killer_moves[ply]:
                                self.killer_moves[ply][1] = self.killer_moves[ply][0]
                                self.killer_moves[ply][0] = move
                                
                            # Update history table
                            piece = board.piece_at(move.from_square)
                            if piece:
                                self.history_table[(piece.piece_type, move.to_square)] += depth * depth
                        
                        break
        
        # Check if we found a valid move
        if moves_searched == 0:
            # No valid moves means either checkmate or stalemate, 
            # but we already checked for checkmate above
            return 0
        
        # Store position in transposition table
        tt_type = EXACT
        if best_score <= alpha:
            tt_type = ALPHA
        elif best_score >= beta:
            tt_type = BETA
            
        self.tt.store(position_hash, depth, best_score, tt_type, best_move)
        
        return best_score
    
    def _quiescence_search(self, board, alpha, beta, ply):
        """
        Quiescence search to handle tactical sequences.
        
        Args:
            board: A chess.Board object
            alpha: Alpha bound
            beta: Beta bound
            ply: Current ply from root
            
        Returns:
            Score in centipawns
        """
        # Update nodes searched count
        self.nodes_searched += 1
        
        # Check if we need to stop the search
        if self.time_limit_ms and (time.time() - self.start_time) * 1000 > self.time_limit_ms:
            self.stop_search = True
            return 0
        
        # Check for immediate draw
        if board.is_stalemate() or board.is_insufficient_material():
            return 0
            
        # Check for checkmate
        if board.is_checkmate():
            return -20000 + ply
        
        # Get static evaluation
        stand_pat = self.evaluator.evaluate(board)
        
        # Stand-pat cutoff
        if stand_pat >= beta:
            return beta
        if stand_pat > alpha:
            alpha = stand_pat
        
        # Generate and order captures
        captures = self._order_captures(board)
        
        for move in captures:
            # Skip bad captures that lose material
            # but always consider recaptures
            # Skip bad captures unless we're in check
            if not board.is_check() and not self._is_good_capture(board, move):
                continue
                
            board.push(move)
            score = -self._quiescence_search(board, -beta, -alpha, ply + 1)
            board.pop()
            
            if self.stop_search:
                return 0
                
            if score > alpha:
                alpha = score
                
                if alpha >= beta:
                    break
        
        return alpha
    
    def _order_moves(self, board, tt_move=None, ply=0):
        """
        Order moves to improve alpha-beta pruning efficiency.
        
        Args:
            board: A chess.Board object
            tt_move: Transposition table move
            ply: Current ply from root
            
        Returns:
            Ordered list of moves
        """
        moves = list(board.legal_moves)
        scores = []
        
        for move in moves:
            score = 0
            
            # PV/TT move gets highest priority
            if tt_move and move == tt_move:
                score = 20000
            # Captures are scored using MVV-LVA (Most Valuable Victim - Least Valuable Attacker)
            elif board.is_capture(move):
                victim = board.piece_at(move.to_square)
                attacker = board.piece_at(move.from_square)
                
                # Handle en passant captures where the victim isn't on the destination square
                if victim is None and board.is_en_passant(move):
                    # En passant capture - the victim is a pawn
                    victim_value = self._get_piece_value(chess.PAWN)
                elif victim is None:
                    # This shouldn't happen for normal captures, but just in case
                    victim_value = 0
                else:
                    victim_value = self._get_piece_value(victim.piece_type)
                
                if attacker is None:
                    # This really shouldn't happen, but just for safety
                    attacker_value = 0
                else:
                    attacker_value = self._get_piece_value(attacker.piece_type)
                
                score = 10000 + victim_value - attacker_value // 10
                
                # Prioritize queen promotions
                if move.promotion == chess.QUEEN:
                    score += 9000
            else:
                # Killer moves
                if ply < len(self.killer_moves) and move in self.killer_moves[ply]:
                    score = 9000 - self.killer_moves[ply].index(move)
                
                # History heuristic
                piece = board.piece_at(move.from_square)
                if piece:
                    score += min(self.history_table[(piece.piece_type, move.to_square)] // 10, 8000)
                    
                # Piece-specific heuristics
                piece_type = board.piece_at(move.from_square).piece_type
                
                # Push pawns forward (slightly favor pawn advances)
                if piece_type == chess.PAWN:
                    # Rank bonus for pawns, especially advanced pawns
                    if board.turn == chess.WHITE:
                        score += chess.square_rank(move.to_square) * 10
                    else:
                        score += (7 - chess.square_rank(move.to_square)) * 10
                
                # Knights and bishops toward center
                elif piece_type in (chess.KNIGHT, chess.BISHOP):
                    # Favor moves toward the center
                    to_file, to_rank = chess.square_file(move.to_square), chess.square_rank(move.to_square)
                    center_dist = abs(to_file - 3.5) + abs(to_rank - 3.5)
                    score += int((4 - center_dist) * 10)
                
                # Rooks to open files
                elif piece_type == chess.ROOK:
                    # Favor moves to files without pawns
                    to_file = chess.square_file(move.to_square)
                    file_mask = chess.BB_FILES[to_file]
                    
                    # Convert SquareSet to integer for bitwise operations
                    white_pawns = int(board.pieces(chess.PAWN, chess.WHITE))
                    black_pawns = int(board.pieces(chess.PAWN, chess.BLACK))
                    
                    if not (white_pawns & file_mask) and not (black_pawns & file_mask):
                        score += 30
                
                # King safety in middlegame, aggression in endgame
                elif piece_type == chess.KING:
                    # Count pieces to determine if we're in endgame
                    piece_count = chess.popcount(board.occupied)
                    
                    if piece_count > 20:  # Middlegame
                        # Penalize king moves that leave the king exposed
                        from_file = chess.square_file(move.from_square)
                        if (from_file <= 2 or from_file >= 5) and chess.square_rank(move.from_square) == (0 if board.turn == chess.WHITE else 7):
                            # Penalize moving away from a castled position
                            score -= 50
                    else:  # Endgame
                        # Encourage king centralization in endgame
                        to_file, to_rank = chess.square_file(move.to_square), chess.square_rank(move.to_square)
                        center_dist = abs(to_file - 3.5) + abs(to_rank - 3.5)
                        score += int((4 - center_dist) * 20)
            
            scores.append(score)
        
        # Sort moves by score in descending order
        return [move for _, move in sorted(zip(scores, moves), key=lambda x: x[0], reverse=True)]
    
    def _order_captures(self, board):
        """
        Order capture moves using MVV-LVA.
        
        Args:
            board: A chess.Board object
            
        Returns:
            Ordered list of capture moves
        """
        captures = []
        
        for move in board.legal_moves:
            if board.is_capture(move):
                captures.append(move)
        
        # Score captures by MVV-LVA
        scores = []
        for move in captures:
            victim = board.piece_at(move.to_square)
            attacker = board.piece_at(move.from_square)
            
            # Handle en passant captures
            if victim is None and board.is_en_passant(move):
                victim_value = self._get_piece_value(chess.PAWN)
            elif victim is None:
                victim_value = 0
            else:
                victim_value = self._get_piece_value(victim.piece_type)
            
            if attacker is None:
                attacker_value = 0
            else:
                attacker_value = self._get_piece_value(attacker.piece_type)
            
            # MVV-LVA: Most Valuable Victim - Least Valuable Attacker
            score = victim_value - attacker_value // 10
            
            # Prioritize queen promotions
            if move.promotion == chess.QUEEN:
                score += 900
                
            scores.append(score)
        
        # Sort moves by score in descending order
        return [move for _, move in sorted(zip(scores, captures), key=lambda x: x[0], reverse=True)]
    
    def _is_tactical(self, board, move):
        """
        Check if a move is tactical (capture, check, promotion).
        
        Args:
            board: A chess.Board object
            move: A chess.Move object
            
        Returns:
            True if the move is tactical, False otherwise
        """
        return board.is_capture(move) or move.promotion is not None or board.gives_check(move)
    
    def _is_good_capture(self, board, move):
        """
        Check if a capture is likely good using SEE (Static Exchange Evaluation).
        This is a simplified version that just checks if the captured piece is worth more than the capturing piece.
        
        Args:
            board: A chess.Board object
            move: A chess.Move object
            
        Returns:
            True if the capture is likely good, False otherwise
        """
        victim = board.piece_at(move.to_square)
        attacker = board.piece_at(move.from_square)
        
        # Handle en passant captures
        if victim is None and board.is_en_passant(move):
            victim_value = self._get_piece_value(chess.PAWN)
        elif victim is None:
            return True  # If no victim but it's a capture, assume it's good (unusual case)
        else:
            victim_value = self._get_piece_value(victim.piece_type)
            
        if attacker is None:
            return True  # Should never happen, but just in case
        else:
            attacker_value = self._get_piece_value(attacker.piece_type)
            
        return victim_value >= attacker_value
    
    def _get_piece_value(self, piece_type):
        """
        Get the value of a piece.
        
        Args:
            piece_type: A chess.PieceType
            
        Returns:
            The value of the piece in centipawns
        """
        values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000
        }
        
        return values.get(piece_type, 0)
    
    def _has_non_pawn_material(self, board, color):
        """
        Check if a side has non-pawn material.
        
        Args:
            board: A chess.Board object
            color: Color to check
            
        Returns:
            True if the side has non-pawn material, False otherwise
        """
        # Convert pieces_mask results to integers and combine with bitwise OR
        return (int(board.pieces_mask(chess.KNIGHT, color)) | 
                int(board.pieces_mask(chess.BISHOP, color)) | 
                int(board.pieces_mask(chess.ROOK, color)) | 
                int(board.pieces_mask(chess.QUEEN, color))) != 0
    
    def _get_pv(self, board, depth):
        """
        Get the principal variation from the PV table.
        
        Args:
            board: A chess.Board object
            depth: Maximum depth
            
        Returns:
            List of move UCI strings
        """
        pv = []
        ply = 0
        
        # Copy board
        temp_board = board.copy()
        position_hash = get_board_hash(temp_board)
        
        # Stop if we reach depth or if the key doesn't exist in the PV table
        while ply < depth and (ply, position_hash) in self.pv_table:
            move = self.pv_table[(ply, position_hash)]
            
            # Validate the move
            if move not in temp_board.legal_moves:
                break
                
            pv.append(move.uci())
            temp_board.push(move)
            ply += 1
            
            # Get hash for the new position
            position_hash = get_board_hash(temp_board)
        
        return pv 