import chess
import time
import random
from collections import defaultdict, OrderedDict
import numpy as np
import threading
import concurrent.futures
import multiprocessing

from src.engine.score import evaluate_position, classical_evaluate
from src.engine.movePick import pick_move
from src.engine.memory import TranspositionTable

# Constants for search
INFINITY = 30000
MATE_SCORE = 29000
DRAW_SCORE = 0
MAX_DEPTH = 50  # Increased to handle deeper searches
QUIESCENCE_DEPTH = 4
MAX_PLY = 100   # Maximum ply depth for search (to prevent stack overflows)

# Transposition table entry flags
EXACT = 0
ALPHA = 1
BETA = 2

class SearchInfo:
    """Class to maintain search information and statistics with thread safety."""
    
    def __init__(self):
        self.start_time = 0
        self.nodes_searched = 0
        self.stop_search = False
        self.depth = 0
        self.time_limit_ms = 0
        self.best_move = None
        self.best_score = -INFINITY
        self.pv_line = []
        self.history_heuristic = defaultdict(int)
        # Initialize with MAX_PLY instead of MAX_DEPTH for deeper searches
        self.killer_moves = [[None for _ in range(2)] for _ in range(MAX_PLY + 1)]
        
        # Use the TranspositionTable class instead of a simple dictionary
        self.transposition_table = TranspositionTable(max_size_mb=128)  # 128MB table
        self.tt_age = 0  # Age counter for the transposition table
        
        # Thread synchronization locks
        self.nodes_lock = threading.Lock()
        self.best_move_lock = threading.Lock()
        self.history_lock = threading.Lock()
    
    def increment_nodes(self):
        """Increment the nodes searched counter (thread-safe)."""
        with self.nodes_lock:
            self.nodes_searched += 1
    
    def get_elapsed_time(self):
        """Get the elapsed time in milliseconds."""
        return int((time.time() - self.start_time) * 1000)
    
    def should_stop(self):
        """Check if the search should stop."""
        # Check if stop flag is set
        if self.stop_search:
            return True
        
        # Check if time limit is reached
        if self.time_limit_ms > 0 and self.get_elapsed_time() >= self.time_limit_ms:
            self.stop_search = True
            return True
        
        return False
    
    def update_best_move(self, move, score, depth, pv_line):
        """Update the best move found (thread-safe)."""
        with self.best_move_lock:
            # Only update if this is a deeper search or a better score
            if depth > self.depth or (depth == self.depth and score > self.best_score):
                self.best_move = move
                self.best_score = score
                self.depth = depth
                self.pv_line = pv_line
    
    def update_killer_move(self, move, ply):
        """
        Update killer moves (thread-safe).
        Killer moves are quiet moves that cause beta cutoffs.
        """
        # Ensure ply is within bounds
        if ply >= MAX_PLY:
            return
            
        # Don't add the same move twice
        if self.killer_moves[ply][0] == move:
            return
            
        # Shift killer moves and add the new one
        self.killer_moves[ply][1] = self.killer_moves[ply][0]
        self.killer_moves[ply][0] = move
            
    def update_history_score(self, move, depth):
        """
        Update history heuristic score for a move (thread-safe).
        Higher depth moves get higher scores.
        """
        from_sq = move.from_square
        to_sq = move.to_square
        with self.history_lock:
            self.history_heuristic[(from_sq, to_sq)] += depth * depth
        
    def hash_position(self, board, depth, alpha, beta, score, move=None, flag=EXACT):
        """
        Store a position in the transposition table (thread-safe).
        
        Args:
            board: Chess board position
            depth: Remaining depth at this position
            alpha: Alpha bound
            beta: Beta bound
            score: Evaluation score
            move: Best move found at this position
            flag: Type of node (EXACT, ALPHA, BETA)
        """
        # Get Zobrist hash of the position
        key = hash(board.fen())
        
        # Store position information (thread-safe)
        self.transposition_table.store(key, depth, score, move, flag, self.tt_age)
        
    def probe_hash(self, board, depth, alpha, beta):
        """
        Probe the transposition table for a position (thread-safe).
        
        Args:
            board: Chess board position
            depth: Remaining depth
            alpha: Alpha bound
            beta: Beta bound
            
        Returns:
            Tuple of (found, score, move) where:
                found: True if position was found and can be used
                score: Score for the position if found
                move: Best move for the position if found
        """
        key = hash(board.fen())
        
        # Thread-safe read from transposition table
        entry = self.transposition_table.probe(key)
        
        if entry is None:
            return False, 0, None
            
        # Only use entries with sufficient depth
        if entry['depth'] < depth:
            return False, 0, entry['move']  # Still return the move for move ordering
            
        score = entry['score']
        flag = entry['flag']
        
        # Handle different node types
        if flag == EXACT:
            return True, score, entry['move']
        elif flag == ALPHA and score <= alpha:
            return True, alpha, entry['move']
        elif flag == BETA and score >= beta:
            return True, beta, entry['move']
            
        # Return the move for move ordering but don't use the score
        return False, 0, entry['move']

def is_futile_move(board, move, margin=100):
    """
    Check if a move is likely futile and can be pruned.
    
    Args:
        board: Chess board position
        move: Move to check
        margin: Margin in centipawns to determine futility
        
    Returns:
        True if the move is likely futile, False otherwise
    """
    # Don't prune captures, promotions, or checks
    if board.is_capture(move) or move.promotion:
        return False
        
    # Check if the move gives check
    board.push(move)
    gives_check = board.is_check()
    board.pop()
    
    if gives_check:
        return False
    
    # Get static evaluation
    static_eval = classical_evaluate(board)
    
    # If we're already doing well, the move might be futile
    if board.turn == chess.WHITE and static_eval > margin:
        return True
    elif board.turn == chess.BLACK and static_eval < -margin:
        return True
    
    return False

def alpha_beta(board, depth, alpha, beta, info, ply=0, null_move_allowed=True):
    """
    Alpha-beta search algorithm with various enhancements.
    
    Args:
        board: Chess board position
        depth: Remaining depth to search
        alpha: Alpha bound
        beta: Beta bound
        info: SearchInfo object
        ply: Current ply (half-move) from root
        null_move_allowed: Whether null move pruning is allowed
        
    Returns:
        Score for the position
    """
    # Safety check for maximum recursion depth
    if ply >= MAX_PLY:
        return evaluate_position(board)
        
    # Check if we should stop the search
    if info.should_stop():
        return 0
        
    # Update node count (thread-safe)
    info.increment_nodes()
    
    # Check for draw by repetition, fifty-move rule, or insufficient material
    if board.is_repetition(2) or board.is_fifty_moves() or board.is_insufficient_material():
        return DRAW_SCORE
    
    # Initialize PV line
    pv_line = []
    
    # Probe transposition table
    hash_hit, hash_score, hash_move = info.probe_hash(board, depth, alpha, beta)
    if hash_hit:
        return hash_score
    
    # If we've reached the maximum depth, perform quiescence search
    if depth <= 0:
        return quiescence_search(board, alpha, beta, info, ply, QUIESCENCE_DEPTH)
    
    # Check for checkmate or stalemate
    if board.is_game_over():
        if board.is_checkmate():
            # Return negative score for checkmate (adjusted by ply to prefer shorter paths)
            return -MATE_SCORE + ply
        else:
            # Stalemate or other draw
            return DRAW_SCORE
    
    # Store original alpha for transposition table
    alpha_orig = alpha
    
    # Null move pruning
    # Skip null move pruning if:
    # 1. We're in check
    # 2. We're at a low depth
    # 3. Null move pruning is disabled for this call
    if (
        null_move_allowed and 
        depth >= 3 and 
        not board.is_check() and
        has_non_pawn_material(board, board.turn)
    ):
        # Make a null move (skip a turn)
        board.push(chess.Move.null())
        
        # Search with reduced depth (R=2)
        null_score = -alpha_beta(
            board, depth - 3, -beta, -beta + 1, info, ply + 1, False
        )
        
        board.pop()
        
        # Check if search should stop
        if info.should_stop():
            return 0
        
        # If the score is good enough, we can prune this branch
        if null_score >= beta:
            return beta
    
    # Internal iterative deepening
    # If we don't have a hash move and we're at a reasonable depth
    if hash_move is None and depth >= 4:
        # Do a shallow search to find a good move
        iid_score = alpha_beta(board, depth - 2, alpha, beta, info, ply, False)
        
        # Check if search should stop
        if info.should_stop():
            return 0
            
        # Get the best move from the shallow search
        hash_hit, hash_score, hash_move = info.probe_hash(board, depth - 2, alpha, beta)
    
    # Get legal moves
    legal_moves = list(board.legal_moves)
    
    # If there are no legal moves, it's checkmate or stalemate
    if not legal_moves:
        if board.is_check():
            return -MATE_SCORE + ply
        else:
            return DRAW_SCORE
    
    # Order moves to improve pruning
    ordered_moves = pick_move(board, legal_moves, info, hash_move, ply)
    
    # Initialize best score and move
    best_score = -INFINITY
    best_move = None
    
    # Count of moves searched
    moves_searched = 0
    
    # Try each move
    for move in ordered_moves:
        # Futility pruning - skip moves that are likely futile
        if depth <= 2 and not board.is_check() and moves_searched > 0:
            if is_futile_move(board, move, margin=100 * depth):
                continue
        
        board.push(move)
        
        # Late move reduction
        # If we've searched several moves and this is a quiet move at a good depth
        if (
            depth >= 3 and 
            moves_searched >= 3 and 
            not board.is_check() and 
            not board.is_capture(move) and 
            move.promotion is None
        ):
            # Search with reduced depth
            score = -alpha_beta(
                board, depth - 2, -alpha - 1, -alpha, info, ply + 1
            )
            
            # If the score is good but not too good, do a full search
            if score > alpha and score < beta:
                score = -alpha_beta(
                    board, depth - 1, -beta, -alpha, info, ply + 1
                )
        else:
            # Principal variation search
            if moves_searched > 0:
                # Search with a null window to see if this move is better than our best so far
                score = -alpha_beta(
                    board, depth - 1, -alpha - 1, -alpha, info, ply + 1
                )
                
                # If the score is good but not too good, do a full search
                if score > alpha and score < beta:
                    score = -alpha_beta(
                        board, depth - 1, -beta, -alpha, info, ply + 1
                    )
            else:
                # Full search for the first move
                score = -alpha_beta(
                    board, depth - 1, -beta, -alpha, info, ply + 1
                )
        
        board.pop()
        
        # Increment moves searched
        moves_searched += 1
        
        # Check if search should stop
        if info.should_stop():
            return 0
        
        # Update best score and move
        if score > best_score:
            best_score = score
            best_move = move
            
            # Update PV line
            pv_line = [move] + info.pv_line
            
            # If this is the root node, update the best move
            if ply == 0:
                info.update_best_move(move, score, depth, pv_line)
            
        # Alpha-beta pruning
        if score > alpha:
            alpha = score
            
        # Beta cutoff
        if score >= beta:
            # If it's a quiet move, update killer moves
            if not board.is_capture(move):
                info.update_killer_move(move, ply)
                info.update_history_score(move, depth)
                
            # Store position in transposition table
            info.hash_position(board, depth, alpha_orig, beta, beta, move, flag=BETA)
            return beta
    
    # Store position in transposition table
    if best_score <= alpha_orig:
        info.hash_position(board, depth, alpha_orig, beta, alpha_orig, best_move, flag=ALPHA)
    else:
        info.hash_position(board, depth, alpha_orig, beta, best_score, best_move, flag=EXACT)
    
    return best_score

def has_non_pawn_material(board, color):
    """Check if a side has any non-pawn material."""
    for piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
        if len(board.pieces(piece_type, color)) > 0:
            return True
    return False

def quiescence_search(board, alpha, beta, info, ply=0, depth_left=QUIESCENCE_DEPTH):
    """
    Quiescence search to resolve tactical sequences.
    
    Args:
        board: Chess board position
        alpha: Alpha bound
        beta: Beta bound
        info: SearchInfo object
        ply: Current ply from root
        depth_left: Maximum depth left for quiescence search
        
    Returns:
        Score for the position
    """
    # Safety check for maximum recursion depth
    if ply >= MAX_PLY or depth_left <= 0:
        return evaluate_position(board)
        
    # Check if we should stop the search
    if info.should_stop():
        return 0
        
    # Update node count (thread-safe)
    info.increment_nodes()
    
    # Stand pat - evaluate the current position
    stand_pat = evaluate_position(board)
    
    # Beta cutoff
    if stand_pat >= beta:
        return beta
        
    # Update alpha if stand pat is better
    if stand_pat > alpha:
        alpha = stand_pat
    
    # Get only captures and promotions
    captures = [move for move in board.legal_moves if board.is_capture(move) or move.promotion]
    
    # Order captures by MVV-LVA
    ordered_captures = pick_move(board, captures, info)
    
    # Try each capture
    for move in ordered_captures:
        board.push(move)
        
        # Recursive quiescence search
        score = -quiescence_search(board, -beta, -alpha, info, ply + 1, depth_left - 1)
        
        board.pop()
        
        # Check if search should stop
        if info.should_stop():
            return 0
        
        # Update alpha if this move is better
        if score > alpha:
            alpha = score
            
        # Beta cutoff
        if score >= beta:
            return beta
    
    return alpha

def iterative_deepening_search(board, info):
    """
    Perform iterative deepening search with parallelization at the root.
    
    Args:
        board: Chess board position
        info: SearchInfo object containing search parameters
        
    Returns:
        Tuple of (best_move, score, depth)
    """
    info.start_time = time.time()
    info.nodes_searched = 0
    info.stop_search = False
    
    # Increment the transposition table age
    info.tt_age += 1
    
    # Calculate maximum time for search (with a buffer)
    max_time_ms = info.time_limit_ms
    if max_time_ms > 0:
        # Use 95% of the time limit to ensure we don't exceed it
        max_time_ms = int(max_time_ms * 0.95)
    
    # Start with depth 1 and increase
    for depth in range(1, MAX_DEPTH + 1):
        info.depth = depth
        
        # Reset PV line for this iteration
        info.pv_line = []
        
        # Calculate time spent so far
        elapsed_ms = info.get_elapsed_time()
        
        # Check if we have enough time for the next iteration
        if max_time_ms > 0:
            # Estimate time for next iteration based on current depth
            # Each depth typically takes 3-5x longer than the previous
            estimated_next_ms = elapsed_ms * 4
            
            # If we don't have enough time, stop the search
            if elapsed_ms + estimated_next_ms > max_time_ms:
                break
        
        # For depth 1-3, use standard alpha-beta (more reliable)
        if depth <= 3:
            score = alpha_beta(board, depth, -INFINITY, INFINITY, info)
        else:
            # Use simplified parallel search for depths > 3
            score = simple_parallel_search(board, depth, info)
        
        # Check if search should stop
        if info.should_stop():
            break
        
        # Print info about this iteration
        elapsed = info.get_elapsed_time()
        nodes = info.nodes_searched
        nps = int(nodes * 1000 / max(elapsed, 1))  # Nodes per second
        
        print(f"info depth {depth} score cp {info.best_score} time {elapsed} nodes {nodes} nps {nps} pv {' '.join(str(move) for move in info.pv_line)}")
    
    return info.best_move, info.best_score, info.depth

def simple_parallel_search(board, depth, info):
    """
    Simple parallel search at the root level.
    
    Args:
        board: Chess board position
        depth: Search depth
        info: SearchInfo object
        
    Returns:
        Best score found
    """
    # Get legal moves
    legal_moves = list(board.legal_moves)
    
    # If there's only one legal move, return it immediately
    if len(legal_moves) == 1:
        info.best_move = legal_moves[0]
        info.best_score = 0  # Neutral score
        return 0
    
    # Order moves to improve pruning
    hash_hit, hash_score, hash_move = info.probe_hash(board, depth - 1, -INFINITY, INFINITY)
    ordered_moves = pick_move(board, legal_moves, info, hash_move)
    
    # Determine number of threads to use (up to number of CPUs)
    num_threads = min(len(ordered_moves), multiprocessing.cpu_count())
    
    # Use ThreadPoolExecutor for parallel search
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit search tasks for each move
        futures = []
        for move in ordered_moves:
            futures.append(executor.submit(
                search_move, board.copy(), move, depth, info
            ))
        
        # Wait for all tasks to complete
        concurrent.futures.wait(futures)
    
    # Return the best score (already updated in info)
    return info.best_score

def search_move(board, move, depth, info):
    """
    Search a single move (used for parallel search).
    
    Args:
        board: Chess board position
        move: Move to search
        depth: Search depth
        info: SearchInfo object
        
    Returns:
        Score for the move
    """
    # Make the move
    board.push(move)
    
    # Search with negamax
    score = -alpha_beta(board, depth - 1, -INFINITY, -info.best_score, info, 1)
    
    # Check if search should stop
    if info.should_stop():
        return 0
    
    # Update best move if this is better
    if score > info.best_score:
        pv_line = [move] + info.pv_line
        info.update_best_move(move, score, depth, pv_line)
    
    return score

def get_best_move(board, time_limit_ms=1000, depth=None):
    """
    Get the best move for a position.
    
    Args:
        board: Chess board position
        time_limit_ms: Time limit in milliseconds (0 for no limit)
        depth: Maximum depth to search (None for iterative deepening)
        
    Returns:
        Tuple of (best_move, score, depth)
    """
    # Create search info
    info = SearchInfo()
    info.time_limit_ms = time_limit_ms
    
    # If depth is specified, search to that depth
    if depth is not None:
        info.depth = depth
        score = alpha_beta(board, depth, -INFINITY, INFINITY, info)
        return info.best_move, score, depth
    
    # Otherwise, use iterative deepening
    return iterative_deepening_search(board, info)
