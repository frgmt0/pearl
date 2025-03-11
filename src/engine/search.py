import chess
import time
import random
from collections import defaultdict, OrderedDict
import numpy as np
import threading
import concurrent.futures
import multiprocessing

from src.engine.score import evaluate_position
from src.engine.movePick import pick_move

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
        self.transposition_table = {}
        
        # Thread synchronization locks
        self.nodes_lock = threading.Lock()
        self.best_move_lock = threading.Lock()
        self.transposition_table_lock = threading.Lock()
        self.history_lock = threading.Lock()
        
    def should_stop(self):
        """Check if search should be stopped due to time constraints."""
        # Always stop if manually requested
        if self.stop_search:
            return True
            
        # Check elapsed time against time limit
        elapsed = self.get_elapsed_time()
        
        # Hard safety cap: Stop after 60 seconds no matter what
        ABSOLUTE_MAX_TIME = 60000  # 60 seconds in ms
        if elapsed >= ABSOLUTE_MAX_TIME:
            print(f"⚠️ SAFETY: Stopping search after reaching absolute max time ({ABSOLUTE_MAX_TIME/1000}s)")
            return True
            
        # Normal time limit check
        if self.time_limit_ms > 0:
            return elapsed >= self.time_limit_ms
            
        return False
        
    def get_elapsed_time(self):
        """Get elapsed time in milliseconds."""
        return int((time.time() - self.start_time) * 1000)
        
    def update_best_move(self, move, score, depth, pv):
        """Update the best move found so far (thread-safe)."""
        with self.best_move_lock:
            self.best_move = move
            self.best_score = score
            self.depth = depth
            self.pv_line = pv.copy() if pv else []
        
    def increment_nodes(self):
        """Increment node counter in a thread-safe way."""
        with self.nodes_lock:
            self.nodes_searched += 1
        
    def update_killer_move(self, move, ply):
        """
        Update killer moves for the given ply.
        Killer moves are quiet moves that cause beta cutoffs.
        """
        # Safety check for ply index to prevent out-of-range errors
        if ply >= len(self.killer_moves):
            return
            
        # Killer moves are ply-specific, so no need for locks
        if self.killer_moves[ply][0] != move:
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
        with self.transposition_table_lock:
            self.transposition_table[key] = {
                'depth': depth,
                'score': score,
                'flag': flag,
                'move': move
            }
        
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
        with self.transposition_table_lock:
            entry = self.transposition_table.get(key)
        
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
    
    # Null Move Pruning
    # If we're not in check and have sufficient material, try a null move
    if (null_move_allowed and depth >= 3 and not board.is_check() and
        has_non_pawn_material(board, board.turn)):
        
        # Make a "null move" (pass the turn)
        board.push(chess.Move.null())
        
        # Search with reduced depth
        null_reduction = 3  # R value in standard null-move pruning
        null_score = -alpha_beta(board, depth - 1 - null_reduction, -beta, -beta + 1, 
                                info, ply + 1, False)
        
        # Unmake the null move
        board.pop()
        
        # If the score exceeds beta, prune this subtree
        if info.should_stop():
            return 0
            
        if null_score >= beta:
            return beta
    
    # Generate moves
    moves = list(board.legal_moves)
    
    # If no legal moves, return score
    if not moves:
        if board.is_check():
            # Checkmate
            return -MATE_SCORE + ply
        else:
            # Stalemate
            return DRAW_SCORE
    
    # Move ordering using history and killer heuristics
    ordered_moves = pick_move(board, moves, info, hash_move, ply)
    
    best_move = None
    best_score = -INFINITY
    alpha_orig = alpha
    
    # Try each move
    for move in ordered_moves:
        board.push(move)
        
        # Search with this move
        score = -alpha_beta(board, depth - 1, -beta, -alpha, info, ply + 1)
        
        board.pop()
        
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

def quiescence_search(board, alpha, beta, info, ply, depth_left):
    """
    Quiescence search to handle tactical positions.
    Only considers captures to reach a "quiet" position.
    
    Args:
        board: Chess board position
        alpha: Alpha bound
        beta: Beta bound
        info: SearchInfo object
        ply: Current ply from root
        depth_left: Maximum quiescence depth left
        
    Returns:
        Score for the position
    """
    # Safety check for maximum recursion depth
    if ply >= MAX_PLY:
        return evaluate_position(board)
    
    # Update node count (thread-safe)
    info.increment_nodes()
    
    # Check if we should stop
    if info.should_stop():
        return 0
    
    # Stand-pat score (static evaluation)
    stand_pat = evaluate_position(board)
    
    # Beta cutoff
    if stand_pat >= beta:
        return beta
    
    # Update alpha
    if stand_pat > alpha:
        alpha = stand_pat
    
    # If we've reached the maximum quiescence depth, return the stand-pat score
    if depth_left <= 0:
        return stand_pat
    
    # Generate capture moves
    captures = [move for move in board.legal_moves if board.is_capture(move)]
    
    # If no captures, return stand-pat score
    if not captures:
        return stand_pat
    
    # Sort captures by MVV-LVA (Most Valuable Victim - Least Valuable Aggressor)
    captures = sort_mvv_lva(board, captures)
    
    # Try each capture
    for move in captures:
        board.push(move)
        
        # Only consider positions where we're not in check
        # This helps filter out some bad captures
        if not board.is_check():
            score = -quiescence_search(board, -beta, -alpha, info, ply + 1, depth_left - 1)
        else:
            score = -alpha_beta(board, 1, -beta, -alpha, info, ply + 1)
        
        board.pop()
        
        # Check if search should stop
        if info.should_stop():
            return 0
        
        # Update alpha
        if score > alpha:
            alpha = score
        
        # Beta cutoff
        if score >= beta:
            return beta
    
    return alpha

def sort_mvv_lva(board, moves):
    """
    Sort moves by Most Valuable Victim - Least Valuable Aggressor.
    
    Args:
        board: Chess board
        moves: List of moves to sort
        
    Returns:
        Sorted list of moves
    """
    move_scores = []
    
    # Piece values (pawn=1, knight=3, bishop=3, rook=5, queen=9)
    piece_values = {
        chess.PAWN: 1, 
        chess.KNIGHT: 3, 
        chess.BISHOP: 3, 
        chess.ROOK: 5, 
        chess.QUEEN: 9, 
        chess.KING: 20
    }
    
    for move in moves:
        # Get the captured piece value (victim)
        victim_square = move.to_square
        victim = board.piece_at(victim_square)
        victim_value = piece_values.get(victim.piece_type, 0) if victim else 0
        
        # Get the moving piece value (aggressor)
        aggressor_square = move.from_square
        aggressor = board.piece_at(aggressor_square)
        aggressor_value = piece_values.get(aggressor.piece_type, 0) if aggressor else 0
        
        # MVV-LVA score: 10 * victim value - aggressor value
        # This prioritizes capturing high-value pieces with low-value pieces
        score = 10 * victim_value - aggressor_value
        
        move_scores.append((move, score))
    
    # Sort by score (descending)
    move_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Return sorted moves
    return [move for move, _ in move_scores]

def has_non_pawn_material(board, color):
    """
    Check if a side has non-pawn material (for null move pruning).
    
    Args:
        board: Chess board
        color: Color to check (True for white, False for black)
        
    Returns:
        True if the side has non-pawn material (knight, bishop, rook, queen)
    """
    for piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
        if board.pieces(piece_type, color):
            return True
    return False

def simple_parallel_search(board, depth, info):
    """
    Simplified parallel search at the root level with safety limits.
    
    Args:
        board: Chess board position
        depth: Current search depth
        info: SearchInfo object
        
    Returns:
        Best score found
    """
    # Generate legal moves at root
    legal_moves = list(board.legal_moves)
    
    # If no legal moves, return appropriate score
    if not legal_moves:
        if board.is_check():
            return -MATE_SCORE  # Checkmate
        else:
            return DRAW_SCORE   # Stalemate
    
    # Use move ordering to prioritize promising moves
    ordered_moves = pick_move(board, legal_moves, info, None, 0)
    
    # Initialize search variables
    best_score = -INFINITY
    best_move = None
    alpha = -INFINITY
    beta = INFINITY
    
    # Safety: Maximum time per move at any depth (3 seconds for depth 3, 6 for depth 4, etc.)
    # This prevents any single move evaluation from hanging
    MAX_MOVE_TIME_MS = 2000 * depth 
    
    # Log the start of parallel search
    print(f"Starting simplified parallel search at depth {depth} with {len(ordered_moves)} moves")
    
    # Get number of CPU cores (limit to 4 max to avoid overhead)
    num_workers = min(4, max(1, multiprocessing.cpu_count() - 1))

    # Create a thread pool with a timeout
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Store futures in a dictionary
        future_results = {}
        
        # Launch a search for each move
        for move in ordered_moves:
            # Skip if we should stop
            if info.should_stop():
                break
                
            # Create a board copy for this move
            board_copy = board.copy()
            board_copy.push(move)
            
            # Submit the search task with a timeout
            def search_task():
                try:
                    # Use a negated search for opponent's perspective
                    return -alpha_beta(board_copy, depth - 1, -beta, -alpha, info, 1)
                except Exception as e:
                    print(f"Search error for move {move}: {e}")
                    return -INFINITY
            
            # Submit the task
            future = executor.submit(search_task)
            future_results[move] = future
        
        # Process completed searches
        completed_moves = 0
        for move, future in future_results.items():
            # Skip if we should stop
            if info.should_stop():
                break
                
            try:
                # Wait for the result with a timeout
                score = future.result(timeout=MAX_MOVE_TIME_MS/1000)
                completed_moves += 1
                
                # Update best move if better
                if score > best_score:
                    best_score = score
                    best_move = move
                    
                    # Create a simple PV line
                    pv_line = [move]
                    
                    # Update the best move
                    info.update_best_move(move, score, depth, pv_line)
                    
                    # Update alpha for pruning
                    if score > alpha:
                        alpha = score
                
                # Log progress
                if completed_moves % 5 == 0 or completed_moves == len(ordered_moves):
                    print(f"Completed {completed_moves}/{len(ordered_moves)} moves at depth {depth}")
                    
            except concurrent.futures.TimeoutError:
                # Log timeout and skip this move
                print(f"Move {move} timed out after {MAX_MOVE_TIME_MS/1000}s at depth {depth}")
                continue
            except Exception as e:
                # Log any other errors
                print(f"Error evaluating move {move}: {e}")
                continue
    
    # Log completion
    print(f"Parallel search completed at depth {depth}. Best move: {best_move}, score: {best_score}")
    
    return best_score

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
    
    # Start with depth 1 and increase
    for depth in range(1, MAX_DEPTH + 1):
        info.depth = depth
        
        # Reset PV line for this iteration
        info.pv_line = []
        
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

def get_best_move(board, time_limit_ms=60000):
    """
    Get the best move for a position using iterative deepening with parallel search.
    
    Args:
        board: Chess board position
        time_limit_ms: Time limit in milliseconds (default: 60000 = 1 minute)
        
    Returns:
        Best move found within the time limit
    """
    # Initialize search info
    info = SearchInfo()
    info.time_limit_ms = time_limit_ms  # Hard cap at 1 minute
    
    # Perform iterative deepening search
    best_move, score, depth = iterative_deepening_search(board, info)
    
    return best_move
