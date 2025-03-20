import time
import chess
from .evaluation.evaluator import Evaluator
from .search.searcher import Searcher
from .util.transposition_table import TranspositionTable

class Engine:
    """
    Main chess engine class that coordinates evaluation and search.
    
    This engine uses classical heuristics for position evaluation and optimized search 
    techniques to find the best move in a given position.
    """
    def __init__(self, transposition_table_size_mb=64):
        """
        Initialize the chess engine.
        
        Args:
            transposition_table_size_mb: Size of the transposition table in MB
        """
        # create our core components
        self.evaluator = Evaluator()
        self.tt = TranspositionTable(size_mb=transposition_table_size_mb)
        self.searcher = Searcher(self.evaluator, self.tt)
        
        # some stats
        self.positions_evaluated = 0
        self.time_spent_evaluating = 0
        self.nodes_searched = 0
        self.time_spent_searching = 0
    
    def search(self, board, depth=4, time_limit_ms=None):
        """
        Search for the best move in the given position.
        
        Args:
            board: A chess.Board object representing the current position
            depth: Maximum search depth
            time_limit_ms: Optional time limit in milliseconds
            
        Returns:
            A tuple (best_move, score, info) where:
                - best_move is a chess.Move object
                - score is the evaluation in centipawns (positive for white advantage)
                - info is a dict with additional information about the search
        """
        return self.searcher.search(board, depth, time_limit_ms)
    
    def get_stats(self):
        """
        Get statistics about engine performance.
        
        Returns:
            A dictionary with performance statistics
        """
        stats = {
            'positions_evaluated': self.positions_evaluated,
            'nodes_searched': self.nodes_searched,
            'nps': int(self.nodes_searched / self.time_spent_searching) if self.time_spent_searching > 0 else 0,
            'eval_speed': int(self.positions_evaluated / self.time_spent_evaluating * 1000) if self.time_spent_evaluating > 0 else 0,
            'tt_hit_rate': self.tt.get_hit_rate(),
            'tt_size': self.tt.get_size_info()
        }
        
        return stats
    
    def reset_stats(self):
        """Reset all performance statistics."""
        self.positions_evaluated = 0
        self.time_spent_evaluating = 0
        self.nodes_searched = 0
        self.time_spent_searching = 0
        self.tt.reset_stats() 