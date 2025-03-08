import chess
import time
import random
from datetime import datetime

from src.engine.search import get_best_move, iterative_deepening_search, SearchInfo
from src.engine.score import evaluate_position, initialize_nnue
from src.engine.nnue.network import NNUE
from src.engine.nnue.weights import save_weights, load_weights, get_latest_weights
from src.engine.finetune import RealtimeFinetuner, initialize_default_weights
from src.engine.utils import evaluate_move_quality
from src.engine.validator import validate_move, debug_board_state, validate_board_state

class NNUEEngine:
    """
    Main chess engine class that integrates NNUE evaluation with
    alpha-beta search to find the best moves.
    """
    def __init__(self, name="Pearl NNUE", depth=5, time_limit_ms=1000, enable_learning=True):
        """
        Initialize a new chess engine.
        
        Args:
            name: Engine name
            depth: Default search depth
            time_limit_ms: Default time limit in milliseconds
            enable_learning: Whether to enable real-time learning
        """
        self.name = name
        self.depth = depth
        self.time_limit_ms = time_limit_ms
        self.enable_learning = enable_learning
        
        # Initialize default weights if they don't exist
        initialize_default_weights()
        
        # Initialize NNUE model
        self.model = initialize_nnue()
        
        # Initialize fine-tuner if learning is enabled
        self.finetuner = RealtimeFinetuner(model=self.model) if enable_learning else None
        
        # Initialize board and game state
        self.board = chess.Board()
        self.game_history = []
        self.move_times = []
        self.total_nodes = 0
        self.game_result = None
        
        # Track previous positions and evaluations for learning
        self.prev_position = None
        self.prev_eval = None
        self.move_quality_history = []
        
    def reset(self):
        """Reset the engine to a new game."""
        self.board = chess.Board()
        self.game_history = []
        self.move_times = []
        self.total_nodes = 0
        self.game_result = None
        self.prev_position = None
        self.prev_eval = None
        self.move_quality_history = []
        
        # Reset to default weights if learning is enabled
        if self.enable_learning and self.finetuner:
            self.finetuner.reset_to_default()
        
    def set_position(self, fen=None):
        """
        Set the current position.
        
        Args:
            fen: FEN string for the position (None for starting position)
        """
        if fen:
            self.board = chess.Board(fen)
        else:
            self.board = chess.Board()
        
        # Reset previous position and evaluation
        self.prev_position = None
        self.prev_eval = None
        
    def make_move(self, move):
        """
        Make a move on the internal board.
        
        Args:
            move: Chess move (string or Move object)
            
        Returns:
            True if move was made, False if illegal
        """
        if isinstance(move, str):
            try:
                move = chess.Move.from_uci(move)
            except ValueError:
                return False
                
        # Verify the move is legal in the current position
        if move not in self.board.legal_moves:
            print(f"Illegal move attempted: {move}")
            return False
            
        # Store current position and evaluation for learning
        if self.enable_learning:
            self.prev_position = chess.Board(self.board.fen())
            self.prev_eval = evaluate_position(self.board)
        
        # Record position before the move
        self.game_history.append(self.board.fen())
        
        # Make the move
        try:
            self.board.push(move)
        except Exception as e:
            print(f"Error making move: {e}")
            return False
        
        # Evaluate move quality for learning if we have previous state
        if self.enable_learning and self.prev_position and self.prev_eval is not None:
            # Get evaluation after the move
            post_eval = -evaluate_position(self.board)  # Negate because now it's from opponent's perspective
            
            # Evaluate move quality
            quality_value, quality_label = evaluate_move_quality(
                self.prev_eval, post_eval, self.prev_position.turn
            )
            
            # Store move quality
            self.move_quality_history.append((move, quality_value, quality_label))
            
            # Adjust model weights based on move quality if fine-tuner is available
            if self.finetuner:
                self.finetuner.adjust_for_move(self.prev_position, move, quality_value)
        
        # Check if game is over
        if self.board.is_game_over():
            result = self.board.result()
            if result == "1-0":
                self.game_result = "white_win"
            elif result == "0-1":
                self.game_result = "black_win"
            else:
                self.game_result = "draw"
        
        return True
    
    def get_best_move(self, time_limit_ms=None, depth=None):
        """
        Get the best move for the current position.
        
        Args:
            time_limit_ms: Time limit in milliseconds (None for default)
            depth: Search depth (None for default)
            
        Returns:
            Best move found
        """
        # Use provided values or defaults
        time_limit = time_limit_ms if time_limit_ms is not None else self.time_limit_ms
        search_depth = depth if depth is not None else self.depth
        
        # Initialize search info
        info = SearchInfo()
        info.time_limit_ms = time_limit
        
        # Record start time
        start_time = time.time()
        
        # Perform search
        best_move, score, reached_depth = iterative_deepening_search(self.board, info)
        
        # Record end time and nodes
        end_time = time.time()
        elapsed_ms = int((end_time - start_time) * 1000)
        self.move_times.append(elapsed_ms)
        self.total_nodes += info.nodes_searched
        
        # Print search statistics
        nps = int(info.nodes_searched * 1000 / max(elapsed_ms, 1))
        print(f"bestmove {best_move} score cp {score} depth {reached_depth} nodes {info.nodes_searched} time {elapsed_ms} nps {nps}")
        
        # Validate that the move is legal in the current position
        if best_move and best_move not in self.board.legal_moves:
            print(f"Warning: Engine suggested illegal move {best_move}. Recalculating...")
            # Get a list of legal moves
            legal_moves = list(self.board.legal_moves)
            if legal_moves:
                # Choose a random legal move as fallback
                best_move = random.choice(legal_moves)
                print(f"Selected alternative move: {best_move}")
            else:
                best_move = None
        
        return best_move
    
    def play_move(self, time_limit_ms=None, depth=None):
        """
        Calculate and play the best move on the internal board.
        
        Args:
            time_limit_ms: Time limit in milliseconds (None for default)
            depth: Search depth (None for default)
            
        Returns:
            Best move found and played
        """
        # Get the current FEN before making a move (for debugging)
        current_fen = self.board.fen()
        
        # Get the best move
        best_move = self.get_best_move(time_limit_ms, depth)
        
        if best_move:
            # Double-check that the move is legal
            if not validate_move(self.board, best_move):
                print(f"Error: Move {best_move} is not legal in position {current_fen}")
                # Get a list of legal moves
                legal_moves = list(self.board.legal_moves)
                if legal_moves:
                    # Choose a random legal move as fallback
                    best_move = random.choice(legal_moves)
                    print(f"Selected alternative move: {best_move}")
                else:
                    return None
            
            # Make the move
            self.board.push(best_move)
            
            # Store the new position for learning
            if self.enable_learning and self.prev_position and self.prev_eval is not None:
                # Get evaluation after the move
                post_eval = -evaluate_position(self.board)  # Negate because now it's from opponent's perspective
                
                # Evaluate move quality
                quality_value, quality_label = evaluate_move_quality(
                    self.prev_eval, post_eval, self.prev_position.turn
                )
                
                # Store move quality
                self.move_quality_history.append((best_move, quality_value, quality_label))
                
                # Adjust model weights based on move quality if fine-tuner is available
                if self.finetuner:
                    self.finetuner.adjust_for_move(self.prev_position, best_move, quality_value)
            
            # Check if game is over
            if self.board.is_game_over():
                result = self.board.result()
                if result == "1-0":
                    self.game_result = "white_win"
                elif result == "0-1":
                    self.game_result = "black_win"
                else:
                    self.game_result = "draw"
            
            # Debug info
            debug_info = debug_board_state(self.board)
            print(f"Board updated to: {self.board.fen()}")
            print(f"New board state: Turn={debug_info['turn']}, Legal moves={debug_info['legal_moves_count']}")
            
        return best_move
    
    def evaluate(self):
        """
        Evaluate the current position.
        
        Returns:
            Evaluation score in centipawns from the current player's perspective
        """
        return evaluate_position(self.board)
    
    def get_game_statistics(self):
        """
        Get statistics about the current game.
        
        Returns:
            Dictionary with game statistics
        """
        avg_time = sum(self.move_times) / max(len(self.move_times), 1)
        total_time = sum(self.move_times)
        
        # Count move quality statistics
        blunders = sum(1 for _, quality, _ in self.move_quality_history if quality == -3)
        mistakes = sum(1 for _, quality, _ in self.move_quality_history if quality == -2)
        inaccuracies = sum(1 for _, quality, _ in self.move_quality_history if quality == -1)
        good_moves = sum(1 for _, quality, _ in self.move_quality_history if quality > 0)
        
        return {
            "moves_played": len(self.board.move_stack),
            "average_time_ms": int(avg_time),
            "total_time_ms": total_time,
            "total_nodes": self.total_nodes,
            "average_nps": int(self.total_nodes * 1000 / max(total_time, 1)),
            "game_result": self.game_result,
            "blunders": blunders,
            "mistakes": mistakes,
            "inaccuracies": inaccuracies,
            "good_moves": good_moves
        }
    
    def save_model(self, name=None):
        """
        Save the current NNUE model weights.
        
        Args:
            name: Optional name for the weights file
            
        Returns:
            Path to the saved weights file
        """
        if self.enable_learning and self.finetuner:
            return self.finetuner.save_model(name)
        else:
            return save_weights(self.model, name)
    
    def load_model(self, path=None):
        """
        Load NNUE model weights.
        
        Args:
            path: Path to weights file (None for latest)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if path is None:
                path = get_latest_weights()
                if path is None:
                    print("No weights found")
                    return False
            
            # Load weights into model
            self.model = load_weights(self.model, path)
            
            # Update fine-tuner model if available
            if self.enable_learning and self.finetuner:
                self.finetuner.model = self.model
            
            return True
        except Exception as e:
            print(f"Error loading weights: {e}")
            return False
    
    def toggle_learning(self, enable=None):
        """
        Toggle real-time learning on/off.
        
        Args:
            enable: True to enable, False to disable, None to toggle
            
        Returns:
            New learning state
        """
        if enable is None:
            self.enable_learning = not self.enable_learning
        else:
            self.enable_learning = enable
        
        # Initialize fine-tuner if needed
        if self.enable_learning and self.finetuner is None:
            self.finetuner = RealtimeFinetuner(model=self.model)
        
        return self.enable_learning
    
    def verify_board_state(self, reference_fen=None):
        """
        Verify that the engine's internal board state is valid and consistent.
        If a reference FEN is provided, ensure the engine's board matches it.
        
        Args:
            reference_fen: Optional reference FEN to compare against
            
        Returns:
            True if the board state is valid, False otherwise
        """
        # Validate the current board state
        validation = validate_board_state(self.board)
        
        if not validation['is_valid']:
            print("Engine's internal board state is invalid:")
            for error in validation['errors']:
                print(f"  - {error}")
            
            # If we have a reference FEN, reset to that
            if reference_fen:
                print(f"Resetting engine board to reference position")
                self.set_position(reference_fen)
                return False
            
            return False
        
        # If a reference FEN is provided, compare against it
        if reference_fen:
            reference_board = chess.Board(reference_fen)
            
            if self.board.fen() != reference_board.fen():
                print("Engine's board state differs from reference position")
                
                # Get detailed differences
                debug_current = debug_board_state(self.board)
                debug_reference = debug_board_state(reference_board)
                
                print(f"Current board: {debug_current['fen']}")
                print(f"Reference board: {debug_reference['fen']}")
                print(f"Current turn: {debug_current['turn']}, Reference turn: {debug_reference['turn']}")
                
                # Reset to the reference position
                print("Resetting engine board to reference position")
                self.set_position(reference_fen)
                return False
        
        return True
