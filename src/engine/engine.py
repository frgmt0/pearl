import chess
import time
import random
import os
from datetime import datetime

from src.engine.search import get_best_move, iterative_deepening_search, SearchInfo
from src.engine.score import evaluate_position, initialize_nnue, save_model, load_model, create_model
from src.engine.finetune import RealtimeFinetuner
from src.engine.utils import evaluate_move_quality
from src.engine.validator import validate_move, debug_board_state, validate_board_state

class NNUEEngine:
    """
    Main chess engine class that integrates NNUE evaluation with
    alpha-beta search to find the best moves.
    """
    def __init__(self, name="Pearl NNUE", depth=7, time_limit_ms=3000, enable_learning=True):
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
        
        # Keep track of game statistics
        if not hasattr(self, 'game_stats'):
            self.game_stats = {
                'games_played': 0,
                'wins': 0,
                'losses': 0,
                'draws': 0,
                'win_rate': 0.0
            }
        
        # Try to load the default model
        model_path = "saved_models/default_weights.pt"
        if os.path.exists(model_path):
            try:
                self.model = load_model(model_path)
                print(f"Loaded model from {model_path}")
                
                # Also update finetuner model if available
                if self.enable_learning and self.finetuner:
                    self.finetuner.model = self.model
            except Exception as e:
                print(f"\033[1;31mError loading model: {e}\033[0m")
                
                # Create a new model if loading fails
                self.model = create_model(self.model_type)
                
                # Update finetuner with the new model
                if self.enable_learning and self.finetuner:
                    self.finetuner.model = self.model
        else:
            # Create a new model if no matching model found
            self.model = create_model(self.model_type)
            
            # Update finetuner with the new model
            if self.enable_learning and self.finetuner:
                self.finetuner.model = self.model
        
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
        Get the best move for the current position using parallel search.
        
        Args:
            time_limit_ms: Time limit in milliseconds (None for default of 60000 = 1 minute)
            depth: Search depth (None for default)
            
        Returns:
            Best move found
        """
        print(f"\033[1;36m🔍 Starting move search (max time: {time_limit_ms or 60000}ms)\033[0m")
        
        # Always use a safe time limit with a hard cap
        time_limit = 30000 if time_limit_ms is None else min(time_limit_ms, 30000)  # Cap at 30 seconds
        search_depth = depth if depth is not None else self.depth
        
        # Count legal moves for debugging
        legal_move_count = len(list(self.board.legal_moves))
        print(f"\033[1;33m🧠 Analyzing position with {legal_move_count} legal moves\033[0m")
        
        # Initialize search info
        info = SearchInfo()
        info.time_limit_ms = time_limit
        
        # Record start time
        start_time = time.time()
        
        try:
            # Perform search with safety checks
            print("\033[1;36m⏱️ Starting iterative deepening search...\033[0m")
            best_move, score, reached_depth = iterative_deepening_search(self.board, info)
            print(f"\033[1;32m✓ Search completed at depth {reached_depth}\033[0m")
        except Exception as e:
            # If search fails, log error and pick a random move
            print(f"\033[1;31m❌ Search error: {e}\033[0m")
            legal_moves = list(self.board.legal_moves)
            best_move = random.choice(legal_moves) if legal_moves else None
            score = 0
            reached_depth = 0
        
        # Record end time and nodes
        end_time = time.time()
        elapsed_ms = int((end_time - start_time) * 1000)
        self.move_times.append(elapsed_ms)
        self.total_nodes += info.nodes_searched
        
        # Print search statistics
        nps = int(info.nodes_searched * 1000 / max(elapsed_ms, 1))
        print(f"\033[1;32mbestmove {best_move} score cp {score} depth {reached_depth} nodes {info.nodes_searched} time {elapsed_ms}ms nps {nps}\033[0m")
        
        # Validate that the move is legal in the current position
        if best_move and best_move not in self.board.legal_moves:
            print(f"\033[1;31m⚠️ Warning: Engine suggested illegal move {best_move}. Using fallback...\033[0m")
            # Get a list of legal moves
            legal_moves = list(self.board.legal_moves)
            if legal_moves:
                # Choose a random legal move as fallback
                best_move = random.choice(legal_moves)
                print(f"\033[1;33m↪️ Selected alternative move: {best_move}\033[0m")
            else:
                best_move = None
                print("\033[1;31m❌ No legal moves available!\033[0m")
        
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
        Save the current neural network model weights.
        
        Args:
            name: Optional name for the weights file
            
        Returns:
            Path to the saved weights file
        """
        if self.enable_learning and self.finetuner:
            # Save through finetuner, which might apply additional logic
            if hasattr(self.finetuner, 'save_model'):
                return self.finetuner.save_model(name)
            
        # Use our new save_model function
        return save_model(self.model, name)
    
    def load_model(self, path=None):
        """
        Load neural network model weights.
        
        Args:
            path: Path to weights file (None for default)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # If path not specified, use default
            if path is None:
                path = os.path.join("saved_models", "default_weights.pt")
            
            # Store the path for reference
            self.current_weights_path = path
            
            # Load model
            self.model = load_model(path)
            
            # Update fine-tuner model if available
            if self.enable_learning and self.finetuner:
                self.finetuner.model = self.model
            
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
            
    def get_current_weights_path(self):
        """
        Get the path of the currently loaded weights file.
        
        Returns:
            Path to the current weights file, or 'Unknown' if not recorded
        """
        return getattr(self, 'current_weights_path', 'Unknown')
    
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
