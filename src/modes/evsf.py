import chess
import time
import sys
import os

from src.game.board import ChessBoard
from src.engine.engine import NNUEEngine
from src.utils.api.stockfish import StockfishAPI, MockStockfishAPI
from src.engine.utils import format_time, evaluate_move_quality

class EngineVsStockfish:
    """
    Engine vs Stockfish game mode.
    
    This mode makes the NNUE engine play against the Stockfish API.
    """
    def __init__(self, engine=None, engine_depth=5, engine_time_ms=1000, 
                 stockfish_depth=10, engine_color=chess.WHITE):
        """
        Initialize the Engine vs Stockfish game.
        
        Args:
            engine: NNUE engine instance (None to create a new one)
            engine_depth: Search depth for the NNUE engine
            engine_time_ms: Time limit for NNUE engine moves in milliseconds
            stockfish_depth: Search depth for Stockfish API
            engine_color: NNUE engine's color (WHITE/BLACK)
        """
        self.board = ChessBoard()
        self.engine = engine or NNUEEngine()
        self.engine_depth = engine_depth
        self.engine_time_ms = engine_time_ms
        self.engine_color = engine_color
        
        # Check if learning is enabled
        self.enable_learning = self.engine.enable_learning if hasattr(self.engine, 'enable_learning') else False
        
        # Try to initialize Stockfish API, fall back to mock if unavailable
        try:
            self.stockfish = StockfishAPI()
            # Test connection
            _, _ = self.stockfish.get_best_move(chess.STARTING_FEN, depth=1)
        except:
            print("Could not connect to Stockfish API, using mock implementation")
            self.stockfish = MockStockfishAPI()
            
        self.stockfish_depth = stockfish_depth
        
        # Game statistics
        self.engine_name = self.engine.name
        self.stockfish_name = "Stockfish"
        self.start_time = None
        self.engine_move_times = []
        self.stockfish_move_times = []
        self.position_evals = []  # Store (fen, eval) for each position
        self.game_history = []    # Store complete game history
        
        # Learning data
        self.engine_move_quality = []  # Store (move_idx, quality_value, quality_label)
        self.stockfish_move_quality = []  # Store (move_idx, quality_value, quality_label)
    
    def reset(self):
        """Reset the game to starting position."""
        self.board.reset()
        self.engine.reset()
        self.start_time = None
        self.engine_move_times = []
        self.stockfish_move_times = []
        self.position_evals = []
        self.game_history = []
        self.engine_move_quality = []
        self.stockfish_move_quality = []
    
    def set_position(self, fen):
        """
        Set the board to a specific position.
        
        Args:
            fen: FEN string
            
        Returns:
            True if position was set, False if invalid FEN
        """
        if self.board.set_position(fen):
            self.engine.set_position(fen)
            self.game_history = [fen]
            return True
        return False
    
    def engine_move(self):
        """
        Make an NNUE engine move.
        
        Returns:
            Tuple of (success, message, move_data)
        """
        # Check if it's engine's turn
        current_turn = chess.WHITE if self.board.get_turn() == 'white' else chess.BLACK
        if current_turn != self.engine_color:
            return False, "It's not engine's turn", None
        
        # Get the current evaluation
        eval_score = self.engine.evaluate()
        
        # Store current position and evaluation
        self.position_evals.append((self.board.get_fen(), eval_score))
        
        # Get start time
        start_time = time.time()
        
        # Get engine move
        move = self.engine.get_best_move(time_limit_ms=self.engine_time_ms)
        
        # Calculate time taken
        elapsed = time.time() - start_time
        self.engine_move_times.append(elapsed)
        
        if not move:
            return False, "Engine couldn't find a move", None
        
        # Check if move is legal before continuing
        if move not in self.board.board.legal_moves:
            return False, f"Engine suggested illegal move: {move}", None
        
        try:
            # Get SAN representation (after verifying legality)
            move_san = self.board.board.san(move)
        except Exception as e:
            # If there's an error with san(), try to continue
            print(f"Warning: Error getting SAN notation - {e}")
            move_san = move.uci()  # Fall back to UCI notation
        
        # Store the position before the move for learning
        prev_position = chess.Board(self.board.get_fen())
        prev_eval = eval_score
        
        # Make the move on the board
        self.board.make_move(move)
        
        # Store the new position in history
        self.game_history.append(self.board.get_fen())
        
        # If learning is enabled, evaluate the move quality
        if self.enable_learning:
            # Get evaluation after the move
            post_eval = -self.engine.evaluate()  # Negate because now it's from opponent's perspective
            
            # Evaluate move quality
            quality_value, quality_label = evaluate_move_quality(prev_eval, post_eval, self.engine_color)
            
            # Store move quality
            move_idx = len(self.board.move_history) - 1
            self.engine_move_quality.append((move_idx, quality_value, quality_label))
        
        move_data = {
            'move': move,
            'move_san': move_san,
            'time': elapsed,
            'eval': eval_score,
            'nodes': self.engine.total_nodes,
            'depth': self.engine_depth
        }
        
        return True, f"{move_san} ({format_time(elapsed * 1000)})", move_data
    
    def stockfish_move(self):
        """
        Make a Stockfish move.
        
        Returns:
            Tuple of (success, message, move_data)
        """
        # Check if it's Stockfish's turn
        current_turn = chess.WHITE if self.board.get_turn() == 'white' else chess.BLACK
        if current_turn == self.engine_color:
            return False, "It's not Stockfish's turn", None
        
        # Get the current evaluation from engine's perspective
        prev_eval = self.engine.evaluate()
        
        # Store the position before the move for learning
        prev_position = chess.Board(self.board.get_fen())
        
        # Get start time
        start_time = time.time()
        
        # Get Stockfish move
        move_uci, stockfish_eval = self.stockfish.get_best_move(
            self.board.get_fen(), 
            depth=self.stockfish_depth
        )
        
        # Calculate time taken
        elapsed = time.time() - start_time
        self.stockfish_move_times.append(elapsed)
        
        if not move_uci:
            return False, "Stockfish couldn't find a move", None
        
        # Parse move
        try:
            move = chess.Move.from_uci(move_uci)
        except ValueError:
            return False, f"Invalid move from Stockfish: {move_uci}", None
        
        # Verify move legality
        if move not in self.board.board.legal_moves:
            # Try to find a legal move as fallback
            legal_moves = list(self.board.board.legal_moves)
            if legal_moves:
                move = legal_moves[0]  # Choose the first legal move
                print(f"Warning: Stockfish suggested illegal move: {move_uci}, using {move} instead")
            else:
                return False, f"Stockfish suggested illegal move: {move_uci} and no legal moves available", None
        
        # Get SAN representation
        try:
            move_san = self.board.board.san(move)
        except Exception as e:
            # If there's an error with san(), try to continue
            print(f"Warning: Error getting SAN notation - {e}")
            move_san = move.uci()  # Fall back to UCI notation
            
        # Make the move
        self.board.make_move(move)
        
        # Store current position and evaluation in history
        self.position_evals.append((self.board.get_fen(), stockfish_eval or 0))
        self.game_history.append(self.board.get_fen())
        
        # If learning is enabled, evaluate the move quality and learn from Stockfish
        if self.enable_learning:
            # Get evaluation after the move
            post_eval = -self.engine.evaluate()  # Negate because now it's from opponent's perspective
            
            # Evaluate move quality
            stockfish_color = not self.engine_color
            quality_value, quality_label = evaluate_move_quality(prev_eval, post_eval, stockfish_color)
            
            # Store move quality
            move_idx = len(self.board.move_history) - 1
            self.stockfish_move_quality.append((move_idx, quality_value, quality_label))
            
            # Learn from Stockfish's move if it was a good move
            if quality_value > 0 and hasattr(self.engine, 'finetuner') and self.engine.finetuner:
                # Stockfish made a good move - our engine should learn from it
                # We use a positive quality value to indicate a good move to learn from
                self.engine.finetuner.adjust_for_move(prev_position, move, abs(quality_value))
                print(f"Engine learned from Stockfish's {quality_label} move")
        
        move_data = {
            'move': move,
            'move_san': move_san,
            'time': elapsed,
            'eval': stockfish_eval,
            'depth': self.stockfish_depth
        }
        
        return True, f"{move_san} ({format_time(elapsed * 1000)})", move_data
    
    def start_game(self, max_moves=200):
        """
        Start and play the entire game automatically.
        
        Args:
            max_moves: Maximum number of moves to play
            
        Returns:
            Dictionary with game result information
        """
        self.start_time = time.time()
        self.reset()
        
        # Set player names
        if self.engine_color == chess.WHITE:
            self.board.set_player_names(self.engine_name, self.stockfish_name)
        else:
            self.board.set_player_names(self.stockfish_name, self.engine_name)
        
        # Store initial position
        self.game_history.append(self.board.get_fen())
        
        # Main game loop
        move_count = 0
        while not self.board.is_game_over() and move_count < max_moves:
            # Determine which side moves
            current_turn = chess.WHITE if self.board.get_turn() == 'white' else chess.BLACK
            
            if current_turn == self.engine_color:
                # NNUE engine move
                success, message, _ = self.engine_move()
                if success:
                    print(f"Engine: {message}")
            else:
                # Stockfish move
                success, message, _ = self.stockfish_move()
                if success:
                    print(f"Stockfish: {message}")
            
            if not success:
                print(f"Error: {message}")
                break
            
            move_count += 1
            
            # Check for draw conditions (repetition or 50-move rule)
            if move_count >= 10 and (self.board.board.is_repetition(3) or self.board.board.is_fifty_moves()):
                self.board.game_result = "1/2-1/2"
                break
        
        # Return game result
        return self.get_game_result()
    
    def get_game_result(self):
        """
        Get the game result.
        
        Returns:
            Dictionary with game result information
        """
        result = self.board.get_result()
        winner = self.board.get_winner()
        
        engine_won = (winner == 'white' and self.engine_color == chess.WHITE) or \
                    (winner == 'black' and self.engine_color == chess.BLACK)
        
        stockfish_won = (winner == 'white' and self.engine_color == chess.BLACK) or \
                        (winner == 'black' and self.engine_color == chess.WHITE)
        
        # Calculate game duration
        duration = time.time() - (self.start_time or time.time())
        
        # Calculate average move times
        avg_engine_time = sum(self.engine_move_times) / max(len(self.engine_move_times), 1)
        avg_stockfish_time = sum(self.stockfish_move_times) / max(len(self.stockfish_move_times), 1)
        
        # Count material difference in final position
        if self.position_evals:
            final_eval = self.position_evals[-1][1]
        else:
            final_eval = 0
        
        # Get move quality statistics if learning was enabled
        engine_good_moves = sum(1 for _, quality, _ in self.engine_move_quality if quality > 0)
        engine_bad_moves = sum(1 for _, quality, _ in self.engine_move_quality if quality < 0)
        stockfish_good_moves = sum(1 for _, quality, _ in self.stockfish_move_quality if quality > 0)
        
        return {
            'result': result,
            'winner': winner,
            'engine_won': engine_won,
            'stockfish_won': stockfish_won,
            'is_draw': winner == 'draw',
            'total_moves': len(self.board.move_history),
            'engine_moves': len(self.engine_move_times),
            'stockfish_moves': len(self.stockfish_move_times),
            'duration': duration,
            'avg_engine_time': avg_engine_time,
            'avg_stockfish_time': avg_stockfish_time,
            'final_eval': final_eval,
            'pgn': self.board.get_pgn(),
            'engine_good_moves': engine_good_moves,
            'engine_bad_moves': engine_bad_moves,
            'stockfish_good_moves': stockfish_good_moves,
            'learned_from_stockfish': len(self.stockfish_move_quality) > 0 and self.enable_learning
        }
    
    def is_game_over(self):
        """
        Check if the game is over.
        
        Returns:
            True if game is over, False otherwise
        """
        return self.board.is_game_over()
    
    def get_pgn(self):
        """
        Get PGN representation of the game.
        
        Returns:
            PGN string
        """
        return self.board.get_pgn()
    
    def save_game_log(self, filename=None):
        """
        Save a detailed game log with positions, evaluations, and times.
        
        Args:
            filename: Filename to save to (None for auto-generated)
            
        Returns:
            Path to the saved file
        """
        if filename is None:
            timestamp = int(time.time())
            filename = f"game_log_{timestamp}.txt"
        
        with open(filename, 'w') as f:
            f.write(f"Game: {self.engine_name} vs {self.stockfish_name}\n")
            f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Engine color: {'White' if self.engine_color == chess.WHITE else 'Black'}\n")
            f.write(f"Engine depth: {self.engine_depth}\n")
            f.write(f"Stockfish depth: {self.stockfish_depth}\n")
            f.write(f"Learning mode: {'Enabled' if self.enable_learning else 'Disabled'}\n")
            f.write(f"Result: {self.board.get_result()}\n\n")
            
            # Write move history
            f.write("Move History:\n")
            history = self.board.get_move_history()
            for i, move in enumerate(history):
                player = self.engine_name if (i % 2 == 0 and self.engine_color == chess.WHITE) or \
                                          (i % 2 == 1 and self.engine_color == chess.BLACK) \
                         else self.stockfish_name
                move_time = self.engine_move_times[i//2] if player == self.engine_name else \
                            self.stockfish_move_times[i//2]
                
                # Add move quality if available
                quality_label = ""
                for idx, quality, label in self.engine_move_quality + self.stockfish_move_quality:
                    if idx == i:
                        quality_label = f" ({label})"
                        break
                
                f.write(f"{i+1}. {player}: {move}{quality_label} ({format_time(move_time * 1000)})\n")
            
            # Write position evaluations
            f.write("\nPosition Evaluations:\n")
            for i, (fen, eval_score) in enumerate(self.position_evals):
                f.write(f"Position {i+1}: {eval_score}\n")
            
            # Write PGN
            f.write("\nPGN:\n")
            f.write(self.board.get_pgn())
        
        return filename

def play_engine_vs_stockfish(engine=None, engine_depth=5, engine_time_ms=1000, stockfish_depth=10, 
                          engine_color=chess.WHITE, max_moves=200):
    """
    Play a game between the NNUE engine and Stockfish API.
    
    Args:
        engine: NNUE engine instance (None to create a new one)
        engine_depth: Search depth for NNUE engine
        engine_time_ms: Time limit for NNUE engine in milliseconds
        stockfish_depth: Search depth for Stockfish API
        engine_color: Color for NNUE engine (WHITE/BLACK)
        max_moves: Maximum number of moves to play
        
    Returns:
        Dictionary with game result
    """
    # Initialize game
    game = EngineVsStockfish(
        engine=engine,
        engine_depth=engine_depth,
        engine_time_ms=engine_time_ms,
        stockfish_depth=stockfish_depth,
        engine_color=engine_color
    )
    
    # Explicitly load the latest model
    print("\033[1;36mLoading latest model weights...\033[0m")
    if game.engine.load_model():
        weights_path = game.engine.get_current_weights_path()
        print(f"\033[1;32m✓ Successfully loaded model: {weights_path}\033[0m")
    else:
        print("\033[1;31m✗ Could not load latest model weights, using current model\033[0m")
    
    # Print game information
    print("Starting Engine vs Stockfish game")
    print(f"Engine color: {'White' if engine_color == chess.WHITE else 'Black'}")
    print(f"Engine depth: {engine_depth}, time limit: {format_time(engine_time_ms)}")
    print(f"Stockfish depth: {stockfish_depth}")
    print(f"Learning mode: {'Enabled' if game.enable_learning else 'Disabled'}")
    
    # Start time
    start_time = time.time()
    
    # Play the game
    result = game.start_game(max_moves)
    
    # End time
    end_time = time.time()
    
    # Print results
    print("\nGame completed!")
    print(f"Result: {result['result']}")
    
    if result['is_draw']:
        print("Game ended in a draw")
    elif result['engine_won']:
        print("Engine won!")
    else:
        print("Stockfish won!")
    
    print(f"Total moves: {result['total_moves']}")
    print(f"Game duration: {format_time((end_time - start_time) * 1000)}")
    print(f"Average engine time: {format_time(result['avg_engine_time'] * 1000)}")
    print(f"Average Stockfish time: {format_time(result['avg_stockfish_time'] * 1000)}")
    
    if game.enable_learning:
        print(f"Engine good moves: {result['engine_good_moves']}")
        print(f"Engine bad moves: {result['engine_bad_moves']}")
        print(f"Stockfish good moves: {result['stockfish_good_moves']}")
        
        if result['learned_from_stockfish']:
            print("Engine learned from Stockfish's good moves")
            
        # Automatically save the trained model
        if hasattr(game.engine, 'save_model'):
            path = game.engine.save_model("nnue_weights_after_stockfish_game")
            print(f"Model automatically saved to {path}")
        else:
            print("Engine does not support saving models")
    
    # Save PGN
    timestamp = int(time.time())
    pgn_filename = f"engine_vs_stockfish_{timestamp}.pgn"
    
    # Save to pgns directory
    os.makedirs("pgns", exist_ok=True)
    pgn_path = os.path.join("pgns", pgn_filename)
    with open(pgn_path, 'w') as f:
        f.write(game.get_pgn())
    
    print(f"Game saved to {pgn_path}")
    
    # Save detailed log
    log_filename = game.save_game_log()
    print(f"Detailed log saved to {log_filename}")
    
    # Update engine statistics
    if hasattr(game.engine, 'game_stats'):
        game.engine.game_stats['games_played'] += 1
        
        if result['is_draw']:
            game.engine.game_stats['draws'] += 1
        elif result['engine_won']:
            game.engine.game_stats['wins'] += 1
        else:
            game.engine.game_stats['losses'] += 1
            
        # Calculate win rate
        total_games = game.engine.game_stats['games_played']
        if total_games > 0:
            win_rate = (game.engine.game_stats['wins'] / total_games) * 100
            game.engine.game_stats['win_rate'] = win_rate
            
        print("\nEngine Statistics:")
        print(f"Games played: {game.engine.game_stats['games_played']}")
        print(f"Wins: {game.engine.game_stats['wins']}")
        print(f"Losses: {game.engine.game_stats['losses']}")
        print(f"Draws: {game.engine.game_stats['draws']}")
        print(f"Win rate: {game.engine.game_stats['win_rate']:.1f}%")
    
    # Automatically finetune the model using this game
    if game.enable_learning and not result['is_draw']:
        # Import here to avoid circular imports
        from src.engine.finetune import finetune_from_pgn
        
        # Prepare feedback based on game result
        feedback = None
        emphasis = 2.0  # Increased emphasis factor
        
        if result['engine_won']:
            # Engine won, reinforce good moves and learn only from engine's moves
            feedback = {
                "result": "win", 
                "emphasis": emphasis,
                "learn_from_winner": True,
                "engine_color": "white" if engine_color == chess.WHITE else "black"
            }
            print("\n\033[1;32m🎮 Automatically finetuning model to reinforce winning strategies...\033[0m")
            print(f"\033[1;33m📊 Using emphasis factor: {emphasis}x\033[0m")
            print(f"\033[1;32m🏆 Learning selectively from winning moves (engine's moves)\033[0m")
        else:
            # Engine lost, learn from Stockfish's winning moves
            # Use inverse learning for losses - learn what NOT to do
            feedback = {
                "result": "loss", 
                "emphasis": emphasis,
                "learn_from_winner": True,
                "engine_color": "white" if engine_color == chess.WHITE else "black",
                "inverse_learning": True  # Learn what NOT to do by inverting evaluations
            }
            print("\n\033[1;35m🎮 Automatically finetuning model to learn from Stockfish's strategies...\033[0m")
            print(f"\033[1;33m📊 Using emphasis factor: {emphasis}x\033[0m")
            print(f"\033[1;32m🏆 Learning selectively from winning moves (Stockfish's moves)\033[0m")
            print(f"\033[1;31m⚠️ Using INVERSE LEARNING: Penalizing engine's losing moves\033[0m")
        
        try:
            # Use more epochs with curriculum learning and memory for better retention
            print("\033[1;36m🚀 Starting advanced finetuning process (50 epochs)...\033[0m")
            print("\033[1;33m🧠 Using position memory and curriculum learning\033[0m")
            
            # Get user confirmation for enhanced model if we'll be creating a new one
            use_enhanced = False
            try:
                # Check if we already have a base.pt model
                import os
                if not os.path.exists("saved_models/base.pt"):
                    # First-time training - offer enhanced architecture
                    print("\033[1;35m⭐ No existing model found - would you like to use the enhanced Stockfish-style architecture?\033[0m")
                    print("\033[1;33m⚠️ Note: This will create a new model from scratch with more parameters (8M vs 260K)\033[0m")
                    print("\033[1;33m⚠️ Training and inference will be slower, but potential for stronger play is higher\033[0m")
                    response = input("Use enhanced architecture? (y/n): ")
                    use_enhanced = response.lower() == 'y'
                    if use_enhanced:
                        print("\033[1;32m🎯 Creating new enhanced model from scratch\033[0m")
                    else:
                        print("\033[1;32m🎯 Creating standard model\033[0m")
                else:
                    # Try to determine if existing model is enhanced
                    # This is just a quick check - we'll create the right type of model
                    # to receive weights during finetuning
                    try:
                        # Create a temporary model to check
                        from src.engine.nnue.network import NNUE
                        temp_model = NNUE(use_enhanced=True)
                        
                        # Try to load it - will fail if architectures don't match
                        from src.engine.nnue.weights import load_weights
                        _ = load_weights(temp_model, "saved_models/base.pt")
                        
                        # If we got here, then enhanced model loaded successfully
                        use_enhanced = True
                        print("\033[1;32m🎯 Using existing enhanced model\033[0m")
                    except:
                        # Failure means it's a standard model
                        use_enhanced = False
                        print("\033[1;32m🎯 Using existing standard model\033[0m")
            except Exception as e:
                print(f"\033[1;31m⚠️ Error detecting model type: {e}\033[0m")
                print("\033[1;33m⚠️ Using standard model for safety\033[0m")
                use_enhanced = False
                
            # Run finetuning with all our enhancements
            # Convert old enhanced flag to model_type
            model_type = "pearl" if use_enhanced else "standard"
            
            finetune_from_pgn(
                pgn_path, 
                epochs=50, 
                batch_size=32, 
                feedback=feedback,
                model_type=model_type,
                use_memory=True
            )
            print("\033[1;32m✅ Finetuning complete! Model updated to base.pt for next game\033[0m")
        except Exception as e:
            print(f"Error during finetuning: {e}")
    
    return result

if __name__ == "__main__":
    # Parse command-line arguments if any
    if len(sys.argv) > 1:
        engine_depth = int(sys.argv[1])
    else:
        engine_depth = 5
        
    if len(sys.argv) > 2:
        stockfish_depth = int(sys.argv[2])
    else:
        stockfish_depth = 10
        
    if len(sys.argv) > 3:
        engine_color = chess.WHITE if sys.argv[3].lower() == 'white' else chess.BLACK
    else:
        engine_color = chess.WHITE
        
    # Create engine with learning enabled and 30-second safe time limit
    engine = NNUEEngine(depth=5, time_limit_ms=30000, enable_learning=True)
    
    play_engine_vs_stockfish(
        engine=engine,
        engine_depth=5,  # Use a more reliable depth
        engine_time_ms=15000,  # 15 seconds per move is safer
        stockfish_depth=stockfish_depth,
        engine_color=engine_color
    )
