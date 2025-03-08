import chess
import time
import sys
import os

from src.game.board import ChessBoard
from src.engine.engine import NNUEEngine
from src.engine.utils import format_time, evaluate_move_quality

class HumanVsEngine:
    """
    Human vs Engine game mode.
    
    This mode allows a human player to play against the NNUE engine.
    """
    def __init__(self, engine=None, depth=5, time_limit_ms=1000, player_color=chess.WHITE, enable_learning=True):
        """
        Initialize the Human vs Engine game.
        
        Args:
            engine: NNUE engine instance (None to create a new one)
            depth: Search depth for the engine
            time_limit_ms: Time limit for engine moves in milliseconds
            player_color: Player's color (WHITE/BLACK)
            enable_learning: Whether to enable real-time learning
        """
        self.board = ChessBoard()
        self.engine = engine or NNUEEngine(enable_learning=enable_learning)
        self.depth = depth
        self.time_limit_ms = time_limit_ms
        self.player_color = player_color
        self.enable_learning = enable_learning
        
        # Game statistics
        self.player_name = "Human"
        self.engine_name = self.engine.name
        self.start_time = None
        self.move_times = []
        self.player_move_evals = []  # Store (prev_eval, post_eval) for player moves
        self.engine_move_evals = []  # Store (prev_eval, post_eval) for engine moves
        self.move_quality = []  # Store (move_number, quality_value, quality_label)
        
        # Feedback options
        self.allow_takeback = True
        self.allow_feedback = True
        self.feedback_history = []  # Store (move, feedback) pairs
    
    def reset(self):
        """Reset the game to starting position."""
        self.board.reset()
        self.engine.reset()
        self.start_time = None
        self.move_times = []
        self.player_move_evals = []
        self.engine_move_evals = []
        self.move_quality = []
        self.feedback_history = []
    
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
            return True
        return False
    
    def player_move(self, move_str):
        """
        Make a player move.
        
        Args:
            move_str: Move in UCI or SAN format
            
        Returns:
            Tuple of (success, message)
        """
        # Check if it's player's turn
        current_turn = chess.WHITE if self.board.get_turn() == 'white' else chess.BLACK
        if current_turn != self.player_color:
            return False, "It's not your turn"
        
        # Get the current evaluation
        prev_eval = self.engine.evaluate()
        
        # Try to make the move
        if not self.board.make_move(move_str):
            return False, "Invalid move"
        
        # Update engine's internal board
        self.engine.set_position(self.board.get_fen())
        
        # Get the new evaluation
        post_eval = -self.engine.evaluate()  # Negate because now it's from opponent's perspective
        
        # Store evaluation pair
        self.player_move_evals.append((prev_eval, post_eval))
        
        # Evaluate move quality
        quality_value, quality_label = evaluate_move_quality(prev_eval, post_eval, self.player_color)
        self.move_quality.append((len(self.board.move_history), quality_value, quality_label))
        
        return True, quality_label
    
    def engine_move(self):
        """
        Make an engine move.
        
        Args:
            feedback: Optional feedback for the previous move (-1 for bad, 0 for neutral, 1 for good)
            
        Returns:
            Tuple of (success, message)
        """
        # Check if it's engine's turn
        current_turn = chess.WHITE if self.board.get_turn() == 'white' else chess.BLACK
        if current_turn == self.player_color:
            return False, "It's not engine's turn"
        
        # Get the current evaluation
        prev_eval = self.engine.evaluate()
        
        # Get start time
        start_time = time.time()
        
        # Get engine move
        move = self.engine.get_best_move(time_limit_ms=self.time_limit_ms)
        
        # Calculate time taken
        elapsed = time.time() - start_time
        self.move_times.append(elapsed)
        
        if not move:
            return False, "Engine couldn't find a move"
        
        # Make the move
        move_san = self.board.board.san(move)
        self.board.make_move(move)
        
        # Get the new evaluation
        post_eval = -self.engine.evaluate()  # Negate because now it's from opponent's perspective
        
        # Store evaluation pair
        self.engine_move_evals.append((prev_eval, post_eval))
        
        return True, f"{move_san} ({format_time(elapsed * 1000)})"
    
    def provide_feedback(self, move_idx, feedback_value):
        """
        Provide feedback on an engine move to help it learn.
        
        Args:
            move_idx: Index of the move in the game history
            feedback_value: Feedback value (-1 for bad, 0 for neutral, 1 for good)
            
        Returns:
            True if feedback was processed
        """
        if not self.enable_learning or not self.engine.enable_learning:
            return False
            
        # Check if the move exists and is an engine move
        if move_idx >= len(self.board.move_history):
            return False
            
        move = self.board.move_history[move_idx]
        
        # Determine if this was an engine move
        is_engine_move = (move_idx % 2 == 0 and self.player_color == chess.BLACK) or \
                         (move_idx % 2 == 1 and self.player_color == chess.WHITE)
        
        if not is_engine_move:
            return False
            
        # Convert feedback to quality value
        if feedback_value < 0:
            quality_value = -2  # Bad move
        elif feedback_value > 0:
            quality_value = 2   # Good move
        else:
            quality_value = 0   # Neutral
            
        # Store feedback
        self.feedback_history.append((move_idx, feedback_value))
        
        # Create a board position before the move
        temp_board = chess.Board()
        for i in range(move_idx):
            temp_board.push(self.board.move_history[i])
            
        # Apply feedback to the engine's learning
        if self.engine.finetuner:
            self.engine.finetuner.adjust_for_move(temp_board, move, quality_value)
            return True
            
        return False
    
    def start_game(self):
        """
        Start the game.
        
        Returns:
            True if the game was started successfully
        """
        self.start_time = time.time()
        
        # Set player names
        if self.player_color == chess.WHITE:
            self.board.set_player_names(self.player_name, self.engine_name)
        else:
            self.board.set_player_names(self.engine_name, self.player_name)
        
        # If engine plays first (player is black), make an engine move
        if self.player_color == chess.BLACK:
            success, _ = self.engine_move()
            return success
        
        return True
    
    def get_game_result(self):
        """
        Get the game result.
        
        Returns:
            Dictionary with game result information
        """
        result = self.board.get_result()
        winner = self.board.get_winner()
        
        player_won = (winner == 'white' and self.player_color == chess.WHITE) or \
                     (winner == 'black' and self.player_color == chess.BLACK)
        
        engine_won = (winner == 'white' and self.player_color == chess.BLACK) or \
                     (winner == 'black' and self.player_color == chess.WHITE)
        
        # Calculate game duration
        duration = time.time() - (self.start_time or time.time())
        
        # Calculate average engine move time
        avg_engine_time = sum(self.move_times) / max(len(self.move_times), 1)
        
        # Get move quality statistics
        player_moves_count = len(self.player_move_evals)
        blunders = sum(1 for _, quality, _ in self.move_quality if quality == -3)
        mistakes = sum(1 for _, quality, _ in self.move_quality if quality == -2)
        inaccuracies = sum(1 for _, quality, _ in self.move_quality if quality == -1)
        good_moves = sum(1 for _, quality, _ in self.move_quality if quality > 0)
        
        # Get feedback statistics
        positive_feedback = sum(1 for _, feedback in self.feedback_history if feedback > 0)
        negative_feedback = sum(1 for _, feedback in self.feedback_history if feedback < 0)
        
        return {
            'result': result,
            'winner': winner,
            'player_won': player_won,
            'engine_won': engine_won,
            'is_draw': winner == 'draw',
            'total_moves': len(self.board.move_history),
            'player_moves': player_moves_count,
            'engine_moves': len(self.board.move_history) - player_moves_count,
            'duration': duration,
            'avg_engine_time': avg_engine_time,
            'blunders': blunders,
            'mistakes': mistakes,
            'inaccuracies': inaccuracies,
            'good_moves': good_moves,
            'positive_feedback': positive_feedback,
            'negative_feedback': negative_feedback
        }
    
    def is_game_over(self):
        """
        Check if the game is over.
        
        Returns:
            True if game is over, False otherwise
        """
        return self.board.is_game_over()
    
    def get_board_unicode(self):
        """
        Get Unicode representation of the current board.
        
        Returns:
            Unicode string
        """
        return self.board.to_unicode(perspective=self.player_color)
    
    def get_board_ascii(self):
        """
        Get ASCII representation of the current board.
        
        Returns:
            ASCII string
        """
        return self.board.to_ascii(perspective=self.player_color)
    
    def get_current_turn(self):
        """
        Get the player who has the current turn.
        
        Returns:
            'player' or 'engine'
        """
        current_turn = chess.WHITE if self.board.get_turn() == 'white' else chess.BLACK
        return 'player' if current_turn == self.player_color else 'engine'
    
    def get_move_history(self):
        """
        Get the game move history.
        
        Returns:
            List of moves in SAN format
        """
        return self.board.get_move_history()
    
    def get_pgn(self):
        """
        Get PGN representation of the game.
        
        Returns:
            PGN string
        """
        return self.board.get_pgn()
    
    def undo_last_moves(self, count=2):
        """
        Undo the last moves (player and engine).
        
        Args:
            count: Number of moves to undo
            
        Returns:
            True if moves were undone, False otherwise
        """
        # Check if there are enough moves to undo
        if len(self.board.move_history) < count:
            return False
        
        # Undo the moves
        for _ in range(count):
            self.board.undo_move()
        
        # Update engine's position
        self.engine.set_position(self.board.get_fen())
        
        # Remove the corresponding evaluations and move times
        player_moves_to_remove = sum(1 for _ in range(count) if self.get_current_turn() == 'player')
        engine_moves_to_remove = count - player_moves_to_remove
        
        if player_moves_to_remove > 0 and self.player_move_evals:
            self.player_move_evals = self.player_move_evals[:-player_moves_to_remove]
        
        if engine_moves_to_remove > 0 and self.engine_move_evals:
            self.engine_move_evals = self.engine_move_evals[:-engine_moves_to_remove]
            self.move_times = self.move_times[:-engine_moves_to_remove]
        
        # Remove move quality entries
        if self.move_quality:
            self.move_quality = self.move_quality[:-count]
        
        return True
    
    def toggle_learning(self, enable=None):
        """
        Toggle learning on/off.
        
        Args:
            enable: True to enable, False to disable, None to toggle
            
        Returns:
            New learning state
        """
        if enable is None:
            self.enable_learning = not self.enable_learning
        else:
            self.enable_learning = enable
            
        # Update engine learning state
        if self.engine:
            self.engine.toggle_learning(self.enable_learning)
            
        return self.enable_learning

def play_human_vs_engine(depth=5, time_limit_ms=1000, player_color=chess.WHITE, enable_learning=True):
    """
    Simple command-line interface for Human vs Engine mode.
    
    Args:
        depth: Search depth for the engine
        time_limit_ms: Time limit for engine moves in milliseconds
        player_color: Player's color (WHITE/BLACK)
        enable_learning: Whether to enable real-time learning
    """
    game = HumanVsEngine(depth=depth, time_limit_ms=time_limit_ms, 
                         player_color=player_color, enable_learning=enable_learning)
    
    print("Starting new game: Human vs Engine")
    print(f"Engine search depth: {depth}, time limit: {format_time(time_limit_ms)}")
    print(f"Learning mode: {'Enabled' if enable_learning else 'Disabled'}")
    print("Type 'help' for available commands")
    
    game.start_game()
    
    while not game.is_game_over():
        # Print the board
        try:
            print(game.get_board_unicode())
        except:
            print(game.get_board_ascii())
        
        # Check whose turn it is
        if game.get_current_turn() == 'player':
            # Player's turn
            move_str = input("Your move (or 'help', 'undo', 'resign'): ")
            
            if move_str.lower() == 'help':
                print("Available commands:")
                print("  help       - Show this help")
                print("  undo       - Undo the last two moves")
                print("  resign     - Resign the game")
                print("  quit       - Exit the game")
                print("  feedback + - Give positive feedback on last engine move")
                print("  feedback - - Give negative feedback on last engine move")
                print("  feedback 0 - Give neutral feedback on last engine move")
                print("  learning on/off - Toggle engine learning")
                print("Enter a move in UCI (e2e4) or SAN (e4) format")
                input("Press Enter to continue...")
                continue
            elif move_str.lower() == 'undo':
                if game.undo_last_moves(2):
                    print("Undoing the last move pair (yours and engine's)")
                else:
                    print("Cannot undo moves")
                continue
            elif move_str.lower() == 'resign':
                print("You resigned.")
                if player_color == chess.WHITE:
                    game.board.game_result = "0-1"
                else:
                    game.board.game_result = "1-0"
                break
            elif move_str.lower() in ['quit', 'exit']:
                print("Exiting game.")
                return
            elif move_str.lower() == 'feedback +':
                # Positive feedback on last engine move
                if len(game.board.move_history) > 0:
                    last_engine_move_idx = len(game.board.move_history) - 1
                    if game.provide_feedback(last_engine_move_idx, 1):
                        print("Positive feedback recorded. Engine will learn from this.")
                    else:
                        print("Could not provide feedback.")
                else:
                    print("No moves to provide feedback on.")
                continue
            elif move_str.lower() == 'feedback -':
                # Negative feedback on last engine move
                if len(game.board.move_history) > 0:
                    last_engine_move_idx = len(game.board.move_history) - 1
                    if game.provide_feedback(last_engine_move_idx, -1):
                        print("Negative feedback recorded. Engine will learn from this.")
                    else:
                        print("Could not provide feedback.")
                else:
                    print("No moves to provide feedback on.")
                continue
            elif move_str.lower() == 'feedback 0':
                # Neutral feedback on last engine move
                if len(game.board.move_history) > 0:
                    last_engine_move_idx = len(game.board.move_history) - 1
                    if game.provide_feedback(last_engine_move_idx, 0):
                        print("Neutral feedback recorded.")
                    else:
                        print("Could not provide feedback.")
                else:
                    print("No moves to provide feedback on.")
                continue
            elif move_str.lower() == 'learning on':
                game.toggle_learning(True)
                print("Learning mode enabled.")
                continue
            elif move_str.lower() == 'learning off':
                game.toggle_learning(False)
                print("Learning mode disabled.")
                continue
            
            # Make the player's move
            success, message = game.player_move(move_str)
            if not success:
                print(f"Error: {message}")
                input("Press Enter to continue...")
                continue
            
            print(f"Your move quality: {message}")
            
            # If the game is over after player's move, break
            if game.is_game_over():
                break
                
            # Make the engine's move
            print("Engine is thinking...")
            success, message = game.engine_move()
            if not success:
                print(f"Error: {message}")
                break
                
            print(f"Engine move: {message}")
            
            # Prompt for feedback if learning is enabled
            if game.enable_learning:
                print("You can provide feedback on this move with 'feedback +/-/0' on your next turn.")
        else:
            # Engine's turn
            print("Engine is thinking...")
            success, message = game.engine_move()
            if not success:
                print(f"Error: {message}")
                break
                
            print(f"Engine move: {message}")
    
    # Game is over
    try:
        print(game.get_board_unicode())
    except:
        print(game.get_board_ascii())
    
    # Print game result
    result = game.get_game_result()
    
    print("\nGame over!")
    if result['is_draw']:
        print("Result: Draw")
    elif result['player_won']:
        print("Result: You win!")
    else:
        print("Result: Engine wins")
    
    print(f"\nGame statistics:")
    print(f"Total moves: {result['total_moves']}")
    print(f"Duration: {format_time(result['duration'] * 1000)}")
    print(f"Your blunders: {result['blunders']}")
    print(f"Your mistakes: {result['mistakes']}")
    print(f"Your inaccuracies: {result['inaccuracies']}")
    print(f"Your good moves: {result['good_moves']}")
    
    if game.enable_learning:
        print(f"Feedback given: {result['positive_feedback']} positive, {result['negative_feedback']} negative")
        
        # Ask to save the trained model
        save = input("\nSave trained engine model? (y/n): ")
        if save.lower() == 'y':
            path = game.engine.save_model("nnue_weights_after_game")
            print(f"Model saved to {path}")
    
    # Ask to save PGN
    save = input("\nSave game to PGN? (y/n): ")
    if save.lower() == 'y':
        filename = input("Enter filename (or press Enter for default): ")
        if not filename:
            filename = f"game_{int(time.time())}.pgn"
        
        pgn = game.get_pgn()
        with open(filename, 'w') as f:
            f.write(pgn)
        
        print(f"Game saved to {filename}")

if __name__ == "__main__":
    play_human_vs_engine()
