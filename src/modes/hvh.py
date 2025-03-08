import chess
import time
import sys
import os

from src.game.board import ChessBoard
from src.engine.engine import NNUEEngine
from src.engine.utils import format_time, evaluate_move_quality

class HumanVsHuman:
    """
    Human vs Human game mode.
    
    This mode allows two human players to play against each other,
    with optional analysis from the NNUE engine.
    """
    def __init__(self, engine=None, analysis_depth=3, white_name="White", black_name="Black"):
        """
        Initialize the Human vs Human game.
        
        Args:
            engine: Optional NNUE engine instance for analysis
            analysis_depth: Depth for engine analysis
            white_name: Name of the white player
            black_name: Name of the black player
        """
        self.board = ChessBoard()
        self.engine = engine
        self.analysis_depth = analysis_depth
        self.show_analysis = engine is not None
        
        # Set player names
        self.board.set_player_names(white_name, black_name)
        
        # Game statistics
        self.start_time = None
        self.move_times = {chess.WHITE: [], chess.BLACK: []}
        self.last_move_time = None
        self.move_evals = []  # Store (move, prev_eval, post_eval, quality_label) for each move
    
    def reset(self):
        """Reset the game to starting position."""
        self.board.reset()
        if self.engine:
            self.engine.reset()
        self.start_time = None
        self.move_times = {chess.WHITE: [], chess.BLACK: []}
        self.last_move_time = None
        self.move_evals = []
    
    def set_position(self, fen):
        """
        Set the board to a specific position.
        
        Args:
            fen: FEN string
            
        Returns:
            True if position was set, False if invalid FEN
        """
        if self.board.set_position(fen):
            if self.engine:
                self.engine.set_position(fen)
            return True
        return False
    
    def make_move(self, move_str):
        """
        Make a move.
        
        Args:
            move_str: Move in UCI or SAN format
            
        Returns:
            Tuple of (success, message, analysis)
        """
        # Record start time if this is the first move
        if self.start_time is None:
            self.start_time = time.time()
        
        # Get current player
        current_player = self.board.get_turn()
        current_color = chess.WHITE if current_player == 'white' else chess.BLACK
        
        # Record move start time
        if self.last_move_time is None:
            self.last_move_time = time.time()
        
        # Get evaluation before the move
        prev_eval = None
        if self.engine and self.show_analysis:
            prev_eval = self.engine.evaluate()
        
        # Try to make the move
        if not self.board.make_move(move_str):
            return False, "Invalid move", None
        
        # Record move time
        move_time = time.time() - self.last_move_time
        self.move_times[current_color].append(move_time)
        self.last_move_time = time.time()
        
        # Update engine's internal board
        if self.engine:
            self.engine.set_position(self.board.get_fen())
        
        # Get evaluation after the move
        post_eval = None
        quality_value = 0
        quality_label = "Unknown"
        
        if self.engine and self.show_analysis:
            post_eval = -self.engine.evaluate()  # Negate because now it's from opponent's perspective
            quality_value, quality_label = evaluate_move_quality(prev_eval, post_eval, current_color)
        
        # Store move evaluation
        if prev_eval is not None and post_eval is not None:
            self.move_evals.append((move_str, prev_eval, post_eval, quality_label))
        
        # Get engine analysis
        analysis = None
        if self.engine and self.show_analysis:
            analysis = {
                'eval': post_eval,
                'quality': quality_label,
                'best_move': None
            }
            
            # Get best move for next player
            best_move = self.engine.get_best_move(depth=self.analysis_depth)
            if best_move:
                analysis['best_move'] = self.board.board.san(best_move)
        
        return True, f"Move played: {move_str}", analysis
    
    def undo_move(self):
        """
        Undo the last move.
        
        Returns:
            The move that was undone, or None if no moves to undo
        """
        move = self.board.undo_move()
        
        if move:
            # Update engine's internal board
            if self.engine:
                self.engine.set_position(self.board.get_fen())
            
            # Remove last move time
            current_player = self.board.get_turn()
            current_color = chess.WHITE if current_player == 'white' else chess.BLACK
            if self.move_times[current_color]:
                self.move_times[current_color].pop()
            
            # Remove last move evaluation
            if self.move_evals:
                self.move_evals.pop()
            
            # Reset last move time
            self.last_move_time = time.time()
        
        return move
    
    def get_current_player(self):
        """
        Get the current player.
        
        Returns:
            'white' or 'black'
        """
        return self.board.get_turn()
    
    def is_game_over(self):
        """
        Check if the game is over.
        
        Returns:
            True if game is over, False otherwise
        """
        return self.board.is_game_over()
    
    def get_game_result(self):
        """
        Get the game result.
        
        Returns:
            Dictionary with game result information
        """
        result = self.board.get_result()
        winner = self.board.get_winner()
        
        # Calculate game duration
        if self.start_time is None:
            duration = 0
        else:
            duration = time.time() - self.start_time
        
        # Calculate average move times
        avg_white_time = sum(self.move_times[chess.WHITE]) / max(len(self.move_times[chess.WHITE]), 1)
        avg_black_time = sum(self.move_times[chess.BLACK]) / max(len(self.move_times[chess.BLACK]), 1)
        
        # Calculate move quality statistics
        blunders = sum(1 for _, _, _, quality in self.move_evals if quality == "Blunder")
        mistakes = sum(1 for _, _, _, quality in self.move_evals if quality == "Mistake")
        inaccuracies = sum(1 for _, _, _, quality in self.move_evals if quality == "Inaccuracy")
        good_moves = sum(1 for _, _, _, quality in self.move_evals if quality in ["Good", "Excellent", "Brilliant"])
        
        return {
            'result': result,
            'winner': winner,
            'is_draw': winner == 'draw',
            'total_moves': len(self.board.move_history),
            'white_moves': len(self.move_times[chess.WHITE]),
            'black_moves': len(self.move_times[chess.BLACK]),
            'duration': duration,
            'avg_white_time': avg_white_time,
            'avg_black_time': avg_black_time,
            'blunders': blunders,
            'mistakes': mistakes,
            'inaccuracies': inaccuracies,
            'good_moves': good_moves
        }
    
    def get_board_ascii(self, perspective=chess.WHITE):
        """
        Get ASCII representation of the current board.
        
        Args:
            perspective: Perspective to view from (WHITE/BLACK)
            
        Returns:
            ASCII string
        """
        return self.board.to_ascii(perspective=perspective)
    
    def get_board_unicode(self, perspective=chess.WHITE):
        """
        Get Unicode representation of the current board.
        
        Args:
            perspective: Perspective to view from (WHITE/BLACK)
            
        Returns:
            Unicode string
        """
        return self.board.to_unicode(perspective=perspective)
    
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
    
    def toggle_analysis(self):
        """
        Toggle analysis on/off.
        
        Returns:
            New analysis state (True/False)
        """
        if self.engine is None:
            return False
            
        self.show_analysis = not self.show_analysis
        return self.show_analysis
    
    def set_engine(self, engine, depth=3):
        """
        Set the analysis engine.
        
        Args:
            engine: NNUE engine instance
            depth: Analysis depth
            
        Returns:
            True if engine was set
        """
        self.engine = engine
        self.analysis_depth = depth
        
        # Update engine's internal board
        if self.engine:
            self.engine.set_position(self.board.get_fen())
            
        return True

def play_human_vs_human(with_analysis=True, white_name="Player 1", black_name="Player 2"):
    """
    Simple command-line interface for Human vs Human mode.
    
    Args:
        with_analysis: Whether to use engine analysis
        white_name: Name of the white player
        black_name: Name of the black player
    """
    # Initialize engine if analysis is enabled
    engine = None
    if with_analysis:
        engine = NNUEEngine()
    
    # Create game
    game = HumanVsHuman(engine=engine, white_name=white_name, black_name=black_name)
    
    print(f"Starting new game: {white_name} vs {black_name}")
    if with_analysis:
        print("Engine analysis is enabled. Type 'analysis off' to disable.")
    print("Type 'help' for available commands")
    
    while not game.is_game_over():
        # Print the board
        try:
            # Show board from current player's perspective
            current_player = game.get_current_player()
            perspective = chess.WHITE if current_player == 'white' else chess.BLACK
            print(game.get_board_unicode(perspective=perspective))
        except:
            print(game.get_board_ascii(perspective=perspective))
        
        # Show whose turn it is
        current_player_name = white_name if current_player == 'white' else black_name
        print(f"\n{current_player_name}'s turn ({current_player})")
        
        # Get input
        move_str = input("Enter your move (or 'help', 'undo', 'resign'): ")
        
        if move_str.lower() == 'help':
            print("Available commands:")
            print("  help        - Show this help")
            print("  undo        - Undo the last move")
            print("  resign      - Resign the game")
            print("  quit/exit   - Exit the game")
            print("  analysis on/off - Toggle engine analysis")
            print("  flip        - Flip the board view")
            print("  save [file] - Save the game to PGN file")
            print("Enter a move in UCI (e2e4) or SAN (e4) format")
            input("Press Enter to continue...")
            continue
        elif move_str.lower() == 'undo':
            if game.undo_move():
                print("Move undone")
            else:
                print("No moves to undo")
            continue
        elif move_str.lower() == 'resign':
            # Current player resigns
            if current_player == 'white':
                game.board.game_result = "0-1"
            else:
                game.board.game_result = "1-0"
            break
        elif move_str.lower() in ['quit', 'exit']:
            print("Exiting game.")
            return
        elif move_str.lower() == 'analysis on':
            if game.engine is None:
                print("No analysis engine available")
            else:
                game.show_analysis = True
                print("Analysis enabled")
            continue
        elif move_str.lower() == 'analysis off':
            game.show_analysis = False
            print("Analysis disabled")
            continue
        elif move_str.lower() == 'flip':
            perspective = not perspective
            continue
        elif move_str.lower().startswith('save'):
            parts = move_str.split(maxsplit=1)
            if len(parts) > 1:
                filename = parts[1]
            else:
                filename = f"game_{int(time.time())}.pgn"
            
            with open(filename, 'w') as f:
                f.write(game.get_pgn())
            
            print(f"Game saved to {filename}")
            continue
        
        # Make the move
        success, message, analysis = game.make_move(move_str)
        
        if not success:
            print(f"Error: {message}")
            input("Press Enter to continue...")
            continue
        
        print(message)
        
        # Show analysis if enabled
        if analysis:
            eval_str = f"{analysis['eval']/100:+.2f}" if abs(analysis['eval']) < 1000 else \
                       f"M{(30000 - abs(analysis['eval']))//100}" if analysis['eval'] > 0 else \
                       f"-M{(30000 - abs(analysis['eval']))//100}"
            
            print(f"Evaluation: {eval_str} ({analysis['quality']})")
            if analysis['best_move']:
                print(f"Engine suggests: {analysis['best_move']}")
    
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
    elif result['winner'] == 'white':
        print(f"Result: {white_name} wins!")
    else:
        print(f"Result: {black_name} wins!")
    
    print(f"\nGame statistics:")
    print(f"Total moves: {result['total_moves']}")
    print(f"Duration: {format_time(result['duration'] * 1000)}")
    print(f"{white_name}'s average time per move: {format_time(result['avg_white_time'] * 1000)}")
    print(f"{black_name}'s average time per move: {format_time(result['avg_black_time'] * 1000)}")
    
    if with_analysis:
        print(f"Blunders: {result['blunders']}")
        print(f"Mistakes: {result['mistakes']}")
        print(f"Inaccuracies: {result['inaccuracies']}")
        print(f"Good moves: {result['good_moves']}")
    
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
    play_human_vs_human()
