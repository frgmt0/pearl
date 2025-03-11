import chess
import time
import sys
import os
import random
import pandas as pd
import csv
from datetime import datetime

from src.game.board import ChessBoard
from src.engine.engine import NNUEEngine
from src.utils.api.stockfish import StockfishAPI, MockStockfishAPI
from src.engine.utils import format_time, evaluate_move_quality

class EngineVsEngine:
    """
    Engine vs Engine self-play mode with occasional Stockfish games.
    
    This mode generates training data by having the engine play against itself,
    with occasional games against Stockfish at varying depths.
    """
    def __init__(self, engine_white=None, engine_black=None, 
                 engine_depth=7, engine_time_ms=3000,
                 stockfish_chance=0.01, stockfish_depths=None):
        """
        Initialize the Engine vs Engine game.
        
        Args:
            engine_white: NNUE engine instance for white (None to create a new one)
            engine_black: NNUE engine instance for black (None to create a new one)
            engine_depth: Search depth for the NNUE engines
            engine_time_ms: Time limit for NNUE engine moves in milliseconds
            stockfish_chance: Probability (0-1) that a game will use Stockfish
            stockfish_depths: List of depths for Stockfish (None for default)
        """
        self.board = ChessBoard()
        
        # Initialize engines (or use provided ones)
        self.engine_white = engine_white or NNUEEngine(name="Pearl White", depth=engine_depth, time_limit_ms=engine_time_ms)
        self.engine_black = engine_black or NNUEEngine(name="Pearl Black", depth=engine_depth, time_limit_ms=engine_time_ms)
        
        # Engine parameters
        self.engine_depth = engine_depth
        self.engine_time_ms = engine_time_ms
        
        # Stockfish parameters
        self.stockfish_chance = stockfish_chance
        self.stockfish_depths = stockfish_depths or [5, 8, 10, 12, 15, 18]
        
        # Try to initialize Stockfish API, fall back to mock if unavailable
        try:
            self.stockfish = StockfishAPI()
            # Test connection
            _, _ = self.stockfish.get_best_move(chess.STARTING_FEN, depth=1)
            self.has_stockfish = True
        except:
            print("\033[1;33mWarning: Could not connect to Stockfish API, using mock implementation\033[0m")
            self.stockfish = MockStockfishAPI()
            self.has_stockfish = self.stockfish.has_engine
        
        # Game statistics
        self.start_time = None
        self.white_move_times = []
        self.black_move_times = []
        self.position_evals = []  # Store (fen, eval) for each position
        self.game_history = []    # Store complete game history
        self.move_sequence = []   # Store moves as UCI strings
        self.move_evaluations = [] # Store evaluations for each move
        
        # Dataset collection
        self.games_data = []      # List of game data (move_sequence, result, evaluations)
        
        # Mode tracking
        self.using_stockfish = False
        self.stockfish_playing_as = None  # 'white' or 'black'
        self.stockfish_depth = 10  # Default depth
        
    def reset(self):
        """Reset the game to starting position."""
        self.board.reset()
        self.engine_white.reset()
        self.engine_black.reset()
        self.start_time = None
        self.white_move_times = []
        self.black_move_times = []
        self.position_evals = []
        self.game_history = []
        self.move_sequence = []
        self.move_evaluations = []
        
        # Sometimes start from a common opening position to increase variety
        self.apply_random_opening()
        
        # Decide if this game will use Stockfish
        if self.has_stockfish and random.random() < self.stockfish_chance:
            self.using_stockfish = True
            # Randomly choose which side Stockfish plays
            self.stockfish_playing_as = random.choice(['white', 'black'])
            # Randomly choose Stockfish depth from provided list
            self.stockfish_depth = random.choice(self.stockfish_depths)
            
            # Color-coded Stockfish info
            color = "\033[1;32m" if self.stockfish_playing_as == 'white' else "\033[1;34m"
            print(f"\033[1;33mStockfish playing as {color}{self.stockfish_playing_as}\033[1;33m at depth \033[1;36m{self.stockfish_depth}\033[0m")
        else:
            self.using_stockfish = False
            self.stockfish_playing_as = None
            
    def apply_random_opening(self):
        """Sometimes start from common opening positions to increase variety"""
        # Decide which type of opening to use
        opening_type = random.random()
        
        # Standard openings (30% of games)
        if opening_type < 0.3:
            self.apply_standard_opening()
        # Tactical positions (10% of games)
        elif opening_type < 0.4:
            self.apply_tactical_position()
        # Default opening (60% of games)
        else:
            return
            
    def apply_standard_opening(self):
        """Apply a standard chess opening"""
        # List of common opening FENs (1-3 moves into the game)
        common_openings = [
            # Open games
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",  # e4 e5
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",  # e4 e5 Nf3
            "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",  # e4 e5 Nf3 Nc6
            
            # Semi-open games
            "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",  # e4 d5
            "rnbqkbnr/pppp1ppp/8/4p3/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2",  # d4 e5
            
            # Closed games
            "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2",  # d4 d5
            "rnbqkbnr/ppp1pppp/8/3p4/3P4/2N5/PPP1PPPP/R1BQKBNR b KQkq - 1 2",  # d4 d5 Nc3
            
            # Flank openings
            "rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 1 1",  # Nf3
            "rnbqkbnr/pppppppp/8/8/1P6/8/P1PPPPPP/RNBQKBNR b KQkq - 0 1",  # b4 (Sokolsky)
            
            # Slightly more complex positions
            "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",  # e4 e5 Nf3 Nf6
            "rnbqkbnr/ppp2ppp/8/3pp3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 3",  # e4 e5 Nf3 d5
            "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",  # e4 e5 Nf3 Nc6 Bc4
        ]
        
        # Select a random opening
        opening_fen = random.choice(common_openings)
        
        # Apply the opening position
        self.board.set_position(opening_fen)
        self.engine_white.set_position(opening_fen)
        self.engine_black.set_position(opening_fen)
        
        # Add the initial position to history
        self.game_history.append(opening_fen)
        
        # Log the selected opening
        print(f"\033[1;35m♘ Starting from standard opening position\033[0m")
        
    def apply_tactical_position(self):
        """Apply a tactical position with winning chances for one side"""
        # List of tactical positions with clear advantages for one side
        tactical_positions = [
            # White has advantage
            {
                "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/2N2N2/PPPP1PPP/R1BQKB1R w KQkq - 0 4",
                "description": "White with development advantage"
            },
            {
                "fen": "rnbqkbnr/ppp2ppp/4p3/3p4/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 0 3",
                "description": "Central advantage for White"
            },
            {
                "fen": "rnbqkb1r/ppp1pppp/5n2/3P4/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 3",
                "description": "White has won a pawn"
            },
            
            # Black has advantage
            {
                "fen": "rnbqkb1r/ppp1pppp/5n2/3p4/3P4/2N5/PPP1PPPP/R1BQKBNR b KQkq - 2 3",
                "description": "Equal position with chances for Black"
            },
            {
                "fen": "rnbqkbnr/pp2pppp/2p5/3p4/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 0 3",
                "description": "Black has better pawn structure"
            },
            {
                "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 4",
                "description": "Standard position with options for Black"
            },
            
            # More dynamic positions
            {
                "fen": "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
                "description": "Common position with chances for either side"
            },
            {
                "fen": "rnbqkb1r/pp2pppp/2p2n2/3p4/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 0 4",
                "description": "Interesting strategic position"
            },
            
            # Material imbalance positions
            {
                "fen": "rnbqkbnr/ppp1p1pp/8/3p1p2/3P4/2N5/PPP1PPPP/R1BQKBNR w KQkq - 0 3",
                "description": "Unusual pawn structure"
            },
            {
                "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p3/3PP3/5N2/PPP2PPP/RNBQKB1R w KQkq - 2 4",
                "description": "Central tension"
            }
        ]
        
        # Select a random tactical position
        position = random.choice(tactical_positions)
        
        # Apply the position
        self.board.set_position(position["fen"])
        self.engine_white.set_position(position["fen"])
        self.engine_black.set_position(position["fen"])
        
        # Add the initial position to history
        self.game_history.append(position["fen"])
        
        # Log the selected opening
        print(f"\033[1;35m♞ Starting from tactical position: {position['description']}\033[0m")
    
    def get_random_opening_move(self):
        """
        Get a random reasonable opening move.
        
        Returns:
            Chess move or None
        """
        # Common opening moves
        common_openings = [
            'e2e4', 'd2d4', 'c2c4', 'g1f3', 'b1c3',  # Common first moves
            'e7e5', 'd7d5', 'c7c5', 'g8f6', 'b8c6'   # Common responses
        ]
        
        # Check current position
        fen = self.board.get_fen()
        board = chess.Board(fen)
        
        # Make sure it's one of the first 2 moves
        if len(board.move_stack) >= 2:
            return None
            
        # Try a random common opening move
        random.shuffle(common_openings)
        for move_uci in common_openings:
            try:
                move = chess.Move.from_uci(move_uci)
                if move in board.legal_moves:
                    return move
            except:
                continue
                
        # If no common opening moves are legal, return None
        return None
    
    def engine_move(self, color):
        """
        Make an engine move.
        
        Args:
            color: 'white' or 'black'
            
        Returns:
            Tuple of (success, message, move_data)
        """
        # Check if it's the right turn
        current_turn = self.board.get_turn()
        if (color == 'white' and current_turn != 'white') or (color == 'black' and current_turn != 'black'):
            return False, f"It's not {color}'s turn", None
        
        # Select the appropriate engine
        engine = self.engine_white if color == 'white' else self.engine_black
        
        # Check if we should use a random opening move
        random_move = self.get_random_opening_move()
        
        # Get start time
        start_time = time.time()
        
        if random_move and random.random() < 0.8:  # 80% chance to use random opening move
            move = random_move
            # Get evaluation for this move
            eval_score = engine.evaluate()
        else:
            # Get current evaluation
            eval_score = engine.evaluate()
            
            # Sync the engine board with our board first
            engine.board = chess.Board(self.board.get_fen())
            
            # Get engine move
            move = engine.get_best_move(time_limit_ms=self.engine_time_ms)
        
        # Calculate time taken
        elapsed = time.time() - start_time
        if color == 'white':
            self.white_move_times.append(elapsed)
        else:
            self.black_move_times.append(elapsed)
        
        # Store current position and evaluation
        self.position_evals.append((self.board.get_fen(), eval_score))
        
        if not move:
            # Get a random legal move as fallback
            legal_moves = list(self.board.board.legal_moves)
            if legal_moves:
                move = random.choice(legal_moves)
                print(f"Engine couldn't find a move. Using random move: {move}")
            else:
                return False, f"Engine couldn't find a move for {color} and no legal moves available", None
        
        # Check if move is legal before continuing
        if move not in self.board.board.legal_moves:
            # Get a random legal move as fallback
            legal_moves = list(self.board.board.legal_moves)
            if legal_moves:
                move = random.choice(legal_moves)
                print(f"Engine suggested illegal move. Using random move: {move}")
            else:
                return False, f"Engine suggested illegal move: {move} and no legal moves available", None
        
        # Get SAN representation
        move_san = self.board.board.san(move)
        
        # Store the move in sequence
        self.move_sequence.append(move.uci())
        self.move_evaluations.append(eval_score)
        
        try:
            # Get SAN representation (catch any potential errors) 
            move_san = self.board.board.san(move)
        except Exception as e:
            # If there's an error with san(), try to continue
            print(f"\033[1;33m⚠ Warning: Error getting SAN notation - {e}\033[0m")
            move_san = move.uci()  # Fall back to UCI notation
        
        # Make the move on the board
        self.board.make_move(move)
        
        # Store the new position in history
        self.game_history.append(self.board.get_fen())
        
        move_data = {
            'move': move,
            'move_san': move_san,
            'time': elapsed,
            'eval': eval_score,
            'depth': self.engine_depth
        }
        
        return True, f"{move_san} ({format_time(elapsed * 1000)})", move_data
    
    def stockfish_move(self, color):
        """
        Make a Stockfish move.
        
        Args:
            color: 'white' or 'black' - which side Stockfish is playing
            
        Returns:
            Tuple of (success, message, move_data)
        """
        # Check if it's the right turn
        current_turn = self.board.get_turn()
        if (color == 'white' and current_turn != 'white') or (color == 'black' and current_turn != 'black'):
            return False, f"It's not {color}'s turn", None
        
        # Get start time
        start_time = time.time()
        
        # Sync engine board with current board
        self.engine_white.board = chess.Board(self.board.get_fen())
        
        # Get current evaluation from engine
        eval_score = self.engine_white.evaluate()
        
        # Get Stockfish move
        move_uci, stockfish_eval = self.stockfish.get_best_move(
            self.board.get_fen(), 
            depth=self.stockfish_depth
        )
        
        # Calculate time taken
        elapsed = time.time() - start_time
        if color == 'white':
            self.white_move_times.append(elapsed)
        else:
            self.black_move_times.append(elapsed)
        
        if not move_uci:
            # Get a random legal move as fallback
            legal_moves = list(self.board.board.legal_moves)
            if legal_moves:
                move = random.choice(legal_moves)
                print(f"Stockfish couldn't find a move. Using random move: {move}")
            else:
                return False, "Stockfish couldn't find a move and no legal moves available", None
        else:
            # Parse move
            try:
                move = chess.Move.from_uci(move_uci)
            except ValueError:
                # Get a random legal move as fallback
                legal_moves = list(self.board.board.legal_moves)
                if legal_moves:
                    move = random.choice(legal_moves)
                    print(f"Invalid move from Stockfish: {move_uci}. Using random move: {move}")
                else:
                    return False, f"Invalid move from Stockfish: {move_uci} and no legal moves available", None
        
        # Verify move legality
        if move not in self.board.board.legal_moves:
            # Get a random legal move as fallback
            legal_moves = list(self.board.board.legal_moves)
            if legal_moves:
                move = random.choice(legal_moves)
                print(f"Stockfish suggested illegal move. Using random move: {move}")
            else:
                return False, f"Stockfish suggested illegal move and no legal moves available", None
        
        try:
            # Get SAN representation (catch any potential errors) 
            move_san = self.board.board.san(move)
        except Exception as e:
            # If there's an error with san(), try to continue
            print(f"\033[1;33m⚠ Warning: Error getting SAN notation - {e}\033[0m")
            move_san = move.uci()  # Fall back to UCI notation
        
        # Store the move in sequence
        self.move_sequence.append(move.uci())
        self.move_evaluations.append(stockfish_eval or eval_score)
        
        # Make the move
        self.board.make_move(move)
        
        # Store current position and evaluation in history
        self.position_evals.append((self.board.get_fen(), stockfish_eval or eval_score))
        self.game_history.append(self.board.get_fen())
        
        move_data = {
            'move': move,
            'move_san': move_san,
            'time': elapsed,
            'eval': stockfish_eval or eval_score,
            'depth': self.stockfish_depth
        }
        
        return True, f"{move_san} ({format_time(elapsed * 1000)})", move_data
    
    def play_game(self, max_moves=200, verbose=True):
        """
        Play a complete game automatically.
        
        Args:
            max_moves: Maximum number of moves to play
            verbose: Whether to print move information
            
        Returns:
            Dictionary with game result information
        """
        self.start_time = time.time()
        self.reset()
        
        # Set player names
        self.board.set_player_names("White", "Black")
        
        # Store initial position
        self.game_history.append(self.board.get_fen())
        
        # Main game loop
        move_count = 0
        while not self.board.is_game_over() and move_count < max_moves:
            # Determine which side moves
            current_turn = self.board.get_turn()
            
            if current_turn == 'white':
                # White to move
                if self.using_stockfish and self.stockfish_playing_as == 'white':
                    success, message, move_data = self.stockfish_move('white')
                    if verbose and success:
                        color_message = message.replace(move_data['move_san'], f"\033[1;36m{move_data['move_san']}\033[0m")
                        print(f"\033[1;33mStockfish (White):\033[0m {color_message}")
                else:
                    success, message, move_data = self.engine_move('white')
                    if verbose and success:
                        color_message = message.replace(move_data['move_san'], f"\033[1;32m{move_data['move_san']}\033[0m")
                        print(f"\033[1;37mEngine (White):\033[0m {color_message}")
            else:
                # Black to move
                if self.using_stockfish and self.stockfish_playing_as == 'black':
                    success, message, move_data = self.stockfish_move('black')
                    if verbose and success:
                        color_message = message.replace(move_data['move_san'], f"\033[1;36m{move_data['move_san']}\033[0m")
                        print(f"\033[1;33mStockfish (Black):\033[0m {color_message}")
                else:
                    success, message, move_data = self.engine_move('black')
                    if verbose and success:
                        color_message = message.replace(move_data['move_san'], f"\033[1;34m{move_data['move_san']}\033[0m")
                        print(f"\033[1;30mEngine (Black):\033[0m {color_message}")
            
            if not success:
                print(f"Error: {message}")
                break
            
            move_count += 1
            
            # Check for draw conditions (repetition or 50-move rule)
            if move_count >= 10 and (self.board.board.is_repetition(3) or self.board.board.is_fifty_moves()):
                self.board.game_result = "1/2-1/2"
                break
        
        # Save game data
        self.save_game_data()
        
        # Return game result
        return self.get_game_result()
    
    def save_game_data(self):
        """Save current game data to the dataset."""
        # Get game result
        result = self.board.get_result()
        
        # Convert to standard chess notation if needed
        if result is None or result == "None":
            # Default to draw if no result
            result = "1/2-1/2"
        elif result == "draw":
            result = "1/2-1/2"
        # Make sure we're using standard notation (1-0, 0-1, 1/2-1/2)
        elif result not in ["1-0", "0-1", "1/2-1/2"]:
            # Default to draw for any other values
            result = "1/2-1/2"
        
        # Store game data
        game_data = {
            'move_sequence': self.move_sequence,
            'result': result,
            'evaluations': self.move_evaluations,
            'using_stockfish': self.using_stockfish,
            'stockfish_playing_as': self.stockfish_playing_as,
            'stockfish_depth': self.stockfish_depth if self.using_stockfish else None,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        self.games_data.append(game_data)
    
    def save_dataset(self, filename="dataset.csv"):
        """
        Save the dataset to CSV, appending to existing file if it exists.
        
        Args:
            filename: Filename to save to (default is dataset.csv)
            
        Returns:
            Path to the saved file
        """
        # Create directory if it doesn't exist
        os.makedirs("dataset", exist_ok=True)
        
        # Create full path
        path = os.path.join("dataset", filename)
        
        # Create DataFrame-compatible dictionary
        data_dict = {
            'move_sequence': [],
            'result': [],
            'evaluations': [],
            'using_stockfish': [],
            'stockfish_playing_as': [],
            'stockfish_depth': [],
            'timestamp': []
        }
        
        # Only process games that haven't been saved yet
        # Track which games have been processed to avoid duplication
        if not hasattr(self, 'saved_game_indices'):
            self.saved_game_indices = set()
            
        # Find games that haven't been saved yet
        new_games = []
        for i, game_data in enumerate(self.games_data):
            if i not in self.saved_game_indices:
                new_games.append(game_data)
                self.saved_game_indices.add(i)
        
        if not new_games:
            print(f"No new games to save to dataset at {path}")
            return path
            
        for game_data in new_games:
            # Standardize result format (use chess notation)
            result = game_data['result']
            if result is None or result == "None":
                # Default to draw if result is missing
                result = "1/2-1/2"
            elif result == "draw":
                result = "1/2-1/2"
                
            data_dict['move_sequence'].append(','.join(game_data['move_sequence']))
            data_dict['result'].append(result)
            data_dict['evaluations'].append(','.join(map(str, game_data['evaluations'])))
            data_dict['using_stockfish'].append(game_data['using_stockfish'])
            data_dict['stockfish_playing_as'].append(game_data['stockfish_playing_as'])
            data_dict['stockfish_depth'].append(game_data['stockfish_depth'])
            data_dict['timestamp'].append(game_data['timestamp'])
        
        # Create DataFrame for new data
        new_df = pd.DataFrame(data_dict)
        
        # Check if file exists to append or create new
        if os.path.exists(path):
            # Read existing file
            try:
                existing_df = pd.read_csv(path)
                # Append new data
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                # Save combined data
                combined_df.to_csv(path, index=False)
                print(f"Appended {len(new_df)} games to existing dataset at {path}")
            except Exception as e:
                print(f"Error appending to existing file: {e}")
                # Save just new data if there was an error
                new_df.to_csv(path, index=False)
                print(f"Created new dataset with {len(new_df)} games at {path}")
        else:
            # File doesn't exist, create new one
            new_df.to_csv(path, index=False)
            print(f"Created new dataset with {len(new_df)} games at {path}")
        
        return path
    
    def get_game_result(self):
        """
        Get the game result.
        
        Returns:
            Dictionary with game result information
        """
        result = self.board.get_result()
        winner = self.board.get_winner()
        
        # Calculate game duration
        duration = time.time() - (self.start_time or time.time())
        
        # Calculate average move times
        avg_white_time = sum(self.white_move_times) / max(len(self.white_move_times), 1)
        avg_black_time = sum(self.black_move_times) / max(len(self.black_move_times), 1)
        
        return {
            'result': result,
            'winner': winner,
            'is_draw': winner == 'draw',
            'total_moves': len(self.board.move_history),
            'white_moves': len(self.white_move_times),
            'black_moves': len(self.black_move_times),
            'duration': duration,
            'avg_white_time': avg_white_time,
            'avg_black_time': avg_black_time,
            'pgn': self.board.get_pgn(),
            'using_stockfish': self.using_stockfish,
            'stockfish_playing_as': self.stockfish_playing_as,
            'stockfish_depth': self.stockfish_depth if self.using_stockfish else None
        }
    
    def get_pgn(self):
        """Get PGN representation of the game."""
        return self.board.get_pgn()
    
    def save_pgn(self, filename=None):
        """
        Save the game in PGN format.
        
        Args:
            filename: Filename to save to (None for auto-generated)
            
        Returns:
            Path to the saved file
        """
        if filename is None:
            timestamp = int(time.time())
            filename = f"game_{timestamp}.pgn"
        
        # Create directory if it doesn't exist
        os.makedirs("pgns", exist_ok=True)
        
        # Create full path
        path = os.path.join("pgns", filename)
        
        # Save PGN
        with open(path, 'w') as f:
            f.write(self.get_pgn())
        
        return path

def generate_dataset(num_games=1000, stockfish_chance=0.01, save_interval=100, verbose=False):
    """
    Generate a dataset of self-play games with occasional Stockfish games.
    
    Args:
        num_games: Number of games to generate
        stockfish_chance: Probability (0-1) that a game will use Stockfish
        save_interval: Save dataset after this many games
        verbose: Whether to print move information
        
    Returns:
        Path to the saved dataset
    """
    # Create engines with different parameters to ensure variety in play
    engine_white = NNUEEngine(name="Pearl White", depth=3, time_limit_ms=10000)  # White with slightly lower depth
    engine_black = NNUEEngine(name="Pearl Black", depth=4, time_limit_ms=8000)   # Black with slightly less time
    
    # Add randomness to evaluation to encourage varied play (avoid draws)
    # We'll use a function that monkeypatches the evaluate_position function
    from src.engine.score import evaluate_position as original_evaluate
    
    def add_randomness_to_evaluation():
        """Add randomness to evaluations to encourage more decisive games"""
        from src.engine import score
        
        # Store the original function
        original_eval = score.evaluate_position
        
        # Define a new function with randomness
        def evaluate_with_randomness(board):
            # Get the original evaluation
            eval_score = original_eval(board)
            
            # Add small random factor (±15 centipawns)
            import random
            randomness = random.uniform(-15, 15)
            
            # Return the slightly randomized score
            return eval_score + randomness
        
        # Replace the original function
        score.evaluate_position = evaluate_with_randomness
        print("\033[1;35m⚄ Added evaluation randomness to encourage more decisive games\033[0m")
    
    # Apply the randomness patch
    add_randomness_to_evaluation()
    
    # Create game manager
    game_manager = EngineVsEngine(
        engine_white=engine_white,
        engine_black=engine_black,
        stockfish_chance=stockfish_chance,
        stockfish_depths=[5, 8, 10, 12, 15, 18]
    )
    
    print(f"\033[1;36mGenerating dataset with \033[1;33m{num_games}\033[1;36m games...\033[0m")
    print(f"\033[1;36mStockfish chance: \033[1;33m{stockfish_chance * 100:.1f}%\033[0m")
    
    # Statistics
    wins_white = 0
    wins_black = 0
    draws = 0
    stockfish_games = 0
    stockfish_wins = 0
    skipped_games = 0
    completed_games = 0
    
    # Play games until we have the requested number of complete games
    i = 0
    while completed_games < num_games:
        i += 1
        # Calculate progress percentage
        progress = (completed_games+1) / num_games * 100
        
        # Print game number with progress bar
        print(f"\n\033[1;36m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\033[0m")
        print(f"\033[1;33mGame {completed_games+1}/{num_games}\033[0m \033[1;37m({progress:.1f}%)\033[0m")
        
        try:
            # Attempt to play a game with timeout protection
            result = game_manager.play_game(verbose=verbose)
            
            # Get move count and check for checkmate
            move_count = len(result.get('move_sequence', []))
            
            # Check for checkmate
            is_checkmate = False
            
            try:
                # First, directly check if the game ended in checkmate
                if game_manager.board.is_checkmate():
                    is_checkmate = True
                    print(f"\033[1;32m♔ Game ended in checkmate!\033[0m")
                
                # If not, check the move history in SAN notation for # symbol
                elif hasattr(game_manager.board, 'get_move_history'):
                    # This gets the moves in SAN notation
                    san_moves = game_manager.board.get_move_history(use_san=True)
                    if san_moves and '#' in san_moves[-1]:
                        is_checkmate = True
                        print(f"\033[1;32m♔ Checkmate detected in move: {san_moves[-1]}\033[0m")
                
                # As a final check, see if the game result indicates a win (not a draw)
                elif game_manager.board.get_result() in ['1-0', '0-1']:
                    is_checkmate = True
                    print(f"\033[1;32m♔ Game ended in a win, likely checkmate!\033[0m")
            except Exception as e:
                # If there's any error checking for checkmate, log it but don't fail
                print(f"\033[1;33m⚠ Error checking for checkmate: {e}\033[0m")
                is_checkmate = False
            
            # Only skip games if they have too few moves AND are not checkmate games AND involve Stockfish
            if move_count < 10 and not is_checkmate and game_manager.using_stockfish:
                # Skip games with too few moves when Stockfish was involved - likely API failure
                print(f"\033[1;33m⚠ Game skipped: Too few moves ({move_count}) with Stockfish - likely API failure\033[0m")
                skipped_games += 1
                continue
                
            # If it's a checkmate in few moves, display a special message
            if is_checkmate and move_count < 10:
                # Get the last move in SAN notation safely
                last_move_san = "?"
                try:
                    san_moves = game_manager.board.get_move_history(use_san=True)
                    if san_moves:
                        last_move_san = san_moves[-1]
                except Exception as e:
                    print(f"\033[1;33m⚠ Error getting last move: {e}\033[0m")
                
                print(f"\033[1;32m♔ Checkmate in {move_count} moves! Last move: {last_move_san}\033[0m")
                print(f"\033[1;32m♔ This is a useful training example - keeping this game\033[0m")
                
            # Count this as a completed game
            completed_games += 1
            
            # Update statistics based on the standard chess result notation
            game_result = result['result']
            if game_result == "1-0":
                wins_white += 1
                result['winner'] = 'white'
            elif game_result == "0-1":
                wins_black += 1
                result['winner'] = 'black'
            else:  # "1/2-1/2" or other
                draws += 1
                result['winner'] = 'draw'
                
            if result['using_stockfish']:
                stockfish_games += 1
                if ((result['stockfish_playing_as'] == 'white' and result['winner'] == 'white') or
                    (result['stockfish_playing_as'] == 'black' and result['winner'] == 'black')):
                    stockfish_wins += 1
            
            # Save PGN
            pgn_path = game_manager.save_pgn()
            
            # Print result with color
            result_color = "\033[1;32m" if result['result'] == "1-0" else "\033[1;34m" if result['result'] == "0-1" else "\033[1;33m"
            winner_color = "\033[1;32m" if result['winner'] == "white" else "\033[1;34m" if result['winner'] == "black" else "\033[1;33m"
            
            print(f"\033[1mResult:\033[0m {result_color}{result['result']}\033[0m")
            print(f"\033[1mWinner:\033[0m {winner_color}{result['winner']}\033[0m")
            print(f"\033[1mMoves:\033[0m \033[1;36m{move_count}\033[0m")
            print(f"\033[1mPGN saved to:\033[0m \033[1;36m{pgn_path}\033[0m")
            
        except Exception as e:
            # If any exception occurs during a game, log it and continue with the next game
            print(f"\033[1;31m✗ Error during game: {str(e)}\033[0m")
            
            # Get a stack trace for debugging
            import traceback
            print(f"\033[1;33m⚠ Error details: {traceback.format_exc()}\033[0m")
            
            # Reset engines and game manager to clean state
            try:
                game_manager.reset()
                game_manager.engine_white.reset()
                game_manager.engine_black.reset()
                print(f"\033[1;33m⚠ Successfully reset engines and game state\033[0m")
            except:
                print(f"\033[1;31m✗ Failed to reset engines, may need to restart\033[0m")
                
            print(f"\033[1;33m⚠ Skipping problematic game and continuing...\033[0m")
            skipped_games += 1
            continue
        
        # Save dataset at intervals
        if completed_games % save_interval == 0 and completed_games > 0:
            dataset_path = game_manager.save_dataset()
            print(f"\n\033[1;36mIntermediate dataset saved after {completed_games} completed games\033[0m")
            print(f"\033[1;35m━━━━━━━━━━━━━━━━━━━ Statistics ━━━━━━━━━━━━━━━━━━━\033[0m")
            print(f"\033[1;37mWhite wins:\033[0m \033[1;32m{wins_white}\033[0m (\033[1;33m{wins_white/completed_games*100:.1f}%\033[0m)")
            print(f"\033[1;37mBlack wins:\033[0m \033[1;34m{wins_black}\033[0m (\033[1;33m{wins_black/completed_games*100:.1f}%\033[0m)")
            print(f"\033[1;37mDraws:     \033[0m \033[1;33m{draws}\033[0m (\033[1;33m{draws/completed_games*100:.1f}%\033[0m)")
            print(f"\033[1;37mStockfish: \033[0m \033[1;36m{stockfish_games}\033[0m (\033[1;33m{stockfish_games/completed_games*100:.1f}%\033[0m)")
            print(f"\033[1;37mSkipped:   \033[0m \033[1;31m{skipped_games}\033[0m")
            if stockfish_games > 0:
                print(f"\033[1;37mStockfish win rate:\033[0m \033[1;33m{stockfish_wins/stockfish_games*100:.1f}%\033[0m")
    
    # Save final dataset
    dataset_path = game_manager.save_dataset()
    
    # Calculate total games attempted
    total_attempted = completed_games + skipped_games
    
    print(f"\n\033[1;32m✓ Dataset generation complete!\033[0m")
    print(f"\033[1;36mDataset saved to: \033[1;33m{dataset_path}\033[0m")
    
    print(f"\n\033[1;35m━━━━━━━━━━━━━━━━━━━ Final Statistics ━━━━━━━━━━━━━━━━━━━\033[0m")
    print(f"\033[1;37mGames completed:   \033[0m \033[1;32m{completed_games}\033[0m")
    print(f"\033[1;37mGames skipped:     \033[0m \033[1;31m{skipped_games}\033[0m (\033[1;33m{skipped_games/total_attempted*100:.1f}%\033[0m)")
    print(f"\033[1;37mWhite wins:        \033[0m \033[1;32m{wins_white}\033[0m (\033[1;33m{wins_white/completed_games*100:.1f}%\033[0m)")
    print(f"\033[1;37mBlack wins:        \033[0m \033[1;34m{wins_black}\033[0m (\033[1;33m{wins_black/completed_games*100:.1f}%\033[0m)")
    print(f"\033[1;37mDraws:             \033[0m \033[1;33m{draws}\033[0m (\033[1;33m{draws/completed_games*100:.1f}%\033[0m)")
    print(f"\033[1;37mStockfish games:   \033[0m \033[1;36m{stockfish_games}\033[0m (\033[1;33m{stockfish_games/completed_games*100:.1f}%\033[0m)")
    if stockfish_games > 0:
        print(f"\033[1;37mStockfish win rate: \033[0m \033[1;33m{stockfish_wins/stockfish_games*100:.1f}%\033[0m")
    print(f"\033[1;35m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\033[0m")
    
    return dataset_path

if __name__ == "__main__":
    # Parse command-line arguments if any
    if len(sys.argv) > 1:
        num_games = int(sys.argv[1])
    else:
        num_games = 100
        
    if len(sys.argv) > 2:
        stockfish_chance = float(sys.argv[2])
    else:
        stockfish_chance = 0.01
        
    if len(sys.argv) > 3:
        save_interval = int(sys.argv[3])
    else:
        save_interval = 10
        
    if len(sys.argv) > 4:
        verbose = sys.argv[4].lower() == 'true'
    else:
        verbose = False
    
    generate_dataset(num_games, stockfish_chance, save_interval, verbose)