import os
import sys
import time
import chess
import re
from blessed import Terminal
from colorama import Fore, Back, Style, init
import random

from src.game.board import ChessBoard
from src.engine.engine import NNUEEngine
from src.utils.api.stockfish import StockfishAPI, MockStockfishAPI
from src.engine.utils import format_time, centipawn_to_win_probability

# Initialize colorama
init()

class ChessUI:
    """
    Text-based user interface for the chess engine.
    """
    def __init__(self, use_unicode=True, use_color=True, use_enhanced_model=True, use_xl_model=False):
        """
        Initialize the chess UI.
        
        Args:
            use_unicode: Use Unicode characters for board display
            use_color: Use color for board display
            use_enhanced_model: Whether to use the enhanced model architecture (backward compatibility)
            use_xl_model: Whether to use the XL model architecture (backward compatibility)
        """
        self.term = Terminal()
        self.board = ChessBoard()
        
        # Convert old parameters to model_type for new system
        if use_enhanced_model and use_xl_model:
            model_type = "pearlxl"
            model_name = "PearlXL"
        elif use_enhanced_model:
            model_type = "pearl"
            model_name = "Pearl"
        else:
            model_type = "standard"
            model_name = "Standard"
            
        print(f"Loading {model_name} model...")
        
        # Initialize the engine with the selected model type
        self.engine = NNUEEngine(
            model_type=model_type
        )
        
        # Try to initialize Stockfish API, fall back to mock if unavailable
        try:
            self.stockfish = StockfishAPI()
            # Test connection
            _, _ = self.stockfish.get_best_move(chess.STARTING_FEN, depth=1)
        except:
            print("Could not connect to Stockfish API, using mock implementation")
            self.stockfish = MockStockfishAPI()
        
        self.use_unicode = use_unicode
        self.use_color = use_color
        self.current_menu = "main"
        self.last_command = None
        self.game_mode = None
        self.engine_depth = 5
        self.engine_time = 1000  # ms
        self.player_color = chess.WHITE
        self.white_name = "Player"
        self.black_name = "Engine"
        self.show_analysis = True
        self.show_hints = False
        self.last_engine_score = 0
        self.game_stats = {
            "moves": 0,
            "start_time": 0,
            "player_wins": 0,
            "engine_wins": 0,
            "draws": 0
        }
    
    def clear_screen(self):
        """Clear the terminal screen."""
        print(self.term.clear())
    
    def wait_for_key(self, prompt="Press any key to continue..."):
        """
        Wait for user to press a key.
        
        Args:
            prompt: Message to display
        """
        print(prompt)
        with self.term.cbreak():
            self.term.inkey()
    
    def print_header(self, title):
        """
        Print a header with the given title.
        
        Args:
            title: Header title
        """
        width = self.term.width
        print(Fore.CYAN + "=" * width + Style.RESET_ALL)
        print(Fore.CYAN + title.center(width) + Style.RESET_ALL)
        print(Fore.CYAN + "=" * width + Style.RESET_ALL)
    
    def print_board(self):
        """Print the current chess board."""
        # Show board from player's perspective
        perspective = self.player_color
        
        if self.use_unicode and self.use_color:
            print(self.board.to_unicode(perspective=perspective))
        else:
            print(self.board.to_ascii(perspective=perspective))
        
        # Show move history
        moves = self.board.get_move_history()
        if moves:
            move_list = []
            for i, move in enumerate(moves):
                if i % 2 == 0:
                    move_list.append(f"{i//2+1}. {move}")
                else:
                    move_list[-1] += f" {move}"
            
            print("\nMove history:")
            # Split into chunks of about 60 chars
            history_line = ""
            for move in move_list:
                if len(history_line) + len(move) + 1 > 60:
                    print(history_line)
                    history_line = move
                else:
                    if history_line:
                        history_line += " " + move
                    else:
                        history_line = move
            
            if history_line:
                print(history_line)
        
        # Show analysis if enabled
        if self.show_analysis and not self.board.is_game_over():
            self._show_analysis()
    
    def _show_analysis(self):
        """Show analysis of the current position."""
        # Get evaluation from engine
        eval_score = self.engine.evaluate()
        self.last_engine_score = eval_score
        
        # Get the current board turn and compute win probability from appropriate perspective
        current_turn = self.board.board.turn  # chess.WHITE or chess.BLACK
        
        # Convert to win probability - for the side to move
        win_prob = centipawn_to_win_probability(eval_score, perspective=current_turn)
        
        # Determine who is better
        if eval_score > 50:
            advantage = "White is better"
        elif eval_score < -50:
            advantage = "Black is better"
        else:
            advantage = "Equal position"
        
        # Format evaluation
        if abs(eval_score) >= 1000:
            # Format as mate
            moves_to_mate = (30000 - abs(eval_score)) // 100
            eval_str = f"M{moves_to_mate}" if eval_score > 0 else f"-M{moves_to_mate}"
        else:
            # Format as centipawns
            eval_str = f"{eval_score/100:+.2f}"
        
        # Clarify which side the win probability is for
        side_to_move = "White" if current_turn == chess.WHITE else "Black"
        print(f"\nAnalysis: {advantage} (Score: {eval_str}, {side_to_move} win probability: {win_prob:.1%})")
        
        # Show best move if hints are enabled
        if self.show_hints:
            best_move = self.engine.get_best_move(time_limit_ms=500)
            if best_move:
                print(f"Best move: {self.board.board.san(best_move)}")
    
    def print_menu(self, options, title="Menu", prompt="Enter your choice: "):
        """
        Print a menu with options and get user choice.
        
        Args:
            options: Dictionary of option keys and descriptions
            title: Menu title
            prompt: Input prompt
            
        Returns:
            User's choice
        """
        self.print_header(title)
        
        for key, desc in options.items():
            print(f"{key}. {desc}")
        
        print()
        return input(prompt)
    
    def main_menu(self):
        """Display the main menu and handle user choice."""
        options = {
            "1": "Human vs Engine",
            "2": "Engine vs Stockfish (automated)",
            "3": "Analysis mode",
            "4": "Settings",
            "5": "Exit"
        }
        
        choice = self.print_menu(options, "NNUE Chess Engine - Main Menu")
        
        if choice == "1":
            self.human_vs_engine_menu()
        elif choice == "2":
            self.engine_vs_stockfish_menu()
        elif choice == "3":
            self.analysis_mode()
        elif choice == "4":
            self.settings_menu()
        elif choice == "5":
            print("Goodbye!")
            sys.exit()
        else:
            print("Invalid choice. Please try again.")
            time.sleep(1)
    
    def human_vs_engine_menu(self):
        """Set up a game between human and engine."""
        self.game_mode = "human_vs_engine"
        
        # Ask for color
        options = {
            "1": "Play as White",
            "2": "Play as Black",
            "3": "Random",
            "4": "Back to main menu"
        }
        
        choice = self.print_menu(options, "Human vs Engine - Choose your color")
        
        if choice == "1":
            self.player_color = chess.WHITE
            self.white_name = "Player"
            self.black_name = "Engine"
        elif choice == "2":
            self.player_color = chess.BLACK
            self.white_name = "Engine"
            self.black_name = "Player"
        elif choice == "3":
            self.player_color = random.choice([chess.WHITE, chess.BLACK])
            if self.player_color == chess.WHITE:
                self.white_name = "Player"
                self.black_name = "Engine"
            else:
                self.white_name = "Engine"
                self.black_name = "Player"
        elif choice == "4":
            return
        else:
            print("Invalid choice. Please try again.")
            time.sleep(1)
            return self.human_vs_engine_menu()
        
        # Set up difficulty
        options = {
            "1": "Easy (Depth 3)",
            "2": "Medium (Depth 5)",
            "3": "Hard (Depth 7)",
            "4": "Custom settings",
            "5": "Back to main menu"
        }
        
        choice = self.print_menu(options, "Human vs Engine - Choose difficulty")
        
        if choice == "1":
            self.engine_depth = 3
            self.engine_time = 500
        elif choice == "2":
            self.engine_depth = 5
            self.engine_time = 1000
        elif choice == "3":
            self.engine_depth = 7
            self.engine_time = 2000
        elif choice == "4":
            # Custom settings
            try:
                depth = int(input("Enter search depth (1-15): "))
                self.engine_depth = max(1, min(15, depth))
                
                time_ms = int(input("Enter time limit in milliseconds (100-10000): "))
                self.engine_time = max(100, min(10000, time_ms))
            except ValueError:
                print("Invalid input. Using default settings.")
                self.engine_depth = 5
                self.engine_time = 1000
        elif choice == "5":
            return
        else:
            print("Invalid choice. Using default settings.")
            self.engine_depth = 5
            self.engine_time = 1000
        
        # Start game
        self.start_game()
    
    def engine_vs_stockfish_menu(self):
        """Set up a game between engine and Stockfish API."""
        self.game_mode = "engine_vs_stockfish"
        
        # Ask for engine color
        options = {
            "1": "Engine plays White",
            "2": "Engine plays Black",
            "3": "Random",
            "4": "Back to main menu"
        }
        
        choice = self.print_menu(options, "Engine vs Stockfish (Automated) - Choose engine color")
        
        if choice == "1":
            self.player_color = chess.WHITE  # Engine plays as player's color
            self.white_name = "Engine"
            self.black_name = "Stockfish"
        elif choice == "2":
            self.player_color = chess.BLACK  # Engine plays as player's color
            self.white_name = "Stockfish"
            self.black_name = "Engine"
        elif choice == "3":
            self.player_color = random.choice([chess.WHITE, chess.BLACK])
            if self.player_color == chess.WHITE:
                self.white_name = "Engine"
                self.black_name = "Stockfish"
            else:
                self.white_name = "Stockfish"
                self.black_name = "Engine"
        elif choice == "4":
            return
        else:
            print("Invalid choice. Please try again.")
            time.sleep(1)
            return self.engine_vs_stockfish_menu()
        
        # Set up difficulty
        options = {
            "1": "Quick game (Engine: Depth 5, Stockfish: Depth 5)",
            "2": "Balanced game (Engine: Depth 7, Stockfish: Depth 7)",
            "3": "Challenge (Engine: Depth 5, Stockfish: Depth 10)",
            "4": "Custom settings",
            "5": "Back to main menu"
        }
        
        choice = self.print_menu(options, "Engine vs Stockfish - Choose settings")
        
        if choice == "1":
            self.engine_depth = 5
            self.engine_time = 1000
            stockfish_depth = 5
        elif choice == "2":
            self.engine_depth = 7
            self.engine_time = 2000
            stockfish_depth = 7
        elif choice == "3":
            self.engine_depth = 5
            self.engine_time = 1000
            stockfish_depth = 10
        elif choice == "4":
            # Custom settings
            try:
                self.engine_depth = int(input("Enter engine search depth (1-15): "))
                self.engine_depth = max(1, min(15, self.engine_depth))
                
                self.engine_time = int(input("Enter engine time limit in milliseconds (100-10000): "))
                self.engine_time = max(100, min(10000, self.engine_time))
                
                stockfish_depth = int(input("Enter Stockfish search depth (1-20): "))
                stockfish_depth = max(1, min(20, stockfish_depth))
            except ValueError:
                print("Invalid input. Using default settings.")
                self.engine_depth = 5
                self.engine_time = 1000
                stockfish_depth = 5
        elif choice == "5":
            return
        else:
            print("Invalid choice. Using default settings.")
            self.engine_depth = 5
            self.engine_time = 1000
            stockfish_depth = 5
        
        # Start automated game
        print("\nStarting automated game between Engine and Stockfish.")
        print("The engines will play against each other without user input.")
        print("Press any key during the game to pause.")
        time.sleep(2)  # Give user time to read the message
        
        # Start the game
        self.start_game(stockfish_depth=stockfish_depth, auto_play=True)
    
    def analysis_mode(self):
        """Enter analysis mode."""
        self.game_mode = "analysis"
        self.show_analysis = True
        
        # Reset board
        self.board = ChessBoard()
        
        self.clear_screen()
        self.print_header("Analysis Mode")
        print("Enter moves or commands. Type 'help' for commands.")
        
        while True:
            self.print_board()
            
            command = input("\nEnter move or command: ").strip()
            
            if command.lower() in ['quit', 'exit', 'q']:
                break
            elif command.lower() == 'help':
                self._show_analysis_help()
            elif command.lower() == 'new':
                self.board = ChessBoard()
            elif command.lower() == 'undo':
                self.board.undo_move()
            elif command.lower().startswith('fen '):
                fen = command[4:].strip()
                if not self.board.set_position(fen):
                    print("Invalid FEN string.")
                    self.wait_for_key()
            elif command.lower().startswith('depth '):
                try:
                    depth = int(command[6:].strip())
                    self.engine_depth = max(1, min(15, depth))
                    print(f"Analysis depth set to {self.engine_depth}")
                except ValueError:
                    print("Invalid depth value.")
                self.wait_for_key()
            elif command.lower() == 'analyze':
                self._deep_analysis()
            elif command.lower() == 'flip':
                self.player_color = not self.player_color
            elif command.lower() == 'pgn':
                print("\nPGN of current game:")
                print(self.board.get_pgn())
                self.wait_for_key()
            else:
                # Try to make the move
                if not self.board.make_move(command):
                    print("Invalid move. Try again.")
                    self.wait_for_key()
    
    def _show_analysis_help(self):
        """Show help for analysis mode."""
        self.clear_screen()
        self.print_header("Analysis Mode Help")
        print("Available commands:")
        print("  help        - Show this help")
        print("  quit/exit/q - Return to main menu")
        print("  new         - Start a new game")
        print("  undo        - Undo the last move")
        print("  fen [fen]   - Set a position from FEN string")
        print("  depth [n]   - Set analysis depth")
        print("  analyze     - Perform deep analysis of current position")
        print("  flip        - Flip the board view")
        print("  pgn         - Show PGN of current game")
        print("\nTo make a move, enter it in UCI (e2e4) or SAN (e4) format.")
        self.wait_for_key()
    
    def _deep_analysis(self):
        """Perform deep analysis of the current position."""
        self.clear_screen()
        self.print_header("Deep Analysis")
        
        print("Analyzing position at higher depth...")
        
        # Increase depth for deep analysis
        analysis_depth = min(12, self.engine_depth + 4)
        
        # Set up search info
        from src.engine.search import SearchInfo
        info = SearchInfo()
        info.time_limit_ms = 10000  # 10 seconds
        
        # Perform search
        from src.engine.search import iterative_deepening_search
        start_time = time.time()
        best_move, score, depth = iterative_deepening_search(self.board.board, info)
        end_time = time.time()
        
        # Get top 3 moves
        top_moves = []
        board_copy = chess.Board(self.board.get_fen())
        
        for move in board_copy.legal_moves:
            board_copy.push(move)
            # Evaluate position after move
            eval_score = -self.engine.evaluate()  # Negated because we're looking from opponent's perspective
            board_copy.pop()
            
            top_moves.append((move, eval_score))
        
        # Sort by score (best first)
        top_moves.sort(key=lambda x: x[1], reverse=True)
        
        # Print results
        print(f"\nAnalysis completed in {format_time((end_time - start_time) * 1000)}")
        print(f"Depth reached: {depth}")
        print(f"Nodes searched: {info.nodes_searched}")
        print(f"NPS: {int(info.nodes_searched / max(end_time - start_time, 0.001))}")
        
        # Print best move
        print(f"\nBest move: {board_copy.san(best_move)} (Score: {score/100:+.2f})")
        
        # Print top 3 moves
        print("\nTop moves:")
        for i, (move, move_score) in enumerate(top_moves[:3]):
            print(f"{i+1}. {board_copy.san(move)} (Score: {move_score/100:+.2f})")
        
        self.wait_for_key()
    
    def settings_menu(self):
        """Display and handle settings menu."""
        while True:
            options = {
                "1": f"Display: {'Unicode' if self.use_unicode else 'ASCII'} mode",
                "2": f"Colors: {'Enabled' if self.use_color else 'Disabled'}",
                "3": f"Analysis: {'Shown' if self.show_analysis else 'Hidden'}",
                "4": f"Hints: {'Shown' if self.show_hints else 'Hidden'}",
                "5": "Engine settings",
                "6": "Back to main menu"
            }
            
            choice = self.print_menu(options, "Settings")
            
            if choice == "1":
                self.use_unicode = not self.use_unicode
            elif choice == "2":
                self.use_color = not self.use_color
            elif choice == "3":
                self.show_analysis = not self.show_analysis
            elif choice == "4":
                self.show_hints = not self.show_hints
            elif choice == "5":
                self._engine_settings()
            elif choice == "6":
                break
            else:
                print("Invalid choice. Please try again.")
                time.sleep(1)
    
    def _engine_settings(self):
        """Handle engine settings."""
        while True:
            options = {
                "1": f"Default depth: {self.engine_depth}",
                "2": f"Default time: {self.engine_time} ms",
                "3": "Load engine from file",
                "4": "Save engine to file",
                "5": "Back to settings menu"
            }
            
            choice = self.print_menu(options, "Engine Settings")
            
            if choice == "1":
                try:
                    depth = int(input("Enter new default depth (1-15): "))
                    self.engine_depth = max(1, min(15, depth))
                except ValueError:
                    print("Invalid input. Using previous setting.")
            elif choice == "2":
                try:
                    time_ms = int(input("Enter new default time in milliseconds (100-10000): "))
                    self.engine_time = max(100, min(10000, time_ms))
                except ValueError:
                    print("Invalid input. Using previous setting.")
            elif choice == "3":
                path = input("Enter weights file path (or leave empty for latest): ")
                if path.strip() == "":
                    path = None
                if self.engine.load_model(path):
                    print("Model loaded successfully.")
                else:
                    print("Failed to load model.")
                self.wait_for_key()
            elif choice == "4":
                name = input("Enter name for saved weights (or leave empty for timestamp): ")
                if name.strip() == "":
                    name = None
                path = self.engine.save_model(name)
                print(f"Model saved to {path}")
                self.wait_for_key()
            elif choice == "5":
                break
            else:
                print("Invalid choice. Please try again.")
                time.sleep(1)
    
    def start_game(self, stockfish_depth=None, auto_play=False):
        """
        Start a new game with the current settings.
        
        Args:
            stockfish_depth: Depth for Stockfish API
            auto_play: Whether to auto-play moves for both sides
        """
        # Reset board and engine
        self.board = ChessBoard()
        self.board.set_player_names(self.white_name, self.black_name)
        
        # Reset the engine and make sure it loads the latest model
        self.engine.reset()
        # Explicitly load the latest model weights before starting the game
        print(self.term.cyan("Loading latest model weights..."))
        if self.engine.load_model():
            weights_path = self.engine.get_current_weights_path()
            print(self.term.green(f"✓ Successfully loaded model: {weights_path}"))
        else:
            print(self.term.red("✗ Could not load latest model weights, using current model"))
        
        # Record start time
        self.game_stats["start_time"] = time.time()
        self.game_stats["moves"] = 0
        
        # Main game loop
        self.clear_screen()
        self.print_header(f"Game: {self.white_name} vs {self.black_name}")
        
        while not self.board.is_game_over():
            # Print board
            self.print_board()
            
            current_turn = self.board.get_turn()
            turn_color = chess.WHITE if current_turn == 'white' else chess.BLACK
            
            # Determine if it's a player or engine/stockfish move
            if auto_play:
                # In auto_play mode (Engine vs Stockfish), determine which engine to use
                if turn_color == self.player_color:
                    # Our engine move
                    print(f"\n{self.engine.name} is thinking...")
                    # Ensure engine's board state is synchronized with the game board
                    reference_fen = self.board.get_fen()
                    self.engine.verify_board_state(reference_fen)
                    move = self.engine.get_best_move(time_limit_ms=self.engine_time)
                else:
                    # Stockfish move
                    print("\nStockfish is thinking...")
                    try:
                        stockfish_move, _ = self.stockfish.get_best_move(
                            self.board.get_fen(), 
                            depth=stockfish_depth or self.engine_depth
                        )
                        
                        if stockfish_move:
                            try:
                                move = chess.Move.from_uci(stockfish_move)
                            except chess.InvalidMoveError as e:
                                print(f"Invalid move from Stockfish: {stockfish_move}")
                                print(f"Error: {e}")
                                # If in auto_play mode, use our engine as a fallback
                                if auto_play:
                                    print("Using our engine as a fallback...")
                                    self.engine.set_position(self.board.get_fen())
                                    move = self.engine.get_best_move(time_limit_ms=self.engine_time)
                                else:
                                    move = None
                        else:
                            print("Stockfish couldn't find a move.")
                            # If in auto_play mode, use our engine as a fallback
                            if auto_play:
                                print("Using our engine as a fallback...")
                                self.engine.set_position(self.board.get_fen())
                                move = self.engine.get_best_move(time_limit_ms=self.engine_time)
                            else:
                                move = None
                    except Exception as e:
                        print(f"Error getting move from Stockfish: {e}")
                        # If in auto_play mode, use our engine as a fallback
                        if auto_play:
                            print("Using our engine as a fallback...")
                            self.engine.set_position(self.board.get_fen())
                            move = self.engine.get_best_move(time_limit_ms=self.engine_time)
                        else:
                            move = None
            elif turn_color == self.player_color:
                # Human player's move
                move_str = input("\nEnter your move (or 'resign', 'help'): ")
                
                if move_str.lower() == 'resign':
                    self.board.game_result = "0-1" if turn_color == chess.WHITE else "1-0"
                    break
                elif move_str.lower() == 'help':
                    self._show_game_help()
                    continue
                elif move_str.lower() == 'undo':
                    if len(self.board.move_history) >= 2:
                        self.board.undo_move()  # Undo opponent's move
                        self.board.undo_move()  # Undo player's move
                    else:
                        print("Cannot undo moves.")
                        self.wait_for_key()
                    continue
                elif move_str.lower() == 'debug':
                    self._show_debug_info()
                    continue
                
                # Try to make the move
                if not self.board.make_move(move_str):
                    print("Invalid move. Try again.")
                    self.wait_for_key()
                    continue
                
                self.game_stats["moves"] += 1
                
                # Continue to next iteration
                continue
            else:
                # Engine move (when playing against human)
                print(f"\n{self.engine.name} is thinking...")
                # Ensure engine's board state is synchronized with the game board
                reference_fen = self.board.get_fen()
                self.engine.verify_board_state(reference_fen)
                move = self.engine.get_best_move(time_limit_ms=self.engine_time)
            
            # Make the move (for engine or stockfish)
            if 'move' in locals() and move:
                try:
                    san_move = self.board.board.san(move)
                    if self.board.make_move(move):
                        self.game_stats["moves"] += 1
                        print(f"Move: {san_move}")
                        time.sleep(0.5)  # Short pause to show the move
                    else:
                        print(f"Illegal move: {move}")
                        if auto_play:
                            self.wait_for_key()
                except Exception as e:
                    print(f"Error making move: {e}")
                    if auto_play:
                        self.wait_for_key()
            elif 'move' in locals():  # move is None but variable exists
                print("Engine couldn't find a move.")
                if auto_play:
                    self.wait_for_key()
                    break
            
            # Check if game is over
            if self.board.is_game_over():
                break
                
            # Auto-play mode: wait a little between moves
            if auto_play:
                time.sleep(1)
            
            self.clear_screen()
            self.print_header(f"Game: {self.white_name} vs {self.black_name}")
        
        # Game is over - show final position
        self.print_board()
        
        # Show game result
        result = self.board.get_result()
        winner = self.board.get_winner()
        
        print("\nGame over!")
        
        if winner == 'draw':
            print("Result: Draw")
            self.game_stats["draws"] += 1
        elif (winner == 'white' and self.white_name == "Player") or \
             (winner == 'black' and self.black_name == "Player"):
            print(f"Result: You win! ({result})")
            self.game_stats["player_wins"] += 1
        elif (winner == 'white' and self.white_name == "Engine") or \
             (winner == 'black' and self.black_name == "Engine"):
            print(f"Result: Engine wins! ({result})")
            self.game_stats["engine_wins"] += 1
        else:
            print(f"Result: {result}")
        
        # Show game statistics
        elapsed = time.time() - self.game_stats["start_time"]
        print(f"\nGame statistics:")
        print(f"Total moves: {self.game_stats['moves']}")
        print(f"Game duration: {format_time(elapsed * 1000)}")
        print(f"Average time per move: {format_time(elapsed * 1000 / max(self.game_stats['moves'], 1))}")
        
        # Option to save PGN
        save_pgn = input("\nSave game to PGN? (y/n): ")
        if save_pgn.lower() == 'y':
            try:
                filename = input("Enter filename (or leave empty for timestamp): ")
                if not filename:
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"game_{timestamp}.pgn"
                
                # Ensure the pgns directory exists
                os.makedirs("pgns", exist_ok=True)
                
                # Save to the pgns directory
                pgn_path = os.path.join("pgns", filename)
                with open(pgn_path, 'w') as f:
                    f.write(self.board.get_pgn())
                
                print(f"Game saved to {pgn_path}")
                
                # Ask if they want to automatically finetune based on this game
                if hasattr(self, 'engine') and self.engine and \
                   hasattr(self.engine, 'enable_learning') and self.engine.enable_learning:
                    finetune_prompt = input("\nAutomatically finetune engine based on this game? (y/n): ")
                    
                    if finetune_prompt.lower() == 'y':
                        # Get game outcome for feedback
                        winner = self.board.get_winner()
                        
                        # Determine if engine won or lost
                        engine_color = None
                        if self.white_name == "Engine":
                            engine_color = chess.WHITE
                        elif self.black_name == "Engine":
                            engine_color = chess.BLACK
                            
                        feedback = None
                        emphasis = 2.0  # Increased emphasis factor
                        
                        if engine_color is not None:
                            engine_won = (engine_color == chess.WHITE and winner == 'white') or \
                                        (engine_color == chess.BLACK and winner == 'black')
                            engine_lost = (engine_color == chess.WHITE and winner == 'black') or \
                                        (engine_color == chess.BLACK and winner == 'white')
                            
                            if engine_won:
                                # Engine won, reinforce good moves
                                feedback = {"result": "win", "emphasis": emphasis}
                                print(self.term.green("🎮 Finetuning engine to reinforce winning strategies..."))
                                print(self.term.yellow(f"📊 Using emphasis factor: {emphasis}x"))
                            elif engine_lost:
                                # Engine lost, learn from loss
                                feedback = {"result": "loss", "emphasis": emphasis}
                                print(self.term.magenta("🎮 Finetuning engine to avoid losing strategies..."))
                                print(self.term.yellow(f"📊 Using emphasis factor: {emphasis}x"))
                            elif winner == 'draw':
                                # Draw, less emphasis
                                print(self.term.cyan("🎮 Finetuning engine from drawn game..."))
                        
                        try:
                            # Import here to avoid circular imports
                            from src.engine.finetune import finetune_from_pgn
                            print(self.term.cyan("🚀 Starting advanced finetuning process (50 epochs)..."))
                            print(self.term.yellow("🧠 Using position memory and curriculum learning"))
                            
                            # Run finetuning with all our enhancements
                            # Get model type from the engine
                            model_type = getattr(self.engine, 'model_type', 'standard')
                            
                            finetune_from_pgn(
                                pgn_path, 
                                epochs=50, 
                                batch_size=32, 
                                feedback=feedback,
                                model_type=model_type,
                                use_memory=True
                            )
                            print(self.term.green("✅ Finetuning complete! Model has been updated to base.pt"))
                        except Exception as e:
                            print(f"Error during finetuning: {e}")
            except Exception as e:
                print(f"Error saving PGN: {e}")
        
        # Return to main menu
        self.wait_for_key()
    
    def _show_game_help(self):
        """Show help for game mode."""
        self.clear_screen()
        self.print_header("Game Help")
        print("Available commands:")
        print("  help   - Show this help")
        print("  resign - Resign the game")
        print("  undo   - Undo your last move and the engine's response")
        print("  debug  - Show debug information about the current board state")
        print("\nTo make a move, enter it in UCI (e2e4) or SAN (e4) format.")
        self.wait_for_key()
    
    def _show_debug_info(self):
        """Show debug information about the current board state."""
        self.clear_screen()
        self.print_header("Debug Information")
        
        # Show current board
        print("Current board state:")
        print(self.board.to_unicode())
        
        # Get FEN
        current_fen = self.board.get_fen()
        print(f"\nFEN: {current_fen}")
        
        # Show turn
        print(f"Turn: {'White' if self.board.board.turn == chess.WHITE else 'Black'}")
        
        # Show game status
        print(f"Check: {self.board.board.is_check()}")
        print(f"Checkmate: {self.board.board.is_checkmate()}")
        print(f"Stalemate: {self.board.board.is_stalemate()}")
        print(f"Game over: {self.board.board.is_game_over()}")
        
        # Show move history
        print("\nMove history:")
        for i, move in enumerate(self.board.get_move_history()):
            print(f"{i+1}. {move}")
        
        # Show legal moves
        print("\nLegal moves:")
        legal_moves = list(self.board.board.legal_moves)
        if legal_moves:
            for move in legal_moves:
                print(f"  {self.board.board.san(move)} ({move.uci()})")
        else:
            print("  No legal moves")
        
        # Show engine state
        print("\nEngine state:")
        print(f"Engine board FEN: {self.engine.board.fen()}")
        print(f"Engine turn: {'White' if self.engine.board.turn == chess.WHITE else 'Black'}")
        
        # Verify board consistency
        print("\nBoard consistency check:")
        if self.engine.board.fen() == current_fen:
            print("  Engine board matches game board ✓")
        else:
            print("  Engine board differs from game board ✗")
            print(f"  Game board: {current_fen}")
            print(f"  Engine board: {self.engine.board.fen()}")
            
            # Offer to fix
            fix = input("\nFix engine board state? (y/n): ")
            if fix.lower() == 'y':
                self.engine.set_position(current_fen)
                print("Engine board state synchronized with game board.")
        
        self.wait_for_key()
    
    def run(self):
        """Main UI loop."""
        while True:
            self.clear_screen()
            self.main_menu()

def select_model():
    """
    Display a model selection menu and return the selected model parameters.
    
    Returns:
        Tuple of (use_enhanced, use_xl)
    """
    from blessed import Terminal
    from colorama import Fore, Style
    
    term = Terminal()
    print(term.clear())
    
    print(Fore.CYAN + "=" * term.width + Style.RESET_ALL)
    print(Fore.CYAN + "Pearl Chess Engine - Select Neural Network Model".center(term.width) + Style.RESET_ALL)
    print(Fore.CYAN + "=" * term.width + Style.RESET_ALL)
    
    print("\nAvailable models:")
    print(Fore.GREEN + "1. " + Style.BRIGHT + "Standard Model" + Style.RESET_ALL + 
          " (260K parameters, fastest, least powerful)")
    print(Fore.YELLOW + "2. " + Style.BRIGHT + "Pearl Model" + Style.RESET_ALL + 
          " (8M parameters, balanced, enhanced evaluation)")
    print(Fore.MAGENTA + "3. " + Style.BRIGHT + "PearlXL Model" + Style.RESET_ALL + 
          " (16M parameters, slowest, most powerful)")
    
    # Import model utilities
    from src.engine.nnue.model_handler import initialize_default_models, list_available_models
    
    # Initialize default models if they don't exist
    initialize_default_models()
    
    # Check which models are available
    models = list_available_models()
    
    # Display model status
    print("\nModel status:")
    
    if models["pearl"]:
        print(Fore.GREEN + "✓ " + Style.RESET_ALL + f"Pearl model found ({os.path.basename(models['pearl'][0])})")
    else:
        print(Fore.YELLOW + "⚠ " + Style.RESET_ALL + "Pearl model will be created if selected")
    
    if models["pearlxl"]:
        print(Fore.GREEN + "✓ " + Style.RESET_ALL + f"PearlXL model found ({os.path.basename(models['pearlxl'][0])})")
    else:
        print(Fore.YELLOW + "⚠ " + Style.RESET_ALL + "PearlXL model will be created if selected")
    
    if models["standard"]:
        print(Fore.GREEN + "✓ " + Style.RESET_ALL + f"Standard model found ({os.path.basename(models['standard'][0])})")
    else:
        print(Fore.YELLOW + "⚠ " + Style.RESET_ALL + "Standard model will be created if selected")
    
    print("\nNote: Larger models provide better chess evaluation but require more time")
    print("for moves and training. First-time creation of a model may take a moment.")
    
    # Get user choice
    choice = None
    while choice not in ["1", "2", "3"]:
        choice = input("\nSelect model (1-3, default=2): ")
        if choice == "":
            choice = "2"  # Default to Pearl model
    
    # Convert choice to model type
    if choice == "1":
        # Standard model
        model_type = "standard"
        print(Fore.GREEN + "\nSelected: Standard Model" + Style.RESET_ALL)
    elif choice == "2":
        # Pearl model
        model_type = "pearl"
        print(Fore.YELLOW + "\nSelected: Pearl Model" + Style.RESET_ALL)
    else:
        # PearlXL model
        model_type = "pearlxl"
        print(Fore.MAGENTA + "\nSelected: PearlXL Model" + Style.RESET_ALL)
    
    # Wait for user confirmation
    print("\nInitializing model, please wait...")
    
    # For backward compatibility (return tuple format)
    use_enhanced = model_type != "standard"
    use_xl = model_type == "pearlxl"
    
    return use_enhanced, use_xl

def main():
    """Main entry point for the chess UI."""
    # First show the model selection menu
    use_enhanced, use_xl = select_model()
    
    # Create UI with the selected model type
    ui = ChessUI(use_enhanced_model=use_enhanced, use_xl_model=use_xl)
    ui.run()

if __name__ == "__main__":
    main()
