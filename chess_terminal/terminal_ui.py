import os
import sys
import time
import select
import termios
import tty
import traceback
from blessed import Terminal
import chess

from chess_engine import ChessGame
from renderer import BoardRenderer

class ChessUI:
    def __init__(self):
        self.term = Terminal()
        self.game = ChessGame()
        self.renderer = BoardRenderer(self.term)
        
        self.last_move = None
        self.command = ""
        self.command_error = None
        self.move_history = []
        
        self.running = True
        self.old_settings = None
        self.fallback_mode = False
        
        # Engine thinking time (in seconds)
        self.engine_thinking_time = 1.0
    
    def start(self):
        """Start the chess game UI."""
        try:
            # Set up terminal for raw input
            self.old_settings = termios.tcgetattr(sys.stdin)
            tty.setraw(sys.stdin.fileno())
            
            # Hide cursor and enable mouse reporting
            print(self.term.hide_cursor)
            
            # Show game mode selection screen
            self.show_game_mode_selection()
            
            # Main game loop
            self.game_loop()
            
        except Exception as e:
            # If we encounter an error, switch to fallback mode
            self.fallback_mode = True
            print(self.term.normal)
            print(f"Error initializing terminal: {e}")
            print("Switching to fallback mode...")
            self.fallback_game_loop()
        finally:
            # Restore terminal settings
            if self.old_settings:
                try:
                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
                except Exception:
                    pass
            
            # Show cursor
            print(self.term.normal_cursor)
    
    def show_game_mode_selection(self):
        """Show the game mode selection screen."""
        # Clear the screen
        print(self.term.clear)
        
        # Draw the selection screen
        title_y = 5
        print(self.term.move_xy(20, title_y) + 
              self.term.bold("Terminal Chess - Game Mode Selection"))
        
        print(self.term.move_xy(20, title_y + 2) + 
              "Select game mode:")
        
        print(self.term.move_xy(20, title_y + 4) + 
              self.term.bright_cyan("1. Human vs Human"))
        
        print(self.term.move_xy(20, title_y + 5) + 
              self.term.bright_yellow("2. Human vs Engine (default model)"))
        
        # Always show the option to select a model
        model_files = self._get_available_models()
        print(self.term.move_xy(20, title_y + 7) + 
              self.term.bright_yellow("3. Human vs Engine (select model)"))
        
        print(self.term.move_xy(20, title_y + 9) + 
              "Press 1, 2, or 3 to select mode...")
        
        # Wait for user input
        while True:
            if select.select([sys.stdin], [], [], 0)[0]:
                key = sys.stdin.read(1)
                
                if key == '1':
                    self.game.set_game_mode("human")
                    break
                elif key == '2':
                    self.game.set_game_mode("engine")
                    break
                elif key == '3':
                    self.game.set_game_mode("engine")
                    if model_files:
                        self._show_model_selection(model_files)
                    else:
                        # No models available, show a message
                        print(self.term.clear)
                        print(self.term.move_xy(20, title_y) + 
                              self.term.bold("No models available"))
                        print(self.term.move_xy(20, title_y + 2) + 
                              "No model files found in the saved_models directory.")
                        print(self.term.move_xy(20, title_y + 4) + 
                              "Press any key to continue with the default model...")
                        # Wait for a keypress
                        while not select.select([sys.stdin], [], [], 0)[0]:
                            time.sleep(0.01)
                        sys.stdin.read(1)
                    break
            
            time.sleep(0.01)
        
        # Clear the screen again
        print(self.term.clear)
    
    def _get_available_models(self):
        """Get a list of available model files."""
        # Get the project root directory (chess2)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        model_dir = os.path.join(project_root, "saved_models")
        
        if not os.path.exists(model_dir):
            return []
        
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.npz')]
        return sorted(model_files)
    
    def _parse_model_info(self, model_name):
        """Parse information from the model filename."""
        info = {}
        
        # Try to extract information from the filename
        try:
            # Check if it's a standard format like chess_g500_d2_i25_20230415_123456.npz
            parts = model_name.replace('.npz', '').split('_')
            
            for part in parts:
                if part.startswith('g') and part[1:].isdigit():
                    info['games'] = int(part[1:])
                elif part.startswith('d') and part[1:].isdigit():
                    info['depth'] = int(part[1:])
                elif part.startswith('i') and part[1:].isdigit():
                    info['save_interval'] = int(part[1:])
        
            # Try to get creation date from file stats
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            model_path = os.path.join(project_root, 'saved_models', model_name)
            if os.path.exists(model_path):
                creation_time = os.path.getctime(model_path)
                info['created'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(creation_time))
        except Exception:
            # If parsing fails, just return the filename
            pass
        
        return info
    
    def _show_model_selection(self, model_files):
        """Show a screen to select a specific model."""
        # Clear the screen
        print(self.term.clear)
        
        # Draw the selection screen
        title_y = 5
        print(self.term.move_xy(20, title_y) + 
              self.term.bold("Terminal Chess - Model Selection"))
        
        print(self.term.move_xy(20, title_y + 2) + 
              "Select a model to use:")
        
        # Display available models with additional information
        for i, model in enumerate(model_files):
            # Parse model info
            info = self._parse_model_info(model)
            
            # Base display with model number and name
            model_display = f"{i+1}. {model}"
            
            # Add additional info if available
            info_text = ""
            if 'games' in info:
                info_text += f" (Games: {info['games']}"
                if 'depth' in info:
                    info_text += f", Depth: {info['depth']}"
                info_text += ")"
            
            # Display the model info
            print(self.term.move_xy(20, title_y + 4 + i*2) + model_display)
            
            # Display creation date on the next line if available
            if 'created' in info:
                # Use normal text instead of dim to avoid terminal capability issues
                print(self.term.move_xy(22, title_y + 4 + i*2 + 1) + 
                      f"Created: {info['created']}")
        
        print(self.term.move_xy(20, title_y + 4 + len(model_files)*2 + 1) + 
              f"Enter 1-{len(model_files)} to select a model...")
        
        # Wait for user input
        while True:
            try:
                # Switch to normal terminal mode for input
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
                print(self.term.normal_cursor)
                
                choice = input()
                
                # Switch back to raw mode
                tty.setraw(sys.stdin.fileno())
                print(self.term.hide_cursor)
                
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(model_files):
                    selected_model = model_files[choice_idx]
                    script_dir = os.path.dirname(os.path.abspath(__file__))
                    project_root = os.path.dirname(script_dir)
                    model_path = os.path.join(project_root, 'saved_models', selected_model)
                    self.game.engine.model_path = model_path
                    
                    # Reinitialize the neural network with the selected model
                    try:
                        self.game.engine._init_neural_network()
                        print(self.term.clear)
                        print(self.term.move_xy(20, title_y) + 
                              self.term.bold_green(f"Selected model: {selected_model}"))
                        time.sleep(1)
                    except Exception as e:
                        print(self.term.clear)
                        print(self.term.move_xy(20, title_y) + 
                              self.term.bold_red(f"Error loading model: {e}"))
                        print(self.term.move_xy(20, title_y + 2) + 
                              "Press any key to continue with the default model...")
                        # Wait for a keypress
                        while not select.select([sys.stdin], [], [], 0)[0]:
                            time.sleep(0.01)
                        sys.stdin.read(1)
                    break
                else:
                    print(self.term.clear)
                    print(self.term.move_xy(20, title_y) + 
                          self.term.bold_red(f"Please enter a number between 1 and {len(model_files)}"))
                    time.sleep(1)
                    # Redisplay the selection screen
                    self._show_model_selection(model_files)
                    return
            except ValueError:
                print(self.term.clear)
                print(self.term.move_xy(20, title_y) + 
                      self.term.bold_red("Please enter a valid number"))
                time.sleep(1)
                # Redisplay the selection screen
                self._show_model_selection(model_files)
                return
        
        # Clear the screen again
        print(self.term.clear)
    
    def game_loop(self):
        """Main game loop."""
        try:
            self.render()
            
            while self.running:
                # If it's the engine's turn, make an engine move
                if (self.game.game_mode == "engine" and 
                    self.game.get_current_player() == "Black" and 
                    not self.game.is_game_over()):
                    
                    # Show thinking message
                    status_y = self.renderer.board_offset_y + 8 * self.renderer.square_height + 12
                    print(self.term.move_xy(0, status_y) + 
                          self.term.bold_yellow("Engine is thinking..."))
                    
                    # Simulate thinking time
                    time.sleep(self.engine_thinking_time)
                    
                    try:
                        # Get the best move from the engine
                        move_result = self.game.engine.get_best_move(self.game.board)
                        
                        # Check if we got a valid move (move_result is now a tuple of (move, eval))
                        if move_result and isinstance(move_result, tuple) and move_result[0]:
                            move = move_result[0]  # Extract the move from the tuple
                            
                            # Get SAN representation BEFORE making the move
                            san_move = self.game.board.san(move)
                            
                            # Make the move
                            self.game.move_history.append(self.game.board.fen())
                            self.game.board.push(move)
                            self.last_move = move
                            
                            # Add to move history
                            self.move_history.append(san_move)
                            
                            # Render the updated board
                            self.render()
                    except Exception as e:
                        print(f"Error making engine move: {e}")
                
                # Handle human input
                if self.handle_input():
                    self.render()
                
                # Small delay to prevent high CPU usage
                time.sleep(0.01)
        except Exception as e:
            # If we encounter an error, switch to fallback mode
            self.fallback_mode = True
            print(self.term.normal)
            print(f"Error in game loop: {e}")
            print("Switching to fallback mode...")
            self.fallback_game_loop()
    
    def print_colored_board(self):
        """Print a colored text representation of the board in fallback mode."""
        board = self.game.board
        print("\n  a b c d e f g h")
        print(" +-----------------+")
        for rank in range(7, -1, -1):
            print(f"{rank+1}|", end="")
            for file in range(8):
                square = chess.square(file, rank)
                piece = board.piece_at(square)
                
                # Determine square background
                is_light = (file + rank) % 2 == 0
                bg = " "  # Default background
                
                if piece is None:
                    # Empty square - just print the background
                    if is_light:
                        print(self.term.white_on_white(bg), end="")
                    else:
                        print(self.term.black_on_black(bg), end="")
                else:
                    # Square with piece - print the piece with its color
                    symbol = piece.symbol()
                    if piece.color == chess.WHITE:
                        # Light blue for white pieces
                        print(self.term.bright_cyan(symbol), end="")
                    else:
                        # Amber/orange for black pieces
                        print(self.term.bright_yellow(symbol), end="")
                print(" ", end="")
            print(f"|{rank+1}")
        print(" +-----------------+")
        print("  a b c d e f g h\n")
    
    def print_move_history(self):
        """Print the move history with newest moves first."""
        print("\nMove History (Newest First):")
        if self.move_history:
            # Reverse the move history to show newest moves first
            reversed_history = list(reversed(self.move_history))
            
            # Display the moves
            for i, move in enumerate(reversed_history):
                # Calculate the original move number
                original_move_num = len(self.move_history) - i
                player = "White" if original_move_num % 2 == 1 else "Black"
                move_text = f"{(original_move_num+1)//2}.{'' if player == 'White' else '..'} {move}"
                
                # Color the move text based on the player
                if player == "White":
                    print(self.term.bright_cyan(move_text))
                else:
                    print(self.term.bright_yellow(move_text))
        else:
            print("No moves yet")
    
    def fallback_game_loop(self):
        """Fallback game loop for when the terminal UI fails."""
        print("\nFallback mode activated. Using simple text interface.\n")
        
        # Show game mode selection
        print("Select game mode:")
        print("1. Human vs Human")
        print("2. Human vs Engine")
        
        while True:
            mode = input("Enter 1 or 2: ")
            if mode == '1':
                self.game.set_game_mode("human")
                break
            elif mode == '2':
                self.game.set_game_mode("engine")
                # Show model selection in fallback mode
                self._fallback_model_selection()
                break
            else:
                print("Invalid selection. Please enter 1 or 2.")
        
        # Print the colored board
        try:
            self.print_colored_board()
        except Exception:
            # If colored board fails, fall back to plain text
            print(self.game.board)
        
        print(f"\nCurrent player: {self.game.get_current_player()}")
        print(f"Game mode: {'Human vs Human' if self.game.game_mode == 'human' else 'Human vs Engine'}")
        
        # Print move history
        try:
            self.print_move_history()
        except Exception:
            # If colored move history fails, fall back to plain text
            print("\nMove History (Newest First):")
            if self.move_history:
                reversed_history = list(reversed(self.move_history))
                for i, move in enumerate(reversed_history):
                    original_move_num = len(self.move_history) - i
                    player = "White" if original_move_num % 2 == 1 else "Black"
                    print(f"{(original_move_num+1)//2}.{'' if player == 'White' else '..'} {move}")
            else:
                print("No moves yet")
        
        while self.running:
            try:
                # Check if it's the engine's turn
                if (self.game.game_mode == "engine" and 
                    self.game.get_current_player() == "Black" and 
                    not self.game.is_game_over()):
                    
                    print("\nEngine is thinking...")
                    time.sleep(self.engine_thinking_time)
                    
                    try:
                        # Get the best move from the engine
                        move_result = self.game.engine.get_best_move(self.game.board)
                        
                        # Check if we got a valid move (move_result is now a tuple of (move, eval))
                        if move_result and isinstance(move_result, tuple) and move_result[0]:
                            move = move_result[0]  # Extract the move from the tuple
                            
                            # Get SAN representation BEFORE making the move
                            san_move = self.game.board.san(move)
                            
                            # Make the move
                            self.game.move_history.append(self.game.board.fen())
                            self.game.board.push(move)
                            self.last_move = move
                            
                            # Add to move history
                            self.move_history.append(san_move)
                            
                            # Print the updated board
                            try:
                                self.print_colored_board()
                            except Exception:
                                print(self.game.board)
                            
                            print(f"\nCurrent player: {self.game.get_current_player()}")
                            
                            # Print move history
                            try:
                                self.print_move_history()
                            except Exception:
                                print("\nMove History (Newest First):")
                                if self.move_history:
                                    reversed_history = list(reversed(self.move_history))
                                    for i, move in enumerate(reversed_history):
                                        original_move_num = len(self.move_history) - i
                                        player = "White" if original_move_num % 2 == 1 else "Black"
                                        print(f"{(original_move_num+1)//2}.{'' if player == 'White' else '..'} {move}")
                                else:
                                    print("No moves yet")
                            
                            # Check for check or checkmate
                            if self.game.board.is_check():
                                print(self.term.bold_red("CHECK!"))
                            if self.game.board.is_checkmate():
                                print(self.term.bold_red("CHECKMATE!"))
                                print(f"Game over: {self.game.get_game_result()}")
                                break
                            elif self.game.board.is_game_over():
                                print(f"Game over: {self.game.get_game_result()}")
                                break
                    except Exception as e:
                        print(f"Error making engine move: {e}")
                        
                    continue
                
                # Get user input
                move_input = input("\nEnter move (or 'q' to quit, 'u' to undo): ")
                
                if move_input.lower() == 'q':
                    print("\nExiting game...")
                    self.running = False
                    break
                elif move_input.lower() == 'u':
                    if self.game.undo_move():
                        if self.move_history:
                            self.move_history.pop()
                        print("Move undone.")
                        
                        # If playing against engine, undo the engine's move too
                        if self.game.game_mode == "engine" and self.game.get_current_player() == "Black":
                            if self.game.undo_move():
                                if self.move_history:
                                    self.move_history.pop()
                    
                    else:
                        print("No moves to undo.")
                    
                    # Print the updated board
                    try:
                        self.print_colored_board()
                    except Exception:
                        print(self.game.board)
                    
                    print(f"\nCurrent player: {self.game.get_current_player()}")
                    
                    # Print move history
                    try:
                        self.print_move_history()
                    except Exception:
                        print("\nMove History (Newest First):")
                        if self.move_history:
                            reversed_history = list(reversed(self.move_history))
                            for i, move in enumerate(reversed_history):
                                original_move_num = len(self.move_history) - i
                                player = "White" if original_move_num % 2 == 1 else "Black"
                                print(f"{(original_move_num+1)//2}.{'' if player == 'White' else '..'} {move}")
                        else:
                            print("No moves yet")
                elif move_input.lower() == 's':
                    filepath = self.game.save_game()
                    print(f"Game saved to {filepath}")
                elif move_input.lower() == 'l':
                    save_dir = self.game.save_dir
                    if os.path.exists(save_dir):
                        saves = [f for f in os.listdir(save_dir) if f.endswith('.json')]
                        if saves:
                            saves.sort(reverse=True)
                            filepath = os.path.join(save_dir, saves[0])
                            if self.game.load_game(filepath):
                                self.last_move = self.game.board.peek() if self.game.board.move_stack else None
                                self.move_history = []
                                for move in self.game.board.move_stack:
                                    self.move_history.append(self.game.board.san(move))
                                print(f"Game loaded from {filepath}")
                                
                                # Print the updated board
                                try:
                                    self.print_colored_board()
                                except Exception:
                                    print(self.game.board)
                                
                                print(f"\nCurrent player: {self.game.get_current_player()}")
                                print(f"Game mode: {'Human vs Human' if self.game.game_mode == 'human' else 'Human vs Engine'}")
                                
                                # Print move history
                                try:
                                    self.print_move_history()
                                except Exception:
                                    print("\nMove History (Newest First):")
                                    if self.move_history:
                                        reversed_history = list(reversed(self.move_history))
                                        for i, move in enumerate(reversed_history):
                                            original_move_num = len(self.move_history) - i
                                            player = "White" if original_move_num % 2 == 1 else "Black"
                                            print(f"{(original_move_num+1)//2}.{'' if player == 'White' else '..'} {move}")
                                    else:
                                        print("No moves yet")
                            else:
                                print("Failed to load game.")
                        else:
                            print("No saved games found.")
                    else:
                        print("No saved games found.")
                elif move_input.lower() == 'r':
                    # Reset the board to starting position
                    self.game.board = chess.Board()
                    self.game.move_history = []
                    self.move_history = []
                    self.last_move = None
                    print("\nGame reset to starting position")
                    
                    # Print the updated board
                    try:
                        self.print_colored_board()
                    except Exception:
                        print(self.game.board)
                    
                    print(f"\nCurrent player: {self.game.get_current_player()}")
                    continue
                else:
                    try:
                        # Try to parse the move in SAN format
                        move = self.game.board.parse_san(move_input)
                        if move in self.game.board.legal_moves:
                            # Record the SAN representation before making the move
                            san_move = self.game.board.san(move)
                            self.move_history.append(san_move)
                            
                            # Make the move
                            self.game.board.push(move)
                            self.last_move = move
                            
                            # Print the updated board
                            try:
                                self.print_colored_board()
                            except Exception:
                                print(self.game.board)
                            
                            print(f"\nCurrent player: {self.game.get_current_player()}")
                            
                            # Print move history
                            try:
                                self.print_move_history()
                            except Exception:
                                print("\nMove History (Newest First):")
                                if self.move_history:
                                    reversed_history = list(reversed(self.move_history))
                                    for i, move in enumerate(reversed_history):
                                        original_move_num = len(self.move_history) - i
                                        player = "White" if original_move_num % 2 == 1 else "Black"
                                        print(f"{(original_move_num+1)//2}.{'' if player == 'White' else '..'} {move}")
                                else:
                                    print("No moves yet")
                            
                            # Check for check or checkmate
                            if self.game.board.is_check():
                                print(self.term.bold_red("CHECK!"))
                            if self.game.board.is_checkmate():
                                print(self.term.bold_red("CHECKMATE!"))
                                print(f"Game over: {self.game.get_game_result()}")
                                break
                            elif self.game.board.is_game_over():
                                print(f"Game over: {self.game.get_game_result()}")
                                break
                        else:
                            print("Illegal move.")
                    except ValueError:
                        print("Invalid move notation.")
            except KeyboardInterrupt:
                print("\nExiting game...")
                self.running = False
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def render(self):
        """Render the game board and command window."""
        # Render the board
        self.renderer.render_board(
            self.game.board,
            last_move=self.last_move,
            command=self.command,
            command_error=self.command_error,
            move_history=self.move_history,
            game_mode=self.game.game_mode
        )
    
    def handle_input(self):
        """Handle user input."""
        try:
            if not select.select([sys.stdin], [], [], 0)[0]:
                return False
            
            key = sys.stdin.read(1)
            
            # Handle key presses
            if key == '\x03':  # Ctrl+C
                self.running = False
                return True
            elif key == 'q':  # Quit
                self.running = False
                return True
            elif key == 'r':  # Reset
                # Reset the board to starting position
                self.game.board = chess.Board()
                self.game.move_history = []
                self.move_history = []
                self.last_move = None
                self.command = ""
                self.command_error = None
                print(self.term.move_xy(0, self.renderer.board_offset_y + 8 * self.renderer.square_height + 12) +
                      "Game reset to starting position")
                time.sleep(1)
                return True
            elif key == 'u':  # Undo
                if self.game.undo_move():
                    self.last_move = self.game.board.peek() if self.game.board.move_stack else None
                    if self.move_history:
                        self.move_history.pop()
                    self.command_error = None
                    
                    # If playing against engine, undo the engine's move too
                    if self.game.game_mode == "engine" and self.game.get_current_player() == "Black":
                        if self.game.undo_move():
                            if self.move_history:
                                self.move_history.pop()
                    
                    return True
            elif key == 's':  # Save
                filepath = self.game.save_game()
                print(self.term.move_xy(0, self.renderer.board_offset_y + 8 * self.renderer.square_height + 12) +
                      f"Game saved to {filepath}")
                time.sleep(1)
                return True
            elif key == 'l':  # Load
                # Simple file selection (just load the most recent save)
                save_dir = self.game.save_dir
                if os.path.exists(save_dir):
                    saves = [f for f in os.listdir(save_dir) if f.endswith('.json')]
                    if saves:
                        saves.sort(reverse=True)  # Sort by name (which includes timestamp)
                        filepath = os.path.join(save_dir, saves[0])
                        if self.game.load_game(filepath):
                            self.last_move = self.game.board.peek() if self.game.board.move_stack else None
                            self.move_history = []
                            for move in self.game.board.move_stack:
                                self.move_history.append(self.game.board.san(move))
                            self.command_error = None
                            print(self.term.move_xy(0, self.renderer.board_offset_y + 8 * self.renderer.square_height + 12) +
                                  f"Game loaded from {filepath}")
                            time.sleep(1)
                            return True
            elif key == '\r' or key == '\n':  # Enter/Return
                if self.command:
                    try:
                        # Try to parse the move in SAN format
                        move = self.game.board.parse_san(self.command)
                        if move in self.game.board.legal_moves:
                            # Record the SAN representation before making the move
                            san_move = self.game.board.san(move)
                            self.move_history.append(san_move)
                            
                            # Make the move
                            self.game.board.push(move)
                            self.last_move = move
                            self.command = ""
                            self.command_error = None
                            return True
                        else:
                            self.command_error = "Illegal move"
                    except ValueError:
                        self.command_error = "Invalid move notation"
                    
                    self.command = ""
                    return True
            elif key == '\x7f' or key == '\x08':  # Backspace
                if self.command:
                    self.command = self.command[:-1]
                    self.command_error = None
                    return True
            elif key.isalnum() or key in ['=', '+', '#', 'x', '-', 'O']:  # Valid move characters
                self.command += key
                self.command_error = None
                return True
            
            return False
        except Exception as e:
            # If we encounter an error, switch to fallback mode
            self.fallback_mode = True
            print(self.term.normal)
            print(f"Error handling input: {e}")
            print("Switching to fallback mode...")
            self.fallback_game_loop()
            return False

    def _fallback_model_selection(self):
        """Fallback method for model selection when terminal UI fails."""
        model_files = self._get_available_models()
        
        if not model_files:
            print("\nNo models found. Using default model.")
            return
        
        print("\nAvailable models:")
        for i, model in enumerate(model_files):
            # Parse model info
            info = self._parse_model_info(model)
            
            # Display model with basic info
            model_display = f"{i+1}. {model}"
            
            # Add additional info if available
            info_text = ""
            if 'games' in info:
                info_text += f" (Games: {info['games']}"
                if 'depth' in info:
                    info_text += f", Depth: {info['depth']}"
                info_text += ")"
            
            print(f"{model_display} {info_text}")
            if 'created' in info:
                print(f"   Created: {info['created']}")
        
        while True:
            try:
                choice = input(f"\nSelect model (1-{len(model_files)}): ")
                if choice.isdigit() and 1 <= int(choice) <= len(model_files):
                    selected_model = model_files[int(choice) - 1]
                    script_dir = os.path.dirname(os.path.abspath(__file__))
                    project_root = os.path.dirname(script_dir)
                    model_path = os.path.join(project_root, "saved_models", selected_model)
                    self.game.engine.model_path = model_path
                    print(f"Selected model: {selected_model}")
                    break
                else:
                    print(f"Invalid selection. Please enter a number between 1 and {len(model_files)}.")
            except Exception as e:
                print(f"Error selecting model: {e}")
                print("Using default model.")
                break
