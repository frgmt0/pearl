import chess
from blessed import Terminal

class BoardRenderer:
    def __init__(self, term):
        self.term = term
        
        # Unicode chess pieces
        self.pieces = {
            chess.PAWN: {chess.WHITE: '♙', chess.BLACK: '♟'},
            chess.KNIGHT: {chess.WHITE: '♘', chess.BLACK: '♞'},
            chess.BISHOP: {chess.WHITE: '♗', chess.BLACK: '♝'},
            chess.ROOK: {chess.WHITE: '♖', chess.BLACK: '♜'},
            chess.QUEEN: {chess.WHITE: '♕', chess.BLACK: '♛'},
            chess.KING: {chess.WHITE: '♔', chess.BLACK: '♚'},
        }
        
        # Board colors - using standard terminal colors for better compatibility
        try:
            self.colors = {
                'light_square': term.on_white,
                'dark_square': term.on_black,
                'highlight': term.on_yellow,
                'valid_move': term.on_green,
                'last_move': term.on_blue,
                'check': term.on_red,
                'command': term.on_black,
            }
            
            # Piece colors
            self.piece_colors = {
                chess.WHITE: term.bright_cyan,  # Light blue for white pieces
                chess.BLACK: term.bright_yellow,  # Amber/orange for black pieces
            }
        except (TypeError, AttributeError):
            # Fallback to simpler colors if terminal doesn't support all colors
            self.colors = {
                'light_square': lambda text: term.white_on_white(text),
                'dark_square': lambda text: term.black_on_black(text),
                'highlight': lambda text: term.yellow_on_yellow(text),
                'valid_move': lambda text: term.green_on_green(text),
                'last_move': lambda text: term.blue_on_blue(text),
                'check': lambda text: term.red_on_red(text),
                'command': lambda text: term.black_on_black(text),
            }
            
            # Fallback piece colors
            self.piece_colors = {
                chess.WHITE: lambda text: text,  # Default color for white pieces
                chess.BLACK: lambda text: text,  # Default color for black pieces
            }
        
        # Square size in terminal characters
        self.square_width = 4
        self.square_height = 2
        
        # Board offset from top-left of terminal
        self.board_offset_x = 5
        self.board_offset_y = 2
        
        # Command window dimensions
        self.command_width = 40
        self.command_height = 3
    
    def get_square_position(self, square):
        """Convert chess square to terminal coordinates."""
        file_idx = chess.square_file(square)
        rank_idx = 7 - chess.square_rank(square)  # Invert rank (0 is the bottom rank in python-chess)
        
        x = self.board_offset_x + file_idx * self.square_width
        y = self.board_offset_y + rank_idx * self.square_height
        
        return x, y
    
    def get_square_from_position(self, x, y):
        """Convert terminal coordinates to chess square."""
        # Adjust for board offset
        x = x - self.board_offset_x
        y = y - self.board_offset_y
        
        # Check if click is within board boundaries
        if x < 0 or y < 0:
            return None
        
        file_idx = x // self.square_width
        rank_idx = y // self.square_height
        
        # Check if click is within board boundaries
        if file_idx >= 8 or rank_idx >= 8:
            return None
        
        # Convert to chess square
        rank = 7 - rank_idx  # Invert rank
        square = chess.square(file_idx, rank)
        
        return square
    
    def render_board(self, board, last_move=None, command="", command_error=None, move_history=None, game_mode="human"):
        """Render the chess board with pieces and highlights."""
        try:
            # Clear the screen
            print(self.term.clear)
            
            # Print board coordinates
            for file_idx in range(8):
                x = self.board_offset_x + file_idx * self.square_width + self.square_width // 2
                y = self.board_offset_y + 8 * self.square_height
                print(self.term.move_xy(x, y) + self.term.bold(chess.FILE_NAMES[file_idx].upper()))
            
            for rank_idx in range(8):
                x = self.board_offset_x - 2
                y = self.board_offset_y + rank_idx * self.square_height + self.square_height // 2
                print(self.term.move_xy(x, y) + self.term.bold(str(8 - rank_idx)))
            
            # Render the board squares
            for square in chess.SQUARES:
                file_idx = chess.square_file(square)
                rank_idx = chess.square_rank(square)
                
                # Get terminal position for this square
                x, y = self.get_square_position(square)
                
                # Determine square color
                is_light = (file_idx + rank_idx) % 2 == 0
                square_color = self.colors['light_square'] if is_light else self.colors['dark_square']
                
                # Apply highlights
                if last_move and (square == last_move.from_square or square == last_move.to_square):
                    square_color = self.colors['last_move']
                elif board.is_check() and square == board.king(board.turn):
                    square_color = self.colors['check']
                
                # Draw the square
                for dy in range(self.square_height):
                    try:
                        print(self.term.move_xy(x, y + dy) + square_color(' ' * self.square_width))
                    except (TypeError, AttributeError):
                        # Fallback to simple rendering if color formatting fails
                        bg_char = '█' if is_light else '▓'
                        print(self.term.move_xy(x, y + dy) + bg_char * self.square_width)
            
            # Render the pieces separately to ensure consistent colors
            for square in chess.SQUARES:
                piece = board.piece_at(square)
                if piece:
                    # Get terminal position for this square
                    x, y = self.get_square_position(square)
                    piece_symbol = self.pieces[piece.piece_type][piece.color]
                    piece_x = x + self.square_width // 2
                    piece_y = y + self.square_height // 2
                    
                    try:
                        # Apply piece color based on the piece's color (white or black)
                        # Use a space as background to ensure consistent piece color
                        colored_piece = self.piece_colors[piece.color](piece_symbol)
                        print(self.term.move_xy(piece_x, piece_y) + colored_piece)
                    except (TypeError, AttributeError):
                        # Fallback to simple rendering if color formatting fails
                        print(self.term.move_xy(piece_x, piece_y) + piece_symbol)
            
            # Print game status
            status_y = self.board_offset_y + 8 * self.square_height + 2
            current_player = "White" if board.turn == chess.WHITE else "Black"
            print(self.term.move_xy(self.board_offset_x, status_y) + 
                  self.term.bold(f"Current player: {current_player}"))
            
            # Print game mode
            game_mode_text = "Human vs Human" if game_mode == "human" else "Human vs Engine"
            print(self.term.move_xy(self.board_offset_x, status_y + 1) + 
                  self.term.bold(f"Game mode: {game_mode_text}"))
            
            if board.is_check():
                print(self.term.move_xy(self.board_offset_x, status_y + 2) + 
                      self.term.bold_red("CHECK!"))
            
            if board.is_game_over():
                result = "1-0" if board.is_checkmate() and board.turn == chess.BLACK else \
                         "0-1" if board.is_checkmate() and board.turn == chess.WHITE else \
                         "1/2-1/2"
                reason = "Checkmate" if board.is_checkmate() else \
                         "Stalemate" if board.is_stalemate() else \
                         "Insufficient material" if board.is_insufficient_material() else \
                         "Fifty-move rule" if board.is_fifty_moves() else \
                         "Threefold repetition" if board.is_repetition() else \
                         "Game over"
                print(self.term.move_xy(self.board_offset_x, status_y + 3) + 
                      self.term.bold_red(f"Game over: {result} - {reason}"))
            
            # Draw command window
            command_x = self.board_offset_x + 8 * self.square_width + 4
            command_y = self.board_offset_y
            
            # Draw command window border
            print(self.term.move_xy(command_x, command_y) + 
                  self.term.bold("┌" + "─" * self.command_width + "┐"))
            
            for dy in range(1, self.command_height + 1):
                print(self.term.move_xy(command_x, command_y + dy) + 
                      self.term.bold("│") + " " * self.command_width + self.term.bold("│"))
            
            print(self.term.move_xy(command_x, command_y + self.command_height + 1) + 
                  self.term.bold("└" + "─" * self.command_width + "┘"))
            
            # Draw command window title
            print(self.term.move_xy(command_x + 2, command_y) + 
                  self.term.bold(" Command Input "))
            
            # Draw command prompt
            print(self.term.move_xy(command_x + 2, command_y + 1) + 
                  self.term.bold("Enter move: ") + command)
            
            # Draw command error if any
            if command_error:
                print(self.term.move_xy(command_x + 2, command_y + 2) + 
                      self.term.bold_red(command_error))
            
            # Draw move history
            history_y = command_y + self.command_height + 3
            print(self.term.move_xy(command_x, history_y) + 
                  self.term.bold("┌" + "─" * self.command_width + "┐"))
            
            print(self.term.move_xy(command_x + 2, history_y) + 
                  self.term.bold(" Move History (Newest First) "))
            
            if move_history:
                # Reverse the move history to show newest moves first
                reversed_history = list(reversed(move_history))
                
                # Display the moves
                for i, move in enumerate(reversed_history[:15]):
                    # Calculate the original move number
                    original_move_num = len(move_history) - i
                    player = "White" if original_move_num % 2 == 1 else "Black"
                    move_text = f"{(original_move_num+1)//2}.{'' if player == 'White' else '..'} {move}"
                    
                    # Color the move text based on the player
                    if player == "White":
                        move_text = self.term.bright_cyan(move_text)
                    else:
                        move_text = self.term.bright_yellow(move_text)
                    
                    print(self.term.move_xy(command_x + 2, history_y + i + 1) + 
                          move_text)
            else:
                print(self.term.move_xy(command_x + 2, history_y + 1) + 
                      "No moves yet")
            
            # Draw the bottom border of move history
            history_height = max(10, min(15, len(move_history) if move_history else 1))
            print(self.term.move_xy(command_x, history_y + history_height + 1) + 
                  self.term.bold("└" + "─" * self.command_width + "┘"))
            
            # Print controls help
            help_y = history_y + history_height + 3
            print(self.term.move_xy(command_x, help_y) + 
                  self.term.bold("Controls:"))
            print(self.term.move_xy(command_x, help_y + 1) + 
                  "- Type algebraic notation (e.g., 'e4', 'Nf3')")
            print(self.term.move_xy(command_x, help_y + 2) + 
                  "- Press Enter to submit move")
            print(self.term.move_xy(command_x, help_y + 3) + 
                  "- Press 'u' to undo the last move")
            print(self.term.move_xy(command_x, help_y + 4) + 
                  "- Press 's' to save the game")
            print(self.term.move_xy(command_x, help_y + 5) + 
                  "- Press 'l' to load a saved game")
            print(self.term.move_xy(command_x, help_y + 6) + 
                  "- Press 'r' to reset the board")
            print(self.term.move_xy(command_x, help_y + 7) + 
                  "- Press 'q' to quit")
            
            # Print piece color legend
            legend_y = help_y + 8
            print(self.term.move_xy(command_x, legend_y) + 
                  self.term.bold("Piece Colors:"))
            print(self.term.move_xy(command_x, legend_y + 1) + 
                  self.term.bright_cyan("■") + " White pieces (Light Blue)")
            print(self.term.move_xy(command_x, legend_y + 2) + 
                  self.term.bright_yellow("■") + " Black pieces (Amber)")
            
            # Print engine info if in engine mode
            if game_mode == "engine":
                engine_y = legend_y + 4
                print(self.term.move_xy(command_x, engine_y) + 
                      self.term.bold("Engine:"))
                print(self.term.move_xy(command_x, engine_y + 1) + 
                      "Neural network with 3 layers (256/32/32)")
                print(self.term.move_xy(command_x, engine_y + 2) + 
                      "Learns from your moves as you play")
            
            # Move cursor to a safe position
            print(self.term.move_xy(0, self.board_offset_y + 8 * self.square_height + 12))
        
        except Exception as e:
            # Fallback to a very simple rendering if all else fails
            print(self.term.clear)
            print(f"Error rendering board: {e}")
            print("\nFallback to simple text mode:\n")
            
            # Print simple text representation of the board
            print(board)
            print(f"\nCurrent player: {'White' if board.turn == chess.WHITE else 'Black'}")
            print(f"Game mode: {'Human vs Human' if game_mode == 'human' else 'Human vs Engine'}")
            
            if board.is_check():
                print("CHECK!")
            if board.is_game_over():
                print(f"Game over: {board.result()}")
            
            print("\nCommand: " + command)
            if command_error:
                print("Error: " + command_error)
            
            print("\nMove history (newest first):")
            if move_history:
                reversed_history = list(reversed(move_history))
                for i, move in enumerate(reversed_history):
                    original_move_num = len(move_history) - i
                    player = "White" if original_move_num % 2 == 1 else "Black"
                    print(f"{(original_move_num+1)//2}.{'' if player == 'White' else '..'} {move}")
            else:
                print("No moves yet")
            
            print("\nControls:")
            print("- Type algebraic notation (e.g., 'e4', 'Nf3')")
            print("- Press Enter to submit move")
            print("- Press 'u' to undo the last move")
            print("- Press 's' to save the game")
            print("- Press 'l' to load a saved game")
            print("- Press 'r' to reset the board")
            print("- Press 'q' to quit")
