"""
Tkinter-based GUI for the chess engine with drag and drop support.

This module provides a graphical user interface for the chess engine,
allowing players to drag and drop pieces to make moves.
"""

import os
import chess
import tkinter as tk
from tkinter import messagebox, PhotoImage

class ChessGUI:
    """
    A Tkinter-based GUI for the chess engine.
    """
    
    def __init__(self, engine=None):
        self.engine = engine
        self.board = chess.Board()
        
        # Initialize the main window
        self.root = tk.Tk()
        self.root.title("Pearl Chess Engine")
        self.root.resizable(False, False)
        
        # Set up the board display
        self.square_size = 64
        self.board_size = self.square_size * 8
        
        # Create a frame for the board
        self.board_frame = tk.Frame(self.root)
        self.board_frame.pack(padx=10, pady=10)
        
        # Create the canvas for the board
        self.canvas = tk.Canvas(self.board_frame, 
                                width=self.board_size, 
                                height=self.board_size)
        self.canvas.pack()
        
        # Create a frame for controls
        self.control_frame = tk.Frame(self.root)
        self.control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Add buttons for controls
        self.new_game_button = tk.Button(self.control_frame, text="New Game", command=self.new_game)
        self.new_game_button.pack(side=tk.LEFT, padx=5)
        
        self.flip_board_button = tk.Button(self.control_frame, text="Flip Board", command=self.flip_board)
        self.flip_board_button.pack(side=tk.LEFT, padx=5)
        
        self.engine_move_button = tk.Button(self.control_frame, text="Engine Move", command=self.make_engine_move)
        self.engine_move_button.pack(side=tk.LEFT, padx=5)
        
        # Add a status bar
        self.status_var = tk.StringVar()
        self.status_var.set("White to move")
        self.status_bar = tk.Label(self.root, textvariable=self.status_var, 
                                 bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)
        
        # Set up piece images
        self.pieces = {}
        self.load_pieces()
        
        # Board orientation (True = white at bottom)
        self.orientation = True
        
        # Variables for drag and drop
        self.selected_piece = None
        self.selected_square = None
        self.drag_piece_image = None
        self.highlighted_squares = []
        
        # Current evaluation
        self.eval_var = tk.StringVar()
        self.eval_var.set("Evaluation: 0.00")
        self.eval_label = tk.Label(self.control_frame, textvariable=self.eval_var)
        self.eval_label.pack(side=tk.RIGHT, padx=10)
        
        # Set up event bindings
        self.canvas.bind("<Button-1>", self.on_square_clicked)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_drop)
        
        # Draw the initial board
        self.draw_board()
        self.draw_pieces()
    
    def load_pieces(self):
        """Load the chess piece images."""
        try:
            # Try to load custom piece images
            piece_path = os.path.join(os.path.dirname(__file__), "../../assets/pieces")
            piece_theme = "cburnett"
            
            if os.path.exists(piece_path):
                # If the pieces directory exists, load the images
                for color in [chess.WHITE, chess.BLACK]:
                    color_name = "w" if color == chess.WHITE else "b"
                    for piece_type in chess.PIECE_TYPES:
                        piece_names = {
                            chess.PAWN: "p",
                            chess.KNIGHT: "n",
                            chess.BISHOP: "b",
                            chess.ROOK: "r",
                            chess.QUEEN: "q",
                            chess.KING: "k"
                        }
                        piece_name = piece_names[piece_type]
                        file_name = f"{color_name}{piece_name}.png"
                        file_path = os.path.join(piece_path, piece_theme, file_name)
                        
                        if os.path.exists(file_path):
                            self.pieces[(color, piece_type)] = PhotoImage(file=file_path)
                        else:
                            self._create_fallback_piece(color, piece_type)
            else:
                # If no image directory exists, create simple pieces
                for color in [chess.WHITE, chess.BLACK]:
                    for piece_type in chess.PIECE_TYPES:
                        self._create_fallback_piece(color, piece_type)
        except Exception as e:
            print(f"Error loading piece images: {e}")
            # Fall back to text-based pieces
            for color in [chess.WHITE, chess.BLACK]:
                for piece_type in chess.PIECE_TYPES:
                    self._create_fallback_piece(color, piece_type)
    
    def _create_fallback_piece(self, color, piece_type):
        """Create a fallback piece image using text."""
        color_fill = "white" if color == chess.WHITE else "black"
        color_outline = "black" if color == chess.WHITE else "white"
        
        piece_symbols = {
            chess.PAWN: "♙" if color == chess.WHITE else "♟",
            chess.KNIGHT: "♘" if color == chess.WHITE else "♞",
            chess.BISHOP: "♗" if color == chess.WHITE else "♝",
            chess.ROOK: "♖" if color == chess.WHITE else "♜", 
            chess.QUEEN: "♕" if color == chess.WHITE else "♛",
            chess.KING: "♔" if color == chess.WHITE else "♚"
        }
        
        # Create a new image
        img = tk.PhotoImage(width=self.square_size, height=self.square_size)
        
        # Create a temporary canvas to draw the piece
        temp_canvas = tk.Canvas(width=self.square_size, height=self.square_size)
        temp_canvas.create_rectangle(0, 0, self.square_size, self.square_size, fill="", outline="")
        
        # Draw the piece symbol
        temp_canvas.create_text(self.square_size//2, self.square_size//2, 
                               text=piece_symbols[piece_type], 
                               fill=color_fill, font=("Arial", 48))
        
        # Store the image
        self.pieces[(color, piece_type)] = img
    
    def draw_board(self):
        """Draw the chess board."""
        colors = ["#eeeed2", "#769656"]  # Light, Dark
        
        for row in range(8):
            for col in range(8):
                x1 = col * self.square_size
                y1 = row * self.square_size
                x2 = x1 + self.square_size
                y2 = y1 + self.square_size
                
                # Determine square color (alternate colors)
                color_idx = (row + col) % 2
                
                # Draw the square
                square_id = self.canvas.create_rectangle(
                    x1, y1, x2, y2, 
                    fill=colors[color_idx], 
                    outline=""
                )
                
                # Add file and rank labels
                if row == 7:
                    # Files (a-h)
                    file_label = "abcdefgh"[col]
                    self.canvas.create_text(
                        x1 + 5, y2 - 5,
                        text=file_label,
                        anchor=tk.SW,
                        fill="#000000" if color_idx == 0 else "#ffffff"
                    )
                
                if col == 0:
                    # Ranks (1-8)
                    rank_label = str(8 - row if self.orientation else row + 1)
                    self.canvas.create_text(
                        x1 + 5, y1 + 5,
                        text=rank_label,
                        anchor=tk.NW,
                        fill="#000000" if color_idx == 0 else "#ffffff"
                    )
    
    def draw_pieces(self):
        """Draw all chess pieces on the board."""
        self.canvas.delete("piece")
        
        for row in range(8):
            for col in range(8):
                # Convert to chess coordinates based on orientation
                chess_row = 7 - row if self.orientation else row
                chess_col = col if self.orientation else 7 - col
                
                square = chess.square(chess_col, chess_row)
                piece = self.board.piece_at(square)
                
                if piece:
                    x = col * self.square_size
                    y = row * self.square_size
                    
                    # Check if we have an image for this piece
                    if (piece.color, piece.piece_type) in self.pieces:
                        piece_image = self.pieces[(piece.color, piece.piece_type)]
                        self.canvas.create_image(
                            x + self.square_size // 2, 
                            y + self.square_size // 2,
                            image=piece_image,
                            tags=("piece", f"square_{square}")
                        )
    
    def draw_highlights(self):
        """Draw highlights for selected square and legal moves."""
        # Clear existing highlights
        self.canvas.delete("highlight")
        
        # Highlight the selected square if there is one
        if self.selected_square is not None:
            row, col = self.get_gui_coords(self.selected_square)
            x1 = col * self.square_size
            y1 = row * self.square_size
            x2 = x1 + self.square_size
            y2 = y1 + self.square_size
            
            self.canvas.create_rectangle(
                x1, y1, x2, y2,
                outline="#ffff00",
                width=3,
                tags="highlight"
            )
        
        # Highlight legal moves
        for square in self.highlighted_squares:
            row, col = self.get_gui_coords(square)
            x1 = col * self.square_size
            y1 = row * self.square_size
            x2 = x1 + self.square_size
            y2 = y1 + self.square_size
            
            # For empty squares, draw a circle
            piece = self.board.piece_at(square)
            if piece is None:
                self.canvas.create_oval(
                    x1 + self.square_size // 4,
                    y1 + self.square_size // 4,
                    x2 - self.square_size // 4,
                    y2 - self.square_size // 4,
                    fill="#aaddaa",
                    outline="",
                    tags="highlight"
                )
            else:
                # For captures, draw a red border
                self.canvas.create_rectangle(
                    x1, y1, x2, y2,
                    outline="#ff0000",
                    width=3,
                    tags="highlight"
                )
    
    def get_square_from_coords(self, x, y):
        """Convert pixel coordinates to a chess square."""
        col = x // self.square_size
        row = y // self.square_size
        
        # Convert to chess coordinates based on orientation
        chess_row = 7 - row if self.orientation else row
        chess_col = col if self.orientation else 7 - col
        
        # Ensure we're within bounds
        if 0 <= chess_row < 8 and 0 <= chess_col < 8:
            return chess.square(chess_col, chess_row)
        return None
    
    def get_gui_coords(self, square):
        """Convert a chess square to GUI coordinates (row, col)."""
        chess_row = chess.square_rank(square)
        chess_col = chess.square_file(square)
        
        # Convert to GUI coordinates based on orientation
        row = 7 - chess_row if self.orientation else chess_row
        col = chess_col if self.orientation else 7 - chess_col
        
        return row, col
    
    def on_square_clicked(self, event):
        """Handle a mouse click on the board."""
        square = self.get_square_from_coords(event.x, event.y)
        if square is None:
            return
        
        piece = self.board.piece_at(square)
        
        # If no square is selected, try to select this one
        if self.selected_square is None:
            if piece and piece.color == self.board.turn:
                self.selected_square = square
                self.selected_piece = piece
                
                # Find legal moves from this square
                self.highlighted_squares = [
                    move.to_square for move in self.board.legal_moves
                    if move.from_square == square
                ]
                
                # Redraw with highlights
                self.draw_highlights()
                
                # Store the current piece for dragging
                row, col = self.get_gui_coords(square)
                self.drag_piece_image = self.canvas.find_withtag(f"square_{square}")
                self.drag_start_x = event.x
                self.drag_start_y = event.y
        else:
            # If clicking on the already selected square, deselect it
            if square == self.selected_square:
                self.selected_square = None
                self.selected_piece = None
                self.highlighted_squares = []
                self.draw_highlights()
            else:
                # If clicking on a different square, check if it's a legal move
                if square in self.highlighted_squares:
                    self.make_move(self.selected_square, square)
                elif piece and piece.color == self.board.turn:
                    # If clicking on another of our pieces, select it instead
                    self.selected_square = square
                    self.selected_piece = piece
                    
                    # Find legal moves from this square
                    self.highlighted_squares = [
                        move.to_square for move in self.board.legal_moves
                        if move.from_square == square
                    ]
                    
                    # Redraw with highlights
                    self.draw_highlights()
                else:
                    # Otherwise, clear selection
                    self.selected_square = None
                    self.selected_piece = None
                    self.highlighted_squares = []
                    self.draw_highlights()
    
    def on_drag(self, event):
        """Handle dragging a piece."""
        if self.selected_square is not None and self.drag_piece_image:
            # Move the piece with the mouse
            dx = event.x - self.drag_start_x
            dy = event.y - self.drag_start_y
            self.canvas.move(self.drag_piece_image, dx, dy)
            self.drag_start_x = event.x
            self.drag_start_y = event.y
    
    def on_drop(self, event):
        """Handle dropping a piece."""
        if self.selected_square is not None:
            target_square = self.get_square_from_coords(event.x, event.y)
            
            if target_square is not None and target_square in self.highlighted_squares:
                # Make the move
                self.make_move(self.selected_square, target_square)
            else:
                # Redraw the board to reset the dragged piece
                self.draw_pieces()
                self.draw_highlights()
    
    def make_move(self, from_square, to_square):
        """Make a move on the board and update the display."""
        # Get all legal moves from the source to target square
        moves = [
            move for move in self.board.legal_moves
            if move.from_square == from_square and move.to_square == to_square
        ]
        
        if not moves:
            # No legal move found
            self.status_var.set("Illegal move")
            return
        
        # If multiple moves possible (e.g., promotion), ask the user
        if len(moves) > 1:
            move = self.handle_promotion(moves)
            if move is None:
                # User canceled
                self.draw_pieces()  # Redraw to original positions
                return
        else:
            move = moves[0]
        
        # Make the move
        san_move = self.board.san(move)
        self.board.push(move)
        
        # Clear selection and redraw
        self.selected_square = None
        self.selected_piece = None
        self.highlighted_squares = []
        self.draw_board()
        self.draw_pieces()
        
        # Update status
        self.status_var.set(f"Move: {san_move}")
        
        # Check for game over
        if self.board.is_game_over():
            result = self.board.result()
            if result == "1-0":
                message = "White wins"
            elif result == "0-1":
                message = "Black wins"
            else:
                message = "Draw"
            
            self.status_var.set(f"Game over: {message}")
            messagebox.showinfo("Game Over", f"Game over: {message}")
            return
        
        # Make engine move if it's available
        if self.engine and not self.board.is_game_over():
            self.make_engine_move()
    
    def make_engine_move(self):
        """Make a move with the engine."""
        if self.board.is_game_over():
            return
        
        self.status_var.set("Engine is thinking...")
        self.root.update()  # Update the UI to show the thinking message
        
        try:
            # Call the engine to search for a move
            best_move, score, info = self.engine.search(self.board, depth=4, time_limit_ms=2000)
            
            if best_move:
                # Display the evaluation
                self.eval_var.set(f"Evaluation: {score / 100:.2f} pawns")
                
                # Convert to SAN format
                san_move = self.board.san(best_move)
                
                # Make the move
                self.board.push(best_move)
                
                # Redraw the board
                self.draw_board()
                self.draw_pieces()
                
                # Update status
                self.status_var.set(f"Engine plays: {san_move}")
                
                # Check for game over
                if self.board.is_game_over():
                    result = self.board.result()
                    if result == "1-0":
                        message = "White wins"
                    elif result == "0-1":
                        message = "Black wins"
                    else:
                        message = "Draw"
                    
                    self.status_var.set(f"Game over: {message}")
                    messagebox.showinfo("Game Over", f"Game over: {message}")
            else:
                self.status_var.set("Engine couldn't find a move")
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
    
    def handle_promotion(self, moves):
        """Handle pawn promotion by asking the user which piece to promote to."""
        # Create a popup dialog
        popup = tk.Toplevel(self.root)
        popup.title("Promotion")
        popup.geometry("250x100")
        popup.resizable(False, False)
        
        # Add a label
        label = tk.Label(popup, text="Choose promotion piece:")
        label.pack(pady=5)
        
        # Variable to store the result
        result = [None]
        
        # Create buttons for each piece type
        piece_types = {
            "Queen": chess.QUEEN,
            "Rook": chess.ROOK,
            "Bishop": chess.BISHOP,
            "Knight": chess.KNIGHT
        }
        
        button_frame = tk.Frame(popup)
        button_frame.pack(pady=5)
        
        for piece_name, piece_type in piece_types.items():
            # Find the matching move for this promotion type
            for move in moves:
                if move.promotion == piece_type:
                    promotion_move = move
                    break
            else:
                continue  # Skip if no matching move
            
            def make_button_command(move=promotion_move):
                return lambda: (result.__setitem__(0, move), popup.destroy())
            
            button = tk.Button(button_frame, text=piece_name, command=make_button_command())
            button.pack(side=tk.LEFT, padx=5)
        
        # Wait for the dialog to close
        self.root.wait_window(popup)
        
        return result[0]
    
    def new_game(self):
        """Start a new game."""
        self.board = chess.Board()
        self.selected_square = None
        self.selected_piece = None
        self.highlighted_squares = []
        self.draw_board()
        self.draw_pieces()
        self.status_var.set("New game started. White to move.")
        self.eval_var.set("Evaluation: 0.00")
    
    def flip_board(self):
        """Flip the board orientation."""
        self.orientation = not self.orientation
        self.draw_board()
        self.draw_pieces()
    
    def run(self):
        """Run the GUI application."""
        self.root.mainloop()


def run_gui(engine=None):
    """
    Run the GUI chess interface.
    
    Args:
        engine: Chess engine object (optional)
    
    Returns:
        The final board state.
    """
    gui = ChessGUI(engine)
    gui.run()
    
    return gui.board 