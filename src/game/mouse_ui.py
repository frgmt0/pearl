"""
Mouse-based interactive chess UI using prompt_toolkit.

This module provides a terminal-based chess interface with mouse support,
allowing players to click on pieces and destination squares to make moves.
"""

import chess
from prompt_toolkit import Application
from prompt_toolkit.layout import Layout, HSplit, VSplit, Window, FormattedTextControl
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.mouse_events import MouseEventType
from prompt_toolkit.styles import Style

class MouseChessUI:
    """
    A terminal-based chess UI with mouse support.
    """
    
    # Unicode chess pieces
    UNICODE_PIECES = {
        chess.PAWN: {chess.WHITE: '♙', chess.BLACK: '♟'},
        chess.KNIGHT: {chess.WHITE: '♘', chess.BLACK: '♞'},
        chess.BISHOP: {chess.WHITE: '♗', chess.BLACK: '♝'},
        chess.ROOK: {chess.WHITE: '♖', chess.BLACK: '♜'},
        chess.QUEEN: {chess.WHITE: '♕', chess.BLACK: '♛'},
        chess.KING: {chess.WHITE: '♔', chess.BLACK: '♚'},
    }
    
    # Board colors
    LIGHT_SQUARE = 'bg:#eeeed2'
    DARK_SQUARE = 'bg:#769656'
    SELECTED_LIGHT_SQUARE = 'bg:#f6f669'
    SELECTED_DARK_SQUARE = 'bg:#baca2b'
    LEGAL_MOVE_LIGHT = 'bg:#f6f669'
    LEGAL_MOVE_DARK = 'bg:#baca2b'
    
    def __init__(self, engine=None):
        self.board = chess.Board()
        self.engine = engine
        self.selected_square = None
        self.legal_moves = set()
        self.last_move = None
        self.message = "Click a piece to move it"
        self.highlighted_squares = set()
        
        # Create the UI components
        self.board_control = FormattedTextControl(self._get_board_tokens, focusable=True)
        self.message_control = FormattedTextControl(self._get_message_tokens)
        
        # Create key bindings
        self.kb = KeyBindings()
        
        @self.kb.add('q')
        def _(event):
            """Exit the application when q is pressed."""
            event.app.exit()
            
        @self.kb.add('n')
        def _(event):
            """Start a new game when n is pressed."""
            self.board = chess.Board()
            self.selected_square = None
            self.legal_moves = set()
            self.last_move = None
            self.message = "New game started. Click a piece to move it."
            self.highlighted_squares = set()
            
        @self.kb.add('r')
        def _(event):
            """Reset the current position."""
            self._deselect()
            
        # Mouse handler for selecting squares and pieces
        @self.kb.add('c-space')  # Ctrl+Space is sent for mouse click events
        def _(event):
            """Handle mouse click events for selecting pieces and making moves."""
            # Get the mouse event data
            data = event.key_sequence[0].data
            
            if data and hasattr(data, 'event_type') and data.event_type == MouseEventType.MOUSE_DOWN:
                # Calculate clicked square
                col = data.position.x // 4  # Each square is 4 chars wide
                row = 7 - (data.position.y // 2)  # Each square is 2 chars high, invert for chess coordinates
                
                if 0 <= col < 8 and 0 <= row < 8:
                    clicked_square = chess.square(col, row)
                    self._handle_square_click(clicked_square)
        
        # Create the layout
        board_window = Window(self.board_control, height=16, width=32)
        message_window = Window(self.message_control, height=1)
        
        layout = Layout(
            HSplit([
                board_window,
                message_window,
            ])
        )
        
        # Create the application without custom style
        self.app = Application(
            layout=layout,
            key_bindings=self.kb,
            mouse_support=True,
            full_screen=True,
        )
    
    def _handle_square_click(self, clicked_square):
        """Handle a click on a chess square."""
        # If no square is selected yet
        if self.selected_square is None:
            # Check if the clicked square has a piece that can be moved
            piece = self.board.piece_at(clicked_square)
            if piece and piece.color == self.board.turn:
                self.selected_square = clicked_square
                
                # Calculate legal moves from this square
                self.legal_moves = {move.to_square for move in self.board.legal_moves 
                                   if move.from_square == clicked_square}
                                   
                self.message = f"Selected {chess.square_name(clicked_square)}"
                self.highlighted_squares = {clicked_square} | self.legal_moves
            else:
                self.message = "Click on one of your pieces to move it"
        
        # If a square is already selected
        else:
            # Check if the clicked square is a legal destination
            if clicked_square in self.legal_moves:
                # Find the exact move (including promotion if needed)
                moves = [move for move in self.board.legal_moves 
                        if move.from_square == self.selected_square and move.to_square == clicked_square]
                
                if len(moves) > 1:
                    # If multiple moves are possible (e.g. promotion), use queen promotion for simplicity
                    # A more sophisticated UI would ask the user which piece to promote to
                    move = [m for m in moves if m.promotion == chess.QUEEN]
                    if move:
                        move = move[0]
                    else:
                        move = moves[0]  # Fallback
                else:
                    move = moves[0]
                
                # Make the move
                self.board.push(move)
                self.last_move = move
                self.message = f"Moved {chess.square_name(self.selected_square)} to {chess.square_name(clicked_square)}"
                
                # Clear selection and legal moves
                self._deselect()
                
                # If game is over, update message
                if self.board.is_game_over():
                    self.message = f"Game over. Result: {self.board.result()}"
                    return
                    
                # If engine is available, make engine move
                if self.engine and not self.board.is_game_over():
                    self.message = "Engine is thinking..."
                    
                    # Need to update the display here
                    self.app.invalidate()
                    
                    try:
                        best_move, score, info = self.engine.search(self.board, depth=4, time_limit_ms=5000)
                        
                        if best_move:
                            move_san = self.board.san(best_move)
                            self.board.push(best_move)
                            self.last_move = best_move
                            self.message = f"Engine plays: {best_move.uci()} ({move_san}). Evaluation: {score / 100:.2f} pawns"
                            
                            # Check for game over
                            if self.board.is_game_over():
                                self.message += f" Game over: {self.board.result()}"
                        else:
                            self.message = "Engine couldn't find a move."
                    except Exception as e:
                        self.message = f"Error during engine search: {str(e)}"
            
            # If it's not a legal destination, handle reselection
            elif piece := self.board.piece_at(clicked_square):
                if piece.color == self.board.turn:
                    # Reselect this piece
                    self.selected_square = clicked_square
                    self.legal_moves = {move.to_square for move in self.board.legal_moves 
                                       if move.from_square == clicked_square}
                    self.message = f"Selected {chess.square_name(clicked_square)}"
                    self.highlighted_squares = {clicked_square} | self.legal_moves
                else:
                    self.message = "That's not your piece"
            else:
                # Deselect if clicked on an empty square that's not a legal destination
                self._deselect()
                self.message = "Canceled selection"
    
    def _deselect(self):
        """Clear the current selection and legal moves."""
        self.selected_square = None
        self.legal_moves = set()
        self.highlighted_squares = set()
        if self.last_move:
            self.highlighted_squares = {self.last_move.from_square, self.last_move.to_square}
    
    def _get_square_style(self, square):
        """Get the style for a square based on its position and state."""
        is_light = (chess.square_rank(square) + chess.square_file(square)) % 2 == 1
        
        # If square is selected
        if square == self.selected_square:
            return self.SELECTED_LIGHT_SQUARE if is_light else self.SELECTED_DARK_SQUARE
        
        # If square is a legal move destination
        if square in self.legal_moves:
            return self.LEGAL_MOVE_LIGHT if is_light else self.LEGAL_MOVE_DARK
        
        # If square is part of the last move
        if self.last_move and square in (self.last_move.from_square, self.last_move.to_square):
            return self.LEGAL_MOVE_LIGHT if is_light else self.LEGAL_MOVE_DARK
        
        # Default square color
        return self.LIGHT_SQUARE if is_light else self.DARK_SQUARE
    
    def _get_board_tokens(self):
        """Format the chess board for display with appropriate styles."""
        result = []
        
        # Add file labels on top
        result.append(('', '    '))
        for file_idx in range(8):
            result.append(('', f"{chess.FILE_NAMES[file_idx]}   "))
        result.append(('', '\n'))
        
        # Add board rows with rank labels
        for row in range(7, -1, -1):
            result.append(('', f"{row + 1} "))  # Rank label
            
            for col in range(8):
                square = chess.square(col, row)
                square_style = self._get_square_style(square)
                
                piece = self.board.piece_at(square)
                if piece:
                    piece_char = self.UNICODE_PIECES[piece.piece_type][piece.color]
                    piece_style = '#ffffff' if piece.color == chess.WHITE else '#000000'
                    result.append((f'{square_style} {piece_style}', f" {piece_char} "))
                else:
                    result.append((square_style, '   '))
            
            result.append(('', '\n'))
        
        # Add file labels on bottom
        result.append(('', '    '))
        for file_idx in range(8):
            result.append(('', f"{chess.FILE_NAMES[file_idx]}   "))
            
        return result
    
    def _get_message_tokens(self):
        """Format the message display."""
        return [('#ansired', self.message)]
    
    def run(self):
        """Run the chess UI application."""
        self.app.run()
        
        
def run_mouse_ui(engine=None):
    """
    Run the mouse-based chess UI.
    
    Args:
        engine: Chess engine object (optional)
    """
    ui = MouseChessUI(engine)
    ui.run()
    
    # Return the final board state if needed
    return ui.board 