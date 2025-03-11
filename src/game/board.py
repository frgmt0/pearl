import chess
import time
import os
import re
from colorama import Fore, Back, Style, init

# Initialize colorama for cross-platform color support
init()

class ChessBoard:
    """
    Chess board representation with visualization and game mechanics.
    """
    def __init__(self, fen=None):
        """
        Initialize a new chess board.
        
        Args:
            fen: Optional FEN string for starting position
        """
        self.board = chess.Board(fen) if fen else chess.Board()
        self.move_history = []
        self.position_history = [self.board.fen()]
        self.last_move = None
        self.last_capture = None
        self.game_result = None
        self.white_player = "White"
        self.black_player = "Black"
        
    def reset(self):
        """Reset the board to starting position."""
        self.board = chess.Board()
        self.move_history = []
        self.position_history = [self.board.fen()]
        self.last_move = None
        self.last_capture = None
        self.game_result = None
        
    def set_position(self, fen):
        """
        Set the board to a specific position.
        
        Args:
            fen: FEN string
            
        Returns:
            True if position was set, False if invalid FEN
        """
        try:
            self.board = chess.Board(fen)
            self.move_history = []
            self.position_history = [fen]
            self.last_move = None
            self.last_capture = None
            self.game_result = None
            return True
        except ValueError:
            return False
    
    def make_move(self, move):
        """
        Make a move on the board.
        
        Args:
            move: Move string in UCI format or chess.Move object
            
        Returns:
            True if move was made, False if illegal
        """
        try:
            # Convert string move to Move object if needed
            if isinstance(move, str):
                try:
                    move = chess.Move.from_uci(move)
                except ValueError:
                    # Try SAN format
                    try:
                        move = self.board.parse_san(move)
                    except ValueError:
                        print(f"Invalid move format: {move}")
                        return False
            
            # Check if move is legal
            if move not in self.board.legal_moves:
                print(f"Illegal move: {move.uci()} not in {[m.uci() for m in list(self.board.legal_moves)[:5]]}...")
                return False
            
            # Double-check: validate move properties
            from_square = move.from_square
            to_square = move.to_square
            
            # Verify the from square has a piece
            piece = self.board.piece_at(from_square)
            if not piece:
                print(f"Invalid move: No piece at {chess.square_name(from_square)}")
                return False
                
            # Verify the piece's color matches the turn
            if piece.color != self.board.turn:
                print(f"Invalid move: {piece.symbol()} at {chess.square_name(from_square)} belongs to the wrong side")
                return False
            
            # Check for capture
            self.last_capture = self.board.is_capture(move)
            
            # Make the move
            self.board.push(move)
            self.last_move = move
            
            # Update history
            self.move_history.append(move)
            self.position_history.append(self.board.fen())
            
            # Check if game is over
            if self.board.is_checkmate():
                self.game_result = "1-0" if self.board.turn == chess.BLACK else "0-1"
            elif self.board.is_stalemate() or self.board.is_insufficient_material():
                self.game_result = "1/2-1/2"
            
            return True
            
        except Exception as e:
            print(f"Error making move {move if isinstance(move, str) else move.uci() if hasattr(move, 'uci') else str(move)}: {e}")
            return False
    
    def undo_move(self):
        """
        Undo the last move.
        
        Returns:
            The move that was undone, or None if no moves to undo
        """
        if not self.move_history:
            return None
        
        move = self.move_history.pop()
        self.position_history.pop()
        self.board.pop()
        
        # Update last move
        self.last_move = self.move_history[-1] if self.move_history else None
        self.last_capture = None
        self.game_result = None
        
        return move
    
    def get_legal_moves(self):
        """
        Get list of legal moves.
        
        Returns:
            List of legal moves in UCI format
        """
        return [move.uci() for move in self.board.legal_moves]
    
    def is_game_over(self):
        """
        Check if the game is over.
        
        Returns:
            True if game is over, False otherwise
        """
        return self.board.is_game_over() or self.game_result is not None
    
    def get_result(self):
        """
        Get the game result.
        
        Returns:
            '1-0' for white win, '0-1' for black win, '1/2-1/2' for draw, or None if game not over
        """
        if self.game_result:
            return self.game_result
        elif self.board.is_game_over():
            return self.board.result()
        return None
    
    def get_winner(self):
        """
        Get the winner of the game.
        
        Returns:
            'white', 'black', 'draw', or None if game not over
        """
        result = self.get_result()
        if not result:
            return None
        elif result == '1-0':
            return 'white'
        elif result == '0-1':
            return 'black'
        else:
            return 'draw'
    
    def get_fen(self):
        """
        Get the FEN string of the current position.
        
        Returns:
            FEN string
        """
        return self.board.fen()
    
    def is_check(self):
        """
        Check if the current player is in check.
        
        Returns:
            True if in check, False otherwise
        """
        return self.board.is_check()
    
    def is_checkmate(self):
        """
        Check if the current player is in checkmate.
        
        Returns:
            True if in checkmate, False otherwise
        """
        return self.board.is_checkmate()
    
    def is_stalemate(self):
        """
        Check if the current position is a stalemate.
        
        Returns:
            True if stalemate, False otherwise
        """
        return self.board.is_stalemate()
    
    def is_insufficient_material(self):
        """
        Check if there is insufficient material for checkmate.
        
        Returns:
            True if insufficient material, False otherwise
        """
        return self.board.is_insufficient_material()
    
    def is_draw(self):
        """
        Check if the position is a draw.
        
        Returns:
            True if draw, False otherwise
        """
        return (self.board.is_stalemate() or
                self.board.is_insufficient_material() or
                self.board.is_seventyfive_moves() or
                self.board.is_fivefold_repetition())
    
    def get_turn(self):
        """
        Get the player to move.
        
        Returns:
            'white' or 'black'
        """
        return 'white' if self.board.turn == chess.WHITE else 'black'
    
    def set_player_names(self, white, black):
        """
        Set player names.
        
        Args:
            white: White player name
            black: Black player name
        """
        self.white_player = white
        self.black_player = black
    
    def get_move_history(self, use_san=True):
        """
        Get move history in human-readable format.
        
        Args:
            use_san: Use SAN notation if True, UCI if False
            
        Returns:
            List of move strings
        """
        if not self.move_history:
            return []
        
        # Create a temporary board to generate SAN notation
        board = chess.Board()
        moves = []
        
        for move in self.move_history:
            if use_san:
                try:
                    # Check if move is legal before using san()
                    if move in board.legal_moves:
                        moves.append(board.san(move))
                    else:
                        # For illegal moves, fall back to UCI notation with a marker
                        moves.append(f"{move.uci()}?")
                except Exception as e:
                    # If anything goes wrong, use UCI notation with a marker
                    moves.append(f"{move.uci()}?")
            else:
                # Always safe to use UCI
                moves.append(move.uci())
                
            try:
                # Make the move (even if it was illegal, to maintain board state)
                board.push(move)
            except Exception as e:
                # If we can't push the move, recreate the board from current position
                print(f"Warning: Error in move history generation - {e}")
                try:
                    board = chess.Board(self.get_fen())
                except:
                    # If all else fails, restart from initial position
                    board = chess.Board()
        
        return moves
    
    def get_pgn(self, headers=None):
        """
        Get the game in PGN format.
        
        Args:
            headers: Optional dictionary with PGN headers
            
        Returns:
            PGN string
        """
        if not headers:
            headers = {}
        
        # Add default headers
        if 'Event' not in headers:
            headers['Event'] = 'Game'
        if 'Site' not in headers:
            headers['Site'] = 'Local'
        if 'Date' not in headers:
            headers['Date'] = time.strftime('%Y.%m.%d')
        if 'Round' not in headers:
            headers['Round'] = '1'
        if 'White' not in headers:
            headers['White'] = self.white_player
        if 'Black' not in headers:
            headers['Black'] = self.black_player
        if 'Result' not in headers:
            headers['Result'] = self.get_result() or '*'
        
        # Build PGN string
        pgn = ""
        
        # Add headers
        for key, value in headers.items():
            pgn += f'[{key} "{value}"]\n'
        pgn += "\n"
        
        # Add moves
        if self.move_history:
            board = chess.Board()
            fullmove_number = 1
            
            for i, move in enumerate(self.move_history):
                # Add move number for white's moves
                if i % 2 == 0:
                    pgn += f"{fullmove_number}. "
                    fullmove_number += 1
                
                try:
                    # Check if move is legal before using san()
                    if move in board.legal_moves:
                        # Add move in algebraic notation
                        pgn += board.san(move) + " "
                    else:
                        # For illegal moves, fall back to UCI notation
                        pgn += move.uci() + "? "
                except Exception as e:
                    # If anything goes wrong, use UCI notation with a marker
                    pgn += move.uci() + "? "
                
                try:
                    # Make the move (even if it was illegal, to maintain board state)
                    board.push(move)
                except Exception as e:
                    # If we can't push the move, create a new board from current position
                    # This is a last resort to avoid breaking the entire PGN generation
                    print(f"Warning: Error in PGN generation - {e}")
                    try:
                        board = chess.Board(self.get_fen())
                    except:
                        # If all else fails, restart from initial position
                        board = chess.Board()
                
                # Add newline every 5 full moves
                if i % 10 == 9:
                    pgn += "\n"
        
        # Add result
        pgn += " " + (self.get_result() or "*")
        
        return pgn
    
    def to_unicode(self, perspective=chess.WHITE, last_move_highlight=True, check_highlight=True):
        """
        Convert board to Unicode string representation.
        
        Args:
            perspective: Perspective to view from (WHITE/BLACK)
            last_move_highlight: Whether to highlight the last move
            check_highlight: Whether to highlight checks
            
        Returns:
            Unicode string representation of the board
        """
        result = []
        
        # Determine rank and file ranges based on perspective
        rank_range = range(7, -1, -1) if perspective == chess.WHITE else range(8)
        file_range = range(8) if perspective == chess.WHITE else range(7, -1, -1)
        
        # Get last move squares for highlighting
        last_move_from = None
        last_move_to = None
        if last_move_highlight and self.last_move:
            last_move_from = self.last_move.from_square
            last_move_to = self.last_move.to_square
        
        # Get king square for check highlighting
        king_square = None
        if check_highlight and self.is_check():
            king_square = self.board.king(self.board.turn)
        
        # Mapping of pieces to Unicode characters
        unicode_pieces = {
            'r': '♜', 'n': '♞', 'b': '♝', 'q': '♛', 'k': '♚', 'p': '♟',
            'R': '♖', 'N': '♘', 'B': '♗', 'Q': '♕', 'K': '♔', 'P': '♙'
        }
        
        # Add file labels at the top
        files = "abcdefgh"
        if perspective == chess.WHITE:
            file_labels = "   " + "  ".join(files) + "  "
        else:
            file_labels = "   " + "  ".join(files[::-1]) + "  "
        result.append(file_labels)
        result.append("  +" + "-+" * 8)
        
        # Add board rows
        for rank in rank_range:
            rank_str = f"{rank+1} |"
            
            for file in file_range:
                square = chess.square(file, rank)
                
                # Determine square background color
                if (square == king_square and check_highlight):
                    # Highlighted check square
                    bg_color = Back.RED
                elif (square == last_move_from or square == last_move_to) and last_move_highlight:
                    # Highlighted last move square
                    bg_color = Back.YELLOW
                elif (rank + file) % 2 == 0:
                    # Light square
                    bg_color = Back.WHITE
                else:
                    # Dark square
                    bg_color = Back.BLUE
                
                # Get piece at square
                piece = self.board.piece_at(square)
                
                # Determine piece representation and color
                if piece:
                    piece_symbol = piece.symbol()
                    piece_unicode = unicode_pieces[piece_symbol]
                    fg_color = Fore.BLACK if piece.color == chess.WHITE else Fore.RED
                    rank_str += f"{bg_color}{fg_color} {piece_unicode} {Style.RESET_ALL}"
                else:
                    rank_str += f"{bg_color}   {Style.RESET_ALL}"
            
            rank_str += f"| {rank+1}"
            result.append(rank_str)
        
        result.append("  +" + "-+" * 8)
        result.append(file_labels)
        
        # Add turn indicator
        turn = "White" if self.board.turn == chess.WHITE else "Black"
        result.append(f"\n{turn} to move")
        
        # Add check indicator
        if self.is_check():
            result.append("CHECK!")
        
        # Add game result if game is over
        if self.is_game_over():
            result.append(f"Game over: {self.get_result()}")
        
        return "\n".join(result)
    
    def to_ascii(self, perspective=chess.WHITE):
        """
        Convert board to ASCII string representation.
        
        Args:
            perspective: Perspective to view from (WHITE/BLACK)
            
        Returns:
            ASCII string representation of the board
        """
        result = []
        
        # Determine rank and file ranges based on perspective
        rank_range = range(7, -1, -1) if perspective == chess.WHITE else range(8)
        file_range = range(8) if perspective == chess.WHITE else range(7, -1, -1)
        
        # Add file labels at the top
        files = "abcdefgh"
        if perspective == chess.WHITE:
            file_labels = "  " + " ".join(files)
        else:
            file_labels = "  " + " ".join(files[::-1])
        result.append(file_labels)
        result.append("  ---------------")
        
        # Add board rows
        for rank in rank_range:
            rank_str = f"{rank+1}|"
            
            for file in file_range:
                square = chess.square(file, rank)
                
                # Get piece at square
                piece = self.board.piece_at(square)
                
                # Determine piece representation
                if piece:
                    rank_str += piece.symbol()
                else:
                    rank_str += "." if (rank + file) % 2 == 0 else " "
                
                rank_str += " "
            
            rank_str += f"|{rank+1}"
            result.append(rank_str)
        
        result.append("  ---------------")
        result.append(file_labels)
        
        # Add turn indicator
        turn = "White" if self.board.turn == chess.WHITE else "Black"
        result.append(f"\n{turn} to move")
        
        # Add check indicator
        if self.is_check():
            result.append("CHECK!")
        
        # Add game result if game is over
        if self.is_game_over():
            result.append(f"Game over: {self.get_result()}")
        
        return "\n".join(result)
    
    def __str__(self):
        """
        String representation of the board (uses Unicode).
        
        Returns:
            Unicode string representation of the board
        """
        try:
            return self.to_unicode()
        except:
            # Fallback to ASCII if Unicode fails
            return self.to_ascii()
