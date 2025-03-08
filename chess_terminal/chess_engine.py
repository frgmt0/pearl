import chess
import json
import os
import random
import time
import numpy as np
from datetime import datetime
import math

# Import the neural network components
try:
    import torch
    from neural_network import ChessNet, board_to_input, move_to_index
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Neural network features will be disabled.")
    print("To enable neural network features, install PyTorch: pip install torch")

class ChessGame:
    def __init__(self):
        self.board = chess.Board()
        self.move_history = []
        self.save_dir = "saved_games"
        self.game_mode = "human"  # "human" or "engine"
        self.engine = ChessEngine()
        
        # Create save directory if it doesn't exist
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
    
    def get_legal_moves(self, square):
        """Get all legal moves for a piece at given square."""
        if square is None:
            return []
        return [move for move in self.board.legal_moves 
                if move.from_square == square]
    
    def make_move(self, from_square, to_square):
        """Make a move if legal."""
        # Check for promotion
        promotion = None
        if self.board.piece_at(from_square) and self.board.piece_at(from_square).piece_type == chess.PAWN:
            # If pawn is moving to the last rank
            if (chess.square_rank(to_square) == 7 and self.board.turn == chess.WHITE) or \
               (chess.square_rank(to_square) == 0 and self.board.turn == chess.BLACK):
                promotion = chess.QUEEN  # Default promotion to queen
        
        # Create the move
        if promotion:
            move = chess.Move(from_square, to_square, promotion=promotion)
        else:
            move = chess.Move(from_square, to_square)
        
        # Make the move if legal
        if move in self.board.legal_moves:
            self.move_history.append(self.board.fen())  # Save the position before the move
            self.board.push(move)
            return True
        return False
    
    def make_move_san(self, san_move):
        """Make a move using standard algebraic notation."""
        try:
            move = self.board.parse_san(san_move)
            if move in self.board.legal_moves:
                self.move_history.append(self.board.fen())  # Save the position before the move
                self.board.push(move)
                return True
        except ValueError:
            pass
        return False
    
    def make_engine_move(self):
        """Let the engine make a move."""
        if self.board.is_game_over():
            return False
        
        try:
            # Get the best move from the engine
            move_result = self.engine.get_best_move(self.board)
            
            # Check if we got a valid move (move_result is now a tuple of (move, eval))
            if move_result and isinstance(move_result, tuple) and move_result[0]:
                move = move_result[0]  # Extract the move from the tuple
                
                # Save the position before the move
                self.move_history.append(self.board.fen())
                
                # Make the move
                self.board.push(move)
                return True
            else:
                print("Engine couldn't find a valid move")
        except Exception as e:
            print(f"Error making engine move: {e}")
            import traceback
            traceback.print_exc()
        
        return False
    
    def undo_move(self):
        """Undo the last move."""
        if self.board.move_stack:
            self.board.pop()
            if self.move_history:
                self.move_history.pop()
            return True
        return False
    
    def is_game_over(self):
        """Check if the game is over."""
        return self.board.is_game_over()
    
    def get_game_result(self):
        """Get the result of the game."""
        if not self.is_game_over():
            return "Game in progress"
        
        if self.board.is_checkmate():
            winner = "Black" if self.board.turn == chess.WHITE else "White"
            return f"{winner} wins by checkmate"
        elif self.board.is_stalemate():
            return "Draw by stalemate"
        elif self.board.is_insufficient_material():
            return "Draw by insufficient material"
        elif self.board.is_fifty_moves():
            return "Draw by fifty-move rule"
        elif self.board.is_repetition():
            return "Draw by threefold repetition"
        return "Game over"
    
    def save_game(self, filename=None):
        """Save the current game state."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chess_game_{timestamp}.json"
        
        filepath = os.path.join(self.save_dir, filename)
        
        game_data = {
            "fen": self.board.fen(),
            "move_history": self.move_history,
            "game_mode": self.game_mode
        }
        
        with open(filepath, 'w') as f:
            json.dump(game_data, f)
        
        return filepath
    
    def load_game(self, filepath):
        """Load a game from a saved file."""
        if not os.path.exists(filepath):
            return False
        
        try:
            with open(filepath, 'r') as f:
                game_data = json.load(f)
            
            self.board = chess.Board(game_data["fen"])
            self.move_history = game_data["move_history"]
            if "game_mode" in game_data:
                self.game_mode = game_data["game_mode"]
            return True
        except Exception as e:
            print(f"Error loading game: {e}")
            return False
    
    def get_piece_at(self, square):
        """Get the piece at the given square."""
        return self.board.piece_at(square)
    
    def get_current_player(self):
        """Get the current player (WHITE or BLACK)."""
        return "White" if self.board.turn == chess.WHITE else "Black"
    
    def is_check(self):
        """Check if the current player is in check."""
        return self.board.is_check()
    
    def set_game_mode(self, mode):
        """Set the game mode (human or engine)."""
        if mode in ["human", "engine"]:
            self.game_mode = mode
            return True
        return False


class ChessEngine:
    """A simple chess engine with evaluation and search capabilities."""
    
    def __init__(self):
        # Piece values (standard values: pawn=1, knight=3, bishop=3, rook=5, queen=9)
        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000
        }
        
        # Position evaluation tables
        self.piece_position_tables = self._init_position_tables()
        
        # Neural network parameters
        self.nn_initialized = False
        self.weights = []
        self.biases = []
        
        # Default model name
        self.model_name = "base_model.npz"
        self.model_path = None
        
        # Get the project root directory (chess2)
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.dirname(self.script_dir)
        self.models_dir = os.path.join(self.project_root, "saved_models")
        
        # Create models directory if it doesn't exist
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir, exist_ok=True)
        
        # Search parameters - increase depth for better play
        self.search_depth = 4  # Increased from 3
        self.max_quiescence_depth = 7  # Increased from 5
        self.use_iterative_deepening = True
        self.use_mcts = True
        self.mcts_simulations = 100
        self.mcts_exploration_weight = 1.4
        
        # Evaluation caches
        self.evaluation_cache = {}
        self.transposition_table = {}
        
        # Training parameters
        self.training_data = []
        self.max_training_samples = 10000
        self.learning_rate = 0.01
        self.batch_size = 32
        
        # Temperature for move selection randomness (0-100)
        self.randomness = 10
        self.temperature = 0.1 + (self.randomness / 100.0)
        
        # Fast mode for training
        self.fast_mode = False
        
        # Try to initialize neural network
        try:
            self._init_neural_network()
        except Exception as e:
            print(f"Error initializing neural network: {e}")
            print("Using heuristic evaluation instead.")
        
        # Add contempt factor to discourage draws
        self.contempt_factor = 25  # Positive value means the engine will avoid draws
    
    def _init_position_tables(self):
        """Initialize position evaluation tables for each piece type."""
        tables = {}
        
        # Pawns: encourage advancement and center control
        tables[chess.PAWN] = [
            0,    0,   0,   0,   0,   0,   0,   0,
            50,  50,  50,  50,  50,  50,  50,  50,
            5,   15,  25,  35,  35,  25,  15,   5,
            0,   10,  20,  40,  40,  20,  10,   0,
            0,    5,  15,  30,  30,  15,   5,   0,
           -5,    0,  10,  15,  15,  10,   0,  -5,
           -5,    0,   5,   5,   5,   5,   0,  -5,
            0,    0,   0,   0,   0,   0,   0,   0
        ]
        
        # Knights: prefer center positions, avoid edges
        tables[chess.KNIGHT] = [
            -50,-40,-30,-30,-30,-30,-40,-50,
            -40,-20,  0,  0,  0,  0,-20,-40,
            -30,  0, 10, 15, 15, 10,  0,-30,
            -30,  5, 15, 20, 20, 15,  5,-30,
            -30,  0, 15, 20, 20, 15,  0,-30,
            -30,  5, 10, 15, 15, 10,  5,-30,
            -40,-20,  0,  5,  5,  0,-20,-40,
            -50,-40,-30,-30,-30,-30,-40,-50
        ]
        
        # Bishops: prefer diagonals, avoid corners
        tables[chess.BISHOP] = [
            -20,-10,-10,-10,-10,-10,-10,-20,
            -10,  0,  0,  0,  0,  0,  0,-10,
            -10,  0, 10, 10, 10, 10,  0,-10,
            -10,  5,  5, 10, 10,  5,  5,-10,
            -10,  0,  5, 10, 10,  5,  0,-10,
            -10,  5,  5,  5,  5,  5,  5,-10,
            -10,  0,  5,  0,  0,  5,  0,-10,
            -20,-10,-10,-10,-10,-10,-10,-20
        ]
        
        # Rooks: prefer open files and 7th rank
        tables[chess.ROOK] = [
            0,  0,  0,  0,  0,  0,  0,  0,
            5, 10, 10, 10, 10, 10, 10,  5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            0,  0,  0,  5,  5,  0,  0,  0
        ]
        
        # Queens: combination of rook and bishop mobility
        tables[chess.QUEEN] = [
            -20,-10,-10, -5, -5,-10,-10,-20,
            -10,  0,  0,  0,  0,  0,  0,-10,
            -10,  0,  5,  5,  5,  5,  0,-10,
            -5,  0,  5,  5,  5,  5,  0, -5,
            0,  0,  5,  5,  5,  5,  0, -5,
            -10,  5,  5,  5,  5,  5,  0,-10,
            -10,  0,  5,  0,  0,  0,  0,-10,
            -20,-10,-10, -5, -5,-10,-10,-20
        ]
        
        # Kings: early/mid game - seek shelter, avoid center
        tables[chess.KING] = [
            -30,-40,-40,-50,-50,-40,-40,-30,
            -30,-40,-40,-50,-50,-40,-40,-30,
            -30,-40,-40,-50,-50,-40,-40,-30,
            -30,-40,-40,-50,-50,-40,-40,-30,
            -20,-30,-30,-40,-40,-30,-30,-20,
            -10,-20,-20,-20,-20,-20,-20,-10,
            20, 20,  0,  0,  0,  0, 20, 20,
            20, 30, 10,  0,  0, 10, 30, 20
        ]
        
        return tables
    
    def _init_neural_network(self):
        """Initialize or load the neural network."""
        try:
            if not TORCH_AVAILABLE:
                print("PyTorch not available. Using simple neural network.")
                self.nn_initialized = False
                return
            
            # Initialize PyTorch model
            self.model = ChessNet()
            
            if self.model_path is not None:
                model_path = self.model_path
            else:
                model_path = os.path.join(self.models_dir, self.model_name)
            
            if os.path.exists(model_path):
                print(f"Loading neural network from {model_path}")
                try:
                    # Try loading as PyTorch model
                    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
                    if 'model_state_dict' in checkpoint:
                        self.model.load_state_dict(checkpoint['model_state_dict'])
                        self.model.eval()  # Set to evaluation mode
                        self.nn_initialized = True
                        print("Successfully loaded PyTorch model")
                    else:
                        # Try loading as numpy arrays
                        print("Not a PyTorch model, trying to load as numpy arrays")
                        self.nn_initialized = False
                except Exception as e:
                    print(f"Error loading model: {e}")
                    self.nn_initialized = False
            else:
                print(f"Model not found at {model_path}. Using default evaluation.")
                self.nn_initialized = False
            
        except Exception as e:
            print(f"Error initializing neural network: {e}")
            print("Using heuristic evaluation instead.")
            self.nn_initialized = False
    
    def _save_neural_network(self, model_name=None):
        """Save the neural network weights to a file."""
        if model_name is None:
            model_name = self.model_name
        
        # Ensure we're saving to the project root's saved_models directory
        if not os.path.isabs(model_name) and not model_name.startswith("saved_models/"):
            save_path = os.path.join(self.models_dir, model_name)
        else:
            save_path = model_name
        
        # Create save directory if it doesn't exist
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        
        # Save the model
        try:
            if self.nn_initialized and hasattr(self, 'model'):
                # Save PyTorch model
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                }, save_path)
                print(f"Neural network weights saved to {save_path}")
            else:
                # Save simple weights if no PyTorch model
                save_dict = {}
                for i, (w, b) in enumerate(zip(self.weights, self.biases)):
                    save_dict[f'w{i+1}'] = w
                    save_dict[f'b{i+1}'] = b
                np.savez(save_path, **save_dict)
                print(f"Simple neural network weights saved to {save_path}")
        except Exception as e:
            print(f"Error saving neural network: {e}")
            import traceback
            traceback.print_exc()
    
    def _board_to_input(self, board):
        """Convert a chess board to neural network input."""
        # Use the board_to_input function from neural_network.py
        # which creates the correct 119-channel input format
        from neural_network import board_to_input
        return board_to_input(board)
    
    def _forward_pass(self, x):
        """Perform a forward pass through the neural network."""
        if not self.nn_initialized:
            return 0
        
        # Forward pass through each layer
        activations = [x]
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(activations[-1], w) + b
            # ReLU activation for hidden layers, linear for output
            a = np.maximum(0, z) if i < len(self.weights) - 1 else z
            activations.append(a)
        
        # Return scalar output
        return float(activations[-1].item())  # Convert to scalar
    
    def _train_neural_network(self, inputs, targets):
        """Train the neural network using backpropagation."""
        if not self.nn_initialized or len(inputs) == 0:
            return
        
        # Convert to numpy arrays
        inputs = np.array(inputs)
        targets = np.array(targets).reshape(-1, 1)
        
        # Simple stochastic gradient descent
        for x, y in zip(inputs, targets):
            x = x.reshape(1, -1)  # Reshape for batch processing
            
            # Forward pass
            activations = [x]
            for i, (w, b) in enumerate(zip(self.weights, self.biases)):
                z = np.dot(activations[-1], w) + b
                # ReLU activation for hidden layers, linear for output
                a = np.maximum(0, z) if i < len(self.weights) - 1 else z
                activations.append(a)
            
            # Backward pass
            delta = activations[-1] - y
            for l in range(len(self.weights) - 1, -1, -1):
                # Gradient for weights and biases
                dw = np.dot(activations[l].T, delta)
                db = np.sum(delta, axis=0, keepdims=True)
                
                # Update weights and biases
                self.weights[l] -= self.learning_rate * dw
                self.biases[l] -= self.learning_rate * db
                
                # Compute delta for previous layer
                if l > 0:
                    # ReLU derivative: 1 if activation > 0, else 0
                    relu_derivative = (activations[l] > 0).astype(float)
                    delta = np.dot(delta, self.weights[l].T) * relu_derivative
        
        # Only save weights periodically, not after every training step
        # We'll save after accumulating a batch of samples instead
        # self._save_neural_network()
    
    def add_training_sample(self, board, evaluation):
        """Add a board position and its evaluation to the training data."""
        # Add to training data regardless of neural network initialization
        # We'll store the board object directly for proper conversion later
        self.training_data.append((board, float(evaluation)))
        
        # Limit the size of training data
        if len(self.training_data) > self.max_training_samples:
            # Remove oldest samples
            self.training_data = self.training_data[-self.max_training_samples:]
    
    def evaluate_position(self, board):
        """Evaluate the current position using neural network if available."""
        # Use cached evaluation if available
        board_hash = hash(board.fen())
        if board_hash in self.evaluation_cache:
            return self.evaluation_cache[board_hash]
        
        # Quick evaluation for terminal positions
        if board.is_checkmate():
            return -10000 if board.turn else 10000
        if board.is_stalemate() or board.is_insufficient_material():
            return 0
        
        # Use neural network if available
        if self.nn_initialized:
            try:
                with torch.no_grad():
                    x = self._board_to_input(board)
                    x = torch.FloatTensor(x).unsqueeze(0)
                    _, value = self.model(x)
                    evaluation = value.item() * 100  # Scale to centipawns
            except Exception as e:
                print(f"Error in neural network evaluation: {e}")
                evaluation = self._evaluate_heuristic(board)
        else:
            evaluation = self._evaluate_heuristic(board)
        
        # Cache the evaluation
        self.evaluation_cache[board_hash] = evaluation
        return evaluation
    
    def _evaluate_material_only(self, board):
        """Simplified material-only evaluation for fast mode."""
        material = 0
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = self.piece_values[piece.piece_type]
                if piece.color == chess.WHITE:
                    material += value
                else:
                    material -= value
        
        # Return from white's perspective
        return material if board.turn == chess.WHITE else -material
    
    def _evaluate_heuristic(self, board):
        """Evaluate the position using heuristic methods."""
        if board.is_checkmate():
            return -10000 if board.turn == chess.WHITE else 10000
        
        if board.is_stalemate() or board.is_insufficient_material():
            return 0
        
        # Material evaluation
        white_material = 0
        black_material = 0
        
        # Position evaluation
        white_position = 0
        black_position = 0
        
        # Mobility (number of legal moves)
        mobility = 0
        
        # Piece safety and protection
        white_piece_safety = 0
        black_piece_safety = 0
        
        # Threat detection
        white_threats = 0
        black_threats = 0
        
        # King safety
        white_king_safety = 0
        black_king_safety = 0
        
        # Track piece locations for later analysis
        white_pieces = {}
        black_pieces = {}
        
        # Evaluate each piece on the board
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                # Material value
                piece_value = self.piece_values[piece.piece_type]
                
                # Position value - adjust for piece color
                square_idx = square
                if piece.color == chess.BLACK:
                    # Flip the square index for black pieces
                    square_idx = chess.square_mirror(square)
                
                position_value = self.piece_position_tables[piece.piece_type][square_idx]
                
                # Store piece location for later analysis
                if piece.color == chess.WHITE:
                    white_pieces[square] = piece
                    white_material += piece_value
                    white_position += position_value
                else:
                    black_pieces[square] = piece
                    black_material += piece_value
                    black_position += position_value
        
        # Calculate mobility for the side to move
        original_turn = board.turn
        
        # Count legal moves for white
        board.turn = chess.WHITE
        white_legal_moves = list(board.legal_moves)
        white_mobility = len(white_legal_moves) * 10  # Weight mobility
        
        # Count legal moves for black
        board.turn = chess.BLACK
        black_legal_moves = list(board.legal_moves)
        black_mobility = len(black_legal_moves) * 10
        
        # Restore original turn
        board.turn = original_turn
        
        # Evaluate piece safety and protection
        for square, piece in white_pieces.items():
            # Check if the piece is protected by a friendly piece
            is_protected = board.is_attacked_by(chess.WHITE, square)
            
            # Check if the piece is under attack by an enemy piece
            is_attacked = board.is_attacked_by(chess.BLACK, square)
            
            # Evaluate piece safety
            if is_protected and not is_attacked:
                # Piece is protected and not attacked - good
                white_piece_safety += piece_value * 0.05
            elif is_protected and is_attacked:
                # Piece is protected but also attacked - neutral
                pass
            elif not is_protected and is_attacked:
                # Piece is attacked and not protected - bad
                white_piece_safety -= piece_value * 0.1
                
                # NEW: Strongly penalize hanging pieces that can be captured for free
                # Check if the attacker is defended
                attackers = list(board.attackers(chess.BLACK, square))
                if attackers:
                    cheapest_attacker_value = min(self.piece_values[board.piece_at(attacker).piece_type] 
                                                for attacker in attackers)
                    # If the piece is more valuable than its cheapest attacker, it's a bad position
                    if piece_value > cheapest_attacker_value:
                        white_piece_safety -= (piece_value - cheapest_attacker_value) * 0.5
            elif not is_protected and not is_attacked:
                # Piece is neither protected nor attacked
                # Slightly penalize unprotected pieces, especially valuable ones
                if piece.piece_type in [chess.QUEEN, chess.ROOK]:
                    white_piece_safety -= piece_value * 0.02
        
        # Same for black pieces
        for square, piece in black_pieces.items():
            is_protected = board.is_attacked_by(chess.BLACK, square)
            is_attacked = board.is_attacked_by(chess.WHITE, square)
            
            if is_protected and not is_attacked:
                black_piece_safety += piece_value * 0.05
            elif is_protected and is_attacked:
                pass
            elif not is_protected and is_attacked:
                black_piece_safety -= piece_value * 0.1
                
                # NEW: Strongly penalize hanging pieces that can be captured for free
                attackers = list(board.attackers(chess.WHITE, square))
                if attackers:
                    cheapest_attacker_value = min(self.piece_values[board.piece_at(attacker).piece_type] 
                                                for attacker in attackers)
                    # If the piece is more valuable than its cheapest attacker, it's a bad position
                    if piece_value > cheapest_attacker_value:
                        black_piece_safety -= (piece_value - cheapest_attacker_value) * 0.5
            elif not is_protected and not is_attacked:
                if piece.piece_type in [chess.QUEEN, chess.ROOK]:
                    black_piece_safety -= piece_value * 0.02
        
        # Evaluate king safety
        white_king_square = board.king(chess.WHITE)
        black_king_square = board.king(chess.BLACK)
        
        if white_king_square is not None:
            # Check pawn shield for white king
            white_king_file = chess.square_file(white_king_square)
            white_king_rank = chess.square_rank(white_king_square)
            
            # King should have pawns in front of it in the opening/middlegame
            if len(board.piece_map()) > 20:  # Not endgame
                # Check for pawns in front of the king
                pawn_shield_count = 0
                for file_offset in [-1, 0, 1]:
                    for rank_offset in [1]:  # Just check one rank ahead for simplicity
                        check_file = white_king_file + file_offset
                        check_rank = white_king_rank + rank_offset
                        
                        if 0 <= check_file < 8 and 0 <= check_rank < 8:
                            check_square = chess.square(check_file, check_rank)
                            check_piece = board.piece_at(check_square)
                            if check_piece and check_piece.piece_type == chess.PAWN and check_piece.color == chess.WHITE:
                                pawn_shield_count += 1
                
                # Bonus for having pawns in front of the king
                white_king_safety += pawn_shield_count * 15
                
                # Penalty for open files near the king
                for file_offset in [-1, 0, 1]:
                    check_file = white_king_file + file_offset
                    if 0 <= check_file < 8:
                        has_pawn = False
                        for rank in range(8):
                            square = chess.square(check_file, rank)
                            piece = board.piece_at(square)
                            if piece and piece.piece_type == chess.PAWN:
                                has_pawn = True
                                break
                        
                        if not has_pawn:
                            # Penalty for open file near king
                            white_king_safety -= 20
        
        # Same for black king
        if black_king_square is not None:
            black_king_file = chess.square_file(black_king_square)
            black_king_rank = chess.square_rank(black_king_square)
            
            if len(board.piece_map()) > 20:  # Not endgame
                pawn_shield_count = 0
                for file_offset in [-1, 0, 1]:
                    for rank_offset in [-1]:  # Check one rank behind for black
                        check_file = black_king_file + file_offset
                        check_rank = black_king_rank + rank_offset
                        
                        if 0 <= check_file < 8 and 0 <= check_rank < 8:
                            check_square = chess.square(check_file, check_rank)
                            check_piece = board.piece_at(check_square)
                            if check_piece and check_piece.piece_type == chess.PAWN and check_piece.color == chess.BLACK:
                                pawn_shield_count += 1
                
                black_king_safety += pawn_shield_count * 15
                
                for file_offset in [-1, 0, 1]:
                    check_file = black_king_file + file_offset
                    if 0 <= check_file < 8:
                        has_pawn = False
                        for rank in range(8):
                            square = chess.square(check_file, rank)
                            piece = board.piece_at(square)
                            if piece and piece.piece_type == chess.PAWN:
                                has_pawn = True
                                break
                        
                        if not has_pawn:
                            black_king_safety -= 20
        
        # Detect tactical threats
        
        # 1. Identify forks (one piece attacking multiple pieces)
        for square in chess.SQUARES:
            # Check white pieces that can attack multiple black pieces
            attackers = board.attackers(chess.WHITE, square)
            if attackers:
                piece_at_square = board.piece_at(square)
                if piece_at_square and piece_at_square.color == chess.BLACK:
                    # This square has a black piece and is attacked by white
                    for attacker_square in attackers:
                        # Check if this attacker is also attacking other black pieces
                        attacker = board.piece_at(attacker_square)
                        if attacker:
                            other_attacks = 0
                            total_value = piece_at_square.piece_type
                            
                            for other_square, other_piece in black_pieces.items():
                                if other_square != square and board.is_attacked_by(chess.WHITE, other_square):
                                    # Check if the same attacker can attack this square
                                    if self._can_piece_attack(board, attacker_square, other_square):
                                        other_attacks += 1
                                        total_value += other_piece.piece_type
                            
                            if other_attacks > 0:
                                # We found a fork! Bonus based on value of pieces being forked
                                white_threats += total_value * 0.2
            
            # Same for black pieces
            attackers = board.attackers(chess.BLACK, square)
            if attackers:
                piece_at_square = board.piece_at(square)
                if piece_at_square and piece_at_square.color == chess.WHITE:
                    for attacker_square in attackers:
                        attacker = board.piece_at(attacker_square)
                        if attacker:
                            other_attacks = 0
                            total_value = piece_at_square.piece_type
                            
                            for other_square, other_piece in white_pieces.items():
                                if other_square != square and board.is_attacked_by(chess.BLACK, other_square):
                                    if self._can_piece_attack(board, attacker_square, other_square):
                                        other_attacks += 1
                                        total_value += other_piece.piece_type
                            
                            if other_attacks > 0:
                                black_threats += total_value * 0.2
        
        # 2. Identify pins (piece can't move because it would expose a more valuable piece)
        for square, piece in white_pieces.items():
            if piece.piece_type != chess.KING:
                # Try removing this piece and see if the king is in check
                board.remove_piece_at(square)
                if board.is_attacked_by(chess.BLACK, board.king(chess.WHITE)):
                    # This piece is pinned
                    black_threats += piece.piece_type * 0.15
                # Restore the piece
                board.set_piece_at(square, piece)
        
        for square, piece in black_pieces.items():
            if piece.piece_type != chess.KING:
                board.remove_piece_at(square)
                if board.is_attacked_by(chess.WHITE, board.king(chess.BLACK)):
                    white_threats += piece.piece_type * 0.15
                board.set_piece_at(square, piece)
        
        # 3. Identify discovered attack potential
        # This is complex, but we can approximate by checking if moving a piece would
        # allow another piece to attack something valuable
        
        # Endgame knowledge
        if len(board.piece_map()) <= 10:  # Endgame
            # King centralization in endgames
            wking_square = board.king(chess.WHITE) if board.king(chess.WHITE) is not None else -1
            bking_square = board.king(chess.BLACK) if board.king(chess.BLACK) is not None else -1
            
            if wking_square != -1:
                # Center distance (0 for e4/d4/e5/d5, higher for corners)
                wking_centralization = 7 - (abs(chess.square_file(wking_square) - 3.5) + 
                                      abs(chess.square_rank(wking_square) - 3.5))
                # In endgames, king centralization is important
                white_position += wking_centralization * 10
            
            if bking_square != -1:
                bking_centralization = 7 - (abs(chess.square_file(bking_square) - 3.5) + 
                                      abs(chess.square_rank(bking_square) - 3.5))
                black_position += bking_centralization * 10
            
            # Pawn promotion incentive
            for square in chess.SQUARES:
                piece = board.piece_at(square)
                if piece and piece.piece_type == chess.PAWN:
                    if piece.color == chess.WHITE:
                        # Distance to promotion (7 - rank)
                        promotion_distance = 7 - chess.square_rank(square)
                        white_position += (7 - promotion_distance) * 5
                    else:
                        promotion_distance = chess.square_rank(square)
                        black_position += (7 - promotion_distance) * 5
        
        # Development incentive in opening
        if len(board.move_stack) < 15:  # Opening phase
            # Penalize unmoved knights and bishops
            if board.piece_at(chess.B1) and board.piece_at(chess.B1).piece_type == chess.KNIGHT:
                white_position -= 20
            if board.piece_at(chess.G1) and board.piece_at(chess.G1).piece_type == chess.KNIGHT:
                white_position -= 20
            if board.piece_at(chess.C1) and board.piece_at(chess.C1).piece_type == chess.BISHOP:
                white_position -= 15
            if board.piece_at(chess.F1) and board.piece_at(chess.F1).piece_type == chess.BISHOP:
                white_position -= 15
            
            # Same for black
            if board.piece_at(chess.B8) and board.piece_at(chess.B8).piece_type == chess.KNIGHT:
                black_position -= 20
            if board.piece_at(chess.G8) and board.piece_at(chess.G8).piece_type == chess.KNIGHT:
                black_position -= 20
            if board.piece_at(chess.C8) and board.piece_at(chess.C8).piece_type == chess.BISHOP:
                black_position -= 15
            if board.piece_at(chess.F8) and board.piece_at(chess.F8).piece_type == chess.BISHOP:
                black_position -= 15
        
        # Add pawn structure evaluation
        white_pawn_structure = 0
        black_pawn_structure = 0
        
        # Penalize doubled pawns
        for file in range(8):
            white_pawns_on_file = 0
            black_pawns_on_file = 0
            for rank in range(8):
                square = chess.square(file, rank)
                piece = board.piece_at(square)
                if piece and piece.piece_type == chess.PAWN:
                    if piece.color == chess.WHITE:
                        white_pawns_on_file += 1
                    else:
                        black_pawns_on_file += 1
            
            # Penalize doubled pawns
            if white_pawns_on_file > 1:
                white_pawn_structure -= 20 * (white_pawns_on_file - 1)
            if black_pawns_on_file > 1:
                black_pawn_structure -= 20 * (black_pawns_on_file - 1)
        
        # Add pawn structure to position evaluation
        white_position += white_pawn_structure
        black_position += black_pawn_structure
        
        # Calculate final evaluation from white's perspective
        material_eval = white_material - black_material
        position_eval = white_position - black_position
        mobility_eval = white_mobility - black_mobility
        safety_eval = white_piece_safety - black_piece_safety
        king_safety_eval = white_king_safety - black_king_safety
        threat_eval = white_threats - black_threats
        
        # Check and checkmate threats
        check_eval = 0
        if board.is_check():
            check_eval = -50 if board.turn == chess.WHITE else 50
        
        # Combine all factors with weights
        evaluation = (
            material_eval * 1.2 +      # Increased material importance
            position_eval * 0.5 +      # Increased position importance for center control
            mobility_eval * 0.2 +      # Mobility remains the same
            safety_eval * 0.4 +        # Increased piece safety for better capture awareness
            king_safety_eval * 0.3 +   # King safety remains the same
            threat_eval * 0.5 +        # Increased tactical threats for better attacking
            check_eval * 1.2           # Increased check importance
        )
        
        # Add contempt factor to avoid draws when ahead
        if abs(evaluation) < 100:  # Close to equal
            evaluation += self.contempt_factor if board.turn == chess.WHITE else -self.contempt_factor
        
        return evaluation
    
    def _can_piece_attack(self, board, from_square, to_square):
        """Check if a piece on from_square can attack to_square."""
        piece = board.piece_at(from_square)
        if not piece:
            return False
            
        # Get all squares the piece attacks
        if piece.piece_type == chess.PAWN:
            # Special case for pawns which can only capture diagonally
            from_file = chess.square_file(from_square)
            from_rank = chess.square_rank(from_square)
            to_file = chess.square_file(to_square)
            to_rank = chess.square_rank(to_square)
            
            # Check if the target square is a diagonal capture
            if piece.color == chess.WHITE:
                return (to_rank == from_rank + 1) and abs(to_file - from_file) == 1
            else:
                return (to_rank == from_rank - 1) and abs(to_file - from_file) == 1
        else:
            # For other pieces, check if the move would be legal
            # ignoring whether the king would be in check
            try:
                move = chess.Move(from_square, to_square)
                return move in board.pseudo_legal_moves
            except:
                return False
    
    def search(self, board, depth=None):
        """Search for the best move using a simple evaluation approach.
        
        Args:
            board: A chess.Board object representing the current position
            depth: Search depth (ignored in this simple implementation)
            
        Returns:
            Tuple of (best_move, evaluation)
        """
        # Use default depth if not specified
        if depth is None:
            depth = self.search_depth
        
        # Get all legal moves
        legal_moves = list(board.legal_moves)
        
        # If no legal moves, return None
        if not legal_moves:
            return None, 0
        
        # If only one legal move, return it
        if len(legal_moves) == 1:
            return legal_moves[0], self.evaluate_position(board)
        
        # Evaluate each legal move
        move_evals = []
        for move in legal_moves:
            # Create a new board with the move applied
            new_board = board.copy()
            new_board.push(move)
            
            # Evaluate the position
            eval_score = self.evaluate_position(new_board)
            
            # Negate the evaluation since we're alternating players
            if board.turn == chess.BLACK:
                eval_score = -eval_score
            
            move_evals.append((move, eval_score))
        
        # Sort moves by evaluation
        move_evals.sort(key=lambda x: x[1], reverse=True)
        
        # Add randomness for variety in play
        if self.randomness > 0 and len(move_evals) > 1:
            # Apply temperature to soften the evaluation differences
            temperature = self.temperature
            
            # Convert evaluations to probabilities using softmax with temperature
            evals = [e[1] for e in move_evals]
            max_eval = max(evals)
            exp_evals = [math.exp((e - max_eval) / temperature) for e in evals]
            sum_exp_evals = sum(exp_evals)
            probs = [e / sum_exp_evals for e in exp_evals]
            
            # Choose move based on probabilities
            r = random.random()
            cumulative_prob = 0
            for i, prob in enumerate(probs):
                cumulative_prob += prob
                if r <= cumulative_prob:
                    best_move, best_eval = move_evals[i]
                    break
            else:
                # Fallback to best move
                best_move, best_eval = move_evals[0]
        else:
            # Just take the best move
            best_move, best_eval = move_evals[0]
        
        return best_move, best_eval

    def _train_neural_network_batch(self):
        """Train the neural network on accumulated training data in batches."""
        if not self.nn_initialized:
            print("Neural network not initialized. Cannot train.")
            return
        
        if not self.training_data:
            print("No training data available.")
            return
        
        try:
            # Store the number of samples for reporting
            num_samples = len(self.training_data)
            
            # Check if we're using PyTorch
            if hasattr(self, 'model') and isinstance(self.model, torch.nn.Module):
                # PyTorch training
                print(f"Training PyTorch model on {num_samples} positions")
                
                # Create optimizer
                optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
                
                # Convert training data to PyTorch tensors
                inputs = []
                values = []
                
                for board, evaluation in self.training_data:
                    # Convert board to input features
                    x = board_to_input(board)
                    inputs.append(x)
                    
                    # Normalize evaluation to [-1, 1] range for tanh activation
                    normalized_eval = np.tanh(evaluation / 1000.0)  # Scale by 1000 (centipawns)
                    values.append(normalized_eval)
                
                # Convert to PyTorch tensors
                inputs = torch.FloatTensor(np.array(inputs))
                values = torch.FloatTensor(np.array(values)).unsqueeze(1)  # Add batch dimension
                
                # Create dummy policy targets (we only care about value training for now)
                policies = torch.zeros((len(inputs), 1968))
                
                # Set model to training mode
                self.model.train()
                
                # Train in batches
                indices = np.arange(num_samples)
                np.random.shuffle(indices)
                
                # Process in batches
                for start_idx in range(0, num_samples, self.batch_size):
                    end_idx = min(start_idx + self.batch_size, num_samples)
                    batch_indices = indices[start_idx:end_idx]
                    
                    batch_inputs = inputs[batch_indices]
                    batch_policies = policies[batch_indices]
                    batch_values = values[batch_indices]
                    
                    # Zero gradients
                    optimizer.zero_grad()
                    
                    # Forward pass
                    policy_out, value_out = self.model(batch_inputs)
                    
                    # Compute loss (MSE for value, ignore policy for now)
                    loss = torch.nn.functional.mse_loss(value_out, batch_values)
                    
                    # Backward pass and optimize
                    loss.backward()
                    optimizer.step()
                
                # Set model back to evaluation mode
                self.model.eval()
                
            else:
                # Simple neural network training
                # Convert training data to numpy arrays
                inputs = []
                targets = []
                
                for board, evaluation in self.training_data:
                    # Convert board to input features
                    x = self._board_to_input(board)
                    inputs.append(x)
                    
                    # Normalize evaluation to [-1, 1] range for tanh activation
                    normalized_eval = np.tanh(evaluation / 1000.0)  # Scale by 1000 (centipawns)
                    targets.append(normalized_eval)
                
                inputs = np.array(inputs)
                targets = np.array(targets)
                
                # Train in batches
                indices = np.arange(num_samples)
                np.random.shuffle(indices)
                
                # Process in batches
                for start_idx in range(0, num_samples, self.batch_size):
                    end_idx = min(start_idx + self.batch_size, num_samples)
                    batch_indices = indices[start_idx:end_idx]
                    
                    batch_inputs = inputs[batch_indices]
                    batch_targets = targets[batch_indices]
                    
                    # Train on this batch
                    self._train_neural_network(batch_inputs, batch_targets)
            
            # Clear training data after training
            self.training_data = []
            
            print(f"Trained neural network on {num_samples} positions.")
            
        except Exception as e:
            print(f"Error training neural network: {e}")
            import traceback
            traceback.print_exc()

    def _iterative_deepening_search(self, board, max_depth):
        """Perform iterative deepening search to find the best move.
        
        Args:
            board: A chess.Board object representing the current position
            max_depth: Maximum search depth
            
        Returns:
            Tuple of (best_move, evaluation)
        """
        best_move = None
        best_eval = 0
        
        # Start with depth 1 and increase
        for depth in range(1, max_depth + 1):
            # Use regular minimax search with the current depth
            move, eval = self.search(board, depth)
            
            # Update best move if we found one
            if move:
                best_move = move
                best_eval = eval
        
        return best_move, best_eval
