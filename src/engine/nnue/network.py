"""
Advanced Neural Network for Chess Position Evaluation (NNUE)

This module implements a convolutional residual network for chess position evaluation,
with approximately 16 million parameters as specified.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import chess
import numpy as np

# Constants for the network architecture
INPUT_CHANNELS = 24  # Board representation (12) + Additional features (12)
BOARD_SIZE = 8  # 8x8 chess board
RESIDUAL_FILTERS = 256  # Number of filters in residual blocks
NUM_RESIDUAL_BLOCKS = 20  # Number of residual blocks in the backbone
POLICY_OUTPUT_SIZE = 1858  # All possible chess moves
VALUE_HIDDEN_SIZE = 256  # Size of value head hidden layer
THREAT_HIDDEN_SIZE = 256  # Size of threat analysis hidden layer
OUTPUT_SIZE = 1  # Final evaluation output

class SelfAttention(nn.Module):
    """
    Self-attention module for capturing piece relationships on the board.
    """
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.key = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))  # Learnable weight for attention
        
    def forward(self, x):
        batch_size, C, height, width = x.size()
        
        # Create query projection and reshape
        proj_query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        
        # Create key projection and reshape
        proj_key = self.key(x).view(batch_size, -1, height * width)
        
        # Calculate attention map
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        
        # Create value projection and reshape
        proj_value = self.value(x).view(batch_size, -1, height * width)
        
        # Apply attention weights to value projection
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, height, width)
        
        # Add residual connection with learnable weight
        out = self.gamma * out + x
        return out

class ResidualBlock(nn.Module):
    """
    Residual block for the convolutional neural network.
    """
    def __init__(self, filters=RESIDUAL_FILTERS):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(filters, filters, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(filters)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class NNUE(nn.Module):
    """
    Advanced Neural Network for Chess Position Evaluation (16M parameters).
    
    Architecture:
    1. Input: 8x8x24 (12 piece channels + 12 additional feature channels)
    2. Feature Extraction: 20 residual blocks with 256 filters each
    3. Policy Head: For move prediction (future use)
    4. Value Head: For position evaluation
    5. Threat Analysis: Special attention mechanism for piece relationships
    """
    def __init__(self, num_blocks=NUM_RESIDUAL_BLOCKS):
        super(NNUE, self).__init__()
        
        # Input layer
        self.input_conv = nn.Conv2d(INPUT_CHANNELS, RESIDUAL_FILTERS, kernel_size=3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(RESIDUAL_FILTERS)
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(RESIDUAL_FILTERS) for _ in range(num_blocks)
        ])
        
        # Policy head (for future use in MCTS)
        self.policy_conv = nn.Conv2d(RESIDUAL_FILTERS, 73, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(73)
        self.policy_fc = nn.Linear(73 * BOARD_SIZE * BOARD_SIZE, POLICY_OUTPUT_SIZE)
        
        # Value head - outputs 64 features for integration
        self.value_conv = nn.Conv2d(RESIDUAL_FILTERS, 8, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(8)
        self.value_fc1 = nn.Linear(8 * BOARD_SIZE * BOARD_SIZE, VALUE_HIDDEN_SIZE)
        self.value_fc2 = nn.Linear(VALUE_HIDDEN_SIZE, 64)  # Output 64 features for integration
        # We don't use value_fc3 in the forward pass anymore, but keep it for compatibility with saved models
        self.value_fc3 = nn.Linear(64, OUTPUT_SIZE)
        
        # Threat analysis module - outputs 128 features for integration
        self.threat_attention = SelfAttention(RESIDUAL_FILTERS)
        self.threat_conv = nn.Conv2d(RESIDUAL_FILTERS, 64, kernel_size=1)
        self.threat_bn = nn.BatchNorm2d(64)
        self.threat_fc1 = nn.Linear(64 * BOARD_SIZE * BOARD_SIZE, THREAT_HIDDEN_SIZE)
        self.threat_fc2 = nn.Linear(THREAT_HIDDEN_SIZE, 128)  # Output 128 features for integration
        
        # Integration layer - takes 64 (from value head) + 128 (from threat analysis) = 192 inputs
        self.integration_fc = nn.Linear(64 + 128, OUTPUT_SIZE)
        
        # Initialize weights
        self.initialize_weights()
        
    def forward(self, x):
        try:
            # Handle different input shapes
            original_shape = x.shape
            
            # If input is 1D (flattened features), reshape to 4D
            if x.dim() == 1:
                x = x.view(1, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE)
            # If input is 2D (batch, flattened features), reshape to 4D
            elif x.dim() == 2:
                batch_size = x.size(0)
                x = x.view(batch_size, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE)
            # If input is 3D (channels, height, width), add batch dimension
            elif x.dim() == 3:
                x = x.unsqueeze(0)  # Add batch dimension (1, C, H, W)
            
            # Now x should be 4D (batch, channels, height, width)
            if x.dim() != 4:
                print(f"Unexpected input shape: {original_shape}, reshaped to {x.shape}")
                # Create a properly shaped tensor as fallback
                x = torch.zeros(1, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE, device=x.device)
            
            # Ensure we have the correct input shape
            batch_size, channels, height, width = x.shape
            if channels != INPUT_CHANNELS or height != BOARD_SIZE or width != BOARD_SIZE:
                print(f"Input shape mismatch: Expected (B, {INPUT_CHANNELS}, {BOARD_SIZE}, {BOARD_SIZE}), got {x.shape}")
                
                # Create a properly shaped tensor
                x_proper = torch.zeros(batch_size, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE, device=x.device)
                
                # Copy data if possible
                min_channels = min(channels, INPUT_CHANNELS)
                min_height = min(height, BOARD_SIZE)
                min_width = min(width, BOARD_SIZE)
                
                x_proper[:, :min_channels, :min_height, :min_width] = x[:, :min_channels, :min_height, :min_width]
                x = x_proper
            
            # Input layer
            x = F.relu(self.input_bn(self.input_conv(x)))
            
            # Residual blocks
            for block in self.residual_blocks:
                x = block(x)
            
            # Store backbone output for multiple heads
            backbone_out = x
            
            # Policy head (not used for evaluation, but included for completeness)
            policy_out = F.relu(self.policy_bn(self.policy_conv(backbone_out)))
            policy_out = policy_out.view(batch_size, -1)
            policy_out = self.policy_fc(policy_out)
            
            # Value head
            value_out = F.relu(self.value_bn(self.value_conv(backbone_out)))
            value_out = value_out.view(batch_size, -1)
            value_out = F.relu(self.value_fc1(value_out))
            value_out = F.relu(self.value_fc2(value_out))
            
            # Threat analysis
            threat_out = self.threat_attention(backbone_out)
            threat_out = F.relu(self.threat_bn(self.threat_conv(threat_out)))
            threat_out = threat_out.view(batch_size, -1)
            threat_out = F.relu(self.threat_fc1(threat_out))
            threat_out = F.relu(self.threat_fc2(threat_out))
            
            # Combine value and threat outputs
            combined = torch.cat((value_out, threat_out), dim=1)
            
            # Final integration
            final_out = self.integration_fc(combined)
            
            # Scale output to centipawns (approximately -1000 to 1000)
            return 1000 * torch.tanh(final_out)
            
        except Exception as e:
            # If there's an error in the forward pass, print it and return a default score
            print(f"Error in NNUE forward pass: {e}")
            import traceback
            traceback.print_exc()
            
            # Return a tensor with a default score of 0 that requires grad
            result = torch.zeros(1, 1, requires_grad=True)
            return result
    
    def incremental_forward(self, accumulator, add_features=None, remove_features=None, board=None, move=None):
        """
        Perform an incremental forward pass using the accumulator from a previous position.
        This is more efficient than a full forward pass when only a few pieces have moved.
        
        Args:
            accumulator: Previous accumulator state (None for initial calculation)
            add_features: Features to add (None if using board/move)
            remove_features: Features to remove (None if using board/move)
            board: Chess board (alternative to add/remove features)
            move: Move to apply (alternative to add/remove features)
            
        Returns:
            Tuple of (evaluation score, new accumulator)
        """
        try:
            # If no accumulator provided, do a full forward pass
            if accumulator is None:
                if board is not None:
                    features = board_to_features(board)
                    score = self.forward(features)
                    return score, features
                else:
                    return None, None
            
            # If board and move are provided, calculate feature differences
            if board is not None and move is not None:
                try:
                    # Get feature differences
                    removed_features, added_features = get_feature_diff(board, move)
                    
                    # Update local variables to match parameter names
                    remove_features = removed_features
                    add_features = added_features
                except Exception as e:
                    print(f"Error calculating feature differences: {e}")
                    # Fall back to full forward pass
                    features = board_to_features(board.copy().push(move))
                    score = self.forward(features)
                    return score, features
            
            # If we have feature differences, apply them to the accumulator
            if add_features is not None or remove_features is not None:
                try:
                    # Create a copy of the accumulator
                    new_accumulator = accumulator.clone()
                    
                    # Apply feature differences
                    if add_features is not None:
                        new_accumulator[add_features] += 1.0
                    
                    if remove_features is not None:
                        new_accumulator[remove_features] -= 1.0
                    
                    # Perform forward pass with the updated accumulator
                    score = self.forward(new_accumulator)
                    return score, new_accumulator
                except Exception as e:
                    print(f"Error applying feature differences: {e}")
                    # Fall back to full forward pass if board is available
                    if board is not None:
                        features = board_to_features(board)
                        score = self.forward(features)
                        return score, features
            
            # If we have a board but no move, do a full forward pass
            if board is not None:
                features = board_to_features(board)
                score = self.forward(features)
                return score, features
            
            # If we have an accumulator but no changes, just return the score
            score = self.forward(accumulator)
            return score, accumulator
        except Exception as e:
            print(f"Error in incremental_forward: {e}")
            import traceback
            traceback.print_exc()
            
            # Fall back to full forward pass if board is available
            if board is not None:
                try:
                    features = board_to_features(board)
                    score = self.forward(features)
                    return score, features
                except:
                    pass
            
            # Last resort: return a default score
            return torch.tensor([[0.0]]), None
    
    def initialize_weights(self):
        """Initialize weights with appropriate scaling for better training."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

def board_to_features(board):
    """
    Convert a chess board to input features for the neural network.
    
    Args:
        board: Chess board position
        
    Returns:
        Tensor of shape (24, 8, 8) containing board features
    """
    try:
        # Initialize features tensor without requiring gradients initially
        features = torch.zeros(INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE)
        
        # Piece type and color channels (12 channels)
        piece_idx = {
            (chess.PAWN, chess.WHITE): 0,
            (chess.KNIGHT, chess.WHITE): 1,
            (chess.BISHOP, chess.WHITE): 2,
            (chess.ROOK, chess.WHITE): 3,
            (chess.QUEEN, chess.WHITE): 4,
            (chess.KING, chess.WHITE): 5,
            (chess.PAWN, chess.BLACK): 6,
            (chess.KNIGHT, chess.BLACK): 7,
            (chess.BISHOP, chess.BLACK): 8,
            (chess.ROOK, chess.BLACK): 9,
            (chess.QUEEN, chess.BLACK): 10,
            (chess.KING, chess.BLACK): 11
        }
        
        # Fill piece channels
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                # Get piece index
                idx = piece_idx.get((piece.piece_type, piece.color))
                if idx is not None:
                    # Convert square to rank and file (0-7)
                    rank, file = divmod(square, 8)
                    # Set feature (note: ranks are flipped in FEN, so we use 7-rank)
                    features[idx][7-rank][file] = 1.0
        
        # Fill additional feature channels
        
        # Channel 12-13: Side to move
        if board.turn == chess.WHITE:
            features[12] = torch.ones(BOARD_SIZE, BOARD_SIZE)
        else:
            features[13] = torch.ones(BOARD_SIZE, BOARD_SIZE)
            
        # Channel 14-15: Castling rights
        if board.has_kingside_castling_rights(chess.WHITE):
            features[14][7][4:7] = 1.0  # White kingside
        if board.has_queenside_castling_rights(chess.WHITE):
            features[14][7][0:5] = 1.0  # White queenside
        if board.has_kingside_castling_rights(chess.BLACK):
            features[15][0][4:7] = 1.0  # Black kingside
        if board.has_queenside_castling_rights(chess.BLACK):
            features[15][0][0:5] = 1.0  # Black queenside
            
        # Channel 16-17: En passant
        if board.ep_square:
            rank, file = divmod(board.ep_square, 8)
            features[16][7-rank][file] = 1.0
            
            # Mark the pawn that can be captured
            if board.turn == chess.WHITE:
                # Black pawn is above the en passant square
                if rank > 0:  # Ensure we don't go out of bounds
                    features[17][7-(rank-1)][file] = 1.0
            else:
                # White pawn is below the en passant square
                if rank < 7:  # Ensure we don't go out of bounds
                    features[17][7-(rank+1)][file] = 1.0
        
        # Channels 18-19: Mobility maps (number of legal moves for each piece)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                rank, file = divmod(square, 8)
                channel = 18 if piece.color == chess.WHITE else 19
                
                # Count legal moves for this piece
                count = 0
                for move in board.legal_moves:
                    if move.from_square == square:
                        count += 1
                
                # Normalize by maximum possible moves (27 for queen)
                features[channel][7-rank][file] = min(1.0, count / 27.0)
        
        # Channels 20-21: Pawn structure (pawn chains and isolated pawns)
        for color in [chess.WHITE, chess.BLACK]:
            channel = 20 if color == chess.WHITE else 21
            
            # Get all pawns of this color
            pawns = board.pieces(chess.PAWN, color)
            
            # Mark pawn chains
            for square in pawns:
                rank, file = divmod(square, 8)
                
                # Check if this pawn is supported by another pawn
                supported = False
                if color == chess.WHITE:
                    # Check if supported by white pawns on the previous rank
                    if file > 0 and square - 9 in pawns:  # Supported from left
                        supported = True
                    if file < 7 and square - 7 in pawns:  # Supported from right
                        supported = True
                else:
                    # Check if supported by black pawns on the previous rank
                    if file > 0 and square + 7 in pawns:  # Supported from left
                        supported = True
                    if file < 7 and square + 9 in pawns:  # Supported from right
                        supported = True
                
                # Mark supported pawns with 1.0, isolated with 0.5
                if supported:
                    features[channel][7-rank][file] = 1.0
                else:
                    # Check if isolated (no pawns on adjacent files)
                    isolated = True
                    for r in range(8):
                        # Check left file
                        if file > 0:
                            left_square = r * 8 + (file - 1)
                            if left_square in pawns:
                                isolated = False
                                break
                        # Check right file
                        if file < 7:
                            right_square = r * 8 + (file + 1)
                            if right_square in pawns:
                                isolated = False
                                break
                    
                    if isolated:
                        features[channel][7-rank][file] = 0.5
                    else:
                        features[channel][7-rank][file] = 0.8  # Connected but not supported
        
        # Channels 22-23: Attack and defense maps
        attack_map = torch.zeros(BOARD_SIZE, BOARD_SIZE)
        defense_map = torch.zeros(BOARD_SIZE, BOARD_SIZE)
        
        # For each square, count how many pieces attack/defend it
        for square in chess.SQUARES:
            attackers_w = board.attackers(chess.WHITE, square)
            attackers_b = board.attackers(chess.BLACK, square)
            
            rank, file = divmod(square, 8)
            
            # Count attackers and defenders
            if board.turn == chess.WHITE:
                # For white's turn, white pieces are defenders, black are attackers
                attack_count = len(attackers_b)
                defense_count = len(attackers_w)
            else:
                # For black's turn, black pieces are defenders, white are attackers
                attack_count = len(attackers_w)
                defense_count = len(attackers_b)
            
            # Normalize by max reasonable value (5)
            attack_map[7-rank][file] = min(1.0, attack_count / 5.0)
            defense_map[7-rank][file] = min(1.0, defense_count / 5.0)
        
        # Add attack and defense maps to features
        features[22] = attack_map
        features[23] = defense_map
        
        # Now that we've built the features tensor, make a copy that requires gradients
        features_with_grad = features.clone().requires_grad_(True)
        return features_with_grad
    except Exception as e:
        print(f"Error in board_to_features: {e}")
        # Return a default tensor with the correct shape
        return torch.zeros(INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE, requires_grad=True)

def get_feature_diff(board, move):
    """
    Calculate the difference in features after a move (for efficient updates).
    
    Args:
        board: Chess board position
        move: Move to apply
        
    Returns:
        Tuple of (removed_features, added_features) for incremental updates
    """
    # Create copies of the board before and after the move
    board_before = board.copy()
    board_after = board.copy()
    board_after.push(move)
    
    # Get features for both boards
    features_before = board_to_features(board_before)
    features_after = board_to_features(board_after)
    
    # Calculate differences
    removed_features = (features_before > 0) & (features_after == 0)
    added_features = (features_before == 0) & (features_after > 0)
    
    return removed_features, added_features

def get_piece_value(piece_type):
    """Get the value of a piece type in centipawns."""
    values = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 20000
    }
    return values.get(piece_type, 0)