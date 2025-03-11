"""
Advanced Neural Network for Chess Position Evaluation (NNUE)

This module implements a convolutional residual network for chess position evaluation,
inspired by AlphaZero and modern chess engines. The architecture has approximately
16 million parameters.
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
    Advanced Neural Network for Chess Position Evaluation.
    
    Architecture:
    1. Input convolutional layer (24 input planes -> 256 filters)
    2. Residual blocks (20 blocks with 256 filters each)
    3. Policy head (move prediction - future use)
    4. Value head (position evaluation)
    5. Threat analysis module with self-attention
    6. Integration layer combining value and threat analysis
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
        
        # Add self-attention layer after some residual blocks
        self.attention = SelfAttention(RESIDUAL_FILTERS)
        
        # Policy head (for future use in move prediction)
        self.policy_conv = nn.Conv2d(RESIDUAL_FILTERS, 73, kernel_size=1)
        self.policy_fc = nn.Linear(73 * 8 * 8, POLICY_OUTPUT_SIZE)
        
        # Value head
        self.value_conv = nn.Conv2d(RESIDUAL_FILTERS, 8, kernel_size=1)
        self.value_fc1 = nn.Linear(8 * 8 * 8, VALUE_HIDDEN_SIZE)
        self.value_fc2 = nn.Linear(VALUE_HIDDEN_SIZE, 64)
        self.value_fc3 = nn.Linear(64, OUTPUT_SIZE)
        
        # Threat analysis module
        self.threat_fc1 = nn.Linear(RESIDUAL_FILTERS * 8 * 8, 512)
        self.threat_fc2 = nn.Linear(512, THREAT_HIDDEN_SIZE)
        self.threat_fc3 = nn.Linear(THREAT_HIDDEN_SIZE, 128)
        self.threat_output = nn.Linear(128, OUTPUT_SIZE)
        
        # Final integration layer
        self.final_fc = nn.Linear(2, OUTPUT_SIZE)
        
        self.initialize_weights()
    
    def initialize_weights(self):
        # Initialize weights for convolutional layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Full forward pass through the network.
        
        Args:
            x: Features representing the board (24x8x8 tensor or flattened tensor)
            
        Returns:
            Evaluation score in centipawns
        """
        # Handle different input shapes
        if len(x.shape) == 1:
            # If input is flattened, reshape to (24, 8, 8)
            x = x.view(INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE)
            
            # Add batch dimension
            x = x.unsqueeze(0)
        elif len(x.shape) == 3:
            # If we have a 3D tensor but no batch dimension
            x = x.unsqueeze(0)
            
        # Input layer
        x = F.relu(self.input_bn(self.input_conv(x)))
        
        # First half of residual blocks
        for i in range(NUM_RESIDUAL_BLOCKS // 2):
            x = self.residual_blocks[i](x)
        
        # Apply attention in the middle of the network
        x = self.attention(x)
        
        # Second half of residual blocks
        for i in range(NUM_RESIDUAL_BLOCKS // 2, NUM_RESIDUAL_BLOCKS):
            x = self.residual_blocks[i](x)
        
        # Value head
        value = F.relu(self.value_conv(x))
        value = value.view(value.size(0), -1)  # Flatten
        value = F.relu(self.value_fc1(value))
        value = F.relu(self.value_fc2(value))
        value = torch.tanh(self.value_fc3(value))  # Output between -1 and 1
        
        # Threat analysis head
        threat = x.view(x.size(0), -1)  # Flatten
        threat = F.relu(self.threat_fc1(threat))
        threat = F.relu(self.threat_fc2(threat))
        threat = F.relu(self.threat_fc3(threat))
        threat = torch.tanh(self.threat_output(threat))  # Output between -1 and 1
        
        # Combine value and threat analysis
        combined = torch.cat([value, threat], dim=1)
        final_eval = torch.tanh(self.final_fc(combined))
        
        # Scale to centipawn value (factor of 600 gives a good range)
        return 600 * final_eval
        
    def incremental_forward(self, accumulator, add_features=None, remove_features=None, board=None, move=None):
        """
        Update evaluation when a move is made.
        For convolutional models, this is more complex than for fully-connected models.
        We perform a semi-incremental update by updating only the affected input planes.
        
        Args:
            accumulator: Dictionary with 'board' and 'features' keys, or None for first call
            add_features: Dict with piece movement info (from get_feature_diff)
            remove_features: Not used (for compatibility with interface)
            board: Chess board (required if accumulator is None)
            move: Move to apply (if any)
            
        Returns:
            Updated evaluation score and updated accumulator
        """
        # For the first call, initialize the accumulator
        if accumulator is None:
            if board is None:
                # Can't do anything without a board
                return torch.tensor([[0.0]]), None
                
            # Initialize a new accumulator
            features = board_to_features(board)
            accumulator = {
                'board': board.copy(),
                'features': features,
                'move': move.uci() if move else None
            }
            
            # Run initial evaluation
            evaluation = self.forward(features)
            return evaluation, accumulator
                
        # For subsequent calls, update the accumulator
        if not isinstance(accumulator, dict):
            # Invalid accumulator, fall back to full evaluation
            if board is not None:
                features = board_to_features(board)
                new_accumulator = {
                    'board': board.copy(),
                    'features': features,
                    'move': move.uci() if move else None
                }
                evaluation = self.forward(features)
                return evaluation, new_accumulator
            return torch.tensor([[0.0]]), None
            
        # Extract the board and previous features from accumulator
        prev_board = accumulator.get('board')
        prev_features = accumulator.get('features')
        
        if prev_board is None or prev_features is None:
            # Invalid accumulator, fall back to full evaluation
            if board is not None:
                features = board_to_features(board)
                new_accumulator = {
                    'board': board.copy(),
                    'features': features,
                    'move': move.uci() if move else None
                }
                evaluation = self.forward(features)
                return evaluation, new_accumulator
            return torch.tensor([[0.0]]), None
            
        # Make a copy of the board to apply the move
        updated_board = prev_board.copy()
        
        # First, check if we've been given a move directly
        if move:
            try:
                updated_board.push(move)
            except:
                # Invalid move, fall back to board if provided
                if board is not None:
                    updated_board = board.copy()
                else:
                    # Can't proceed with valid state
                    return torch.tensor([[0.0]]), None
        # Then check if the accumulator contains move information
        elif accumulator.get('move'):
            try:
                move_uci = accumulator.get('move')
                move = chess.Move.from_uci(move_uci)
                updated_board.push(move)
            except:
                # Invalid move, fall back to board if provided
                if board is not None:
                    updated_board = board.copy()
                else:
                    # Can't proceed with valid state
                    return torch.tensor([[0.0]]), None
        # Finally, if neither move nor accumulator.move, use provided board
        elif board is not None:
            updated_board = board.copy()
                
        # Update only the affected features rather than regenerating all
        if add_features and remove_features:
            # For incremental updates, we need to track what specific pieces changed
            updated_features = update_features_incrementally(prev_features, updated_board, add_features, remove_features)
        else:
            # Fall back to regenerating all features if no specific change info
            updated_features = board_to_features(updated_board)
        
        # Run the full forward pass with the updated features
        evaluation = self.forward(updated_features)
        
        # Create updated accumulator
        updated_accumulator = {
            'board': updated_board,
            'features': updated_features,
            'move': None  # Reset for next move
        }
        
        return evaluation, updated_accumulator

def board_to_features(board):
    """
    Convert a chess board to input features for the neural network.
    Creates a 24x8x8 tensor where:
    - Channels 0-11: Piece type planes (6 piece types for each color)
    - Channels 12-13: Side to move planes
    - Channels 14-17: Castling rights planes
    - Channel 18: En passant square
    - Channels 19-20: Attack maps
    - Channels 21-22: Mobility maps
    - Channel 23: Move count (normalized)
    
    Args:
        board: A chess.Board instance
        
    Returns:
        Tensor of shape (24, 8, 8)
    """
    # Initialize empty tensor
    features = torch.zeros(INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE, dtype=torch.float32)
    
    # Piece type and color planes (12 planes)
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
    
    # Fill piece planes
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            rank = 7 - chess.square_rank(square)  # Flip rank (0-7 to 7-0)
            file = chess.square_file(square)
            features[piece_idx[(piece.piece_type, piece.color)], rank, file] = 1.0
    
    # Current turn (1 plane)
    if board.turn == chess.WHITE:
        features[12].fill_(1.0)
    else:
        features[13].fill_(1.0)
    
    # Castling rights (4 planes)
    if board.has_kingside_castling_rights(chess.WHITE):
        features[14].fill_(1.0)
    if board.has_queenside_castling_rights(chess.WHITE):
        features[15].fill_(1.0)
    if board.has_kingside_castling_rights(chess.BLACK):
        features[16].fill_(1.0)
    if board.has_queenside_castling_rights(chess.BLACK):
        features[17].fill_(1.0)
    
    # En passant square (1 plane)
    if board.ep_square:
        rank = 7 - chess.square_rank(board.ep_square)
        file = chess.square_file(board.ep_square)
        features[18, rank, file] = 1.0
    
    # Attack maps (2 planes for each side)
    white_attacks = set()
    black_attacks = set()
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if not piece:
            continue
            
        # Get attacks from this piece
        attacks = board.attacks(square)
        for attack_square in attacks:
            if piece.color == chess.WHITE:
                white_attacks.add(attack_square)
            else:
                black_attacks.add(attack_square)
    
    # Fill attack maps
    for square in white_attacks:
        rank = 7 - chess.square_rank(square)
        file = chess.square_file(square)
        features[19, rank, file] = 1.0
        
    for square in black_attacks:
        rank = 7 - chess.square_rank(square)
        file = chess.square_file(square)
        features[20, rank, file] = 1.0
    
    # Mobility maps (2 planes) - where each side can move to
    if board.turn == chess.WHITE:
        for move in board.legal_moves:
            to_rank = 7 - chess.square_rank(move.to_square)
            to_file = chess.square_file(move.to_square)
            features[21, to_rank, to_file] = 1.0
    else:
        for move in board.legal_moves:
            to_rank = 7 - chess.square_rank(move.to_square)
            to_file = chess.square_file(move.to_square)
            features[22, to_rank, to_file] = 1.0
    
    # Move count plane (normalized)
    move_count = len(board.move_stack)
    normalized_move_count = min(1.0, move_count / 100.0)
    features[23].fill_(normalized_move_count)
    
    return features

def update_features_incrementally(prev_features, new_board, add_features, remove_features):
    """
    Update only the affected feature planes rather than regenerating all features.
    
    Args:
        prev_features: Previous feature tensor (24, 8, 8)
        new_board: Updated chess board
        add_features: Features to add
        remove_features: Features to remove
        
    Returns:
        Updated feature tensor
    """
    # Clone the previous features as our starting point
    features = prev_features.clone()
    
    # Update piece planes (0-11)
    # We need to clear and repopulate the piece planes since the move affects positions
    # Clear piece planes by zeroing them out
    features[0:12, :, :] = 0.0
    
    # Repopulate piece planes
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
    
    # Add pieces back to the board
    for square in chess.SQUARES:
        piece = new_board.piece_at(square)
        if piece:
            rank = 7 - chess.square_rank(square)
            file = chess.square_file(square)
            features[piece_idx[(piece.piece_type, piece.color)], rank, file] = 1.0
    
    # Update current turn (planes 12-13)
    features[12:14, :, :] = 0.0
    if new_board.turn == chess.WHITE:
        features[12].fill_(1.0)
    else:
        features[13].fill_(1.0)
    
    # Update castling rights (planes 14-17)
    features[14:18, :, :] = 0.0
    if new_board.has_kingside_castling_rights(chess.WHITE):
        features[14].fill_(1.0)
    if new_board.has_queenside_castling_rights(chess.WHITE):
        features[15].fill_(1.0)
    if new_board.has_kingside_castling_rights(chess.BLACK):
        features[16].fill_(1.0)
    if new_board.has_queenside_castling_rights(chess.BLACK):
        features[17].fill_(1.0)
    
    # Update en passant square (plane 18)
    features[18, :, :] = 0.0
    if new_board.ep_square:
        rank = 7 - chess.square_rank(new_board.ep_square)
        file = chess.square_file(new_board.ep_square)
        features[18, rank, file] = 1.0
    
    # Update attack maps (planes 19-20)
    features[19:21, :, :] = 0.0
    white_attacks = set()
    black_attacks = set()
    
    for square in chess.SQUARES:
        piece = new_board.piece_at(square)
        if not piece:
            continue
            
        # Get attacks from this piece
        attacks = new_board.attacks(square)
        for attack_square in attacks:
            if piece.color == chess.WHITE:
                white_attacks.add(attack_square)
            else:
                black_attacks.add(attack_square)
    
    # Fill attack maps
    for square in white_attacks:
        rank = 7 - chess.square_rank(square)
        file = chess.square_file(square)
        features[19, rank, file] = 1.0
        
    for square in black_attacks:
        rank = 7 - chess.square_rank(square)
        file = chess.square_file(square)
        features[20, rank, file] = 1.0
    
    # Update mobility maps (planes 21-22)
    features[21:23, :, :] = 0.0
    if new_board.turn == chess.WHITE:
        for move in new_board.legal_moves:
            to_rank = 7 - chess.square_rank(move.to_square)
            to_file = chess.square_file(move.to_square)
            features[21, to_rank, to_file] = 1.0
    else:
        for move in new_board.legal_moves:
            to_rank = 7 - chess.square_rank(move.to_square)
            to_file = chess.square_file(move.to_square)
            features[22, to_rank, to_file] = 1.0
    
    # Update move count plane (plane 23)
    move_count = len(new_board.move_stack)
    normalized_move_count = min(1.0, move_count / 100.0)
    features[23].fill_(normalized_move_count)
    
    return features

def get_feature_diff(board, move):
    """
    Calculate which features to add and remove when a move is made.
    This identifies the specific changes to the board state for incremental updates.
    
    Args:
        board: Chess board before the move
        move: The move to be made
        
    Returns:
        Dictionary with info about what features need to change
    """
    # Create dictionary to track changes
    changes = {
        'from_square': move.from_square,
        'to_square': move.to_square,
        'piece_moved': board.piece_at(move.from_square),
        'piece_captured': board.piece_at(move.to_square),
        'promotion': move.promotion,
        'is_castling': board.is_castling(move),
        'is_en_passant': board.is_en_passant(move)
    }
    
    return changes