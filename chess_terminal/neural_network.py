import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import chess

class ChessNet(nn.Module):
    def __init__(self, residual_blocks=19, channels=256):
        super(ChessNet, self).__init__()
        # Input: 8x8x119 (8x8 board, 119 planes for piece positions, move counters, etc.)
        self.conv1 = nn.Conv2d(119, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        
        # Residual blocks
        self.num_residual = residual_blocks  # Configurable (default: 19 like AlphaZero)
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(channels) for _ in range(self.num_residual)
        ])
        
        # Policy head (outputs move probabilities)
        self.policy_conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.policy_bn = nn.BatchNorm2d(channels)
        self.policy_fc = nn.Linear(channels * 8 * 8, 1968)  # 1968 possible moves
        
        # Value head (outputs position evaluation)
        self.value_conv = nn.Conv2d(channels, 1, 1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(8 * 8, channels)
        self.value_fc2 = nn.Linear(channels, 1)
        
    def forward(self, x):
        # Initial convolution block
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Residual tower
        for block in self.residual_blocks:
            x = block(x)
        
        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(-1, self.policy_conv.out_channels * 8 * 8)
        policy = self.policy_fc(policy)
        policy = F.log_softmax(policy, dim=1)
        
        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(-1, 8 * 8)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        identity = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += identity
        x = F.relu(x)
        return x

def board_to_input(board):
    """Convert a chess board to neural network input planes."""
    planes = np.zeros((119, 8, 8), dtype=np.float32)
    
    # 1-6: White pieces (Pawn, Knight, Bishop, Rook, Queen, King)
    # 7-12: Black pieces (Pawn, Knight, Bishop, Rook, Queen, King)
    piece_plane = {
        (True, chess.PAWN): 0,
        (True, chess.KNIGHT): 1,
        (True, chess.BISHOP): 2,
        (True, chess.ROOK): 3,
        (True, chess.QUEEN): 4,
        (True, chess.KING): 5,
        (False, chess.PAWN): 6,
        (False, chess.KNIGHT): 7,
        (False, chess.BISHOP): 8,
        (False, chess.ROOK): 9,
        (False, chess.QUEEN): 10,
        (False, chess.KING): 11
    }
    
    # Set piece positions
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            rank, file = divmod(square, 8)
            planes[piece_plane[piece.color, piece.piece_type]][rank][file] = 1
    
    # Additional features (repetitions, move count, etc.)
    current_plane = 12
    
    # Repetition counters (3 planes)
    repetitions = board.is_repetition(2)
    if repetitions:
        planes[current_plane:current_plane + repetitions] = 1
    current_plane += 3
    
    # Color (1 plane)
    if board.turn:
        planes[current_plane] = 1
    current_plane += 1
    
    # Move count (8 planes)
    move_count = bin(board.fullmove_number)[2:].zfill(8)
    for i, bit in enumerate(move_count):
        if bit == '1':
            planes[current_plane + i] = 1
    current_plane += 8
    
    # Castling rights (4 planes)
    if board.has_kingside_castling_rights(chess.WHITE):
        planes[current_plane] = 1
    current_plane += 1
    if board.has_queenside_castling_rights(chess.WHITE):
        planes[current_plane] = 1
    current_plane += 1
    if board.has_kingside_castling_rights(chess.BLACK):
        planes[current_plane] = 1
    current_plane += 1
    if board.has_queenside_castling_rights(chess.BLACK):
        planes[current_plane] = 1
    current_plane += 1
    
    # No-progress count (50 move rule) (8 planes)
    no_progress = bin(board.halfmove_clock)[2:].zfill(8)
    for i, bit in enumerate(no_progress):
        if bit == '1':
            planes[current_plane + i] = 1
    current_plane += 8
    
    # Remaining planes are all zero (unused features)
    
    return planes

def move_to_index(move):
    """Convert a chess move to policy index."""
    from_square = move.from_square
    to_square = move.to_square
    
    # Regular moves
    if not move.promotion:
        return from_square * 64 + to_square
    
    # Promotion moves
    promotion_offset = 64 * 64
    piece_offsets = {
        chess.KNIGHT: 0,
        chess.BISHOP: 1,
        chess.ROOK: 2,
        chess.QUEEN: 3
    }
    return promotion_offset + piece_offsets[move.promotion] * 64 + to_square

def index_to_move(index):
    """Convert policy index to chess move."""
    if index < 64 * 64:
        from_square = index // 64
        to_square = index % 64
        return chess.Move(from_square, to_square)
    
    # Promotion moves
    index -= 64 * 64
    promotion_piece = [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN][index // 64]
    to_square = index % 64
    # Find the pawn that can promote to this square
    from_square = to_square - 8 if to_square >= 8 else to_square + 8
    return chess.Move(from_square, to_square, promotion=promotion_piece) 