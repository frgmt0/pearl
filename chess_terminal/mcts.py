import math
import chess
import numpy as np
import torch
from neural_network import board_to_input, move_to_index, index_to_move

class Node:
    def __init__(self, board, parent=None, move=None, prior=0.0):
        self.board = board.copy()
        self.parent = parent
        self.move = move
        self.prior = prior
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0
        self.is_expanded = False
        
    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count
    
    def select_child(self, c_puct=1.0):
        """Select child using PUCT algorithm."""
        best_score = float('-inf')
        best_child = None
        
        # Total visit count of parent
        total_count = sum(child.visit_count for child in self.children.values())
        
        for child, node in self.children.items():
            score = node.value() + c_puct * node.prior * math.sqrt(total_count) / (1 + node.visit_count)
            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child 

    def search(self, board):
        """Perform MCTS search and return move probabilities."""
        root = Node(board)
        
        for _ in range(self.num_simulations):
            node = root
            scratch_board = board.copy()
            
            # Selection
            while node.is_expanded and not node.board.is_game_over():
                child_move = node.select_child(self.c_puct)
                if child_move is None:
                    break
                node = node.children[child_move]
                scratch_board.push(child_move)
            
            # Expansion and evaluation
            if not node.board.is_game_over():
                # Prepare input for neural network
                x = board_to_input(scratch_board)
                x = torch.FloatTensor(x).unsqueeze(0)
                
                # Get policy and value predictions
                with torch.no_grad():
                    policy, value = self.model(x)
                
                policy = torch.exp(policy).squeeze()
                value = value.item()
                
                # Create children for all legal moves
                for move in scratch_board.legal_moves:
                    child_board = scratch_board.copy()
                    child_board.push(move)
                    move_idx = move_to_index(move)
                    node.children[move] = Node(
                        child_board,
                        parent=node,
                        move=move,
                        prior=policy[move_idx].item()
                    )
                node.is_expanded = True
            else:
                # Game is over, use real game result
                if node.board.is_checkmate():
                    value = -1 if node.board.turn else 1
                else:
                    value = 0  # Draw
            
            # Backpropagation
            while node is not None:
                node.visit_count += 1
                node.value_sum += value
                value = -value  # Flip value for opponent
                node = node.parent
        
        # Calculate move probabilities based on visit counts
        moves = []
        probs = []
        total_visits = sum(child.visit_count for child in root.children.values())
        
        for move, child in root.children.items():
            moves.append(move)
            probs.append(child.visit_count / total_visits)
        
        return moves, probs 