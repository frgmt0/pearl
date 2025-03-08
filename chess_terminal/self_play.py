#!/usr/bin/env python3
"""
Self-play training script for the chess engine.

This script allows the chess engine to play against itself repeatedly,
learning and improving with each game.
"""

import os
import sys
import select
import time
import argparse
import chess
import chess.pgn
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

# Get the project root directory (chess2)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MODELS_DIR = os.path.join(PROJECT_ROOT, "saved_models")

# Create models directory if it doesn't exist
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR, exist_ok=True)

def setup_engine(args):
    """Initialize the chess engine with the specified parameters."""
    print(f"\n{'='*50}")
    print(f"Initializing Chess Engine with parameters:")
    print(f"{'='*50}")
    print(f"Search depth: {args.depth}")
    print(f"Base model: {args.base_model if args.base_model else 'New model'}")
    print(f"MCTS simulations: {args.mcts_sims}")
    print(f"Temperature: {args.temperature}")
    print(f"Exploration constant: {args.exploration_constant}")
    print(f"Fast mode: {args.fast_mode}")
    print(f"{'='*50}\n")
    
    # Initialize engine with specified parameters
    engine = ChessEngine(
        search_depth=args.depth,
        model_path=args.base_model,
        mcts_simulations=args.mcts_sims,
        temperature=args.temperature,
        exploration_constant=args.exploration_constant,
        training_data_max_size=50000,
        fast_mode=args.fast_mode
    )
    
    # Create a directory for saved models if it doesn't exist
    os.makedirs(os.path.dirname(os.path.join("../saved_models", args.model_name)), exist_ok=True)
    
    return engine

def self_play_game(engine, game_id, args):
    """Play a full game of chess using self-play."""
    # Create a new board
    board = chess.Board()
    
    # Initialize game metadata
    game = chess.pgn.Game()
    game.headers["Event"] = f"Self-play Training Game {game_id}"
    game.headers["Site"] = "Training Session"
    game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
    game.headers["Round"] = str(game_id)
    game.headers["White"] = "ChessEngine (White)"
    game.headers["Black"] = "ChessEngine (Black)"
    
    # Record moves and evaluations for visualization
    move_history = []
    eval_history = []
    
    # Create game node for PGN recording
    node = game
    
    # Track game statistics
    moves_count = 0
    start_time = time.time()
    
    # Play the game until it's over
    while not board.is_game_over(claim_draw=True):
        # Adjust temperature based on move number
        if len(move_history) >= args.temperature_drop:
            engine.temperature = 0.1  # Lower temperature for later moves
        else:
            engine.temperature = args.temperature  # Use specified temperature for early moves
            
        # Get the start time
        start_time = time.time()
        
        # Get engine's move for the current position
        move_result = engine.search(board)
        
        if move_result is None or not move_result[0]:
            # If no valid move is found, resign
            print(f"Game {game_id}, Move {moves_count+1}: No valid move found, resigning")
            break
            
        move, evaluation = move_result
        
        # Record move and evaluation
        move_history.append(move)
        eval_history.append(evaluation)
        
        # Add move to PGN
        node = node.add_variation(move)
        
        # Display progress every few moves
        if args.verbose or moves_count % 10 == 0:
            print(f"Game {game_id}, Move: {moves_count+1}, {'White' if board.turn else 'Black'} played {move.uci()}")
            print(f"Evaluation: {evaluation:.2f}, Time: {(time.time() - start_time):.2f}s")
            if args.verbose:
                print(board)
                print()
        
        # Make the move on the board
        board.push(move)
        moves_count += 1
        
        # Add position to training data
        if not board.is_game_over(claim_draw=True):
            engine.add_training_sample(board, evaluation)
        
        # Avoid excessively long games
        if moves_count >= 200:
            print(f"Game {game_id}: Move limit reached, ending game")
            game.headers["Termination"] = "adjudication"
            break
    
    # Record game result
    if board.is_checkmate():
        result = "0-1" if board.turn else "1-0"
        winner = "Black" if board.turn else "White"
        game.headers["Termination"] = "checkmate"
    elif board.is_stalemate():
        result = "1/2-1/2"
        winner = "Draw (stalemate)"
        game.headers["Termination"] = "stalemate"
    elif board.is_insufficient_material():
        result = "1/2-1/2"
        winner = "Draw (insufficient material)"
        game.headers["Termination"] = "insufficient material"
    elif board.can_claim_fifty_moves():
        result = "1/2-1/2"
        winner = "Draw (fifty-move rule)"
        game.headers["Termination"] = "fifty-move rule"
    elif board.can_claim_threefold_repetition():
        result = "1/2-1/2"
        winner = "Draw (threefold repetition)"
        game.headers["Termination"] = "threefold repetition"
    elif moves_count >= 200:
        result = "1/2-1/2"
        winner = "Draw (move limit)"
    else:
        result = "1/2-1/2"
        winner = "Draw (unknown reason)"
        game.headers["Termination"] = "unterminated"
    
    game.headers["Result"] = result
    
    # Game statistics
    game_time = time.time() - start_time
    avg_move_time = game_time / max(1, moves_count)
    
    game_stats = {
        "id": game_id,
        "result": result,
        "winner": winner,
        "moves": moves_count,
        "time": game_time,
        "avg_move_time": avg_move_time,
        "move_history": move_history,
        "eval_history": eval_history
    }
    
    return game_stats, game

def train_model(engine, iteration, args):
    """Train the neural network on collected data."""
    print(f"\n{'='*50}")
    print(f"Training Neural Network - Iteration {iteration}")
    print(f"{'='*50}")
    print(f"Training on {len(engine.training_data)} positions")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    
    metrics = {}
    
    # Split training data into batches
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        # Train the model using batch training
        loss, accuracy = engine._train_neural_network_batch(batch_size=args.batch_size)
        
        if loss is not None and accuracy is not None:
            metrics[f"epoch_{epoch+1}"] = {
                "loss": loss,
                "accuracy": accuracy
            }
            print(f"  Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%")
        else:
            print("  No training performed (neural network not initialized)")
    
    return metrics

def plot_game_statistics(all_stats, save_path):
    """Plot game statistics and save to file."""
    # Prepare data
    games = [stat["id"] for stat in all_stats]
    moves = [stat["moves"] for stat in all_stats]
    times = [stat["time"] for stat in all_stats]
    results = [stat["result"] for stat in all_stats]
    
    # Count results
    white_wins = results.count("1-0")
    black_wins = results.count("0-1")
    draws = results.count("1/2-1/2")
    
    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
    
    # Plot moves per game
    ax1.bar(games, moves, color='blue', alpha=0.7)
    ax1.set_title('Moves per Game')
    ax1.set_xlabel('Game Number')
    ax1.set_ylabel('Number of Moves')
    ax1.grid(True, alpha=0.3)
    
    # Plot game duration
    ax2.bar(games, times, color='green', alpha=0.7)
    ax2.set_title('Game Duration')
    ax2.set_xlabel('Game Number')
    ax2.set_ylabel('Time (seconds)')
    ax2.grid(True, alpha=0.3)
    
    # Plot results pie chart
    labels = ['White Wins', 'Black Wins', 'Draws']
    sizes = [white_wins, black_wins, draws]
    colors = ['#ff9999', '#66b3ff', '#c2c2f0']
    explode = (0.1, 0.1, 0.1)
    
    ax3.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax3.axis('equal')
    ax3.set_title('Game Results')
    
    # Add stats as text
    fig.text(0.1, 0.02, 
             f"Total Games: {len(games)}\n"
             f"White Wins: {white_wins} ({white_wins/len(games)*100:.1f}%)\n"
             f"Black Wins: {black_wins} ({black_wins/len(games)*100:.1f}%)\n"
             f"Draws: {draws} ({draws/len(games)*100:.1f}%)\n"
             f"Average Moves: {np.mean(moves):.1f}",
             fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"Game statistics plotted and saved to {save_path}")

def plot_training_metrics(iterations, metrics, save_path):
    """Plot training metrics across iterations."""
    if not metrics:
        print("No training metrics to plot")
        return
        
    # Prepare data structure
    iterations_list = []
    losses = []
    accuracies = []
    
    # Extract metrics
    for iter_idx, iter_metrics in metrics.items():
        for epoch, epoch_metrics in iter_metrics.items():
            if 'loss' in epoch_metrics and 'accuracy' in epoch_metrics:
                iterations_list.append(f"{iter_idx}_{epoch}")
                losses.append(epoch_metrics['loss'])
                accuracies.append(epoch_metrics['accuracy'])
    
    # Plot metrics
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Loss plot
    ax1.plot(losses, 'r-', marker='o')
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(accuracies, 'b-', marker='o')
    ax2.set_title('Move Prediction Accuracy')
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Accuracy (%)')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"Training metrics plotted and saved to {save_path}")

def self_play_training(args):
    """Run the full self-play training loop."""
    # Initialize engine
    engine = setup_engine(args)
    
    # Check if the neural network is initialized
    if not engine.nn_initialized and args.base_model is None:
        print("Neural network not initialized. Creating a new model...")
        try:
            # Create a new neural network model
            import torch
            from neural_network import ChessNet
            
            model = ChessNet()
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'epoch': 0,
                'val_loss': float('inf')
            }
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.join("../saved_models", args.model_name)), exist_ok=True)
            
            # Save the new model
            torch.save(checkpoint, os.path.join("../saved_models", args.model_name))
            print(f"Created and saved new model to {os.path.join('../saved_models', args.model_name)}")
            
            # Reinitialize engine with the new model
            engine = setup_engine(args)
        except Exception as e:
            print(f"Error creating new model: {e}")
    
    # Statistics to track
    all_games_stats = []
    training_metrics = {}
    total_white_wins = 0
    total_black_wins = 0
    total_draws = 0
    
    start_time = time.time()
    
    # Create directory for plots
    plots_dir = os.path.join("../training_plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Run training iterations
    for iteration in range(args.iterations):
        print(f"\n{'='*50}")
        print(f"Starting Iteration {iteration+1}/{args.iterations}")
        print(f"{'='*50}")
        
        # Play multiple games per iteration
        iteration_stats = []
        pgn_nodes = []
        
        # Play games in this iteration
        for game_id in range(1, args.games + 1):
            # Play a self-play game
            game_stats, pgn_node = self_play_game(engine, game_id, args)
            
            # Collect statistics
            iteration_stats.append(game_stats)
            pgn_nodes.append(pgn_node)
            
            # Update win/loss/draw statistics
            if game_stats["result"] == "1-0":
                total_white_wins += 1
            elif game_stats["result"] == "0-1":
                total_black_wins += 1
            else:
                total_draws += 1
            
            # Print game summary
            print(f"Game {game_id} completed: {game_stats['result']} ({game_stats['winner']})")
            print(f"Moves: {game_stats['moves']}, Time: {game_stats['time']:.2f}s, "
                  f"Avg time per move: {game_stats['avg_move_time']:.2f}s")
        
        # Calculate statistics for this iteration
        games_played = len(iteration_stats)
        total_moves = sum(stat["moves"] for stat in iteration_stats)
        avg_moves = total_moves / games_played if games_played > 0 else 0
        
        # Print iteration summary
        print(f"\nIteration {iteration+1} Summary:")
        print(f"Games played: {games_played}")
        print(f"Total moves: {total_moves}")
        print(f"Average moves per game: {avg_moves:.1f}")
        
        # Save PGN file for this iteration
        pgn_path = os.path.join("../training_pgns", f"iteration_{iteration+1}.pgn")
        os.makedirs(os.path.dirname(pgn_path), exist_ok=True)
        
        with open(pgn_path, "w") as pgn_file:
            for pgn_node in pgn_nodes:
                print(pgn_node, file=pgn_file)
                print("", file=pgn_file)
        
        print(f"PGN file saved to {pgn_path}")
        
        # Train the neural network
        if args.epochs > 0:
            iter_metrics = train_model(engine, iteration+1, args)
            training_metrics[f"iteration_{iteration+1}"] = iter_metrics
        
        # Save the model at specified intervals
        if (iteration+1) % args.save_interval == 0 or iteration == args.iterations - 1:
            save_path = os.path.join("../saved_models", f"iter_{iteration+1}_{args.model_name}")
            engine._save_neural_network(save_path)
            print(f"Model saved to {save_path}")
        
        # Update all game statistics
        all_games_stats.extend(iteration_stats)
        
        # Plot statistics for this iteration
        plot_game_statistics(iteration_stats, 
                             os.path.join(plots_dir, f"iteration_{iteration+1}_stats.png"))
    
    # Calculate overall training statistics
    total_games = len(all_games_stats)
    total_moves = sum(stat["moves"] for stat in all_games_stats)
    avg_moves_per_game = total_moves / total_games if total_games > 0 else 0
    
    # Plot overall game statistics
    plot_game_statistics(all_games_stats, os.path.join(plots_dir, "overall_stats.png"))
    
    # Plot training metrics
    plot_training_metrics(args.iterations, training_metrics, 
                          os.path.join(plots_dir, "training_metrics.png"))
    
    # Calculate training time
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    # Save the final model
    final_model_path = os.path.join("../saved_models", args.model_name)
    engine._save_neural_network(final_model_path)
    
    # Print final summary
    print(f"\n{'='*50}")
    print(f"Training Complete")
    print(f"{'='*50}")
    print(f"Games played: {total_games}")
    print(f"White wins: {total_white_wins} ({total_white_wins/total_games*100:.1f}%)")
    print(f"Black wins: {total_black_wins} ({total_black_wins/total_games*100:.1f}%)")
    print(f"Draws: {total_draws} ({total_draws/total_games*100:.1f}%)")
    print(f"Average moves per game: {avg_moves_per_game:.1f}")
    print(f"Total training time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"Time per game: {total_time/total_games:.2f}s")
    print(f"Final model saved as: {final_model_path}")
    print(f"{'='*50}")
    
    return engine

def main():
    parser = argparse.ArgumentParser(description='Self-play training for the chess engine')
    
    # Game generation parameters
    parser.add_argument('--games', type=int, default=10,
                        help='Number of games to play per iteration')
    parser.add_argument('--iterations', type=int, default=10,
                        help='Number of training iterations')
    
    # MCTS and search parameters
    parser.add_argument('--mcts-sims', type=int, default=800,
                        help='Number of MCTS simulations per move')
    parser.add_argument('--depth', type=int, default=3,
                        help='Search depth for the engine')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Temperature for move selection (higher = more exploration)')
    parser.add_argument('--temperature-drop', type=int, default=20,
                        help='Move number to drop temperature to 0.1 for more focused play')
    parser.add_argument('--exploration-constant', type=float, default=1.0,
                        help='Exploration constant for MCTS (higher = more exploration)')
    parser.add_argument('--fast-mode', action='store_true',
                        help='Use faster but less accurate evaluation')
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size for neural network training')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of epochs to train per iteration')
    
    # Model parameters
    parser.add_argument('--model-name', type=str, default='refined_model.pt',
                        help='Name for the saved model file')
    parser.add_argument('--base-model', type=str, default=None,
                        help='Path to an existing model to use as a base')
    parser.add_argument('--save-interval', type=int, default=1,
                        help='Save model every N iterations')
    
    # Misc parameters
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed information during play')
    
    args = parser.parse_args()
    
    # Run the training
    self_play_training(args)
    
    print("Training complete!")

if __name__ == "__main__":
    main() 