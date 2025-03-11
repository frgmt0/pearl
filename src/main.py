#!/usr/bin/env python3
"""
Pearl NNUE Chess Engine

A chess engine using NNUE (Efficiently Updatable Neural Network) for evaluation.
"""

import sys
import argparse
import os

from src.game.ui import ChessUI, main as ui_main
from src.modes.hve import play_human_vs_engine
from src.modes.evsf import play_engine_vs_stockfish
from src.modes.hvh import play_human_vs_human
from src.engine.engine import NNUEEngine
from src.engine.finetune import auto_finetune, initialize_default_weights, finetune_from_pgn
from src.engine.score import initialize_nnue

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Pearl NNUE Chess Engine")
    
    # Model path option
    parser.add_argument("--model-path", type=str, default=None,
                      help="Custom path to model weights file")
    
    # Main mode selection
    parser.add_argument('--mode', '-m', choices=['ui', 'hve', 'evsf', 'hvh', 'finetune', 'finetune-pgn', 'finetune-recent'],
                        default='ui', help='Game mode (default: ui)')
    
    # PGN file for finetuning
    parser.add_argument('--pgn', type=str, help='PGN file for finetune-pgn mode')
    
    # Engine settings
    parser.add_argument('--depth', '-d', type=int, default=5,
                        help='Search depth (default: 5)')
    parser.add_argument('--time', '-t', type=int, default=1000,
                        help='Time limit in milliseconds (default: 1000)')
    
    # Player settings
    parser.add_argument('--color', '-c', choices=['white', 'black', 'random'],
                        default='white', help='Player color for human vs engine (default: white)')
    
    # Stockfish settings
    parser.add_argument('--stockfish-depth', '-sd', type=int, default=10,
                        help='Stockfish search depth (default: 10)')
    
    # Fine-tuning settings
    parser.add_argument('--games', '-g', type=int, default=10,
                        help='Number of self-play games for fine-tuning (default: 10)')
    parser.add_argument('--epochs', '-e', type=int, default=5,
                        help='Number of training epochs for fine-tuning (default: 5)')
    parser.add_argument('--num-files', '-n', type=int, default=5,
                        help='Number of recent PGN files to use for finetuning (default: 5)')
    parser.add_argument('--emphasis', '-E', type=float, default=1.5,
                        help='Emphasis factor for finetuning feedback (default: 1.5)')
    parser.add_argument('--result-filter', '-r', choices=['win', 'loss', 'all'], default='all',
                        help='Filter games by result for finetuning (default: all)')
    
    # Analysis settings
    parser.add_argument('--no-analysis', action='store_true',
                        help='Disable engine analysis in human vs human mode')
    
    # Learning settings
    parser.add_argument('--no-learning', action='store_true',
                        help='Disable real-time learning')
    parser.add_argument('--reset-weights', action='store_true',
                        help='Reset to default weights before starting')
    
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_args()
    
    # Convert color string to chess.WHITE/BLACK
    import chess
    if args.color == 'white':
        player_color = chess.WHITE
    elif args.color == 'black':
        player_color = chess.BLACK
    else:  # random
        import random
        player_color = random.choice([chess.WHITE, chess.BLACK])
    
    # Ensure saved_models directory exists
    os.makedirs("saved_models", exist_ok=True)
    
    # Get model path from arguments
    model_path = args.model_path
    
    # Print model info
    print(f"Using Pearl Chess Engine (16M parameters)")
    if model_path:
        print(f"Loading weights from {model_path}")
    
    # Initialize the model for global use
    initialize_nnue(model_path=model_path)
    
    # Determine if learning is enabled
    enable_learning = not args.no_learning
    
    # Reset weights if requested
    if args.reset_weights:
        initialize_default_weights()
        print("Reset to default weights")
    
    # Run the selected mode
    if args.mode == 'ui':
        # Start the UI
        ui_main()
    elif args.mode == 'hve':
        # Human vs Engine mode
        engine = NNUEEngine(depth=args.depth, time_limit_ms=args.time, enable_learning=enable_learning)
        play_human_vs_engine(engine, player_color)
    elif args.mode == 'evsf':
        # Engine vs Stockfish mode
        engine = NNUEEngine(depth=args.depth, time_limit_ms=args.time, enable_learning=enable_learning)
        play_engine_vs_stockfish(engine, args.stockfish_depth)
    elif args.mode == 'hvh':
        # Human vs Human mode
        engine = NNUEEngine(depth=args.depth, time_limit_ms=args.time, enable_learning=enable_learning) if not args.no_analysis else None
        play_human_vs_human(engine)
    elif args.mode == 'finetune':
        # Auto-finetune mode
        auto_finetune(num_games=args.games, epochs=args.epochs, emphasis=args.emphasis)
    elif args.mode == 'finetune-pgn':
        # Finetune from PGN file
        if not args.pgn:
            print("Error: --pgn argument is required for finetune-pgn mode")
            return 1
        finetune_from_pgn(args.pgn, epochs=args.epochs, emphasis=args.emphasis, result_filter=args.result_filter)
    elif args.mode == 'finetune-recent':
        # Finetune from recent games
        from src.engine.finetune import finetune_from_recent
        finetune_from_recent(num_files=args.num_files, epochs=args.epochs, emphasis=args.emphasis, result_filter=args.result_filter)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
