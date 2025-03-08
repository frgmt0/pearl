#!/usr/bin/env python3
"""
Pearl NNUE Chess Engine

A chess engine using NNUE (Efficiently Updatable Neural Network) for evaluation.
"""

import sys
import argparse

from src.game.ui import ChessUI, main as ui_main
from src.modes.hve import play_human_vs_engine
from src.modes.evsf import play_engine_vs_stockfish
from src.modes.hvh import play_human_vs_human
from src.engine.engine import NNUEEngine
from src.engine.finetune import auto_finetune, initialize_default_weights

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Pearl NNUE Chess Engine")
    
    # Main mode selection
    parser.add_argument('--mode', '-m', choices=['ui', 'hve', 'evsf', 'hvh', 'finetune'],
                        default='ui', help='Game mode (default: ui)')
    
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
    
    # Initialize default weights if they don't exist
    initialize_default_weights()
    
    # Determine if learning is enabled
    enable_learning = not args.no_learning
    
    # Run the selected mode
    if args.mode == 'ui':
        # Start the text-based UI
        ui_main()
    elif args.mode == 'hve':
        # Human vs Engine mode
        play_human_vs_engine(
            depth=args.depth,
            time_limit_ms=args.time,
            player_color=player_color,
            enable_learning=enable_learning
        )
    elif args.mode == 'evsf':
        # Engine vs Stockfish mode
        # Create engine with learning enabled/disabled
        engine = NNUEEngine(
            depth=args.depth,
            time_limit_ms=args.time,
            enable_learning=enable_learning
        )
        
        # Reset to default weights if requested
        if args.reset_weights and engine.finetuner:
            engine.finetuner.reset_to_default()
            
        play_engine_vs_stockfish(
            engine=engine,
            engine_depth=args.depth,
            engine_time_ms=args.time,
            stockfish_depth=args.stockfish_depth,
            engine_color=player_color
        )
    elif args.mode == 'hvh':
        # Human vs Human mode
        play_human_vs_human(
            with_analysis=not args.no_analysis
        )
    elif args.mode == 'finetune':
        # Fine-tune the engine
        engine = NNUEEngine(
            depth=args.depth,
            time_limit_ms=args.time,
            enable_learning=True  # Always enable learning for fine-tuning
        )
        
        # Reset to default weights if requested
        if args.reset_weights and engine.finetuner:
            engine.finetuner.reset_to_default()
            
        auto_finetune(
            engine=engine,
            games=args.games,
            epochs=args.epochs
        )
    else:
        print(f"Unknown mode: {args.mode}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
