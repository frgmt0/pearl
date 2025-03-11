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
from src.engine.finetune import auto_finetune, initialize_default_weights, finetune_from_pgn
from src.engine.score import initialize_nnue

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Pearl NNUE Chess Engine")
    
    # Add model selection options
    parser.add_argument("--model", choices=["standard", "pearl", "pearlxl"], default="pearl",
                      help="Select model architecture (standard=small, pearl=enhanced, pearlxl=extra large)")
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
    
    # Initialize default models if they don't exist
    from src.engine.nnue.model_handler import initialize_default_models
    initialize_default_models()
    
    # Get model type from arguments
    model_type = args.model.lower()
    model_path = args.model_path
    
    # Print model info
    if model_type == "pearlxl":
        model_name = "PearlXL (extra large)"
    elif model_type == "pearl":
        model_name = "Pearl (enhanced)"
    else:
        model_name = "Standard"
        
    print(f"Using {model_name} model architecture")
    if model_path:
        print(f"Loading weights from {model_path}")
    
    # Initialize the model for global use
    initialize_nnue(model_type=model_type, model_path=model_path)
    
    # Determine if learning is enabled
    enable_learning = not args.no_learning
    
    # Run the selected mode
    if args.mode == 'ui':
        # Import the model selection function from UI
        from src.game.ui import select_model, ChessUI
        
        # If specific model path provided, use command line args
        if args.model_path:
            # Start the text-based UI with command line specified model
            ui_main()
        else:
            # Show the model selection UI first, then start the game UI
            use_enhanced, use_xl = select_model()
            
            # Convert back to model_type for consistency
            if use_enhanced and use_xl:
                model_type = "pearlxl"
            elif use_enhanced:
                model_type = "pearl"
            else:
                model_type = "standard"
            
            # Create and run UI with the selected model
            ui = ChessUI(use_enhanced_model=use_enhanced, use_xl_model=use_xl)
            ui.run()
            return 0  # Exit directly since we've taken control of the UI
    elif args.mode == 'hve':
        # Human vs Engine mode
        play_human_vs_engine(
            depth=args.depth,
            time_limit_ms=args.time,
            player_color=player_color,
            enable_learning=enable_learning,
            model_type=model_type
        )
    elif args.mode == 'evsf':
        # Engine vs Stockfish mode
        # Create engine with learning enabled/disabled
        engine = NNUEEngine(
            depth=args.depth,
            time_limit_ms=args.time,
            enable_learning=enable_learning,
            model_type=model_type
        )
        
        # Reset to default weights if requested
        if args.reset_weights and engine.finetuner:
            if hasattr(engine.finetuner, 'reset_to_default'):
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
            enable_learning=True,  # Always enable learning for fine-tuning
            model_type=model_type
        )
        
        # Reset to default weights if requested
        if args.reset_weights and engine.finetuner:
            if hasattr(engine.finetuner, 'reset_to_default'):
                engine.finetuner.reset_to_default()
            
        auto_finetune(
            engine=engine,
            games=args.games,
            epochs=args.epochs
        )
    elif args.mode == 'finetune-pgn':
        # Fine-tune from PGN file
        if not args.pgn:
            print("Error: --pgn parameter is required for finetune-pgn mode")
            return 1
        
        # Prepare feedback based on result filter
        feedback = None
        if args.result_filter != 'all':
            feedback = {
                "result": args.result_filter,
                "emphasis": args.emphasis
            }
            
        print(f"Finetuning model from PGN file: {args.pgn} with {args.epochs} epochs")
        if feedback:
            print(f"Using feedback: {feedback}")
        finetune_from_pgn(
            args.pgn, 
            epochs=args.epochs, 
            feedback=feedback, 
            model_type=model_type,
            use_memory=True
        )
    
    elif args.mode == 'finetune-recent':
        # Fine-tune from recent PGN files
        from src.engine.finetune import finetune_from_recent_pgns
        
        # Prepare feedback based on result filter
        feedback = None
        if args.result_filter != 'all':
            feedback = {
                "result": args.result_filter,
                "emphasis": args.emphasis
            }
        
        print(f"Finetuning model from {args.num_files} recent PGN files with {args.epochs} epochs")
        if feedback:
            print(f"Using feedback: {feedback}")
        
        finetune_from_recent_pgns(
            num_files=args.num_files,
            epochs=args.epochs,
            batch_size=64,
            feedback=feedback,
            model_type=model_type
        )
    else:
        print(f"Unknown mode: {args.mode}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
