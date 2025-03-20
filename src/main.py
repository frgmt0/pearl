#!/usr/bin/env python3
"""
Pearl Classical Chess Engine

A high-performance classical heuristic-based chess engine that makes strong moves
through optimized evaluation functions rather than relying solely on deep search.
"""

import sys
import argparse
import os
import chess

# Import our new engine
from src.engine import Engine
# Import the user interfaces
from src.game.mouse_ui import run_mouse_ui
from src.game.gui import run_gui

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Pearl Classical Chess Engine")
    
    # Main mode selection
    parser.add_argument('--mode', '-m', choices=['interactive', 'position', 'uci', 'mouse', 'gui'],
                        default='gui', help='Engine mode (default: gui)')
    
    # Engine settings
    parser.add_argument('--depth', '-d', type=int, default=4,
                        help='Search depth (default: 4)')
    parser.add_argument('--time', '-t', type=int, default=1000,
                        help='Time limit in milliseconds (default: 1000)')
    
    # Position evaluation
    parser.add_argument('--fen', type=str, 
                        help='FEN string for position evaluation')
    
    # Transposition table size
    parser.add_argument('--tt-size', type=int, default=64,
                        help='Transposition table size in MB (default: 64)')
    
    # Output verbosity
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show verbose output')
    
    return parser.parse_args()

def run_position_test(engine, fen=None, depth=4, verbose=False):
    """Run a position test to evaluate engine performance."""
    board = chess.Board(fen) if fen else chess.Board()
    
    print(f"Position: {board.fen()}")
    print(board)
    print()
    print(f"Evaluating position to depth {depth}...")
    
    # Reset engine stats
    engine.reset_stats()
    
    # Search with a time limit of 10 seconds
    start_time = import_time_and_return()
    best_move, score, info = engine.search(board, depth=depth, time_limit_ms=10000)
    end_time = import_time_and_return()
    
    # Print results
    print(f"Best move: {best_move}")
    print(f"Score: {score / 100:.2f} pawns {'for white' if score > 0 else 'for black' if score < 0 else '(equal)'}")
    print(f"Depth reached: {info['depth']}")
    print(f"Nodes searched: {info['nodes']:,}")
    print(f"Time: {end_time - start_time:.2f} seconds")
    print(f"Nodes per second: {info['nps']:,}")
    
    if 'pv' in info and info['pv']:
        print(f"Principal variation: {' '.join(info['pv'])}")
        
    # Get more detailed stats if verbose
    if verbose:
        stats = engine.get_stats()
        print("\nEngine Statistics:")
        print(f"Positions evaluated: {stats['positions_evaluated']:,}")
        print(f"Evaluation speed: {stats['eval_speed']:,} positions/sec")
        print(f"Transposition table hit rate: {stats['tt_hit_rate']:.1f}%")
        print(f"Transposition table size: {stats['tt_size']['used_entries']:,}/{stats['tt_size']['max_entries']:,} entries ({stats['tt_size']['usage_percent']:.1f}%)")
    
    return best_move, score, info

def import_time_and_return():
    """Import time module and return the current time."""
    import time
    return time.time()

def interactive_mode(engine):
    """
    Run an interactive command-line interface for the chess engine.
    
    Args:
        engine: The chess engine object
    """
    board = chess.Board()
    
    print("Pearl Classical Chess Engine - Interactive Mode")
    print("\nEnter 'quit' to exit, 'new' for a new game, or a move in UCI format (e.g., 'e2e4').")
    print(board)
    
    while True:
        print("\nYour move (or command): ", end='')
        user_input = input().strip()
        
        # Skip empty input
        if not user_input:
            continue
        
        if user_input == 'quit':
            break
        
        if user_input == 'new':
            board = chess.Board()
            print("\nNew game started.")
            print(board)
            continue
        
        if user_input == 'fen':
            print(f"Current FEN: {board.fen()}")
            continue
            
        if user_input.startswith('setboard '):
            try:
                fen = user_input[9:].strip()
                board = chess.Board(fen)
                print("\nBoard set to new position.")
                print(board)
            except Exception as e:
                print(f"Error setting board: {e}")
            continue
            
        if user_input == 'go':
            # Engine's turn to move
            print("\nEngine is thinking...")
            try:
                best_move, score, info = engine.search(board, depth=4, time_limit_ms=5000)
                
                if best_move:
                    move_san = board.san(best_move)
                    board.push(best_move)
                    print(f"Engine plays: {best_move.uci()} ({move_san})")
                    print(f"Evaluation: {score / 100:.2f} pawns")
                    print(board)
                    
                    # Check for game over
                    if board.is_game_over():
                        print("\nGame over:", board.result())
                else:
                    print("Engine couldn't find a move.")
            except Exception as e:
                print(f"Error during engine search: {e}")
            continue
        
        # Try to parse the input as a move
        try:
            # Try UCI format first
            move = chess.Move.from_uci(user_input)
            # Check if the move is legal
            if move in board.legal_moves:
                board.push(move)
                print(board)
                
                # Engine's turn
                if not board.is_game_over():
                    print("\nEngine is thinking...")
                    try:
                        best_move, score, info = engine.search(board, depth=4, time_limit_ms=5000)
                        
                        if best_move:
                            move_san = board.san(best_move)
                            board.push(best_move)
                            print(f"Engine plays: {best_move.uci()} ({move_san})")
                            print(f"Evaluation: {score / 100:.2f} pawns")
                            print(board)
                            
                            # Check for game over
                            if board.is_game_over():
                                print("\nGame over:", board.result())
                        else:
                            print("Engine couldn't find a move.")
                    except Exception as e:
                        print(f"Error during engine search: {e}")
                else:
                    print("\nGame over:", board.result())
            else:
                print("Illegal move.")
        except ValueError:
            try:
                # Try SAN format as a fallback
                move = board.parse_san(user_input)
                board.push(move)
                print(board)
                
                # Engine's turn
                if not board.is_game_over():
                    print("\nEngine is thinking...")
                    try:
                        best_move, score, info = engine.search(board, depth=4, time_limit_ms=5000)
                        
                        if best_move:
                            move_san = board.san(best_move)
                            board.push(best_move)
                            print(f"Engine plays: {best_move.uci()} ({move_san})")
                            print(f"Evaluation: {score / 100:.2f} pawns")
                            print(board)
                            
                            # Check for game over
                            if board.is_game_over():
                                print("\nGame over:", board.result())
                        else:
                            print("Engine couldn't find a move.")
                    except Exception as e:
                        print(f"Error during engine search: {e}")
                else:
                    print("\nGame over:", board.result())
            except ValueError:
                # Neither UCI nor SAN format matched
                print("Invalid move or command. Type 'quit' to exit, 'new' for a new game, or enter a move.")
            except Exception as e:
                print(f"Error: {e}")
        except Exception as e:
            print(f"Error: {e}")

def uci_mode(engine):
    """Run in UCI (Universal Chess Interface) mode."""
    import time
    
    print("Pearl Classical Chess Engine")
    print("by Jason and Claude")
    
    board = chess.Board()
    depth = 4
    time_limit_ms = None
    
    while True:
        if not sys.stdin.isatty():
            # If not connected to a terminal, wait for input
            try:
                cmd = input()
            except EOFError:
                break
        else:
            # If connected to a terminal, use readline
            try:
                cmd = input()
            except KeyboardInterrupt:
                break
        
        if not cmd:
            continue
        
        # Split command into tokens
        tokens = cmd.split()
        command = tokens[0].lower()
        
        if command == "uci":
            print("id name Pearl Classical Engine")
            print("id author Jason and Claude")
            print("option name Hash type spin default 64 min 1 max 1024")
            print("option name Depth type spin default 4 min 1 max 100")
            print("uciok")
        
        elif command == "isready":
            print("readyok")
        
        elif command == "setoption":
            if len(tokens) >= 5 and tokens[1].lower() == "name" and tokens[3].lower() == "value":
                option_name = tokens[2].lower()
                option_value = tokens[4]
                
                if option_name == "hash":
                    try:
                        hash_size = int(option_value)
                        # Create a new engine with the specified hash size
                        engine = Engine(transposition_table_size_mb=hash_size)
                    except ValueError:
                        pass
                
                elif option_name == "depth":
                    try:
                        depth = int(option_value)
                    except ValueError:
                        pass
        
        elif command == "ucinewgame":
            board = chess.Board()
            engine.tt.clear()  # Clear transposition table
        
        elif command == "position":
            if len(tokens) < 2:
                continue
            
            if tokens[1] == "startpos":
                board = chess.Board()
                move_idx = 3 if len(tokens) > 2 and tokens[2] == "moves" else 2
            
            elif tokens[1] == "fen":
                fen_end_idx = 7
                if len(tokens) <= fen_end_idx:
                    continue
                
                fen = " ".join(tokens[2:fen_end_idx + 1])
                try:
                    board = chess.Board(fen)
                except ValueError:
                    continue
                
                move_idx = fen_end_idx + 2 if len(tokens) > fen_end_idx + 1 and tokens[fen_end_idx + 1] == "moves" else fen_end_idx + 1
            
            else:
                continue
            
            # Apply moves if provided
            if len(tokens) > move_idx:
                for move_str in tokens[move_idx:]:
                    try:
                        move = chess.Move.from_uci(move_str)
                        if move in board.legal_moves:
                            board.push(move)
                        else:
                            break
                    except ValueError:
                        break
        
        elif command == "go":
            time_limit_ms = None
            # Parse time control parameters
            for i in range(1, len(tokens), 2):
                if i+1 < len(tokens):
                    if tokens[i] == "depth" and len(tokens) > i+1:
                        try:
                            depth = int(tokens[i+1])
                        except ValueError:
                            pass
                    
                    elif tokens[i] == "movetime" and len(tokens) > i+1:
                        try:
                            time_limit_ms = int(tokens[i+1])
                        except ValueError:
                            pass
                    
                    elif tokens[i] in ("wtime", "btime") and len(tokens) > i+1:
                        try:
                            side_time = int(tokens[i+1])
                            # Set the time limit to a portion of the available time
                            if (tokens[i] == "wtime" and board.turn == chess.WHITE) or \
                               (tokens[i] == "btime" and board.turn == chess.BLACK):
                                # Use about 1/20th of the remaining time, but at least 100ms
                                time_limit_ms = max(side_time // 20, 100)
                        except ValueError:
                            pass
            
            # Start the search
            best_move, _, _ = engine.search(board, depth=depth, time_limit_ms=time_limit_ms)
            
            if best_move:
                print(f"bestmove {best_move.uci()}")
            else:
                print("bestmove (none)")
        
        elif command == "quit":
            break

def main():
    """Main entry point."""
    args = parse_args()
    
    # Initialize engine with specified transposition table size
    print(f"Initializing Pearl Classical Chess Engine (TT size: {args.tt_size}MB)...")
    engine = Engine(transposition_table_size_mb=args.tt_size)
    
    # Run the selected mode
    if args.mode == 'interactive':
        interactive_mode(engine)
    elif args.mode == 'position':
        run_position_test(engine, args.fen, args.depth, args.verbose)
    elif args.mode == 'uci':
        uci_mode(engine)
    elif args.mode == 'mouse':
        print("Starting mouse-based terminal UI...")
        run_mouse_ui(engine)
    elif args.mode == 'gui':
        print("Starting graphical UI...")
        run_gui(engine)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
