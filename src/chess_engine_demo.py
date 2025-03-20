#!/usr/bin/env python3
import chess
import time
import argparse
from engine import Engine

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
    start_time = time.time()
    best_move, score, info = engine.search(board, depth=depth, time_limit_ms=10000)
    end_time = time.time()
    
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

def interactive_mode(engine):
    """Run the engine in interactive mode."""
    board = chess.Board()
    print("Interactive mode. Enter 'quit' to exit, 'new' for a new game, or a move in UCI format (e.g., 'e2e4').")
    print(board)
    
    while True:
        # Get user input
        user_input = input("\nYour move (or command): ").strip().lower()
        
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
            best_move, score, info = engine.search(board, depth=4, time_limit_ms=5000)
            
            if best_move:
                move_san = board.san(best_move)
                board.push(best_move)
                print(f"Engine plays: {best_move.uci()} ({move_san})")
                print(f"Evaluation: {score / 100:.2f} pawns")
                print(board)
            else:
                print("Engine couldn't find a move.")
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
            except ValueError:
                print("Invalid move or command. Try again.")

def main():
    parser = argparse.ArgumentParser(description='Chess Engine Demo')
    parser.add_argument('--fen', help='FEN string for position evaluation')
    parser.add_argument('--depth', type=int, default=4, help='Search depth (default: 4)')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    parser.add_argument('--verbose', action='store_true', help='Show verbose output')
    parser.add_argument('--tt-size', type=int, default=64, help='Transposition table size in MB (default: 64)')
    args = parser.parse_args()
    
    # Initialize engine
    print(f"Initializing chess engine with {args.tt_size}MB transposition table...")
    engine = Engine(transposition_table_size_mb=args.tt_size)
    
    if args.interactive:
        interactive_mode(engine)
    else:
        run_position_test(engine, args.fen, args.depth, args.verbose)

if __name__ == "__main__":
    main() 