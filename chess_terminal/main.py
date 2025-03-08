#!/usr/bin/env python3
"""
Terminal Chess - A terminal-based chess game with standard algebraic notation input.
"""

import sys
import time
import signal
from utils import check_terminal_support, print_platform_info, clear_screen, print_title
from terminal_ui import ChessUI

def signal_handler(sig, frame):
    """Handle Ctrl+C to exit gracefully."""
    print("\nExiting Terminal Chess...")
    sys.exit(0)

def main():
    """Main entry point for the chess game."""
    # Register signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    # Clear the screen
    clear_screen()
    
    # Print the title
    print_title()
    
    # Wait for a key press
    try:
        input()
    except KeyboardInterrupt:
        print("\nExiting Terminal Chess...")
        sys.exit(0)
    
    # Check terminal support
    if not check_terminal_support():
        print("\nPress Enter to continue anyway, or Ctrl+C to exit...")
        try:
            input()
        except KeyboardInterrupt:
            print("\nExiting Terminal Chess...")
            sys.exit(0)
    
    # Print platform information
    print_platform_info()
    print("\nStarting Terminal Chess...")
    time.sleep(1)
    
    # Start the chess UI
    ui = ChessUI()
    ui.start()

if __name__ == "__main__":
    main()
