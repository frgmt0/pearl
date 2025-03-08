import os
import sys
import platform

def check_terminal_support():
    """Check if the terminal supports required features."""
    # Check if running in a terminal
    if not sys.stdout.isatty():
        print("Error: This program must be run in a terminal.")
        return False
    
    # Check terminal size
    try:
        import shutil
        columns, rows = shutil.get_terminal_size()
        if columns < 80 or rows < 24:
            print(f"Warning: Terminal size ({columns}x{rows}) may be too small. Recommended: 80x24 or larger.")
    except Exception:
        print("Warning: Could not determine terminal size.")
    
    # Check if the terminal supports Unicode
    term_env = os.environ.get('TERM', '')
    if 'xterm' not in term_env and 'rxvt' not in term_env and 'screen' not in term_env:
        print(f"Warning: Your terminal ({term_env}) may not support Unicode characters properly.")
        print("This program works best with xterm, rxvt, or screen terminals.")
    
    return True

def get_platform_info():
    """Get information about the platform."""
    return {
        'system': platform.system(),
        'release': platform.release(),
        'version': platform.version(),
        'machine': platform.machine(),
        'python_version': platform.python_version(),
        'terminal': os.environ.get('TERM', 'unknown')
    }

def print_platform_info():
    """Print information about the platform."""
    info = get_platform_info()
    print(f"System: {info['system']} {info['release']} ({info['version']})")
    print(f"Machine: {info['machine']}")
    print(f"Python: {info['python_version']}")
    print(f"Terminal: {info['terminal']}")

def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_title():
    """Print the game title."""
    title = """
    ╔═══════════════════════════════════════════════════╗
    ║                                                   ║
    ║   ████████╗███████╗██████╗ ███╗   ███╗ ██████╗   ║
    ║   ╚══██╔══╝██╔════╝██╔══██╗████╗ ████║██╔════╝   ║
    ║      ██║   █████╗  ██████╔╝██╔████╔██║██║        ║
    ║      ██║   ██╔══╝  ██╔══██╗██║╚██╔╝██║██║        ║
    ║      ██║   ███████╗██║  ██║██║ ╚═╝ ██║╚██████╗   ║
    ║      ╚═╝   ╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝ ╚═════╝   ║
    ║                                                   ║
    ║                  ██████╗██╗  ██╗███████╗███████╗ ║
    ║                 ██╔════╝██║  ██║██╔════╝██╔════╝ ║
    ║                 ██║     ███████║█████╗  ███████╗ ║
    ║                 ██║     ██╔══██║██╔══╝  ╚════██║ ║
    ║                 ╚██████╗██║  ██║███████╗███████║ ║
    ║                  ╚═════╝╚═╝  ╚═╝╚══════╝╚══════╝ ║
    ║                                                   ║
    ╚═══════════════════════════════════════════════════╝
    """
    print(title)
    print("  A terminal-based chess game with standard algebraic notation")
    print("  for 2 players.\n")
    print("  Press any key to start...")
