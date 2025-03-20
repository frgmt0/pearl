import time

class TranspositionTable:
    """
    Zobrist hash-based transposition table for storing and retrieving positions.
    
    The transposition table uses a fixed-size array and employs a replacement
    strategy to determine which entries to keep when collisions occur.
    """
    def __init__(self, size_mb=64):
        """
        Initialize the transposition table.
        
        Args:
            size_mb: Size of the table in MB
        """
        # calc how many entries we can fit in size_mb
        # each entry is a dict with 5 fields: key, depth, value, type, best_move
        # which we assume is about 64 bytes-ish per entry (rough estimate)
        # 1 MB = 1024 * 1024 bytes
        entry_size_bytes = 64
        size_bytes = size_mb * 1024 * 1024
        self.size = size_bytes // entry_size_bytes
        
        # Initialize an empty table
        self.table = [None] * self.size
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.stores = 0
        self.collisions = 0
        
        # Age for replacement strategy
        self.age = 0
        
    def get(self, key):
        """
        Get an entry from the table.
        
        Args:
            key: Zobrist hash key (or any other unique position identifier)
            
        Returns:
            The entry if found, None otherwise
        """
        index = key % self.size
        entry = self.table[index]
        
        if entry is not None and entry['key'] == key:
            self.hits += 1
            return entry
        
        self.misses += 1
        return None
    
    def store(self, key, depth, value, node_type, best_move=None):
        """
        Store an entry in the table.
        
        Args:
            key: Zobrist hash key
            depth: Search depth
            value: Position evaluation
            node_type: Type of node (exact, alpha, beta)
            best_move: Best move found for this position
        """
        index = key % self.size
        entry = self.table[index]
        
        # Replacement strategy: deeper depth or newer age
        if entry is None or depth >= entry['depth'] or self.age - entry['age'] > 2:
            # yeah i know there's a memory leak here because we never actually
            # do anything with this info, but i assume gc will handle this
            if entry is not None:
                self.collisions += 1
                
            self.table[index] = {
                'key': key,
                'depth': depth,
                'value': value,
                'type': node_type,
                'best_move': best_move,
                'age': self.age
            }
            
            self.stores += 1
    
    def get_hit_rate(self):
        """
        Get the hit rate of the table.
        
        Returns:
            Hit rate as a percentage
        """
        total = self.hits + self.misses
        return (self.hits / total) * 100 if total > 0 else 0
    
    def get_size_info(self):
        """
        Get information about the table size and usage.
        
        Returns:
            A dictionary with size info
        """
        used_entries = sum(1 for entry in self.table if entry is not None)
        
        return {
            'size_mb': (self.size * 64) / (1024 * 1024),  # Estimate 64 bytes per entry
            'max_entries': self.size,
            'used_entries': used_entries,
            'usage_percent': (used_entries / self.size) * 100
        }
    
    def clear(self):
        """Clear the table."""
        self.table = [None] * self.size
        self.reset_stats()
        
    def reset_stats(self):
        """Reset statistics."""
        self.hits = 0
        self.misses = 0
        self.stores = 0
        self.collisions = 0
        
    def increment_age(self):
        """Increment the age counter (should be called at the start of each search)."""
        self.age += 1
        
    def prefetch(self, key):
        """
        Prefetch an entry from the table (for performance optimization).
        
        Args:
            key: Zobrist hash key
        """
        # This is a no-op in Python, but in a real engine implementation
        # you might want to prefetch the entry to CPU cache
        pass 