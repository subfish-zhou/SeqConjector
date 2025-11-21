"""
Utility functions for sequence length splitting across the project
"""

def compute_split(total_len):
    """
    Compute train/validation split for a sequence.
    
    Args:
        total_len: Total length of the sequence
        
    Returns:
        (n_in, n_chk): 
            - n_in: number of items for training (input)
            - n_chk: number of items for validation/checking
            
    Rules:
        - For sequences <= 10: last 2 items for validation
        - For sequences > 10: 30% (floor) for validation
    """
    if total_len <= 10:
        # For short sequences: last 2 items for validation
        n_chk = 2
        n_in = max(1, total_len - 2)
    else:
        # For longer sequences: 30% (floor) for validation
        n_chk = int(total_len * 0.3)
        n_in = total_len - n_chk
    
    return n_in, n_chk


def get_n_in(total_len):
    """Convenience function to get just n_in"""
    n_in, _ = compute_split(total_len)
    return n_in


def get_n_chk(total_len):
    """Convenience function to get just n_chk"""
    _, n_chk = compute_split(total_len)
    return n_chk

