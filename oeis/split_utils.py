"""
Utility functions for sequence length splitting across the project
"""

def compute_split(total_len):
    """
    Compute train/validation split for a sequence.
    
    Args:
        total_len: Total length of the sequence
        
    Returns:
        (n_in, n_validate): 
            - n_in: number of items for training (input)
            - n_validate: number of items for validation
            
    Rules:
        - For sequences <= 10: last 2 items for validation
        - For sequences > 10: 30% (floor) for validation
    """
    if total_len <= 10:
        # For short sequences: last 2 items for validation
        n_validate = 2
        n_in = max(1, total_len - 2)
    else:
        # For longer sequences: 30% (floor) for validation
        n_validate = int(total_len * 0.3)
        n_in = total_len - n_validate
    
    return n_in, n_validate


def get_n_in(total_len):
    """Convenience function to get just n_in"""
    n_in, _ = compute_split(total_len)
    return n_in


def get_n_validate(total_len):
    """Convenience function to get just n_validate"""
    _, n_validate = compute_split(total_len)
    return n_validate

