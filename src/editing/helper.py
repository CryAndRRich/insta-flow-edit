def calculate_shift(image_seq_len: int,
                    base_seq_len: int = 256,
                    max_seq_len: int = 4096,
                    base_shift: float = 0.5,
                    max_shift: float = 1.16) -> float:
    """
    Calculates the time shift value for FLUX scheduling.
    FLUX adjusts the noise schedule based on image resolution (sequence length)
    to optimize detail generation

    Parameters:
        image_seq_len: Number of tokens in the current image (H * W / patch_size)
        base_seq_len: Base sequence length
        max_seq_len: Maximum sequence length
        base_shift: Base shift value
        max_shift: Maximum shift value

    Returns:
        float: The "mu" value used to adjust the timestep schedule
    """
    # Calculate slope for shift interpolation
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    # Calculate bias
    b = base_shift - m * base_seq_len
    # Calculate specific shift for this image length
    mu = image_seq_len * m + b
    return mu


def lr_hump_beta(k: int, 
                 N: int, 
                 alpha_max: float,
                 a: float = 3.0, 
                 b: float = 6.0) -> float:
    """
    Compute learning rate using beta distribution hump shape
    
    Parameters:
        k: Current step (1-based)
        N: Total number of steps
        alpha_max: Maximum learning rate
        a: Beta distribution parameter a
        b: Beta distribution parameter b
    
    Returns:
        float: Learning rate at step k
    """
    if not (1 <= k <= N):
        return 0.0 # Handle edge cases
    
    x = (k - 1) / (N - 1)
    pdf = x ** (a - 1) * (1 - x) ** (b - 1)
    peak = ((a - 1) / (a + b - 2)) ** (a - 1) * ((b - 1)/(a + b - 2)) ** (b - 1)
    # Avoid division by zero if peak is too small
    if peak == 0: 
        return 0.0
    return alpha_max * pdf / peak

def lr_hump_tail_beta(k: int, 
                      N: int, 
                      alpha_max: float, 
                      beta: float,
                      a: float = 3.0, 
                      b: float = 6.0) -> float:
    """
    Compute learning rate using beta distribution hump with linear tail
    
    Parameters:
        k: Current step (1-based)
        N: Total number of steps
        alpha_max: Maximum learning rate
        beta: Linear tail slope
        a: Beta distribution parameter a
        b: Beta distribution parameter b
    
    Returns:
        float: Learning rate at step k
    """
    x = (k - 1) / (N - 1)
    hump = lr_hump_beta(k, N, alpha_max - beta, a, b)
    tail = beta * x
    return hump + tail