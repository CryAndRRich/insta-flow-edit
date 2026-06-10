def calculate_shift(image_seq_len: int,
                    base_seq_len: int = 256,
                    max_seq_len: int = 4096,
                    base_shift: float = 0.5,
                    max_shift: float = 1.16) -> float:
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def lr_hump_beta(k: int,
                 N: int,
                 alpha_max: float,
                 a: float = 3.0,
                 b: float = 6.0) -> float:
    if not (1 <= k <= N):
        return 0.0
    x = (k - 1) / (N - 1)
    pdf = x ** (a - 1) * (1 - x) ** (b - 1)
    peak = ((a - 1) / (a + b - 2)) ** (a - 1) * ((b - 1) / (a + b - 2)) ** (b - 1)
    if peak == 0:
        return 0.0
    return alpha_max * pdf / peak


def lr_hump_tail_beta(k: int,
                      N: int,
                      alpha_max: float,
                      beta: float,
                      a: float = 3.0,
                      b: float = 6.0) -> float:
    x = (k - 1) / (N - 1)
    hump = lr_hump_beta(k, N, alpha_max - beta, a, b)
    tail = beta * x
    return hump + tail