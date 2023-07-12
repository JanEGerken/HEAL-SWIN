import math

import torch


def window_partition(x, window_size):
    """
    Args:
        x: (B, N, C)
        window_size (int): Must be a power of 2 in the healpy grid.

    Returns:
        windows: (num_windows*B, window_size * window_size, C)
    """
    # assert that window_size is a power of 2
    assert (math.log(window_size) / math.log(2)) % 1 == 0

    B, N, C = x.shape
    x = x.view(B, N // window_size, window_size, C)
    windows = x.contiguous().view(-1, window_size, C)
    return windows


def window_reverse(windows, window_size, N):
    """
    Args:
        windows: (num_windows*B, window_size, C)
        window_size (int): Must be a power of 2 in the healpy grid
        N (int): Number of pixels in the healpy grid

    Returns:
        x: (B, N, C)
    """
    # assert that window_size is a power of 2
    assert (math.log(window_size) / math.log(2)) % 1 == 0

    B = int(windows.shape[0] / (N // window_size))
    x = windows.view(B, N // window_size, window_size, -1)
    x = x.contiguous().view(B, N, -1)
    return x


def get_nest_win_idcs(window_size):
    """Returns a sqrt(window_size) x sqrt(window_size) tensor with the indices from the nested index
    scheme"""
    result = torch.zeros((int(window_size**0.5), int(window_size**0.5)), dtype=torch.int64)

    def fill_quadrant(idx, x, y, size):
        if size == 2:
            result[x, y + 1] = idx
            result[x, y] = idx + 1
            result[x + 1, y + 1] = idx + 2
            result[x + 1, y] = idx + 3
        else:
            fill_quadrant(idx, x, y + size // 2, size // 2)
            fill_quadrant(idx + size**2 // 4, x, y, size // 2)
            fill_quadrant(idx + 2 * (size**2 // 4), x + size // 2, y + size // 2, size // 2)
            fill_quadrant(idx + 3 * (size**2 // 4), x + size // 2, y, size // 2)

    fill_quadrant(0, 0, 0, int(window_size**0.5))

    return result
