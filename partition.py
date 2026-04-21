import torch
from torch.nn import functional as F
from typing import Optional, Tuple, Type
def window_partition(x: torch.Tensor, m: int, n:int, num_heads:int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = (m - H % m) % m
    pad_w = (n - W % n) % n
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, m, Hp//m,  n, Wp // n, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).reshape(m*n, Hp//m*Wp//n, num_heads, C//num_heads).permute(0,2,1,3)
    return windows