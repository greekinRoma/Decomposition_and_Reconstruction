import torch
from torch.nn import functional as F
from typing import Tuple

def window_partition(x: torch.Tensor, m: int, n: int) -> Tuple[torch.Tensor, Tuple[int, int], Tuple[int, int]]:
    """
    Args:
        x: 输入特征图 [B, C, H, W]
        m: 垂直方向的窗口数量 (rows of windows)
        n: 水平方向的窗口数量 (cols of windows)
    """
    B, C, H, W = x.shape

    # 1. 计算每个窗口的大小 (window_h, window_w)
    # 如果不能整除，先进行 Padding
    pad_h = (m - H % m) % m
    pad_w = (n - W % n) % n

    if pad_h > 0 or pad_w > 0:
        # F.pad 在 (B, C, H, W) 下，最后四个参数对应 (W_left, W_right, H_top, H_bottom)
        x = F.pad(x, (0, pad_w, 0, pad_h))
    # print(x.shape)
    Hp, Wp = H + pad_h, W + pad_w
    win_h, win_w = Hp // m, Wp // n

    # 2. 重塑维度进行分区
    # 目标是将 H 拆分为 m * win_h, W 拆分为 n * win_w
    # 形状变化: [B, C, m, win_h, n, win_w]
    x = x.view(B, C, m, win_h, n, win_w)
    
    # 3. 置换维度
    # 我们希望窗口被堆叠在 Batch 维度: [B * m * n, C, win_h, win_w]
    # 顺序: (B, m, n, C, win_h, win_w)
    windows = x.permute(0, 2, 4, 1, 3, 5).contiguous()
    windows = windows.view(-1, C, win_h, win_w)

    return windows, (win_h, win_w), (Hp, Wp)

def window_reverse(windows: torch.Tensor, 
                   m: int, n: int, 
                   paged_shape: Tuple[int, int]) -> torch.Tensor:
    """
    Args:
        windows: [B * m * n, C, win_h, win_w]
        m, n: 窗口数量
        paged_shape: (Hp, Wp)
    """
    Hp, Wp = paged_shape
    C = windows.shape[1]
    win_h, win_w = windows.shape[2], windows.shape[3]
    
    # 1. 计算原始 Batch size
    B = windows.shape[0] // (m * n)
    
    # 2. View 重塑回多维
    # [B, m, n, C, win_h, win_w]
    x = windows.view(B, m, n, C, win_h, win_w)
    
    # 3. Permute 还原维度顺序
    # 对应 partition 的 permute(0, 2, 4, 1, 3, 5)
    # 我们要回到 [B, C, m, win_h, n, win_w]
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
    
    # 4. 合并为大图 [B, C, Hp, Wp]
    x = x.view(B, C, Hp, Wp)
        
    return x