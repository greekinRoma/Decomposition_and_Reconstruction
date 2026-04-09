import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple, Type

class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (16, 16),
        padding: Tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int):  embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        return x
import torch

def get_2d_rope(h, w, dim, base=10000.0, device='cpu'):
    """
    生成二维的 Rotary Positional Embedding (2D RoPE)
    
    参数:
        h (int): 图像/网格的高度 (对应 rows)
        w (int): 图像/网格的宽度 (对应 cols)
        dim (int): 每个 token 的特征维度 (必须是偶数，通常是 head_dim)
        base (float): 频率底数
    
    返回:
        cos, sin: 形状均为 (h * w, dim) 的张量，可直接用于 Attention
    """
    assert dim % 2 == 0, "Dimension must be divisible by 2"
    
    # 因为维度要分给 H 和 W 各一半，所以每个维度的有效大小是 dim // 2
    half_dim = dim // 2
    
    # 1. 计算频率系数 inv_freq (长度为 half_dim // 2)
    # theta_i = base^(-2i / half_dim)
    inv_freq = 1.0 / (base ** (torch.arange(0, half_dim, 2).float().to(device) / half_dim))
    
    # 2. 生成 Y 轴 (高度) 和 X 轴 (宽度) 的位置索引
    pos_y = torch.arange(h, device=device, dtype=inv_freq.dtype)
    pos_x = torch.arange(w, device=device, dtype=inv_freq.dtype)
    
    # 3. 计算各个轴的频率外积
    # freqs_y 形状: (h, half_dim // 2)
    # freqs_x 形状: (w, half_dim // 2)
    freqs_y = torch.outer(pos_y, inv_freq)
    freqs_x = torch.outer(pos_x, inv_freq)
    
    # 4. 利用广播机制扩展到二维网格 (h, w)
    # 将 freqs_y 扩展为 (h, w, half_dim // 2)
    freqs_y = freqs_y.unsqueeze(1).expand(h, w, -1)
    # 将 freqs_x 扩展为 (h, w, half_dim // 2)
    freqs_x = freqs_x.unsqueeze(0).expand(h, w, -1)
    
    # 5. 拼接 H 和 W 的频率
    # 此时形状为 (h, w, half_dim)
    freqs_2d = torch.cat([freqs_y, freqs_x], dim=-1)
    
    # 6. 复制一份以匹配复数旋转的实数等价拼接 (类似于 1D RoPE 中的重复)
    # 最终形状为 (h, w, dim)
    emb = torch.cat((freqs_2d, freqs_2d), dim=-1)
    
    # 7. 展平为一维序列 (h * w, dim)，因为 Transformer 通常按 1D 序列处理
    emb = emb.view(h * w, dim)
    
    # 返回 cos 和 sin
    return emb.cos(), emb.sin()