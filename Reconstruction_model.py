import torch
from torch import nn
import torch.nn.functional as F
from partition import window_partition, window_reverse

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 使用 PyTorch 内置的 layer_norm 效率更高
        return F.layer_norm(
            x.permute(0, 2, 3, 1), 
            (x.size(1),), 
            self.weight, self.bias, self.eps
        ).permute(0, 3, 1, 2)

class ReconstructionModel(nn.Module):
    def __init__(self, num_head=16, m=2, n=2, embed_dim=48, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.m = m
        self.n = n
        self.num_head = num_head
        self.embed_dim = embed_dim
        
        # 对应 DecompositionModel 中的 ratio 逻辑
        self.h_num_ratio = 8
        self.w_num_ratio = 8
        self.num_ratio = self.h_num_ratio * self.w_num_ratio
        
        # 网络层
        self.downsample = nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=1)
        
        # 这里的通道数需要根据上文的 origin_embed_dim // ratio 匹配
        # 假设上文 ratio=64, origin_embed_dim=48, 则此处输入通道为 48/64 (需确保除尽)
        recon_channels = self.embed_dim // self.num_ratio
        
        self.out_conv = nn.Sequential(
            nn.Conv2d(recon_channels, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x, attn, origin_win=None, origin_shape=None):
        """
        x: DecompositionModel 的输出 [B, embed_dim, H_res, W_res]
        attn: 注意力矩阵 [B*m*n, num_head, Q_len, K_len]
        origin_win: 原始图像的分辨率 [H_orig, W_orig]
        origin_shape: pad 后的分辨率 [H_pad, W_pad]
        """
        B, C, H, W = x.shape
        # 1. 特征解构：将全局特征重新排列回比例块
        # 此处逻辑应与 DecompositionModel 的 n_x 重排逻辑互逆
        # y 形状变化: [B, C, H, W] -> [B, C/ratio, H, 8, W, 8] -> [B, C/ratio, H*8, W*8]
        y = x.view(B, C // self.num_ratio, self.h_num_ratio, self.w_num_ratio, H, W)
        y = y.permute(0, 1, 4, 2, 5, 3).reshape(B, C // self.num_ratio, H * self.h_num_ratio, W * self.w_num_ratio)
        
        # 2. 窗口分区
        y_win, _, y_shape = window_partition(y, self.m, self.n)
        
        # 3. 注意力还原
        # y_win: [B*m*n, C_sub, Win_H, Win_W] -> [B*m*n, num_head, head_dim, -1]
        C_sub = y_win.shape[1]
        y_win = y_win.reshape(B * self.m * self.n, self.num_head, C_sub // self.num_head, -1)
        
        # 核心：通过 attn 将低分辨率特征映射回原始窗口分布
        # attn 形状假设为 [Batch, Heads, L_target, L_source]
        # 使用 transpose 是为了处理矩阵乘法的维度对齐
        # print(attn.shape,y_win.shape)
        y_res = y_win @ attn
        
        # 4. 逆向还原图像
        # 还原到原始窗口大小
        target_h = origin_shape[0] // self.m
        target_w = origin_shape[1] // self.n
        y_res = y_res.reshape(B * self.m * self.n, C_sub, target_h, target_w)
        
        # 窗口合并
        y_out = window_reverse(y_res, self.m, self.n, paged_shape=origin_shape)
        
        # 5. 裁剪掉 Pad 部分并输出
        if origin_win is not None:
            y_out = y_out[:, :, :origin_win[0], :origin_win[1]]
            
        return self.out_conv(y_out)