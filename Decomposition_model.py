import torch
from torch import nn
import torch.nn.functional as F
from partition import window_partition, window_reverse
from mix_rope import MultiplyMatrixWithRoPE

class DecompositionModel(nn.Module):
    def __init__(self, num_head=16, m=2, n=2, origin_patch_size=4, resize_patch_size=16, 
                 origin_embed_dim=48, resize_embed_dim=48, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 基础参数
        self.m, self.n = m, n
        self.num_heads = num_head
        self.ratio = 64
        self.h_num_ratio, self.w_num_ratio = 8, 8
        self.num_ratio = self.h_num_ratio * self.w_num_ratio
        self.resize_embed_dim = resize_embed_dim
        self.origin_embed_dim = origin_embed_dim
        
        # 维度计算
        head_dim_b = origin_embed_dim // (num_head * self.ratio)
        head_dim_q = resize_embed_dim // (num_head * self.num_ratio)
        
        # 网络层 - 移除 .cuda()
        self.patch_embed = nn.Conv2d(resize_embed_dim // self.ratio, resize_embed_dim, 
                                     kernel_size=resize_patch_size, stride=resize_patch_size)
        
        self.b_proj = nn.Linear(head_dim_b, head_dim_b)
        self.q_proj = nn.Linear(head_dim_q, head_dim_b) # 注意：q和b投影后的维度通常需一致才能做matmul
        
        # 位置编码预设
        # 建议根据实际输入的 resize 动态生成或使用固定的 Max Size
        pos_len = 1024 * 1024 * self.num_ratio // (m * n * resize_patch_size**2)
        self.pos = nn.Parameter(torch.randn(1, num_head, pos_len, head_dim_b))
        
        self.norm_b = nn.LayerNorm(head_dim_b)
        self.norm_q = nn.LayerNorm(head_dim_q)
        
        self.q_attn = nn.Sequential(
            nn.Linear(head_dim_b, 1),
            nn.Sigmoid()
        )
        
        self.proj = nn.Conv2d(resize_embed_dim // self.ratio * self.num_ratio, 
                              resize_embed_dim, kernel_size=1)
        
        self.mat_mul = MultiplyMatrixWithRoPE(num_heads=num_head, 
                                              head_dim=head_dim_b, 
                                              rope_theta=10.0)
        
        self.x_win_conv = nn.Conv2d(3, origin_embed_dim // self.ratio, kernel_size=3, padding=1)
        self.r_win_conv = nn.Conv2d(3, origin_embed_dim // self.ratio, kernel_size=3, padding=1)

    def forward(self, x):
        B, C, H, W = x.shape

        # 1. 局部特征提取与分区
        x_feat = self.x_win_conv(x)
        x_win, x_win_size, x_shape = window_partition(x_feat, m=self.m, n=self.n)

        # 2. 插值与全局/缩放特征处理
        # 优化：合并 pad 和 interpolate
        pad_h, pad_w = x_shape[0] - H, x_shape[1] - W
        res_x = F.interpolate(F.pad(x, (0, pad_w, 0, pad_h)), size=(1024, 1024), 
                              mode='bilinear', align_corners=False)
        
        res_x = self.r_win_conv(res_x)
        res_x_base = self.patch_embed(res_x) # 保持这个特征用于最后的残差连接
        
        # 3. 维度重排优化 (Flattening indices)
        # 目标: 将补齐后的特征重新排列以适配 window 逻辑
        B_idx, C_idx, R = B, self.resize_embed_dim // self.num_ratio, 1024 // 16
        res_x_reshaped = res_x_base.view(B_idx, C_idx, self.h_num_ratio, self.w_num_ratio, R, R)
        res_x_reshaped = res_x_reshaped.permute(0, 1, 4, 2, 5, 3).reshape(B_idx, C_idx, R*self.h_num_ratio, R*self.w_num_ratio)

        rx_win, rx_win_size, rx_shape = window_partition(res_x_reshaped, m=self.m, n=self.n)

        # 4. 准备 Attention 的 Q, B
        # 这里的 view 转换可以直接合并入 linear 前后的处理
        # print(x_win.shape)
        x_win = x_win.view(B * self.m * self.n, self.num_heads, -1, x_win_size[-1]*x_win_size[-2]).transpose(-2, -1)
        rx_win = rx_win.view(B * self.m * self.n, self.num_heads, -1, rx_win_size[-1]*rx_win_size[-2]).transpose(-2, -1)

        b = self.b_proj(self.norm_b(x_win))
        q = self.q_proj(self.norm_q(rx_win))
        
        # 5. Attention 逻辑
        # 避免不必要的 clone，直接使用 b
        b_norm = F.normalize(b, dim=-1)
        gate = self.q_attn(q)
        q = F.normalize(q * gate + self.pos * (1. - gate), dim=-1)
        
        attn_map = self.mat_mul.multiply(q=q, b=b_norm, 
                                        end_x_xb=x_win_size[0], end_y_xb=x_win_size[1], 
                                        end_x_xq=rx_win_size[0], end_y_xq=rx_win_size[1])
        
        # 应用注意力
        out_win = attn_map @ b # 使用原始投影后的 b

        # 6. 逆变换回图像空间
        out_win = out_win.transpose(-2, -1).reshape(B * self.m * self.n, -1, rx_win_size[0], rx_win_size[1])
        n_x = window_reverse(out_win, m=self.m, n=self.n, paged_shape=rx_shape)
        
        # 复杂的逆重排
        n_x = n_x.view(B, C_idx, R, self.h_num_ratio, R, self.w_num_ratio)
        n_x = n_x.permute(0, 1, 3, 5, 2, 4).reshape(B, -1, R, R)

        # 7. 残差与投影
        x_final = self.proj(n_x + res_x_base)
        
        return x_final, attn_map, [H, W], x_shape