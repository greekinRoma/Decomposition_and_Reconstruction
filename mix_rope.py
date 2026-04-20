import torch
import math # 推荐使用 math.pi，比 torch.pi 更 Pythonic
from typing import Tuple
from torch import nn

class MultiplyMatrixWithRoPE(nn.Module):
    def __init__(self, head_dim: int, num_heads: int, rope_theta: float = 10.0):
        super().__init__() # 关键修复1：继承 nn.Module
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.rope_theta = rope_theta
        
        freqs = self.init_random_2d_freqs(head_dim, num_heads, rope_theta)
        # 关键修复2：移除 .cuda()，交由模型统一管理设备
        self.rope_freqs = nn.Parameter(freqs, requires_grad=True)
        self.weight = nn.Parameter(torch.rand(1,self.num_heads,1,1),requires_grad=True)
        
        # 关键修复3：删除了未使用的 weight_delta, bias, scales, weight 避免 DDP 报错

    def init_random_2d_freqs(self, head_dim: int, num_heads: int, theta: float):
        # 关键修复4：完全向量化，消灭 for 循环
        mag = 1 / (theta ** (torch.arange(0, head_dim, 8)[: (head_dim // 8)].float() / head_dim))
        
        # 直接生成形状为 [num_heads, 1] 的随机角度
        angles = torch.rand(num_heads, 1) * 2 * math.pi
        
        # 利用广播机制直接计算出形状为 [num_heads, head_dim // 4] 的特征
        fx = torch.cat([mag * torch.cos(angles), mag * torch.cos(math.pi/2 + angles)], dim=-1)
        fy = torch.cat([mag * torch.sin(angles), mag * torch.sin(math.pi/2 + angles)], dim=-1)
        
        # 最终 shape: [2, num_heads, head_dim // 4]
        freqs = torch.stack([fx, fy], dim=0)
        return freqs

    def reshape_for_broadcast(self, freqs_cis: torch.Tensor, x: torch.Tensor):
        ndim = x.ndim
        assert 0 <= 1 < ndim
        if freqs_cis.shape == (x.shape[-2], x.shape[-1]):
            shape = [d if i >= ndim-2 else 1 for i, d in enumerate(x.shape)]
        elif freqs_cis.shape == (x.shape[-3], x.shape[-2], x.shape[-1]):
            shape = [d if i >= ndim-3 else 1 for i, d in enumerate(x.shape)]
        else:
            raise ValueError(f"Unsupported shapes: freqs_cis {freqs_cis.shape}, x {x.shape}")
        return freqs_cis.view(*shape)
    
    def init_t_xy(self, end_x: int, end_y: int, device: torch.device):
        # 关键修复5：直接在对应设备上生成张量，避免前向传播时的 CPU->GPU 拷贝
        t = torch.arange(end_x * end_y, dtype=torch.float32, device=device)
        t_x = (t % end_x)
        t_y = torch.div(t, end_x, rounding_mode='floor')
        return t_x, t_y
    
    def apply_rotary_emb(
        self, xq: torch.Tensor, xk: torch.Tensor, xq_freqs_cis: torch.Tensor, xk_freqs_cis: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 增加鲁棒性检查：确保最后一维是偶数才能复数化
        assert xq.shape[-1] % 2 == 0, "Feature dimension must be even for RoPE"
        
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        
        xq_freqs_cis = self.reshape_for_broadcast(xq_freqs_cis, xq_)
        xk_freqs_cis = self.reshape_for_broadcast(xk_freqs_cis, xk_)
        
        xq_out = torch.view_as_real(xq_ * xq_freqs_cis).flatten(3)
        xk_out = torch.view_as_real(xk_ * xk_freqs_cis).flatten(3)
        return xq_out.type_as(xq), xk_out.type_as(xk)
    
    def compute_cis(self, freqs: torch.Tensor, t_x: torch.Tensor, t_y: torch.Tensor):
        with torch.autocast(device_type='cuda', enabled=False): # 使用更现代的 API
            freqs_x = (t_x.unsqueeze(-1) @ freqs[0].float().unsqueeze(-2))
            freqs_y = (t_y.unsqueeze(-1) @ freqs[1].float().unsqueeze(-2))
            freqs = torch.cat([freqs_x, freqs_y], dim=-1)
            freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis
    
    def multiply(self, q: torch.Tensor, b: torch.Tensor, end_x_xq: int, end_y_xq: int, end_x_xb: int, end_y_xb: int):        
        device = q.device # 获取当前张量的设备
        
        t_x_xb, t_y_xb = self.init_t_xy(end_x_xb, end_y_xb, device)
        t_x_xq, t_y_xq = self.init_t_xy(end_x_xq, end_y_xq, device)
        
        # 保持你正确修改的插值逻辑
        t_x_xb = torch.floor((t_x_xb + 0.5) / end_x_xb * end_x_xq - 0.5)
        t_y_xb = torch.floor((t_y_xb + 0.5) / end_y_xb * end_y_xq - 0.5)
        
        xk_freqs_cls = self.compute_cis(self.rope_freqs, t_x_xb, t_y_xb)
        xq_freqs_cls = self.compute_cis(self.rope_freqs, t_x_xq, t_y_xq)
        delta = torch.exp(-torch.sqrt((t_x_xb.unsqueeze(0) - t_x_xq.unsqueeze(-1))**2 + (t_y_xb.unsqueeze(0) - t_y_xq.unsqueeze(-1))**2).unsqueeze(0).unsqueeze(0)*torch.nn.functional.leaky_relu(self.weight))
        xq, xb = self.apply_rotary_emb(q, b, xq_freqs_cls, xk_freqs_cls)
        attn = xq @ xb.transpose(-2, -1)*delta
        return attn