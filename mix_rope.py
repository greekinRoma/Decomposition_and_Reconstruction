import torch
from typing import Tuple
from torch import nn
class rope():
    def __init__(self, head_dim, num_heads, rope_theta):
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.rope_theta = rope_theta
        freqs = self.init_random_2d_freqs(
            head_dim=self.head_dim, num_heads=self.num_heads, theta=rope_theta
        )
        self.rope_freqs = nn.Parameter(freqs, requires_grad=True).cuda()
        self.q_scale = nn.Parameter(torch.ones(1,1,head_dim//2),requires_grad=True).cuda()
        self.b_scale = nn.Parameter(torch.ones(1,1,head_dim//2),requires_grad=True).cuda()
        # print(self.rope_freqs.shape)
        # print(self.num_heads)
        # print(self.head_dim)
    
    def init_random_2d_freqs(self, head_dim: int, num_heads: int, theta: float = 10.0):
        freqs_x = []
        freqs_y = []
        theta = theta
        mag = 1 / (theta ** (torch.arange(0, head_dim, 8)[: (head_dim // 8)].float() / head_dim))
        for i in range(num_heads):
            angles = torch.rand(1) * 2 * torch.pi
            fx = torch.cat([mag * torch.cos(angles), mag * torch.cos(torch.pi/2 + angles)], dim=-1)
            fy = torch.cat([mag * torch.sin(angles), mag * torch.sin(torch.pi/2 + angles)], dim=-1)
            # 转角矩阵
            freqs_x.append(fx)
            freqs_y.append(fy)
        freqs_x = torch.stack(freqs_x, dim=0)
        freqs_y = torch.stack(freqs_y, dim=0)
        freqs = torch.stack([freqs_x, freqs_y], dim=0)
        # 2, num_heads, head_dim//2 
        return freqs

        
    def reshape_for_broadcast(self, freqs_cis: torch.Tensor, x: torch.Tensor):
        ndim = x.ndim
        assert 0 <= 1 < ndim
        # print(freqs_cis.shape)
        # print((x.shape[-3], x.shape[-2], x.shape[-1]))
        # print(freqs_cis.shape == (x.shape[-3], x.shape[-2], x.shape[-1]))
        if freqs_cis.shape == (x.shape[-2], x.shape[-1]):
            shape = [d if i >= ndim-2 else 1 for i, d in enumerate(x.shape)]
        elif freqs_cis.shape == (x.shape[-3], x.shape[-2], x.shape[-1]):
            shape = [d if i >= ndim-3 else 1 for i, d in enumerate(x.shape)]
        else:
            raise
        return freqs_cis.view(*shape)
    
    def init_t_xy(self, end_x: int, end_y: int, zero_center=False):
        t = torch.arange(end_x * end_y, dtype=torch.float32)
        t_x = (t % end_x).float()
        t_y = torch.div(t, end_x, rounding_mode='floor').float()
        return t_x.cuda(), t_y.cuda()
    
    def apply_rotary_emb(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        xq_freqs_cis: torch.Tensor,
        xk_freqs_cis: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        xq_freqs_cis = self.reshape_for_broadcast(xq_freqs_cis, xq_).to(xq.device)
        xk_freqs_cis = self.reshape_for_broadcast(xk_freqs_cis, xk_).to(xk.device)
        xq_out = torch.view_as_real(xq_ * xq_freqs_cis).flatten(3)
        xk_out = torch.view_as_real(xk_ * xk_freqs_cis).flatten(3)
        return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)
    
    def compute_cis(self, freqs: torch.Tensor, t_x: torch.Tensor, t_y: torch.Tensor, scale: torch.Tensor):
        N = t_x.shape[0]
        # print(freqs.shape)
        with torch.cuda.amp.autocast(enabled=False):
            freqs_x = (t_x.unsqueeze(-1) @ freqs[0].unsqueeze(-2))
            freqs_y = (t_y.unsqueeze(-1) @ freqs[1].unsqueeze(-2))
            scale_1, scale_2 = torch.chunk(scale,dim=-1,chunks=2)
            freqs = torch.concat([torch.cos(freqs_x)*scale_1, torch.cos(freqs_y)*scale_2, torch.sin(freqs_x)*scale_1, torch.sin(freqs_y)*scale_2], dim=-1)            
        return freqs
    
    def multiply(self, q, b, end_x_xq, end_y_xq, end_x_xb, end_y_xb):        
        t_x_xb, t_y_xb = self.init_t_xy(end_x_xb, end_y_xb)
        t_x_xq, t_y_xq = self.init_t_xy(end_x_xq, end_y_xq)

        t_x_xq = t_x_xq / end_x_xq * end_x_xb
        t_y_xq = t_y_xq / end_y_xq * end_y_xb

        xq_freqs_cls = self.compute_cis(self.rope_freqs, t_x_xq, t_y_xq)
        xk_freqs_cls = self.compute_cis(self.rope_freqs, t_x_xb, t_y_xb)
        xq, xb = self.apply_rotary_emb(q[...,:self.head_dim], b[...,:self.head_dim], xq_freqs_cls, xk_freqs_cls)

        attn = xq  @ xb.transpose(-2, -1) + q[...,self.head_dim:]@b[...,self.head_dim:].transpose(-2,-1)
        return attn
    
    def compute_rope(self, end_x_xq, end_y_xq, end_x_xb, end_y_xb):
        t_x_xb, t_y_xb = self.init_t_xy(end_x_xb, end_y_xb)
        t_x_xq, t_y_xq = self.init_t_xy(end_x_xq, end_y_xq)

        t_x_xq = t_x_xq / end_x_xq * end_x_xb
        t_y_xq = t_y_xq / end_y_xq * end_y_xb

        xq_freqs_cls = self.compute_cis(self.rope_freqs, t_x_xq, t_y_xq, self.q_scale).cuda()
        xk_freqs_cls = self.compute_cis(self.rope_freqs, t_x_xb, t_y_xb, self.b_scale).cuda()
        return xq_freqs_cls, xk_freqs_cls



