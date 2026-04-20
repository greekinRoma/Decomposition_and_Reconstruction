from os import path
from IPython import embed
from cv2 import resize
from httpx import get
import torch
from torch import nn
from torch.nn.functional import unfold
from torch.nn.functional import pad
from partition import window_unpartition, window_partition
from mix_rope import MultiplyMatrixWithRoPE
class DecompositionModel(nn.Module):
    def __init__(self, num_head=16, m=2, n =2, origin_patch_size=4, resize_patch_size=16, origin_embed_dim = 48, resize_embed_dim=48, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.origin_embed_dim = origin_embed_dim
        self.resize_embed_dim = resize_embed_dim
        self.resize_patch_size = resize_patch_size
        self.origin_patch_size = origin_patch_size
        self.m = m
        self.n = n
        self.original_patch_embed = nn.Conv2d(in_channels=3,out_channels=self.origin_embed_dim,kernel_size=self.origin_patch_size,stride=self.origin_patch_size,groups=3).cuda()
        self.resize_patch_embed = nn.Conv2d(in_channels=3,out_channels=self.resize_embed_dim,kernel_size=self.resize_patch_size,stride=self.resize_patch_size,groups=3).cuda()
        self.b = nn.Linear(self.origin_embed_dim, self.origin_embed_dim)
        self.q  = nn.Linear(self.resize_embed_dim, self.resize_embed_dim)
        self.num_heads = num_head
        self.resize = 1024 // self.resize_patch_size
        self.pos_embed = nn.Parameter(torch.rand(1, 1024 // self.resize_patch_size//self.m, 1024 // self.resize_patch_size//self.n, self.resize_embed_dim),requires_grad=True)
        self.pos = nn.Parameter(torch.rand(self.m*self.n, self.num_heads, self.resize * self.resize//self.m //self.n, self.resize_embed_dim // self.num_heads),requires_grad=True)
        self.norm_b = nn.LayerNorm(self.origin_embed_dim)
        self.norm_q = nn.LayerNorm(self.resize_embed_dim)
        self.q_attn = nn.Sequential(*[
            nn.Linear(self.resize_embed_dim//self.num_heads,1),
            nn.Sigmoid()
        ])
        self.down = nn.Sequential(
            nn.Linear(self.resize_embed_dim//self.num_heads, self.origin_embed_dim//self.num_heads)
        )
        self.proj = nn.Linear(self.resize_embed_dim, self.resize_embed_dim)
        self.mat_mul = MultiplyMatrixWithRoPE(num_heads=num_head,head_dim=self.origin_embed_dim//self.num_heads, rope_theta=10.0)

    def forward(self, x):
        B, C, H, W = x.shape
        
        # 1. Padding the raw image first to ensure it's divisible by (m * origin_patch_size)
        # This is necessary if you want the windows to contain whole patches.
        H_pad = (self.origin_patch_size * self.m - H % (self.origin_patch_size * self.m)) % (self.origin_patch_size * self.m)
        W_pad = (self.origin_patch_size * self.n - W % (self.origin_patch_size * self.n)) % (self.origin_patch_size * self.n)
        x_pad = pad(x, (0, W_pad, 0, H_pad), mode='constant', value=0)
        
        # 2. Interpolate for the resize branch before partitioning
        resize_x = torch.nn.functional.interpolate(x, size=(1024, 1024), mode='bilinear', align_corners=False)

        # ==========================================
        # 3. WINDOW PARTITION (Before Embedding)
        # We treat windows as a batch dimension: [B, C, H, W] -> [B*m*n, C, H/m, W/n]
        # ==========================================
        # Original branch partition
        x_win, [win_h, win_w] = window_partition(x_pad.permute(0, 2, 3, 1), m=self.m, n=self.n)
        x_win = x_win.permute(0, 3, 1, 2) # Back to [B_win, C, H_win, W_win]
        
        # Resize branch partition
        rx_win, [rwin_h, rwin_w] = window_partition(resize_x.permute(0, 2, 3, 1), m=self.m, n=self.n)
        rx_win = rx_win.permute(0, 3, 1, 2)

        # ==========================================
        # 4. EMBEDDING (Now operating on windows)
        # ==========================================
        origin_patches = self.original_patch_embed(x_win).permute(0, 2, 3, 1)
        resize_patches = self.resize_patch_embed(rx_win).permute(0, 2, 3, 1)

        # Get spatial dimensions for Attention
        _, oh, ow, _ = origin_patches.shape
        _, rh, rw, _ = resize_patches.shape

        # Normalization and Position Bias
        origin_patches = self.norm_b(origin_patches)
        resize_patches = self.norm_q(resize_patches + self.pos_embed)

        # Head Splitting and Linear Projections
        B_win = B * self.m * self.n
        head_dim_q = self.resize_embed_dim // self.num_heads
        head_dim_b = self.origin_embed_dim // self.num_heads

        q = self.q(resize_patches).view(B_win, rh * rw, self.num_heads, head_dim_q).permute(0, 2, 1, 3)
        b = self.b(origin_patches).view(B_win, oh * ow, self.num_heads, head_dim_b).permute(0, 2, 1, 3)

        # ... (Rest of your Attention and Matrix Multiplication logic remains the same) ...
        tmp_b = b.clone()
        attn = self.q_attn(q)
        q = torch.nn.functional.normalize(q * attn + self.pos * (1. - attn), dim=-1)
        b = torch.nn.functional.normalize(b, dim=-1)

        attn_map = self.mat_mul.multiply(q=q, b=b, end_x_xq=rw, end_y_xq=rh, end_x_xb=ow, end_y_xb=oh)
        out = attn_map @ tmp_b 

        # Merge Heads and Unpartition
        out = out.permute(0, 2, 1, 3).contiguous().view(B_win, rh, rw, self.resize_embed_dim)
        
        # Reconstruct full image from windows
        x = out.view(B, self.m, self.n, rh, rw, self.resize_embed_dim)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, self.resize, self.resize, self.resize_embed_dim)

        x = self.proj(x).permute(0, 3, 1, 2)
        
        return x, attn_map, [oh, ow], [H, W]