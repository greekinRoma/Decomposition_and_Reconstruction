from os import path
from IPython import embed
from cv2 import resize
from httpx import get
import torch
from torch import nn
from torch.nn.functional import unfold
from torch.nn.functional import pad
from partition import window_unpartition, window_partition
from mix_rope import rope
class DecompositionModel(nn.Module):
    def __init__(self, num_head=16, m=2, n =2, origin_patch_size=4, resize_patch_size=16, origin_embed_dim = 48, resize_embed_dim=48, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.origin_embed_dim = origin_embed_dim
        self.resize_embed_dim = resize_embed_dim
        self.resize_patch_size = resize_patch_size
        self.origin_patch_size = origin_patch_size
        self.num_heads = num_head
        self.m = m
        self.n = n
        self.original_patch_embed = nn.Conv2d(in_channels=3,out_channels=self.origin_embed_dim,kernel_size=self.origin_patch_size,stride=self.origin_patch_size,groups=3).cuda()
        self.resize_patch_embed = nn.Conv2d(in_channels=3,out_channels=self.resize_embed_dim,kernel_size=self.resize_patch_size,stride=self.resize_patch_size,groups=3).cuda()
        self.b = nn.Linear(self.origin_embed_dim, self.origin_embed_dim)
        self.q  = nn.Linear(self.resize_embed_dim, self.resize_embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.resize_embed_dim, 1024 // self.resize_patch_size, 1024 // self.resize_patch_size),requires_grad=True)
        self.norm_b = nn.LayerNorm(self.origin_embed_dim)
        self.norm_q = nn.LayerNorm(self.resize_embed_dim)
        self.num_heads = num_head
        self.reszie = 1024 // self.resize_patch_size
        self.down = nn.Sequential(
            nn.Linear(self.resize_embed_dim//self.num_heads, self.origin_embed_dim//self.num_heads)
        )
        self.proj = nn.Linear(self.resize_embed_dim, self.resize_embed_dim)
        self.rope_encoder = rope(num_heads=self.num_heads,head_dim=self.origin_embed_dim//self.num_heads//6, rope_theta=10.0)

    def forward(self, x):
        B, C, H, W = x.shape
        resize_x = torch.nn.functional.interpolate(x, size=(1024, 1024), mode='bilinear', align_corners=False)
        H_pad = (self.origin_patch_size - H % self.origin_patch_size) % self.origin_patch_size
        W_pad = (self.origin_patch_size - W % self.origin_patch_size) % self.origin_patch_size
        x_pad = pad(x, (0, W_pad, 0, H_pad), mode='constant', value=0)

        origin_patches = self.original_patch_embed(x_pad).permute(0,2,3,1)
        resize_patches = (self.resize_patch_embed(resize_x)+ self.pos_embed).permute(0,2,3,1)

        origin_patches, [oh, ow] = window_partition(origin_patches, m=self.m, n=self.n)
        resize_patches, [rh, rw] = window_partition(resize_patches, m=self.m, n=self.n)

        origin_patches = self.norm_b(origin_patches)
        resize_patches = self.norm_q(resize_patches)

        q = self.q(resize_patches)
        b = self.b(origin_patches)
        
        b = b.reshape(B*self.m*self.n, oh, ow, self.num_heads, self.origin_embed_dim // self.num_heads).permute(0,3,1,2,4).reshape(B*self.m*self.n, self.num_heads, oh * ow, self.origin_embed_dim // self.num_heads)
        q = q.reshape(B*self.m*self.n, rh, rw, self.num_heads, self.resize_embed_dim // self.num_heads).permute(0,3,1,2,4).reshape(B*self.m*self.n, self.num_heads, rh * rw, self.resize_embed_dim // self.num_heads)
        tmp_b = b.clone()
        xq_freqs_cls, xb_freqs_cls = self.rope_encoder.compute_rope( end_x_xq=rw, end_y_xq=rh, end_x_xb=ow, end_y_xb=oh)
        xq_freqs_cls = xq_freqs_cls.reshape(1, self.num_heads,rh*rw,-1).expand(self.m*self.n, -1, -1, -1)
        xb_freqs_cls = xb_freqs_cls.reshape(1, self.num_heads,oh*ow,-1).expand(self.m*self.n, -1, -1, -1)
        q = torch.concat([q, xq_freqs_cls],dim=-1)
        b = torch.concat([b, xb_freqs_cls],dim=-1)

        q = torch.nn.functional.normalize(q, dim=-1)
        b = torch.nn.functional.normalize(b, dim=-1)

        
        attn = q  @ b.transpose(-2, -1)
        x = (attn @ tmp_b).reshape(B, self.m, self.n, self.num_heads, self.reszie//self.m, self.reszie//self.n, -1).permute(0, 1, 4, 2, 5, 3, 6).reshape(B, self.reszie, self.reszie, self.resize_embed_dim)

        x = self.proj(x).permute(0, 3, 1, 2)
        return x, attn, [oh, ow], [H, W]
