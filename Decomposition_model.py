from os import path
from IPython import embed
from cv2 import resize
from httpx import get
import torch
from torch import nn
from torch.nn.functional import unfold
from torch.nn.functional import pad
class DecompositionModel(nn.Module):
    def __init__(self, patch_size=16, num_head=16, embed_dim=768, m=2, n =2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.origin_patch_size = patch_size
        self.original_patch_embed = nn.Conv2d(in_channels=3,out_channels=embed_dim,kernel_size=self.origin_patch_size,stride=self.origin_patch_size,groups=3).cuda()
        self.resize_patch_embed = nn.Conv2d(in_channels=3,out_channels=embed_dim,kernel_size=patch_size,stride=patch_size,groups=3).cuda()
        self.b = nn.Linear(embed_dim, embed_dim)
        self.q  = nn.Linear(embed_dim, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, 1024 // patch_size, 1024 // patch_size))
        self.norm_b = nn.LayerNorm(embed_dim)
        self.norm_q = nn.LayerNorm(embed_dim)
        self.num_heads = num_head
        head_dim = embed_dim// self.num_heads
        self.scale = head_dim**-0.5
        self.resize = 1024 // patch_size
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.m = m
        self.n = n
    def forward(self, x):
        B, C, H, W = x.shape
        resized_x = torch.nn.functional.interpolate(x, size=(1024, 1024), mode='bilinear', align_corners=False)
        # H_pad = self.origin_patch_size*self.m - H % (self.origin_patch_size*self.m)
        # W_pad = self.origin_patch_size*self.n - W % (self.origin_patch_size*self.n)
        # x_pad = pad(x, (0, W_pad, 0, H_pad), mode='constant', value=0)
        # B, C, H_pad, W_pad = x_pad.shape
        # H_pad_patch = H_pad // self.origin_patch_size
        # W_pad_patch = W_pad // self.origin_patch_size
        
        resized_patches = (self.resize_patch_embed(resized_x) + self.pos_embed).permute(0,2,3,1).reshape(B, self.resize * self.resize, self.embed_dim)

        # original_patches = self.original_patch_embed(x_pad).permute(0,2,3,1).reshape(B, H_pad_patch* W_pad_patch, self.embed_dim)
        resized_patches = self.norm_q(resized_patches)
        # original_patches = self.norm_b(original_patches)

        q = self.q(resized_patches).reshape(B, self.resize * self.resize, self.num_heads, -1).permute(0, 2, 1, 3).reshape(B * self.num_heads, self.resize * self.resize, -1)
        # original_b = self.b(original_patches).reshape(B, H_pad_patch* W_pad_patch, self.num_heads, -1).permute(0,2,1,3)

        # b = original_b.reshape(B * self.num_heads, H_pad_patch* W_pad_patch, -1)
        x = q.reshape(B, self.num_heads, self.resize, self.resize, -1).permute(0, 2, 3, 1, 4).reshape(B, self.resize, self.resize, -1)
        x = self.proj(x).permute(0, 3, 1, 2)
        

        # b = original_b.reshape(B*self.num_heads, self.m , H_pad_patch//self.m, self.n , W_pad_patch//self.n, -1).permute(0, 1, 3, 2, 4, 5).reshape(B*self.num_heads*self.m*self.n, H_pad_patch* W_pad_patch//(self.m*self.n), -1)
        # q = q.reshape(B*self.num_heads, self.m, self.resize//self.m, self.n, self.resize//self.n, -1).permute(0, 1, 3, 2, 4, 5).reshape(B*self.num_heads*self.m*self.n, self.resize * self.resize//(self.m*self.n), -1)
        # tmp_b = b.clone()
        # q = torch.nn.functional.normalize(q, dim=-1)
        # b = torch.nn.functional.normalize(b, dim=-1)
        # attn = q  @ b.transpose(-2, -1)
        # x = (attn @ tmp_b).reshape(B, self.num_heads, self.m, self.n, self.resize//self.m, self.resize//self.n, -1).permute(0, 2, 4, 3, 5, 1, 6).reshape(B, self.resize, self.resize, self.embed_dim)
        
        return x, None, [self.resize, self.resize], [H, W]
