from torch import nn
from torch.nn import LayerNorm,GELU
import torch
import torch.nn.functional as F
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
class ReconstructionModel(nn.Module):
    def __init__(self, patch_size, num_head, m, n, embed_dim=768,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.m = m
        self.n = n
        self.patch_size = patch_size
        self.num_head = num_head
        self.embed_dim = embed_dim
        self.downsample = nn.Linear(self.embed_dim, self.embed_dim)
        self.proj = nn.Conv2d(self.embed_dim, 3, kernel_size=1, stride=1)
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(embed_dim // 4),
            GELU(),
            nn.ConvTranspose2d(embed_dim// 4, embed_dim// 8, kernel_size=2, stride=2),
            LayerNorm2d(embed_dim // 8),
            GELU(),
            nn.ConvTranspose2d(embed_dim// 8, embed_dim// 16, kernel_size=2, stride=2),
            LayerNorm2d(embed_dim // 16),
            GELU(),
            nn.Conv2d(embed_dim//16, embed_dim//16, kernel_size=1, stride=1),
            GELU(),
        )
        self.out_conv = nn.Sequential(*[nn.Conv2d(embed_dim//16, 3, kernel_size=1, stride=1),
                                        nn.Sigmoid()])
        
    def forward(self, x, attn, pad_size=None, orgin_size=None):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H*W, self.embed_dim)
        y = self.downsample(x).permute(0,2,1).reshape(B, self.embed_dim, H, W)
        # y = (x.transpose(-1,-2) @ attn).reshape(B*self.num_head,self.m, self.n ,-1, pad_size[0]//self.m, pad_size[1]//self.n).permute(0, 3, 1, 4, 2, 5).reshape(B, self.embed_dim, pad_size[0], pad_size[1])
        y = self.output_upscaling(y)
        y =  F.interpolate(
            y,
            (orgin_size[0], orgin_size[1]),
            mode="bilinear",
            align_corners=False,
        )
        y = self.out_conv(y)
        return y 