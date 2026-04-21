from torch import nn
from torch.nn import LayerNorm,GELU
import torch
import torch.nn.functional as F
from partition import window_partition
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
        self.downsample = nn.Conv2d(self.embed_dim, self.embed_dim,kernel_size=1,stride=1)
        self.proj = nn.Conv2d(self.embed_dim, 3, kernel_size=1, stride=1)
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(embed_dim // 4),
            GELU(),
            nn.ConvTranspose2d(embed_dim//4, embed_dim // 16, kernel_size=2, stride=2),
            LayerNorm2d(embed_dim // 16),
            GELU(),
            nn.ConvTranspose2d(embed_dim//16, embed_dim // 32, kernel_size=2, stride=2),
            LayerNorm2d(embed_dim // 32),
            GELU(),
            nn.ConvTranspose2d(embed_dim//32, embed_dim // 64, kernel_size=2, stride=2),
            GELU(),
        )
        self.out_conv = nn.Sequential(*[nn.Conv2d(embed_dim//64, 3, kernel_size=1, stride=1),
                                        nn.Sigmoid()])
        
    def forward(self, x, attn, pad_size=None, orgin_size=None):
        B, C, H, W = x.shape
        x = self.downsample(x)
        y = self.output_upscaling(x).permute(0,2,3,1)
        y = window_partition(y,self.m,self.n,num_heads=self.num_head).permute(0,1,3,2)
        # print(y.shape)
        y = y @ attn
        # rx_win = rx_win.reshape(B*self.m*self.n,self.num_heads,self.resize_embed_dim//self.num_heads//self.ratio,-1)
        y = y.reshape(B,self.m,self.n,self.num_head,self.embed_dim//64//self.num_head,pad_size[0]//self.m,pad_size[1]//self.n).permute(0,3,4,1,5,2,6).reshape(B,self.embed_dim//64,pad_size[0],pad_size[1])
        y =  y[:,:, :orgin_size[0], :orgin_size[1]]
        y = self.out_conv(y)
        return y 