from torch import nn
from torch.nn import LayerNorm,GELU
import torch
import torch.nn.functional as F
from partition import window_partition,window_reverse
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
        
    def forward(self, x, attn, origin_win=None, origin_shape=None):
        '''
        size_t [B,self.m,self.n,H,W]
        '''
        B, C, H, W = x.shape
        x = self.downsample(x)
        y = self.output_upscaling(x)
        y, _, _ = window_partition(y,self.m,self.n)
        y = y.reshape(B*self.m*self.n,self.num_head,self.embed_dim//self.num_head//64,-1)
        y = y @ attn
        y = y.reshape(B*self.m*self.n,self.embed_dim//64,origin_shape[0]//self.m,origin_shape[1]//self.n)
        y = window_reverse(windows=y,m=self.m,n=self.n,paged_shape=origin_shape)
        # print(y[:,:,:origin_win[0],:origin_win[1]].shape)
        # print(origin_win)
        y = self.out_conv(y[:,:,:origin_win[0],:origin_win[1]])
        return y 