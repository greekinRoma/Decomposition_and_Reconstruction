import torch
from torch import nn
from torch.nn.functional import pad
from partition import window_partition, window_reverse
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
        self.ratio = 96
        self.resize = 1024 // self.resize_patch_size
        self.num_heads = num_head
        self.patch_embed = nn.Conv2d(in_channels=self.resize_embed_dim//self.ratio,out_channels=self.resize_embed_dim,kernel_size=self.resize_patch_size,stride=self.resize_patch_size).cuda()
        self.b = nn.Linear(self.origin_embed_dim//self.num_heads//self.ratio, self.origin_embed_dim//self.num_heads//self.ratio)
        self.q  = nn.Linear(self.resize_embed_dim//self.num_heads//self.ratio, self.resize_embed_dim//self.num_heads//self.ratio)
        
        self.pos = nn.Parameter(torch.rand(1, self.num_heads, 1024 * 1024//self.m //self.n, self.resize_embed_dim // self.num_heads//self.ratio),requires_grad=True)
        self.pos_emd = nn.Parameter(torch.rand(1, self.num_heads, 1024 * 1024//self.m //self.n, self.resize_embed_dim//self.num_heads//self.ratio),requires_grad=True)
       
        self.norm_b = nn.LayerNorm(self.origin_embed_dim//self.num_heads//self.ratio)
        self.norm_q = nn.LayerNorm(self.resize_embed_dim//self.num_heads//self.ratio)
        self.q_attn = nn.Sequential(*[
            nn.Linear(self.resize_embed_dim//self.num_heads//self.ratio,1),
            nn.Sigmoid()
        ])
        self.down = nn.Sequential(
            nn.Linear(self.resize_embed_dim//self.num_heads, self.origin_embed_dim//self.num_heads)
        )
        self.proj = nn.Conv2d(self.resize_embed_dim,self.resize_embed_dim,kernel_size=1,stride=1)
        self.mat_mul = MultiplyMatrixWithRoPE(num_heads=num_head,head_dim=self.origin_embed_dim//self.ratio//self.num_heads, rope_theta=10.0)
        self.x_win_conv = nn.Conv2d(3,self.origin_embed_dim//self.ratio,kernel_size=3,padding=1)
        self.r_win_conv = nn.Conv2d(3,self.origin_embed_dim//self.ratio,kernel_size=3,padding=1)

    def forward(self, x):
        B, C, H, W = x.shape

        
        x_pad = self.x_win_conv(x)
        x_win, x_win_size, x_shape = window_partition(x_pad, m=self.m, n=self.n)

        resize_x = torch.nn.functional.interpolate(torch.nn.functional.pad(x, (0,x_shape[1]-W, 0, x_shape[0]-H)), size=(1024, 1024), mode='bilinear', align_corners=False)
        resize_x = self.r_win_conv(resize_x)
        rx_win, rx_win_size, rx_shape = window_partition(resize_x, m=self.m, n=self.n)

        x_win = x_win.reshape(B*self.m*self.n,self.num_heads,self.origin_embed_dim//self.num_heads//self.ratio,-1).permute(0,1,3,2)
        rx_win = rx_win.reshape(B*self.m*self.n,self.num_heads,self.resize_embed_dim//self.num_heads//self.ratio,-1).permute(0,1,3,2)


        b = self.norm_b(x_win)
        q = self.norm_q(rx_win)

        b = self.b(b)
        q = self.q(q)
        tmp_b = b.clone()

        b = torch.nn.functional.normalize(b,dim=-1)
        attn = self.q_attn(q)
        q = torch.nn.functional.normalize(q*attn+self.pos*(1. -attn),dim=-1)

        attn_map = self.mat_mul.multiply(q=q, b=b, end_x_xb=x_win_size[0], end_y_xb=x_win_size[1], end_x_xq=rx_win_size[0], end_y_xq=rx_win_size[1])
        n_x = attn_map @ tmp_b

        n_x = n_x.permute(0,1,3,2).reshape(B*self.m*self.n,self.resize_embed_dim//self.ratio,rx_win_size[0], rx_win_size[1])
        n_x = window_reverse(windows=n_x,m=self.m,n=self.n,paged_shape=rx_shape)

        patches = self.patch_embed(n_x)

        x = self.proj(patches)
        
        return x, attn_map, [H,W], x_shape