import torch
from torch import nn
from torch.nn.functional import pad
from partition import window_partition
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
        self.ratio = 32
        self.num_heads = num_head
        self.original_patch_embed = nn.Conv2d(in_channels=3,out_channels=self.origin_embed_dim,kernel_size=self.origin_patch_size,stride=self.origin_patch_size,groups=3).cuda()
        self.resize_patch_embed = nn.Conv2d(in_channels=self.resize_embed_dim//self.ratio,out_channels=self.resize_embed_dim,kernel_size=self.resize_patch_size,stride=self.resize_patch_size,groups=self.num_heads//2).cuda()
        self.b = nn.Linear(self.origin_embed_dim//self.ratio, self.origin_embed_dim//self.ratio)
        self.q  = nn.Linear(self.resize_embed_dim//self.ratio, self.resize_embed_dim//self.ratio)
        self.resize = 1024 // self.resize_patch_size
        self.pos = nn.Parameter(torch.rand(1, 1024, 1024, self.resize_embed_dim//self.ratio),requires_grad=True)
        self.norm_b = nn.LayerNorm(self.origin_embed_dim//self.ratio)
        self.norm_q = nn.LayerNorm(self.resize_embed_dim//self.ratio)
        self.q_attn = nn.Sequential(*[
            nn.Linear(self.resize_embed_dim//self.num_heads,1),
            nn.Sigmoid()
        ])
        self.down = nn.Sequential(
            nn.Linear(self.resize_embed_dim//self.num_heads, self.origin_embed_dim//self.num_heads)
        )
        self.proj = nn.Conv2d(self.resize_embed_dim,self.resize_embed_dim,kernel_size=1,stride=1)
        print(self.origin_embed_dim//self.ratio)
        self.mat_mul = MultiplyMatrixWithRoPE(num_heads=1,head_dim=self.origin_embed_dim//self.ratio, rope_theta=10.0)
        self.x_win_conv = nn.Conv2d(3,self.origin_embed_dim//self.ratio,kernel_size=3,padding=1)
        self.r_win_conv = nn.Conv2d(3,self.origin_embed_dim//self.ratio,kernel_size=3,padding=1)

    def forward(self, x):
        B, C, H, W = x.shape
        
        # 1. Padding the raw image first to ensure it's divisible by (m * origin_patch_size)
        # This is necessary if you want the windows to contain whole patches.
        H_pad = (self.m - H % (self.m)) % (self.m)
        W_pad = (self.n - W % (self.n)) % (self.n)
        x_pad = pad(x, (0, W_pad, 0, H_pad), mode='constant', value=0)
        H_n, W_n = H+H_pad, W+W_pad
        
        # 2. Interpolate for the resize branch before partitioning
        resize_x = torch.nn.functional.interpolate(x, size=(1024, 1024), mode='bilinear', align_corners=False)

        # ==========================================
        # 3. WINDOW PARTITION (Before Embedding)
        # We treat windows as a batch dimension: [B, C, H, W] -> [B*m*n, C, H/m, W/n]
        # ==========================================
        # Original branch partition
        x_pad = self.x_win_conv(x_pad)
        resize_x = self.r_win_conv(resize_x)
        # x_win, [win_h, win_w] = window_partition(x_pad, m=self.m, n=self.n)
        
        # Resize branch partition
        # rx_win, [rwin_h, rwin_w] = window_partition(resize_x, m=self.m, n=self.n)

        x_win = x_pad.reshape(B,self.origin_embed_dim//self.ratio, H_n, W_n).permute(0,2,3,1)
        rx_win = resize_x.reshape(B,self.resize_embed_dim//self.ratio, 1024, 1024).permute(0,2,3,1)


        q = self.norm_q(rx_win)
        b = self.norm_b(x_win)

        b = self.b(b)
        q = self.q(q)
        tmp_b = b.clone()

        b = torch.nn.functional.normalize(b,dim=-1)
        q = torch.nn.functional.normalize(q+self.pos,dim=-1)

        q, b, delta = self.mat_mul.add_rope(q=q, b=b, end_x_xq=1024, end_y_xq=1024, end_x_xb=H_pad+H, end_y_xb=W_pad+W,m=self.m,n=self.n)
        q = window_partition(q, m=self.m, n=self.n,num_heads=self.num_heads)
        b = window_partition(b, m=self.m, n=self.n,num_heads=self.num_heads)
        tmp_b = window_partition(tmp_b, m=self.m, n=self.n,num_heads=self.num_heads)
        # print(q.shape)
        # print(b.shape)
        attn_map = q @ b.transpose(-2, -1)*delta

        n_x = attn_map @ tmp_b

        n_x = n_x.permute(0,1,3,2).reshape(B,self.m,self.n,self.num_heads,self.resize_embed_dim//self.ratio//self.num_heads,1024//self.m,1024//self.n).permute(0,3,4,1,5,2,6).reshape(B,self.resize_embed_dim//self.ratio,1024,1024)
        patches = self.resize_patch_embed(n_x)

        x = self.proj(patches)
        
        return x, attn_map, [H_pad+H,W_pad+W], [H, W]