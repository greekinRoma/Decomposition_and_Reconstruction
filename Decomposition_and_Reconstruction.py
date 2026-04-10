from torch import nn
from Decomposition_model import DecompositionModel
from Reconstruction_model import ReconstructionModel
class DeRemodel(nn.Module):
    def __init__(self, num_head=16, m=1, n=1, origin_patch_size=4, resize_patch_size=16, origin_embed_dim = 48, resize_embed_dim =48, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decomposition_model = DecompositionModel(num_head=num_head, m=m, n=n, origin_patch_size=origin_patch_size, resize_patch_size=resize_patch_size, origin_embed_dim=origin_embed_dim, resize_embed_dim=resize_embed_dim)
        self.reconstruction_model = ReconstructionModel(patch_size=origin_patch_size, num_head=num_head, m=m, n=n, embed_dim=resize_embed_dim)
    def forward(self, x):
        patch, attn, pad_size, orgin_size = self.decomposition_model(x)
        re_img = self.reconstruction_model(x=patch, attn=attn, pad_size=pad_size, orgin_size=orgin_size)
        return re_img
    def encode(self, x):
        patch, attn, pad_size, orgin_size = self.decomposition_model(x)
        return patch, attn, pad_size, orgin_size
    def decode(self, patch, attn, pad_size, orgin_size):
        re_img = self.reconstruction_model(x=patch, attn=attn, pad_size=pad_size, orgin_size=orgin_size)
        return re_img