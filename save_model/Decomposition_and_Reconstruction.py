from torch import nn
from Decomposition_model import DecompositionModel
from Reconstruction_model import ReconstructionModel
class DeRemodel(nn.Module):
    def __init__(self, patch_size=16, num_head=16, embed_dim=768,m=1, n=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decomposition_model = DecompositionModel(patch_size=patch_size, num_head=num_head, embed_dim=embed_dim,m=m, n=n)
        self.reconstruction_model = ReconstructionModel(patch_size=patch_size, num_head=num_head, embed_dim=embed_dim,m=m, n=n)
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