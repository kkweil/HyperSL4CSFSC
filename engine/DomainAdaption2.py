import torch
from torch import nn
import torch.nn.functional as F

from model import SpectralSharedEncoder, Attention, Corss_Attention

from torch.autograd import Function
from timm.models.vision_transformer import Block, Mlp
from timm.models.layers import DropPath


class Encoder(nn.Module):
    def __init__(self, model_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if model_size == 'small':
            self.embedding_dim = 256
            self.spectral_encoder = SpectralSharedEncoder(
                embedding_dim=256,
                encoder_depth=8,
                decoder_depth=4,
                num_heads=8,
            )
        if model_size == 'base':
            self.embedding_dim = 512
            self.spectral_encoder = SpectralSharedEncoder(
                embedding_dim=512,
                max_band=500,
                encoder_depth=24,
                decoder_depth=12,
                num_heads=16,
                mlp_ratio=4.,
                norm_layer=nn.LayerNorm
            )
        if model_size == 'large':
            self.embedding_dim = 1024
            self.spectral_encoder = SpectralSharedEncoder(
                embedding_dim=1024,
                max_band=500,
                encoder_depth=32,
                decoder_depth=16,
                num_heads=32,
                mlp_ratio=4.,
                norm_layer=nn.LayerNorm
            )

        if model_size == 'huge':
            self.embedding_dim = 2048
            self.spectral_encoder = SpectralSharedEncoder(
                embedding_dim=2048,
                max_band=500,
                encoder_depth=48,
                decoder_depth=24,
                num_heads=32,
                mlp_ratio=4.,
                norm_layer=nn.LayerNorm
            )
        self.feature_extract = nn.ModuleList([TransformerEncoderLayer(dim=256,num_heads=8) for i in range(4)])
        # self.convblock = nn.Sequential(
        #     nn.Conv2d(self.embedding_dim, self.embedding_dim, 1, 1),
        #     nn.BatchNorm2d(self.embedding_dim),
        #     nn.GELU(),
        #     nn.Conv2d(self.embedding_dim, self.embedding_dim, 3, 2, 1),  # h,w /2
        #     nn.BatchNorm2d(self.embedding_dim),
        #     nn.GELU(),
        #     nn.Conv2d(self.embedding_dim, self.embedding_dim, 3, 2, 1),  # h,w /2
        #     nn.Conv2d(self.embedding_dim, self.embedding_dim, 1, 1),
        #     nn.BatchNorm2d(self.embedding_dim),
        #     nn.GELU(),

        # )

    def forward(self, x, wavelength, mask_ratio):

        x, _, _, _, _, _, shape = self.spectral_encoder.encoder_forward(x, wavelength, mask_ratio)

        B, W, H, C = shape

        x = x.view(B, W*H, -1)
        feature = self.feature_extract(x)
        feature = feature[:,feature.size(1)//2]

        return feature


class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x, _, _,_ = self.attn(self.norm1(x))
        x = x + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


if __name__ == '__main__':
    batch_size = 4
    source_data = torch.randn(batch_size, 5, 5, 224).cuda()
    target_data = torch.randn(batch_size, 5, 5, 224).cuda()

    source_wavelength = torch.randn(batch_size, 224).cuda()
    target_wavelength = torch.randn(batch_size, 224).cuda()

    model = Encoder(model_size='small').cuda()
    tr = TransformerEncoderLayer(dim=256,num_heads=8).cuda()
   
    for i in range(10000):
        s_feature = model(source_data, source_wavelength, mask_ratio=0.3)
        t1_feature = model(target_data, target_wavelength, mask_ratio=0.3)
        t2_feature = model(target_data, target_wavelength, mask_ratio=0.3)
        # features = torch.cat([s_feature,t1_feature,t2_feature], dim=1)
        # features = tr(s_feature)
        # x = features[:,features.size(1)//2,:]



    a = 0
