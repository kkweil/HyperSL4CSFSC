import torch
from torch import nn
from timm.layers import Mlp, DropPath, use_fused_attn
from .model import SpectralSharedEncoder
from torch.autograd import Function
from typing import Any, Callable, Dict, Optional, Set, Tuple, Type, Union, List
import torch.nn.functional as F
from torch.jit import Final

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_bias: bool = True,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
                attn_mask=mask
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerScale(nn.Module):
    def __init__(
            self,
            dim: int,
            init_values: float = 1e-5,
            inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_bias: bool = True,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: Type[nn.Module] = nn.GELU,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            mlp_layer: Type[nn.Module] = Mlp,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            bias=proj_bias,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), mask)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x



import torch
import torch.nn as nn
import torch.nn.functional as F


class Adapter(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        
        self.adapter = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        
    def forward(self, x):
        h = self.adapter(x)
        return h  


class SpectralChannelGatingAdapter(nn.Module):
    """
    Spectral-Channel Adapter for token sequence shaped (B*N, C+1, D),
    where token dimension corresponds to spectral channels (+ CLS).

    It learns a per-channel (per token) scalar gate to rescale each spectral token.
    Returns a residual delta with the same shape as input, so you can do:
        x = x + adapter(x)
    """

    def __init__(
        self,
        d_model: int,
        gate_scale: float = 0.1,
        use_tanh: bool = True,
        keep_cls: bool = False,
    ):
        super().__init__()
        self.keep_cls = keep_cls
        self.use_tanh = use_tanh
        self.gate_scale = gate_scale

        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, 1)
        self.adapter = Adapter(d_model)

        # Initialize gate to be near zero so it starts as ~identity
        nn.init.zeros_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (BN, C+1, D)
        return delta: (BN, C+1, D)
        """
        assert x.dim() == 3, "Expected x to be (BN, C+1, D)"
        BN, L, D = x.shape
        assert L >= 2, "Need at least CLS + 1 spectral token"

        # Separate CLS and spectral tokens
        if self.keep_cls:
            cls = x[:, :1, :]          # (BN, 1, D)
            spec = x[:, 1:, :]         # (BN, C, D)
        else:
            spec = x

        # Token-wise gating: one scalar gate per spectral token
        h = self.norm(spec)                               # (BN, C, D)
        g = self.fc1(h)                # (BN, C, 1)

        # Make gate signed and small (so delta can increase or decrease)
        # if self.use_tanh:
        #     g = torch.tanh(g)                              # (-1, 1)
        # else:
        #     g = 2.0 * torch.sigmoid(g) - 1.0              # (-1, 1)

        g = g * self.gate_scale                           # scale adaptation strength

        # Residual delta: elementwise rescaling of each spectral token
        delta_spec = self.adapter(spec)                              # (BN, C, D)
        delta_spec = delta_spec * g + delta_spec

        # Assemble delta with CLS unchanged (delta=0 for CLS)
        if self.keep_cls:
            delta = torch.zeros_like(x)
            delta[:, 1:, :] = delta_spec
            return delta
        else:
            return delta_spec




class Classifier(nn.Module):
    def __init__(self, embedding_dim=256, class_num=9):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(embedding_dim),
            # nn.Linear(embedding_dim, class_num),
            nn.Linear(embedding_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            # nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            # nn.Dropout(0.3),
            nn.Linear(256, class_num),
            # nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)




class CosineClassifier(nn.Module):
    def __init__(self, embedding_dim=256, class_num=9, scale=10.0):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(class_num, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.scale = scale  # 温度/缩放因子

    def forward(self, x):
        # x: [B, embedding_dim]
        x_norm = F.normalize(x, dim=-1)                # 特征归一化
        w_norm = F.normalize(self.weight, dim=-1)      # 权重归一化
        logits = self.scale * x_norm @ w_norm.t()      # [B, C]
        return logits

class FeatureDiscriminator(nn.Module):
    def __init__(self, embedding_dim=256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Linear(128, 2),
        )

    def forward(self, feat):
        logits = self.net(feat)
        return logits





class SuperpixelPosEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, centers):  # centers: [N, 2] or [B, N, 2]
        return self.mlp(centers)
    

import math

class LearnableGatingGlobalAggregation(nn.Module):
    """
    LGA module (as in the diagram):
      - token-wise similarity via W^G and W^S
      - reweight tokens (×)
      - GAP over tokens
      - residual add to global token (+)
      - LayerNorm + Linear (Normal + Linear)

    Input:
        x: (B, N+1, D), where x[:,0] is global token, x[:,1:] are region/patch tokens
    Output:
        x_out: (B, N+1, D), with updated global token at position 0
        attn:  (B, N) attention weights over non-global tokens
    """

    def __init__(self, dim: int, out_dim: int = None, temperature: float = None, dropout: float = 0.0):
        super().__init__()
        out_dim = out_dim if out_dim is not None else dim

        # W^G and W^S in the diagram
        self.Wg = nn.Linear(dim, dim, bias=False)
        self.Ws = nn.Linear(dim, dim, bias=False)

        # Normal + Linear in the diagram
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, out_dim, bias=True)

        self.dropout = nn.Dropout(dropout)

        # optional temperature for softmax; default: sqrt(D)
        self.temperature = temperature

    def forward(self, x: torch.Tensor):
        assert x.dim() == 3, f"Expected (B, N+1, D), got {tuple(x.shape)}"
        B, T, D = x.shape
        assert T >= 2, "Need at least 1 global token + 1 non-global token."

        g = x[:, 0, :]          # (B, D)
        s = x[:, 1:, :]         # (B, N, D)
        N = s.size(1)

        # --- Blue box: compute learnable token->global weights ---
        q = self.Wg(g)          # (B, D)
        k = self.Ws(s)          # (B, N, D)

        # dot-product similarity (B, N)
        scale = self.temperature if self.temperature is not None else math.sqrt(D)
        logits = (k * q.unsqueeze(1)).sum(dim=-1) / (scale + 1e-6)
        attn = F.softmax(logits, dim=1)  # (B, N)
        attn = self.dropout(attn)

        # --- × : reweight tokens ---
        weighted_s = s * attn.unsqueeze(-1)  # (B, N, D)

        # --- GAP : average pool over tokens ---
        pooled = weighted_s.mean(dim=1)      # (B, D)

        # --- + : residual add to global token ---
                 
        g_upd = self.proj(self.norm(g + pooled))  # (B, out_dim)
    

        # write back updated global token
      
        if g_upd.shape[-1] != D:
            # if out_dim != dim, expand the sequence to out_dim for consistency
            # simplest: project all tokens to out_dim (optional design choice)
            # Here we keep other tokens in original dim; you can change if needed.
            raise ValueError("out_dim != dim: for sequence output, please keep out_dim == dim "
                             "or add a token projection for non-global tokens.")


        return g_upd



    
class AdapterEncoder(nn.Module):
    def __init__(self,
                 backbone,
                 class_num=9,
                 unfrez_block_num=1,
                 frez_patch_embeding = False,
                 frez_normal = False,
                 frez_global_token = False,):
        super().__init__()
        self.embedding_dim = backbone.embedding_dim
        self.max_band = backbone.max_band
        self.global_token = backbone.global_token
        
        self.cls = nn.Parameter(torch.zeros(1, 1, self.embedding_dim))
        torch.nn.init.normal_(self.cls, std=.02)
        # 冻结主干 
        if frez_normal:
            for p in backbone.parameters():
                p.requires_grad = False
            print('🧊 Whole backbone have been frozen.')
        else:
            for name, param in backbone.named_parameters():
                if "norm" in name:      
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            print('🔥 Normal layers have been unfrezed, 🧊 other layers have been frozen.')
        
            
        if frez_patch_embeding:
            print('🧊 Embedding layers have been frozen.')
        else:
            for p in backbone.embedding_layer_1.parameters():
                p.requires_grad = True
            for p in backbone.embedding_layer_2.parameters():
                p.requires_grad = True
            for p in backbone.embedding_layer_3.parameters():
                p.requires_grad = True
            for p in backbone.embedding_layer_4.parameters():
                p.requires_grad = True
            print('🔥 Embedding layers have been unfrozened.')
            
        
        if unfrez_block_num > 0:    
            total = len(backbone.encoder_blocks)
            start = total - unfrez_block_num
            
            for i in range(start, total):
                for p in backbone.encoder_blocks[i].parameters():
                    p.requires_grad = True
                    
            print(f"Total blocks: {total}, 🔥 unfreezing blocks[{start}:{total}]")
        
        if frez_global_token:
            self.global_token.requires_grad = False
            print('🧊 Global token have been frozen.')
        else:
            self.global_token.requires_grad = True
            print('🔥 Global token have been unfrozen.')
            
            
        
        self.embedding_layer_1 = backbone.embedding_layer_1

        self.embedding_layer_2 = backbone.embedding_layer_2

        self.embedding_layer_3 = backbone.embedding_layer_3

        self.embedding_layer_4 = backbone.embedding_layer_4
        
        self.div_term = backbone.div_term

        self.pe = backbone.pe
        
        # self.encoder_blocks = backbone.encoder_blocks
        self.encoder_blocks = nn.ModuleList([
            Block(self.embedding_dim, 8, 4., qkv_bias=True, norm_layer=nn.LayerNorm)
            for i in range(len(backbone.encoder_blocks))])
        
        self._load_pretrained_blocks(backbone.encoder_blocks)
        
        self.embed_adapter = SpectralChannelGatingAdapter(self.embedding_dim, gate_scale=0.3)
        # self.embed_adapters = Adapter(self.embedding_dim)
        
        self.superpixelposencoding = SuperpixelPosEncoding(self.embedding_dim)

        self.adapters = nn.ModuleList([
           SpectralChannelGatingAdapter(self.embedding_dim, gate_scale=0.3)
            for i in range(len(self.encoder_blocks))])
        
        # self.spec_adapter = SpectralChannelGatingAdapter(self.embedding_dim, gate_scale=0.3)

        # self.proj1 = nn.Linear(self.embedding_dim, self.embedding_dim)

        # self.proj2 = nn.Linear(self.embedding_dim, self.embedding_dim)

        self.lgga = LearnableGatingGlobalAggregation(self.embedding_dim, self.embedding_dim)
        # self.norm = nn.LayerNorm(self.embedding_dim)
        
        self.blocks = nn.ModuleList([
            Block(self.embedding_dim, 8, 2., qkv_bias=True, norm_layer=nn.LayerNorm)
            for i in range(3)])
        
        self.class_classifier = Classifier(self.embedding_dim, class_num)
        

    def embedding(self, x):
        x1 = self.embedding_layer_1(x)
        x2 = self.embedding_layer_2(x)
        x3 = self.embedding_layer_3(x)
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.embedding_layer_4(x)
        return x



    def encoder_forward(self, x, wave):
        B,N,C = x.shape  # (B,N,C)
        x = self.embedding(x.reshape(-1, 1, C))
        # (B,Wave) -> (B,1,Wave) -> (B,H*W,Wave) -> (B,H*W,Wave,1) -> (B*H*W,Wave,1)
        wave = ((wave - 400) / 10).unsqueeze(1).repeat(1, N, 1).unsqueeze(-1).view(-1, C, 1)
        x = x.transpose(2, 1)
        
        x = x + self.embed_adapter(x)
        
        
        # 计算中心波段的pos embedding tokens
        pos_tokens = self.pe.repeat(x.shape[0], 1, 1)
        pos_tokens[:, :x.shape[1], 0::2] = torch.sin(self.div_term * wave)
        pos_tokens[:, :x.shape[1], 1::2] = torch.cos(self.div_term * wave)
        x = x + pos_tokens[:, :x.shape[1]]

        # 添加全局变量
        global_tokens = self.global_token.expand(x.shape[0], -1, -1)
        x = torch.cat((global_tokens, x), dim=1)
        # Encoder层
        for idx, block in enumerate(self.encoder_blocks):
            x = block(x)
            x = x + self.adapters[idx](x)
        # x = x + self.spec_adapter(x)
        
        features = self.lgga(x).reshape(B,N,-1)
        
        # features = x[:,0].reshape(B,N,-1)
        return features


    def forward(self, x, wave, pos, mask):
        features = self.encoder_forward(x, wave)
        # features = self.proj1(features) + features
        # features = self.lgga(features)
        
        
        
        pos_encodings = self.superpixelposencoding(pos)
        
        
        
        features = features + pos_encodings
        
        cls_tokens = self.cls.expand(features.shape[0], -1, -1)
        features = torch.cat((cls_tokens, features), dim=1)
        
        cls_mask = torch.ones((features.shape[0], 1), dtype=torch.bool, device=mask.device)
        mask = torch.cat([cls_mask, mask], dim=1)
        mask = mask.unsqueeze(1).unsqueeze(1)
        attn_mask = torch.zeros_like(mask, dtype=x.dtype)

            # 2. 把 "无效(False)" 的地方填成 -inf
        attn_mask.masked_fill_(~mask, float("-inf"))
        
        
        for idx, block in enumerate(self.blocks):
            features = block(features, attn_mask)
            
        # feats = features[:,0]  # + self.proj2(features[:,0])# + features[:,1:].mean(dim=1)
        feats = features[:,1:].mean(dim=1)
        
        class_output = self.class_classifier(feats)

        return class_output, feats
    
    
    def _load_pretrained_blocks(self, old_blocks):
        
        for new_blk, old_blk in zip(self.encoder_blocks, old_blocks):
            # ---- 1. 先迁移权重 ----
            old_state = old_blk.state_dict()
            new_state = new_blk.state_dict()

            filtered = {
                k: v for k, v in old_state.items()
                if k in new_state and new_state[k].shape == v.shape
            }
            new_blk.load_state_dict(filtered, strict=True)

            # ---- 2. 再迁移 requires_grad 状态 ----
            old_param_dict = dict(old_blk.named_parameters())
            for name, new_p in new_blk.named_parameters():
                if name in old_param_dict:
                    new_p.requires_grad = old_param_dict[name].requires_grad
                # else: 新增参数（比如你自己加的东西），保持默认 requires_grad=True 或你自己再改

        print("✔ 已继承预训练 Block 的权重和冻结状态(requires_grad)。")
        
        
        
class DAEncoder(nn.Module):
    def __init__(self,
                 backbone,
                 class_num=9,
                 unfrez_block_num=1,
                 frez_patch_embeding = False,
                 frez_normal = False,
                 frez_global_token = False,):
        super().__init__()
        self.embedding_dim = backbone.embedding_dim
        self.max_band = backbone.max_band
        self.global_token = backbone.global_token
        
        self.cls = nn.Parameter(torch.zeros(1, 1, self.embedding_dim))
        torch.nn.init.normal_(self.cls, std=.02)
        # 冻结主干 
        if frez_normal:
            for p in backbone.parameters():
                p.requires_grad = False
            print('🧊 Whole backbone have been frozen.')
        else:
            for name, param in backbone.named_parameters():
                if "norm" in name:      
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            print('🔥 Normal layers have been unfrezed, 🧊 other layers have been frozen.')
        
            
        if frez_patch_embeding:
            print('🧊 Embedding layers have been frozen.')
        else:
            for p in backbone.embedding_layer_1.parameters():
                p.requires_grad = True
            for p in backbone.embedding_layer_2.parameters():
                p.requires_grad = True
            for p in backbone.embedding_layer_3.parameters():
                p.requires_grad = True
            for p in backbone.embedding_layer_4.parameters():
                p.requires_grad = True
            print('🔥 Embedding layers have been unfrozened.')
            
        
        if unfrez_block_num > 0:    
            total = len(backbone.encoder_blocks)
            start = total - unfrez_block_num
            
            for i in range(start, total):
                for p in backbone.encoder_blocks[i].parameters():
                    p.requires_grad = True
                    
            print(f"Total blocks: {total}, 🔥 unfreezing blocks[{start}:{total}]")
        
        if frez_global_token:
            self.global_token.requires_grad = False
            print('🧊 Global token have been frozen.')
        else:
            self.global_token.requires_grad = True
            print('🔥 Global token have been unfrozen.')
            
            
        
        self.embedding_layer_1 = backbone.embedding_layer_1

        self.embedding_layer_2 = backbone.embedding_layer_2

        self.embedding_layer_3 = backbone.embedding_layer_3

        self.embedding_layer_4 = backbone.embedding_layer_4
        
        self.div_term = backbone.div_term

        self.pe = backbone.pe
        
        # self.encoder_blocks = backbone.encoder_blocks
        self.encoder_blocks = nn.ModuleList([
            Block(self.embedding_dim, 8, 4., qkv_bias=True, norm_layer=nn.LayerNorm)
            for i in range(len(backbone.encoder_blocks))])
        
        self._load_pretrained_blocks(backbone.encoder_blocks)
        
        
        self.embed_adapters = Adapter(self.embedding_dim)
        
        self.superpixelposencoding = SuperpixelPosEncoding(self.embedding_dim)

        self.adapters = nn.ModuleList([
           Adapter(self.embedding_dim)
            for i in range(len(self.encoder_blocks))])
        
        self.proj1 = nn.Linear(self.embedding_dim, self.embedding_dim)
        # self.proj2 = nn.Linear(self.embedding_dim, self.embedding_dim)
        
        self.blocks = nn.ModuleList([
            Block(self.embedding_dim, 8, 2., qkv_bias=True, norm_layer=nn.LayerNorm)
            for i in range(3)])
        
        self.class_classifier = Classifier(self.embedding_dim, class_num)
        
        self.domain_classifier = FeatureDiscriminator(embedding_dim=self.embedding_dim)
        
        self.grad_revers = ReverseLayerF()

    def embedding(self, x):
        x1 = self.embedding_layer_1(x)
        x2 = self.embedding_layer_2(x)
        x3 = self.embedding_layer_3(x)
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.embedding_layer_4(x)
        return x



    def encoder_forward(self, x, wave):
        B,N,C = x.shape  # (B,N,C)
        x = self.embedding(x.reshape(-1, 1, C))
        # (B,Wave) -> (B,1,Wave) -> (B,H*W,Wave) -> (B,H*W,Wave,1) -> (B*H*W,Wave,1)
        wave = ((wave - 400) / 10).unsqueeze(1).repeat(1, N, 1).unsqueeze(-1).view(-1, C, 1)
        x = x.transpose(2, 1)
        
        x = x + self.embed_adapters(x)
        
        
        # 计算中心波段的pos embedding tokens
        pos_tokens = self.pe.repeat(x.shape[0], 1, 1)
        pos_tokens[:, :x.shape[1], 0::2] = torch.sin(self.div_term * wave)
        pos_tokens[:, :x.shape[1], 1::2] = torch.cos(self.div_term * wave)
        x = x + pos_tokens[:, :x.shape[1]]

        # 添加全局变量
        global_tokens = self.global_token.expand(x.shape[0], -1, -1)
        x = torch.cat((global_tokens, x), dim=1)
        # Encoder层
        for idx, block in enumerate(self.encoder_blocks):
            x = block(x)
            x = x + self.adapters[idx](x)
        features = x[:,0].reshape(B,N,-1)
        return features


    def forward(self, x, wave, pos, mask, alpha):
        features = self.encoder_forward(x, wave)
        # features = self.proj1(features) + features
        
        
        
        
        pos_encodings = self.superpixelposencoding(pos)
        
        
        
        features = features + pos_encodings
        
        cls_tokens = self.cls.expand(features.shape[0], -1, -1)
        features = torch.cat((cls_tokens, features), dim=1)
        
        cls_mask = torch.ones((features.shape[0], 1), dtype=torch.bool, device=mask.device)
        mask = torch.cat([cls_mask, mask], dim=1)
        mask = mask.unsqueeze(1).unsqueeze(1)
        attn_mask = torch.zeros_like(mask, dtype=x.dtype)

            # 2. 把 "无效(False)" 的地方填成 -inf
        attn_mask.masked_fill_(~mask, float("-inf"))
        
        
        for idx, block in enumerate(self.blocks):
            features = block(features, attn_mask)
            
        feats = features[:,0]  # + self.proj2(features[:,0])# + features[:,1:].mean(dim=1)
        
        class_output = self.class_classifier(feats)
        
        revers_feats = self.grad_revers.apply(feats, alpha)
        
        domain_output = self.domain_classifier(revers_feats)

        return class_output, domain_output, feats
    
    
    def _load_pretrained_blocks(self, old_blocks):
        
        for new_blk, old_blk in zip(self.encoder_blocks, old_blocks):
            # ---- 1. 先迁移权重 ----
            old_state = old_blk.state_dict()
            new_state = new_blk.state_dict()

            filtered = {
                k: v for k, v in old_state.items()
                if k in new_state and new_state[k].shape == v.shape
            }
            new_blk.load_state_dict(filtered, strict=True)

            # ---- 2. 再迁移 requires_grad 状态 ----
            old_param_dict = dict(old_blk.named_parameters())
            for name, new_p in new_blk.named_parameters():
                if name in old_param_dict:
                    new_p.requires_grad = old_param_dict[name].requires_grad
                # else: 新增参数（比如你自己加的东西），保持默认 requires_grad=True 或你自己再改

        print("✔ 已继承预训练 Block 的权重和冻结状态(requires_grad)。")

def print_trainable_parameters(model):
    total = 0
    trainable = 0

    # print("\n=== Parameter Grad Status ===")
    for name, param in model.named_parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
            status = "TRAIN ✔"
        else:
            status = "FROZEN ❄"
        
        # print(f"{name:50s} {str(param.shape):20s} {status}")

    print("\n=== Summary ===")
    print(f"Total parameters:     {total}")
    print(f"Trainable parameters: {trainable}")
    print(f"Frozen parameters:    {total - trainable}")
    print("============================\n")


if __name__ == '__main__':
    batch_size = 4
    
    source_data = torch.randn(batch_size, 50, 224).cuda()
    target_data = torch.randn(batch_size, 50, 235).cuda()

    source_pos = torch.randn(batch_size, 50, 2).cuda()
    target_pos = torch.randn(batch_size, 50, 2).cuda()
    
    source_mask = torch.zeros(batch_size, 50, dtype=bool).cuda()
    target_mask = torch.zeros(batch_size, 50, dtype=bool).cuda()
    
    source_mask[:, :30] = 1 
    target_mask[:, :30] = 1 
    
    source_wavelength = torch.randn(batch_size, 224).cuda()
    target_wavelength = torch.randn(batch_size, 235).cuda()
    
    foundationmodel = SpectralSharedEncoder(
                        embedding_dim=256,
                        encoder_depth=8,
                        decoder_depth=4,
                        num_heads=8,)

    model = AdapterEncoder(backbone=foundationmodel,
                           class_num=9,
                           frez_patch_embeding=True)
    
    print_trainable_parameters(model)
    model.cuda()
    for i in range(10000):
        
        class_output_s, domain_output_s = model(source_data, source_wavelength, source_pos, source_mask, alpha=1.)
        # _, domain_output_t = model(target_data, target_wavelength, target_pos, alpha=1.)


    a = 0
