import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple, Type
from .fuse_block import FusePre, FusePro, ModelMetaInfo
class FuseArch(nn.Module):
    def __init__(self,
                 model_meta_list: [ModelMetaInfo] = [],
                 img_size: int = 1024,
                 patch_size: int = 16,
                 embed_dim: int = 768,
                 depth: int = 12,
                 num_heads: int = 12,
                 mlp_ratio: float = 4.0,
                 out_chans: int = 256,
                 qkv_bias: bool = True,
                 norm_layer: Type[nn.Module] = nn.LayerNorm,
                 act_layer: Type[nn.Module] = nn.GELU,
                 use_abs_pos: bool = True,
                 use_rel_pos: bool = False,
                 rel_pos_zero_init: bool = True,
                 window_size: int = 0,
                 global_attn_indexes: Tuple[int, ...] = (),
                 ):
        super(FuseArch,self).__init__()
        self.embed_dim = embed_dim
        self.out_chans = out_chans
        self.patch_size = patch_size
        num_model = len(model_meta_list)
        self.fues_pre = FusePre(img_size = img_size,
                           patch_size = patch_size,
                           meta_list=model_meta_list,
                           out_chans = embed_dim,
                           )
        self.fuse_pro = FusePro(img_size = img_size,
                            patch_size = patch_size,
                            in_chans = embed_dim,
                            embed_dim= embed_dim,
                            depth= depth,
                            num_heads = num_heads,
                            mlp_ratio = mlp_ratio,
                            out_chans = out_chans,
                            qkv_bias = qkv_bias,
                            norm_layer = norm_layer,
                            act_layer = act_layer,
                            use_abs_pos = use_abs_pos,
                            use_rel_pos = use_rel_pos,
                            rel_pos_zero_init = rel_pos_zero_init,
                            window_size = window_size,
                            global_attn_indexes = global_attn_indexes,
                            num_model = num_model,)
    def forward(self, xs):
        xs = self.fues_pre(xs)
        x = self.fues_pro(xs)
        return x

