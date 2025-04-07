import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple, Type

from .common import LayerNorm2d, MLPBlock
from .image_encoder import window_partition, window_unpartition, add_decomposed_rel_pos


# def atten_vis(attn, token_id=0, apedix = '',vis=False):
#     out_dir = './data/vis_temp'
#     import numpy as np
#     import os.path as osp
#     att_map = attn[0, :, :, token_id].cpu().numpy()
#     norm_map = 255 * (att_map - att_map.min()) / (att_map.max() - att_map.min())
#     norm_map = np.uint8(norm_map)
#     import PIL
#     import PIL.Image as Im
#     att_im = Im.fromarray(norm_map)
#     if vis:
#         att_im.save(osp.join(out_dir, f'{apedix}_token_id{token_id}.jpg'))
#     return att_im

# def attns_vis(attns,apendix = ''):
#     for idx_model in range(len(attns)):
#         for idx_token in range(attns[0].shape[-1]):
#             atten_vis(attns[idx_model],idx_token,apedix=f'{apendix}_model{idx_model}',vis=True)
#     return


class CrossBlock_tokenkv(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
        require_atten: bool = False,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then
                use global attention.
            input_size (int or None): Input resolution for calculating the relative positional
                parameter size.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CrossAttention_tokenkv(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            require_atten = require_atten,
        )
        self.require_atten = require_atten
        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)

        self.window_size = window_size

    def forward(self, x: torch.Tensor, token: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)  # [B * num_windows, window_size, window_size, C]

        if not self.require_atten:
            x = self.attn(x, token)
        else:
            x, atten = self.attn(x, token)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        if not self.require_atten:
            return x
        else:
            return x, atten

class CrossBlock(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
        require_atten: bool = False,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then
                use global attention.
            input_size (int or None): Input resolution for calculating the relative positional
                parameter size.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CrossAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            require_atten = require_atten
        )
        self.require_atten = require_atten
        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)

        self.window_size = window_size

    def forward(self, xq: torch.Tensor, xkv: torch.Tensor) -> torch.Tensor:
        shortcut = xq
        xq = self.norm1(xq)
        # Window partition
        if self.window_size > 0:
            H, W = xq.shape[1], xq.shape[2]
            xq, pad_hw = window_partition(xq, self.window_size)  # [B * num_windows, window_size, window_size, C]

        if not self.require_atten:
            xq = self.attn(xkv=xkv, xq=xq)
        else:
            xq, atten = self.attn(xkv=xkv, xq=xq)
        # Reverse window partition
        if self.window_size > 0:
            xq = window_unpartition(xq, self.window_size, pad_hw, (H, W))

        xq = shortcut + xq
        xq = xq + self.mlp(self.norm2(xq))

        if not self.require_atten:
            return xq
        else:
            return xq, atten


# class CrossAttenBlock_2model(nn.Module):
#     """Transformer blocks with support of window attention and residual propagation blocks"""
#
#     def __init__(
#         self,
#         dim: int,
#         num_heads: int,
#         mlp_ratio: float = 4.0,
#         qkv_bias: bool = True,
#         norm_layer: Type[nn.Module] = nn.LayerNorm,
#         act_layer: Type[nn.Module] = nn.GELU,
#         use_rel_pos: bool = False,
#         rel_pos_zero_init: bool = True,
#         window_size: int = 0,
#         input_size: Optional[Tuple[int, int]] = None,
#         num_model: int = 3,
#     ) -> None:
#         """
#         Args:
#             dim (int): Number of input channels.
#             num_heads (int): Number of attention heads in each ViT block.
#             mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
#             qkv_bias (bool): If True, add a learnable bias to query, key, value.
#             norm_layer (nn.Module): Normalization layer.
#             act_layer (nn.Module): Activation layer.
#             use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
#             rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
#             window_size (int): Window size for window attention blocks. If it equals 0, then
#                 use global attention.
#             input_size (int or None): Input resolution for calculating the relative positional
#                 parameter size.
#         """
#         super().__init__()
#         self.num_model = num_model
#         assert num_model == 2
#         self.cblocks = nn.ModuleList()
#         for i in range(self.num_model):
#             cblock = CrossBlock(
#                 dim=dim,
#                 num_heads=num_heads,
#                 mlp_ratio=mlp_ratio,
#                 qkv_bias=qkv_bias,
#                 norm_layer=norm_layer,
#                 act_layer=act_layer,
#                 use_rel_pos=use_rel_pos,
#                 rel_pos_zero_init=rel_pos_zero_init,
#                 window_size=window_size,
#                 input_size=input_size,
#             )
#             self.cblocks.append(cblock)
#
#         self.window_size = window_size
#
#     def forward(self, xs: [torch.Tensor])\
#             -> [torch.Tensor]:
#         xs_out = []
#         # model 0
#         idx_ori = 0
#         idx_oth = 1
#         x_ori = xs[idx_ori]
#         x_oth = xs[idx_oth]
#         cblock = self.cblocks[idx_ori]
#         x_out = cblock(xkv=x_oth, xq=x_ori)
#         xs_out.append(x_out)
#         # model 1
#         idx_ori = 1
#         idx_oth = 0
#         x_ori = xs[idx_ori]
#         x_oth = xs[idx_oth]
#         cblock = self.cblocks[idx_ori]
#         x_out = cblock(xkv=x_oth, xq=x_ori)
#         xs_out.append(x_out)
#         return xs
#
# class SelfAttenBlock_2model(nn.Module):
#     """Transformer blocks with support of window attention and residual propagation blocks"""
#
#     def __init__(
#         self,
#         dim: int,
#         num_heads: int,
#         mlp_ratio: float = 4.0,
#         qkv_bias: bool = True,
#         norm_layer: Type[nn.Module] = nn.LayerNorm,
#         act_layer: Type[nn.Module] = nn.GELU,
#         use_rel_pos: bool = False,
#         rel_pos_zero_init: bool = True,
#         window_size: int = 0,
#         input_size: Optional[Tuple[int, int]] = None,
#         num_model: int = 3,
#     ) -> None:
#         """
#         Args:
#             dim (int): Number of input channels.
#             num_heads (int): Number of attention heads in each ViT block.
#             mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
#             qkv_bias (bool): If True, add a learnable bias to query, key, value.
#             norm_layer (nn.Module): Normalization layer.
#             act_layer (nn.Module): Activation layer.
#             use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
#             rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
#             window_size (int): Window size for window attention blocks. If it equals 0, then
#                 use global attention.
#             input_size (int or None): Input resolution for calculating the relative positional
#                 parameter size.
#         """
#         super().__init__()
#         self.num_model = num_model
#         assert num_model == 2
#         self.cblocks = nn.ModuleList()
#         for i in range(self.num_model):
#             cblock = CrossBlock(
#                 dim=dim,
#                 num_heads=num_heads,
#                 mlp_ratio=mlp_ratio,
#                 qkv_bias=qkv_bias,
#                 norm_layer=norm_layer,
#                 act_layer=act_layer,
#                 use_rel_pos=use_rel_pos,
#                 rel_pos_zero_init=rel_pos_zero_init,
#                 window_size=window_size,
#                 input_size=input_size,
#             )
#             self.cblocks.append(cblock)
#
#         self.window_size = window_size
#
#     def forward(self, xs: [torch.Tensor])\
#             -> [torch.Tensor]:
#         xs_out = []
#         # model 0
#         idx_ori = 0
#         idx_oth = 0
#         x_ori = xs[idx_ori]
#         x_oth = xs[idx_oth]
#         cblock = self.cblocks[idx_ori]
#         x_out = cblock(xkv=x_oth, xq=x_ori)
#         xs_out.append(x_out)
#         # model 1
#         idx_ori = 1
#         idx_oth = 1
#         x_ori = xs[idx_ori]
#         x_oth = xs[idx_oth]
#         cblock = self.cblocks[idx_ori]
#         x_out = cblock(xkv=x_oth, xq=x_ori)
#         xs_out.append(x_out)
#         return xs
# class CrossAttenBlock_Act_Token(nn.Module):
#     """Transformer blocks with support of window attention and residual propagation blocks"""
#
#     def __init__(
#         self,
#         dim: int,
#         num_heads: int,
#         mlp_ratio: float = 4.0,
#         qkv_bias: bool = True,
#         norm_layer: Type[nn.Module] = nn.LayerNorm,
#         act_layer: Type[nn.Module] = nn.GELU,
#         use_rel_pos: bool = False,
#         rel_pos_zero_init: bool = True,
#         window_size: int = 0,
#         input_size: Optional[Tuple[int, int]] = None,
#         num_model: int = 3,
#         fuse_token_num_base: int = 256*3,
#     ) -> None:
#         """
#         Args:
#             dim (int): Number of input channels.
#             num_heads (int): Number of attention heads in each ViT block.
#             mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
#             qkv_bias (bool): If True, add a learnable bias to query, key, value.
#             norm_layer (nn.Module): Normalization layer.
#             act_layer (nn.Module): Activation layer.
#             use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
#             rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
#             window_size (int): Window size for window attention blocks. If it equals 0, then
#                 use global attention.
#             input_size (int or None): Input resolution for calculating the relative positional
#                 parameter size.
#         """
#         super().__init__()
#         self.num_model = num_model
#         self.fuse_token_num_base = fuse_token_num_base
#         self.cblocks = nn.ModuleList()
#         for i in range(self.num_model):
#             cblock = CrossBlock(
#                 dim=dim,
#                 num_heads=num_heads,
#                 mlp_ratio=mlp_ratio,
#                 qkv_bias=qkv_bias,
#                 norm_layer=norm_layer,
#                 act_layer=act_layer,
#                 use_rel_pos=use_rel_pos,
#                 rel_pos_zero_init=rel_pos_zero_init,
#                 window_size=window_size,
#                 input_size=input_size,
#             )
#             self.cblocks.append(cblock)
#
#         self.window_size = window_size
#
#     def forward(self, xs: [torch.Tensor], token: torch.Tensor)\
#             -> [torch.Tensor]:
#         '''
#         xs: list of [B,H,W,C]
#         token:[1,N,C]
#         '''
#         token_out = []
#         for i in range(self.num_model):
#             s = self.fuse_token_num_base*(i)
#             e = s+self.fuse_token_num_base
#             token_ = token[:,s:e,:]
#             x_ = xs[i]
#             cblock = self.cblocks[i]
#             token_out_ = cblock(xkv=x_, xq=token_) #[1,1,N,C]
#             token_out.append(token_out_[0,...])
#         token_out = torch.cat(token_out,dim=-2)
#
#         return token_out




class FuseProBlock(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
        num_model: int = 3,
        require_atten: bool = False,

    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then
                use global attention.
            input_size (int or None): Input resolution for calculating the relative positional
                parameter size.
        """
        super().__init__()
        self.num_model = num_model
        self.require_atten = require_atten
        self.cblocks = nn.ModuleList()
        for i in range(self.num_model):
            cblock = CrossBlock_tokenkv(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size,
                input_size=input_size,
                require_atten=require_atten,
            )
            self.cblocks.append(cblock)

        self.window_size = window_size

    def forward(self, xs: [torch.Tensor], mm_token: torch.Tensor)\
            -> [torch.Tensor]:

        if not self.require_atten:
            for i in range(self.num_model):
                x = xs[i]
                cblock = self.cblocks[i]
                xs[i] = cblock(x, token=mm_token)
            return xs
        else:
            attns = []
            for i in range(self.num_model):
                x = xs[i]
                cblock = self.cblocks[i]
                xs[i], attn = cblock(x, token=mm_token)
                attns.append(attn)
            return xs, attns



# class FuseProBlock_TokenDict(nn.Module):
#     """Transformer blocks with support of window attention and residual propagation blocks"""
#
#     def __init__(
#         self,
#         dim: int,
#         num_heads: int,
#         mlp_ratio: float = 4.0,
#         qkv_bias: bool = True,
#         norm_layer: Type[nn.Module] = nn.LayerNorm,
#         act_layer: Type[nn.Module] = nn.GELU,
#         use_rel_pos: bool = False,
#         rel_pos_zero_init: bool = True,
#         window_size: int = 0,
#         input_size: Optional[Tuple[int, int]] = None,
#         num_model: int = 3,
#     ) -> None:
#         """
#         Args:
#             dim (int): Number of input channels.
#             num_heads (int): Number of attention heads in each ViT block.
#             mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
#             qkv_bias (bool): If True, add a learnable bias to query, key, value.
#             norm_layer (nn.Module): Normalization layer.
#             act_layer (nn.Module): Activation layer.
#             use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
#             rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
#             window_size (int): Window size for window attention blocks. If it equals 0, then
#                 use global attention.
#             input_size (int or None): Input resolution for calculating the relative positional
#                 parameter size.
#         """
#         super().__init__()
#         self.num_model = num_model
#         self.cblocks = nn.ModuleList()
#         for i in range(self.num_model):
#             cblock = CrossBlock_tokenkv(
#                 dim=dim,
#                 num_heads=num_heads,
#                 mlp_ratio=mlp_ratio,
#                 qkv_bias=qkv_bias,
#                 norm_layer=norm_layer,
#                 act_layer=act_layer,
#                 use_rel_pos=use_rel_pos,
#                 rel_pos_zero_init=rel_pos_zero_init,
#                 window_size=window_size,
#                 input_size=input_size,
#                 require_atten=True, #require atten
#             )
#             self.cblocks.append(cblock)
#         from .atd_arch import TokenDict_MSA
#         self.td_msa = TokenDict_MSA(dim=dim,
#                                     num_model=num_model,
#                                     num_heads=num_heads,
#                                     category_size=32,
#                                     qkv_bias=qkv_bias)
#
#         self.window_size = window_size
#
#     def forward(self, xs: [torch.Tensor], mm_token: torch.Tensor)\
#             -> [torch.Tensor]:
#         attens2 = []
#         xs2 = []
#         for i in range(self.num_model):
#             cblock = self.cblocks[i]
#             x = xs[i]
#             x, atten = cblock(x, token=mm_token)
#             xs2.append(x)
#             attens2.append(atten)
#         xs3 = self.td_msa(xs,attens2)
#         xs_sum = []
#         for i in range(self.num_model):
#             x_ = xs[i] + xs2[i] + xs3[i]
#             xs_sum.append(x_)
#
#         return xs_sum

class CrossAttention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
        require_atten: bool = False,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool:  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (int or None): Input resolution for calculating the relative positional
                parameter size.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.require_atten = require_atten

        self.kv_lin = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q_lin = nn.Linear(dim, dim * 1, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert (
                input_size is not None
            ), "Input size must be provided if using relative positional encoding."
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim)) # [2M-1,head_dim]
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, xkv: torch.Tensor, xq: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = xkv.shape
        # qkv with shape (3, B, nHead, H * W, C)
        kv = self.kv_lin(xkv).reshape(B, H * W, 2, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q in shape [1,1,C] or [B,Hq,Wq,C]
        if len(xq.shape) < 4:
            xq = xq.unsqueeze(0).repeat((B, 1, 1, 1))
        _, Hq, Wq, Cq = xq.shape
        # print(f'1 q:{q.shape}')
        q = self.q_lin(xq).reshape(B, Hq*Wq, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4) # check
        # print(f'1 q:{q.shape}')
        # q, k, v with shape (B * nHead, H * W, C)
        k, v = kv.reshape(2, B * self.num_heads, H * W, -1).unbind(0)  #k,v in [B*num_heads,HW,C]
        q = q.reshape(1, B * self.num_heads, Hq*Wq, -1)[0]
        # print(f'2 q:{q.shape},   k:{k.shape}')

        attn = (q * self.scale) @ k.transpose(-2, -1) #atten [B*num_heads,HqWq,HW]
        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (Hq, Wq), (H, W))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, Hq, Wq, -1).permute(0, 2, 3, 1, 4).reshape(B, Hq, Wq, -1) #[B,Hq,Wq,Cori]
        x = self.proj(x)

        # ac_msa
        attn = attn.reshape(B,self.num_heads,Hq,Wq,H,W).mean(dim=1,keepdim=False)#atten [B,Hq,Wq,H,W]

        if not self.require_atten:
            return x
        else:
            return x, attn


class CrossAttention_tokenkv(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        require_atten: bool = False,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool:  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (int or None): Input resolution for calculating the relative positional
                parameter size.
        """
        super().__init__()
        self.require_atten = require_atten
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.kv_lin = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q_lin = nn.Linear(dim, dim * 1, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, token: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        # kv = self.kv_lin(x).reshape(B, H * W, 2, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q in shape [1,1,C] or [B,Hq,Wq,C]
        if len(token.shape) < 4:
            token = token.unsqueeze(0).repeat((B, 1, 1, 1))
        _, Ht, Wt, Ct = token.shape
        # print(f'1 q:{q.shape}')
        # token = self.q_lin(token).reshape(B, Ht*Wt, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4) # check

        kv = self.kv_lin(token).reshape(B, Ht * Wt, 2, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q = self.q_lin(x).reshape(B, H*W, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4) # check

        # print(f'1 q:{q.shape}')
        # q, k, v with shape (B * nHead, H * W, C)
        k, v = kv.reshape(2, B * self.num_heads, Ht * Wt, -1).unbind(0)  #k,v in [B*num_heads,HtWt,C]
        q = q.reshape(1, B * self.num_heads, H*W, -1)[0]
        # print(f'2 q:{q.shape},   k:{k.shape}')

        attn = (q * self.scale) @ k.transpose(-2, -1) #atten [B*num_heads,HW,HtWt]

        attn = attn.softmax(dim=1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1) #[B,H,W,Cori]
        x = self.proj(x)

        # ac_msa
        attn = attn.reshape(B,self.num_heads,H,W,Ht*Wt).mean(dim=1,keepdim=False)#atten [B,HW,HtWt]
        # qkv = self.qkv_lin(x).reshape(B, H*W, -1)
        # x_msa = self.ac_msa(qkv, attn, x_size=(0,0))
        # print(f"debug: {attn.std()}")
        # x = x + x_msa

        if not self.require_atten:
            return x
        else:
            return x, attn

# This class and its supporting functions below lightly adapted from the ViTDet backbone available at: https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/vit.py # noqa
class FusePro(nn.Module):
    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
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
        num_model: int = 3,
    ) -> None:
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            global_attn_indexes (list): Indexes for blocks using global attention.
        """
        super().__init__()
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.num_model = num_model
        self.depth = depth
        # self.pos_embed: Optional[nn.Parameter] = None
        # if use_abs_pos:
        #     # Initialize absolute positional embedding with pretrain image size.
        #     self.pos_embed = nn.Parameter(
        #         torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim)
        #     )

        self.blocks = nn.ModuleList()
        self.mm_tokens = nn.ParameterList()
        for i in range(depth):
            mm_token = nn.Parameter(torch.randn(1, 1, embed_dim))
            self.mm_tokens.append(mm_token)
            block = FuseProBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
                num_model = self.num_model
            )
            self.blocks.append(block)

        self.neck = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                out_chans,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
            nn.Conv2d(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
        )

    def forward(self, xs: [torch.Tensor]) -> torch.Tensor:
        '''
        xs: List[x], x:[B,H,W,C]
        '''
        # x = self.patch_embed(x)  # pre embed: [1, 3, 1024, 1024], post embed: [1, 64, 64, 768]
        # if self.pos_embed is not None:
        #     x = x + self.pos_embed
        for idx in range(self.depth):
            blk = self.blocks[idx]
            mm_token = self.mm_tokens[idx]
            xs = blk(xs,mm_token=mm_token)
        N = len(xs)
        assert N == self.num_model
        x = xs[0]
        for idx in range(1,N):
            x = x + xs[idx]
        x = self.neck(x.permute(0, 3, 1, 2))  # [b, c, h, w], [1, 256, 64, 64]
        return x

class FusePro_Parallel_ShareToken(nn.Module):
    def __init__(self,
                 img_size: int = 1024,
                 patch_size: int = 16,
                 in_chans: int = 3,
                 embed_dim: int = 768,
                 depth: int = 12,
                 share_token_num: int = 1,
                 num_heads: int = 12,
                 mlp_ratio: float = 4.0,
                 out_chans: int = 256,
                 qkv_bias: bool = True,
                 norm_layer: Type[nn.Module] = nn.LayerNorm,
                 act_layer: Type[nn.Module] = nn.GELU,
                 use_rel_pos: bool = False,
                 rel_pos_zero_init: bool = True,
                 window_size: int = 0,
                 global_attn_indexes: Tuple[int, ...] = (),
                 num_model: int = 3,
                 require_atten: bool = False,

                 ):
        super().__init__()
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.num_model = num_model
        self.depth = depth
        self.share_token_num = share_token_num
        self.require_atten = require_atten
        self.blocks = nn.ModuleList()
        self.mm_tokens = nn.ParameterList()
        for i in range(depth):
            mm_token = nn.Parameter(torch.randn(1, self.share_token_num, embed_dim))
            self.mm_tokens.append(mm_token)
            block = FuseProBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
                num_model = self.num_model,
                require_atten = require_atten,
            )
            self.blocks.append(block)

    def forward(self,xs: [torch.Tensor]) -> [torch.Tensor]:
        """
        xs: [tensor], shape in [b,h,w,c]
        out: shape same as xs
        """
        for idx in range(self.depth):
            blk = self.blocks[idx]
            mm_token = self.mm_tokens[idx]
            if not self.require_atten:
                xs = blk(xs,mm_token=mm_token)
            else:
                xs, attns = blk(xs,mm_token=mm_token)
            # attns_vis(attns,f'depth{idx}')
        return xs
# class FusePro_Parallel_ActToken(nn.Module):
#     def __init__(self,
#                  img_size: int = 1024,
#                  patch_size: int = 16,
#                  in_chans: int = 3,
#                  embed_dim: int = 768,
#                  depth: int = 12,
#                  share_token_num: int = 1,
#                  num_heads: int = 12,
#                  mlp_ratio: float = 4.0,
#                  out_chans: int = 256,
#                  qkv_bias: bool = True,
#                  norm_layer: Type[nn.Module] = nn.LayerNorm,
#                  act_layer: Type[nn.Module] = nn.GELU,
#                  use_rel_pos: bool = False,
#                  rel_pos_zero_init: bool = True,
#                  window_size: int = 0,
#                  global_attn_indexes: Tuple[int, ...] = (),
#                  num_model: int = 3,
#                  require_atten: bool = False,
#
#                  ):
#         super().__init__()
#         self.img_size = img_size
#         self.embed_dim = embed_dim
#         self.patch_size = patch_size
#         self.in_chans = in_chans
#         self.out_chans = out_chans
#         self.num_model = num_model
#         self.depth = depth
#         self.share_token_num = share_token_num
#         self.require_atten = require_atten
#         self.cblocks = nn.ModuleList()
#         self.tblocks = nn.ModuleList()
#
#         self.mm_tokens = nn.ParameterList()
#         for i in range(depth):
#             mm_token = nn.Parameter(torch.randn(1, self.share_token_num*self.num_model, embed_dim))
#             self.mm_tokens.append(mm_token)
#             tblock = CrossAttenBlock_Act_Token(
#                 dim=embed_dim,
#                 num_heads=num_heads,
#                 mlp_ratio=mlp_ratio,
#                 qkv_bias=qkv_bias,
#                 norm_layer=norm_layer,
#                 act_layer=act_layer,
#                 use_rel_pos=use_rel_pos,
#                 rel_pos_zero_init=rel_pos_zero_init,
#                 window_size=window_size if i not in global_attn_indexes else 0,
#                 input_size=(img_size // patch_size, img_size // patch_size),
#                 num_model=self.num_model,
#                 fuse_token_num_base = self.share_token_num,
#             )
#             self.tblocks.append(tblock)
#             cblock = FuseProBlock(
#                 dim=embed_dim,
#                 num_heads=num_heads,
#                 mlp_ratio=mlp_ratio,
#                 qkv_bias=qkv_bias,
#                 norm_layer=norm_layer,
#                 act_layer=act_layer,
#                 use_rel_pos=use_rel_pos,
#                 rel_pos_zero_init=rel_pos_zero_init,
#                 window_size=window_size if i not in global_attn_indexes else 0,
#                 input_size=(img_size // patch_size, img_size // patch_size),
#                 num_model = self.num_model,
#                 require_atten = require_atten,
#             )
#             self.cblocks.append(cblock)
#
#     def forward(self,xs: [torch.Tensor]) -> [torch.Tensor]:
#         """
#         xs: [tensor], shape in [b,h,w,c]
#         out: shape same as xs
#         """
#         for idx in range(self.depth):
#             tblk = self.tblocks[idx]
#             cblk = self.cblocks[idx]
#             mm_token = self.mm_tokens[idx]
#             act_token = tblk(xs,mm_token)
#
#             if not self.require_atten:
#                 xs = cblk(xs,mm_token=act_token)
#             else:
#                 xs, attns = cblk(xs,mm_token=act_token)
#             # attns_vis(attns,f'depth{idx}')
#         return xs

# class FusePro_Parallel_CrossAttention(nn.Module):
#     def __init__(self,
#                  img_size: int = 1024,
#                  patch_size: int = 16,
#                  in_chans: int = 3,
#                  embed_dim: int = 768,
#                  depth: int = 12,
#                  num_heads: int = 12,
#                  mlp_ratio: float = 4.0,
#                  out_chans: int = 256,
#                  qkv_bias: bool = True,
#                  norm_layer: Type[nn.Module] = nn.LayerNorm,
#                  act_layer: Type[nn.Module] = nn.GELU,
#                  use_rel_pos: bool = False,
#                  rel_pos_zero_init: bool = True,
#                  window_size: int = 0,
#                  global_attn_indexes: Tuple[int, ...] = (),
#                  num_model: int = 3,
#                  ):
#         super().__init__()
#         self.img_size = img_size
#         self.embed_dim = embed_dim
#         self.patch_size = patch_size
#         self.in_chans = in_chans
#         self.out_chans = out_chans
#         self.num_model = num_model
#         self.depth = depth
#         assert num_model == 2
#         self.blocks = nn.ModuleList()
#         for i in range(depth):
#             block = CrossAttenBlock_2model(
#                 dim=embed_dim,
#                 num_heads=num_heads,
#                 mlp_ratio=mlp_ratio,
#                 qkv_bias=qkv_bias,
#                 norm_layer=norm_layer,
#                 act_layer=act_layer,
#                 use_rel_pos=use_rel_pos,
#                 rel_pos_zero_init=rel_pos_zero_init,
#                 window_size=window_size if i not in global_attn_indexes else 0,
#                 input_size=(img_size // patch_size, img_size // patch_size),
#                 num_model = self.num_model
#             )
#             self.blocks.append(block)
#     def forward(self,xs: [torch.Tensor]) -> [torch.Tensor]:
#         """
#         xs: [tensor], shape in [b,h,w,c]
#         out: shape same as xs
#         """
#         for idx in range(self.depth):
#             blk = self.blocks[idx]
#             xs = blk(xs)
#         return xs

# class FusePro_Parallel_SelfAttention(nn.Module):
#     def __init__(self,
#                  img_size: int = 1024,
#                  patch_size: int = 16,
#                  in_chans: int = 3,
#                  embed_dim: int = 768,
#                  depth: int = 12,
#                  num_heads: int = 12,
#                  mlp_ratio: float = 4.0,
#                  out_chans: int = 256,
#                  qkv_bias: bool = True,
#                  norm_layer: Type[nn.Module] = nn.LayerNorm,
#                  act_layer: Type[nn.Module] = nn.GELU,
#                  use_rel_pos: bool = False,
#                  rel_pos_zero_init: bool = True,
#                  window_size: int = 0,
#                  global_attn_indexes: Tuple[int, ...] = (),
#                  num_model: int = 3,
#                  ):
#         super().__init__()
#         self.img_size = img_size
#         self.embed_dim = embed_dim
#         self.patch_size = patch_size
#         self.in_chans = in_chans
#         self.out_chans = out_chans
#         self.num_model = num_model
#         self.depth = depth
#         assert num_model == 2
#         self.blocks = nn.ModuleList()
#         for i in range(depth):
#             block = SelfAttenBlock_2model(
#                 dim=embed_dim,
#                 num_heads=num_heads,
#                 mlp_ratio=mlp_ratio,
#                 qkv_bias=qkv_bias,
#                 norm_layer=norm_layer,
#                 act_layer=act_layer,
#                 use_rel_pos=use_rel_pos,
#                 rel_pos_zero_init=rel_pos_zero_init,
#                 window_size=window_size if i not in global_attn_indexes else 0,
#                 input_size=(img_size // patch_size, img_size // patch_size),
#                 num_model = self.num_model
#             )
#             self.blocks.append(block)
#     def forward(self,xs: [torch.Tensor]) -> [torch.Tensor]:
#         """
#         xs: [tensor], shape in [b,h,w,c]
#         out: shape same as xs
#         """
#         for idx in range(self.depth):
#             blk = self.blocks[idx]
#             xs = blk(xs)
#         return xs


# class FusePro_Parallel_TokenDict(nn.Module):
#     def __init__(self,
#                  img_size: int = 1024,
#                  patch_size: int = 16,
#                  in_chans: int = 3,
#                  embed_dim: int = 768,
#                  depth: int = 12,
#                  share_token_num: int = 1,
#                  num_heads: int = 12,
#                  mlp_ratio: float = 4.0,
#                  out_chans: int = 256,
#                  qkv_bias: bool = True,
#                  norm_layer: Type[nn.Module] = nn.LayerNorm,
#                  act_layer: Type[nn.Module] = nn.GELU,
#                  use_rel_pos: bool = False,
#                  rel_pos_zero_init: bool = True,
#                  window_size: int = 0,
#                  global_attn_indexes: Tuple[int, ...] = (),
#                  num_model: int = 3,
#                  ):
#         super().__init__()
#         self.img_size = img_size
#         self.embed_dim = embed_dim
#         self.patch_size = patch_size
#         self.in_chans = in_chans
#         self.out_chans = out_chans
#         self.num_model = num_model
#         self.depth = depth
#         self.share_token_num = share_token_num
#         self.blocks = nn.ModuleList()
#         self.mm_tokens = nn.ParameterList()
#         for i in range(depth):
#             mm_token = nn.Parameter(torch.randn(1, self.share_token_num, embed_dim))
#             self.mm_tokens.append(mm_token)
#             block = FuseProBlock_TokenDict(
#                 dim=embed_dim,
#                 num_heads=num_heads,
#                 mlp_ratio=mlp_ratio,
#                 qkv_bias=qkv_bias,
#                 norm_layer=norm_layer,
#                 act_layer=act_layer,
#                 use_rel_pos=use_rel_pos,
#                 rel_pos_zero_init=rel_pos_zero_init,
#                 window_size=window_size if i not in global_attn_indexes else 0,
#                 input_size=(img_size // patch_size, img_size // patch_size),
#                 num_model = self.num_model
#             )
#             self.blocks.append(block)
#         self.init_parameters()
#     def init_parameters(self) -> None:
#         for mm_token in self.mm_tokens:
#             nn.init.kaiming_uniform_(mm_token, a=math.sqrt(5))
#
#     def forward(self,xs: [torch.Tensor]) -> [torch.Tensor]:
#         """
#         xs: [tensor], shape in [b,h,w,c]
#         out: shape same as xs
#         """
#         for idx in range(self.depth):
#             blk = self.blocks[idx]
#             mm_token = self.mm_tokens[idx]
#             xs = blk(xs,mm_token=mm_token)
#         return xs

# class FusePro_Parallel_None(nn.Module):
#     def __init__(self,
#                  img_size: int = 1024,
#                  ):
#         super().__init__()
#         self.img_size = img_size
#
#
#     def forward(self,xs: [torch.Tensor]) -> [torch.Tensor]:
#         """
#         xs: [tensor], shape in [b,h,w,c]
#         out: shape same as xs
#         """
#         return xs

# class FusePro_Abla_CrossAttention(nn.Module):
#     def __init__(self,
#                  img_size: int = 1024,
#                  patch_size: int = 16,
#                  in_chans: int = 3,
#                  embed_dim: int = 768,
#                  depth: int = 12,
#                  num_heads: int = 12,
#                  mlp_ratio: float = 4.0,
#                  out_chans: int = 256,
#                  qkv_bias: bool = True,
#                  norm_layer: Type[nn.Module] = nn.LayerNorm,
#                  act_layer: Type[nn.Module] = nn.GELU,
#                  use_rel_pos: bool = False,
#                  rel_pos_zero_init: bool = True,
#                  window_size: int = 0,
#                  global_attn_indexes: Tuple[int, ...] = (),
#                  num_model: int = 3,
#                  ):
#         super().__init__()
#         self.img_size = img_size
#         self.embed_dim = embed_dim
#         self.patch_size = patch_size
#         self.in_chans = in_chans
#         self.out_chans = out_chans
#         self.num_model = num_model
#         self.depth = depth
#         assert num_model == 2
#         self.blocks = nn.ModuleList()
#         for i in range(depth):
#             block = CrossAttenBlock_2model(
#                 dim=embed_dim,
#                 num_heads=num_heads,
#                 mlp_ratio=mlp_ratio,
#                 qkv_bias=qkv_bias,
#                 norm_layer=norm_layer,
#                 act_layer=act_layer,
#                 use_rel_pos=use_rel_pos,
#                 rel_pos_zero_init=rel_pos_zero_init,
#                 window_size=window_size if i not in global_attn_indexes else 0,
#                 input_size=(img_size // patch_size, img_size // patch_size),
#                 num_model = self.num_model
#             )
#             self.blocks.append(block)
#     def forward(self,xs: [torch.Tensor]) -> [torch.Tensor]:
#         """
#         xs: [tensor], shape in [b,h,w,c]
#         out: shape same as xs
#         """
#         for idx in range(self.depth):
#             blk = self.blocks[idx]
#             xs = blk(xs)
#         return xs

# class FusePro_Abla_SelfAttention(nn.Module):
#     def __init__(self,
#                  img_size: int = 1024,
#                  patch_size: int = 16,
#                  in_chans: int = 3,
#                  embed_dim: int = 768,
#                  depth: int = 12,
#                  num_heads: int = 12,
#                  mlp_ratio: float = 4.0,
#                  out_chans: int = 256,
#                  qkv_bias: bool = True,
#                  norm_layer: Type[nn.Module] = nn.LayerNorm,
#                  act_layer: Type[nn.Module] = nn.GELU,
#                  use_rel_pos: bool = False,
#                  rel_pos_zero_init: bool = True,
#                  window_size: int = 0,
#                  global_attn_indexes: Tuple[int, ...] = (),
#                  num_model: int = 3,
#                  ):
#         super().__init__()
#         self.img_size = img_size
#         self.embed_dim = embed_dim
#         self.patch_size = patch_size
#         self.in_chans = in_chans
#         self.out_chans = out_chans
#         self.num_model = num_model
#         self.depth = depth
#         assert num_model == 2
#         self.blocks = nn.ModuleList()
#         for i in range(depth):
#             block = SelfAttenBlock_2model(
#                 dim=embed_dim,
#                 num_heads=num_heads,
#                 mlp_ratio=mlp_ratio,
#                 qkv_bias=qkv_bias,
#                 norm_layer=norm_layer,
#                 act_layer=act_layer,
#                 use_rel_pos=use_rel_pos,
#                 rel_pos_zero_init=rel_pos_zero_init,
#                 window_size=window_size if i not in global_attn_indexes else 0,
#                 input_size=(img_size // patch_size, img_size // patch_size),
#                 num_model = self.num_model
#             )
#             self.blocks.append(block)
#     def forward(self,xs: [torch.Tensor]) -> [torch.Tensor]:
#         """
#         xs: [tensor], shape in [b,h,w,c]
#         out: shape same as xs
#         """
#         # obtain spatial size
#         ht,wt = None, None
#         for x_ in xs:
#             _, h, w, _ = x_.shape
#             ht = h if ht is None else ht
#             wt = w if wt is None else wt
#             ht,wt = min(ht,h),min(wt,w)
#         for idx in range(len(xs)):
#             x_ = xs[idx]
#             x_ = x_.permute(0,3,1,2)
#             x_ = F.interpolate(x_,size=(ht,wt),mode='bilinear')
#             x_ = x_.permute(0,2,3,1)
#             xs[idx] = x_
#         for idx in range(self.depth):
#             blk = self.blocks[idx]
#             xs = blk(xs)
#         return xs


# class FusePro_Abla_Sum(nn.Module):
#     def __init__(self,
#                  img_size: int = 1024,
#                  ):
#         super().__init__()
#         self.img_size = img_size
#
#
#     def forward(self,xs: [torch.Tensor]) -> [torch.Tensor]:
#         """
#         xs: [tensor], shape in [b,h,w,c]
#         out: shape same as xs
#         """
#         return xs

# class FusePro_Abla_Cat(nn.Module):
#     def __init__(self,
#                  num_model: int  = 2,
#                  embed_dim: int = 768,
#                  ):
#         super().__init__()
#         self.num_model = num_model
#         self.proj = nn.Sequential(
#                     nn.Linear(embed_dim*num_model, embed_dim, bias=False),  # 从 c -> c/r
#                     nn.GELU(),
#                     # nn.Linear(embed_dim, embed_dim, bias=False),
#         )
#
#     def forward(self,xs: [torch.Tensor]) -> [torch.Tensor]:
#         """
#         xs: [tensor], shape in [b,h,w,c]
#         out: shape same as xs
#         """
#         # obtain spatial size
#         ht,wt = None, None
#         for x_ in xs:
#             _, h, w, _ = x_.shape
#             ht = h if ht is None else ht
#             wt = w if wt is None else wt
#             ht,wt = min(ht,h),min(wt,w)
#         for idx in range(len(xs)):
#             x_ = xs[idx]
#             x_ = x_.permute(0,3,1,2)
#             x_ = F.interpolate(x_,size=(ht,wt),mode='bilinear')
#             x_ = x_.permute(0,2,3,1)
#             xs[idx] = x_
#
#         x_cat = torch.cat(xs,dim=-1)
#         x_cat = self.proj(x_cat)
#         Nmodel = len(xs)
#         xs2 = []
#         for i in range(Nmodel):
#             xs2.append(x_cat/Nmodel)
#         return xs2

# class FusePro_Abla_CatLittle(nn.Module):
#     def __init__(self,
#                  num_model: int  = 2,
#                  embed_dim: int = 768,
#                  ):
#         super().__init__()
#         self.num_model = num_model
#         assert num_model == 2
#         assert embed_dim%2 == 0
#         self.projs = nn.ModuleList()
#         for idx in range(num_model):
#             proj = nn.Sequential(
#                     nn.Linear(embed_dim, embed_dim//2, bias=False),  # 从 c -> c/r
#                     nn.GELU(),)
#             self.projs.append(proj)
#
#
#     def forward(self,xs: [torch.Tensor]) -> [torch.Tensor]:
#         """
#         xs: [tensor], shape in [b,h,w,c]
#         out: shape same as xs
#         """
#         # obtain spatial size
#         ht,wt = None, None
#         for x_ in xs:
#             _, h, w, _ = x_.shape
#             ht = h if ht is None else ht
#             wt = w if wt is None else wt
#             ht,wt = min(ht,h),min(wt,w)
#         for idx in range(len(xs)):
#             x_ = xs[idx]
#             x_ = x_.permute(0,3,1,2)
#             x_ = F.interpolate(x_,size=(ht,wt),mode='bilinear')
#             x_ = x_.permute(0,2,3,1)
#             xs[idx] = x_
#         xs2 = []
#         for idx in range(len(xs)):
#             x_ = xs[idx]
#             proj = self.projs[idx]
#             x_ = proj(x_)
#             xs2.append(x_)
#         x_cat = torch.cat(xs2,dim=-1)
#         Nmodel = len(xs)
#         xs2 = []
#         for i in range(Nmodel):
#             xs2.append(x_cat/Nmodel)
#         return xs2


# class SE_Block(nn.Module):
#     def __init__(self, inchannel, ratio=16):
#         super(SE_Block, self).__init__()
#         self.res = True
#         self.gap = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Sequential(
#             nn.Linear(inchannel, inchannel // ratio, bias=False),  # 从 c -> c/r
#             nn.ReLU(),
#             nn.Linear(inchannel // ratio, inchannel, bias=False),  # 从 c/r -> c
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#             b, c, h, w = x.size()
#             y = self.gap(x).view(b, c)
#             y = self.fc(y).view(b, c, 1, 1)
#             out = x * y.expand_as(x)
#             if self.res:
#                 out = out + x
#             return out

# class SA_Block(nn.Module):
#     '''
#     spatial attention/gate
#     '''
#     def __init__(self, inchannel, ratio=16):
#         super(SA_Block, self).__init__()
#         self.res = True
#         self.fc = nn.Sequential(
#             nn.Linear(inchannel, inchannel // ratio, bias=False),  # 从 c -> c/r
#             nn.GELU(),
#             nn.Linear(inchannel // ratio, 1, bias=False),  # 从 c/r -> c
#             nn.Sigmoid()
#         )
#         self.proj = nn.Sequential(
#             nn.Linear(inchannel, inchannel // ratio, bias=False),  # 从 c -> c/r
#             nn.GELU(),
#             nn.Linear(inchannel // ratio, inchannel, bias=False),  # 从 c/r -> c
#         )
#
#     def forward(self, x):
#         '''
#         x : [B,C,H,W]
#         '''
#         b, c, h, w = x.size()
#         gate = self.fc(x.permute(0,2,3,1)).permute(0, 3, 1, 2)
#         out = self.proj(x) * gate
#         if self.res:
#             out = out + x
#         return out

# class MultiModelPrompter(nn.Module):
#     def __init__(self,
#                  embed_dim: int = 768,
#                  mlp_shrink_ratio: float = 8.0,
#                  num_model: int = 3,
#                  ):
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.mlp_shrink_ratio = mlp_shrink_ratio
#         self.num_model = num_model
#         shrink_dim = int(embed_dim//self.mlp_shrink_ratio)
#         per_cat_dim = int(embed_dim//num_model)
#         self.qs = nn.ModuleList()
#         for i in range(self.num_model):
#             q_ = nn.Sequential(
#             nn.Linear(embed_dim, shrink_dim),
#             nn.GELU(),
#             nn.Linear(shrink_dim, per_cat_dim),
#             )
#             self.qs.append(q_)
#         self.proj = nn.Sequential(
#             nn.Linear(per_cat_dim*self.num_model, shrink_dim),
#             nn.GELU(),
#             nn.Linear(shrink_dim, embed_dim),
#             )
#
#         self.se_block = SE_Block(inchannel=embed_dim,ratio=mlp_shrink_ratio)
#         self.gap = nn.AdaptiveAvgPool2d((1, 1))
#
#
#     def forward(self,xs: [torch.Tensor]) -> torch.Tensor:
#         """
#         xs: [tensor], shape in [b,h,w,c]
#         out: shape same as xs
#         """
#         xs1d = []
#         for i in range(self.num_model):
#             q_ = self.qs[i]
#             x_ = q_(xs[i])
#             x_ = self.gap(x_.permute(0,3,1,2)).permute(0,2,3,1)
#             xs1d.append(x_)
#
#
#         x_fuse = torch.cat(xs1d,dim=-1)
#         x_fuse = self.proj(x_fuse)
#         x_fuse = self.se_block(x_fuse.permute(0,3,1,2))
#         x_fuse = x_fuse.permute(0,2,3,1)
#         return x_fuse
#
#
# class MultiModelPrompter_GAPlater(nn.Module):
#     def __init__(self,
#                  embed_dim: int = 768,
#                  mlp_shrink_ratio: float = 8.0,
#                  num_model: int = 3,
#                  ):
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.mlp_shrink_ratio = mlp_shrink_ratio
#         self.num_model = num_model
#         shrink_dim = int(embed_dim//self.mlp_shrink_ratio)
#         self.q_tot = nn.Sequential(
#             nn.Linear(embed_dim*3, shrink_dim*3),
#             nn.GELU(),
#             nn.Linear(shrink_dim*3, embed_dim),
#         )
#         self.se_block = SE_Block(inchannel=embed_dim,ratio=mlp_shrink_ratio)
#         self.gap = nn.AdaptiveAvgPool2d((1, 1))
#
#
#     def forward(self,xs: [torch.Tensor]) -> torch.Tensor:
#         """
#         xs: [tensor], shape in [b,h,w,c]
#         out: shape same as xs
#         """
#         xs3 = torch.cat(xs,dim=-1)
#         xs3 = self.q_tot(xs3)
#         xs3 = xs3.permute(0,3,1,2)
#         xs3 = self.se_block(xs3)
#         xs3 = self.gap(xs3)
#         xs3 = xs3.permute(0,2,3,1)
#         return xs3
#
# class MultiModelPrompter2d(nn.Module):
#     def __init__(self,
#                  embed_dim: int = 768,
#                  mlp_shrink_ratio: float = 8.0,
#                  num_model: int = 3,
#                  ):
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.mlp_shrink_ratio = mlp_shrink_ratio
#         self.num_model = num_model
#         shrink_dim = int(embed_dim // self.mlp_shrink_ratio)
#         self.se_proj = nn.Sequential(
#             nn.Linear((embed_dim // num_model) * num_model, shrink_dim),
#             nn.GELU(),
#             nn.Linear(shrink_dim, embed_dim),
#         )
#         self.q_tot = nn.Sequential(
#             nn.Linear(embed_dim*3, shrink_dim*3),
#             nn.GELU(),
#             nn.Linear(shrink_dim*3, embed_dim),
#         )
#         self.se_block = SE_Block(inchannel=embed_dim, ratio=mlp_shrink_ratio)
#         self.dconv = nn.Conv2d(in_channels=embed_dim,
#                                out_channels=embed_dim,
#                                kernel_size=3,
#                                padding=1,
#                                stride=1,
#                                groups=embed_dim)
#         self.pconv = nn.Conv2d(in_channels=embed_dim,
#                                out_channels=embed_dim,
#                                kernel_size=1,
#                                stride=1)
#
#     def forward(self, xs: [torch.Tensor]) -> torch.Tensor:
#         """
#         xs: [tensor], shape in [b,h,w,c]
#         out: shape same as xs
#         """
#         xs2 = torch.cat(xs,dim=-1)
#         xs2 = self.q_tot(xs2)
#         xs2 = xs2.permute(0, 3, 1, 2)
#         xs2 = self.dconv(xs2)
#         xs2 = self.se_block(xs2)
#         xs2 = self.pconv(xs2)
#         xs2 = xs2.permute(0, 2, 3, 1)
#         return xs2

# class FusePro_Parallel_Prompt(nn.Module):
#     def __init__(self,
#                  img_size: int = 1024,
#                  patch_size: int = 16,
#                  in_chans: int = 3,
#                  embed_dim: int = 768,
#                  depth: int = 12,
#                  num_heads: int = 12,
#                  mlp_ratio: float = 4.0,
#                  out_chans: int = 256,
#                  qkv_bias: bool = True,
#                  norm_layer: Type[nn.Module] = nn.LayerNorm,
#                  act_layer: Type[nn.Module] = nn.GELU,
#                  use_rel_pos: bool = False,
#                  rel_pos_zero_init: bool = True,
#                  window_size: int = 0,
#                  global_attn_indexes: Tuple[int, ...] = (),
#                  num_model: int = 3,
#                  ):
#         super().__init__()
#         self.img_size = img_size
#         self.embed_dim = embed_dim
#         self.patch_size = patch_size
#         self.in_chans = in_chans
#         self.out_chans = out_chans
#         self.num_model = num_model
#         self.depth = depth
#         self.blocks = nn.ModuleList()
#         self.mm_tokens = nn.ParameterList()
#         self.prompters = nn.ModuleList()
#         for i in range(depth):
#             prompter = MultiModelPrompter(
#                             embed_dim=embed_dim,
#                             mlp_shrink_ratio=8,
#                             num_model=self.num_model
#                             )
#             self.prompters.append(prompter)
#             block = FuseProBlock(
#                 dim=embed_dim,
#                 num_heads=num_heads,
#                 mlp_ratio=mlp_ratio,
#                 qkv_bias=qkv_bias,
#                 norm_layer=norm_layer,
#                 act_layer=act_layer,
#                 use_rel_pos=use_rel_pos,
#                 rel_pos_zero_init=rel_pos_zero_init,
#                 window_size = window_size if i not in global_attn_indexes else 0,
#                 input_size=(img_size // patch_size, img_size // patch_size),
#                 num_model = self.num_model
#             )
#             self.blocks.append(block)
#     def forward(self,xs: [torch.Tensor]) -> [torch.Tensor]:
#         """
#         xs: [tensor], shape in [b,h,w,c]
#         out: shape same as xs
#         """
#         for idx in range(self.depth):
#             blk = self.blocks[idx]
#             prompter = self.prompters[idx]
#             prompt = prompter(xs)
#             xs = blk(xs,mm_token=prompt)
#         return xs

# class FusePro_Parallel_Prompt_GAPlater(nn.Module):
#     def __init__(self,
#                  img_size: int = 1024,
#                  patch_size: int = 16,
#                  in_chans: int = 3,
#                  embed_dim: int = 768,
#                  depth: int = 12,
#                  num_heads: int = 12,
#                  mlp_ratio: float = 4.0,
#                  out_chans: int = 256,
#                  qkv_bias: bool = True,
#                  norm_layer: Type[nn.Module] = nn.LayerNorm,
#                  act_layer: Type[nn.Module] = nn.GELU,
#                  use_rel_pos: bool = False,
#                  rel_pos_zero_init: bool = True,
#                  window_size: int = 0,
#                  global_attn_indexes: Tuple[int, ...] = (),
#                  num_model: int = 3,
#                  ):
#         super().__init__()
#         self.img_size = img_size
#         self.embed_dim = embed_dim
#         self.patch_size = patch_size
#         self.in_chans = in_chans
#         self.out_chans = out_chans
#         self.num_model = num_model
#         self.depth = depth
#         self.blocks = nn.ModuleList()
#         self.mm_tokens = nn.ParameterList()
#         self.prompters = nn.ModuleList()
#         for i in range(depth):
#             prompter = MultiModelPrompter_GAPlater(
#                             embed_dim=embed_dim,
#                             mlp_shrink_ratio=8,
#                             num_model=self.num_model
#                             )
#             self.prompters.append(prompter)
#             block = FuseProBlock(
#                 dim=embed_dim,
#                 num_heads=num_heads,
#                 mlp_ratio=mlp_ratio,
#                 qkv_bias=qkv_bias,
#                 norm_layer=norm_layer,
#                 act_layer=act_layer,
#                 use_rel_pos=use_rel_pos,
#                 rel_pos_zero_init=rel_pos_zero_init,
#                 window_size = window_size if i not in global_attn_indexes else 0,
#                 input_size=(img_size // patch_size, img_size // patch_size),
#                 num_model = self.num_model
#             )
#             self.blocks.append(block)
#     def forward(self,xs: [torch.Tensor]) -> [torch.Tensor]:
#         """
#         xs: [tensor], shape in [b,h,w,c]
#         out: shape same as xs
#         """
#         for idx in range(self.depth):
#             blk = self.blocks[idx]
#             prompter = self.prompters[idx]
#             prompt = prompter(xs)
#             xs = blk(xs,mm_token=prompt)
#         return xs
#
# class FusePro_Parallel_Prompt2d(nn.Module):
#     def __init__(self,
#                  img_size: int = 1024,
#                  patch_size: int = 16,
#                  in_chans: int = 3,
#                  embed_dim: int = 768,
#                  depth: int = 12,
#                  num_heads: int = 12,
#                  mlp_ratio: float = 4.0,
#                  out_chans: int = 256,
#                  qkv_bias: bool = True,
#                  norm_layer: Type[nn.Module] = nn.LayerNorm,
#                  act_layer: Type[nn.Module] = nn.GELU,
#                  use_rel_pos: bool = False,
#                  rel_pos_zero_init: bool = True,
#                  window_size: int = 0,
#                  global_attn_indexes: Tuple[int, ...] = (),
#                  num_model: int = 3,
#                  require_atten: bool = True,
#                  ):
#         super().__init__()
#         self.img_size = img_size
#         self.embed_dim = embed_dim
#         self.patch_size = patch_size
#         self.in_chans = in_chans
#         self.out_chans = out_chans
#         self.num_model = num_model
#         self.depth = depth
#         self.blocks = nn.ModuleList()
#         self.mm_tokens = nn.ParameterList()
#         self.prompters = nn.ModuleList()
#         for i in range(depth):
#             prompter = MultiModelPrompter2d(
#                             embed_dim=embed_dim,
#                             mlp_shrink_ratio=8,
#                             num_model=self.num_model
#                             )
#             self.prompters.append(prompter)
#             block = FuseProBlock(
#                 dim=embed_dim,
#                 num_heads=num_heads,
#                 mlp_ratio=mlp_ratio,
#                 qkv_bias=qkv_bias,
#                 norm_layer=norm_layer,
#                 act_layer=act_layer,
#                 use_rel_pos=use_rel_pos,
#                 rel_pos_zero_init=rel_pos_zero_init,
#                 window_size = window_size if i not in global_attn_indexes else 0,
#                 input_size=(img_size // patch_size, img_size // patch_size),
#                 num_model = self.num_model,
#                 require_atten = require_atten,
#             )
#             self.blocks.append(block)
#     def forward(self,xs: [torch.Tensor]) -> [torch.Tensor]:
#         """
#         xs: [tensor], shape in [b,h,w,c]
#         out: shape same as xs
#         """
#         for idx in range(self.depth):
#             blk = self.blocks[idx]
#             prompter = self.prompters[idx]
#             prompt = prompter(xs)
#             xs = blk(xs,mm_token=prompt)
#         return xs

class FusePro_Final_Sum(nn.Module):
    def __init__(self,
                 embed_dim: int = 768,
                 out_chans: int = 256,
                 num_model: int = 3,
                 ):
        super().__init__()
        self.num_model = num_model
        self.fin_fuser = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                out_chans,
                kernel_size=3,
                bias=False,
            ),
            nn.GELU(),
            nn.Conv2d(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
        )
    def forward(self,xs: [torch.Tensor]) -> torch.Tensor:
        """
        xs: [tensor], all tensors in shape [b, h, w, c]
        out:
        x: [b, c, h, w]
        """
        N = len(xs)
        assert N == self.num_model
        x = xs[0]
        for idx in range(1,N):
            x = x + xs[idx]
        x = self.fin_fuser(x.permute(0, 3, 1, 2))  # [b, c, h, w], [1, 256, 64, 64]

        return x

# class FusePro_Final_MultiSE(nn.Module):
#     def __init__(self,
#                  embed_dim: int = 768,
#                  out_chans: int = 256,
#                  num_model: int = 3,
#                  ):
#         super().__init__()
#         self.num_model = num_model
#         self.aligners = nn.ModuleList()
#         self.attners = nn.ModuleList()
#         self.multipliers = nn.ModuleList()
#
#         for i in range(self.num_model):
#             aligner = nn.Sequential(MLPBlock(embedding_dim=embed_dim,
#                                              mlp_dim=embed_dim//8),
#                                     LayerNorm2d_forMLP(embed_dim)
#                                     )
#             self.aligners.append(aligner)
#             attner = nn.Sequential(
#                                     nn.Linear(embed_dim, embed_dim//8),
#                                     nn.GELU(),
#                                     nn.Linear(embed_dim//8, out_chans),
#                                     nn.Sigmoid()
#             )
#             self.attners.append(attner)
#             multiplier = nn.Sequential(
#                                     nn.Linear(embed_dim, embed_dim//8),
#                                     nn.GELU(),
#                                     nn.Linear(embed_dim//8, out_chans),
#                                     )
#             self.multipliers.append(multiplier)
#         self.out_norm = LayerNorm2d_forMLP(out_chans)
#
#     def forward(self,xs: [torch.Tensor]) -> torch.Tensor:
#         """
#         xs: [tensor], all tensors in shape [b, h, w, c]
#         out:
#         x: [b, c, h, w]
#         """
#         N = len(xs)
#         assert N == self.num_model
#         xs2 = []
#         for idx in range(N):
#             aligner = self.aligners[idx]
#             x2 = aligner(xs[idx])
#             xs2.append(x2)
#         x2 = torch.stack(xs2,dim=0).sum(dim=0)
#         x3s = []
#         for idx in range(N):
#             attener = self.attners[idx]
#             hm = attener(x2)
#             multiplier = self.multipliers[idx]
#             x3 = multiplier(x2) * hm
#             x3s.append(x3)
#         x_out = torch.stack(x3s,dim=0).sum(dim=0)
#         x_out = self.out_norm(x_out).permute(0, 3, 1, 2) # [b, c, h, w]
#
#         return x_out

# This class and its supporting functions below lightly adapted from the ViTDet backbone available at: https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/vit.py # noqa
class FusePro_Select(nn.Module):
    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        share_token_num: int = 1,
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
        num_model: int = 3,
        parallel_type: str = 'share_token',
        cfg = None,
    ) -> None:
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            global_attn_indexes (list): Indexes for blocks using global attention.
        """
        super().__init__()
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.num_model = num_model
        self.depth = depth
        self.cfg = cfg
        # self.pos_embed: Optional[nn.Parameter] = None
        # if use_abs_pos:
        #     # Initialize absolute positional embedding with pretrain image size.
        #     self.pos_embed = nn.Parameter(
        #         torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim)
        #     )

        # assert parallel_type in ['share_token','self_prompt','token_dict','act_token',
        #                          'sum','cat','cat_little','ca','sa']#for ablation
        if parallel_type == 'share_token':
            self.para_fuser = FusePro_Parallel_ShareToken(
                            img_size = img_size,
                            patch_size = patch_size,
                            in_chans = in_chans,
                            embed_dim = embed_dim,
                            depth = depth,
                            share_token_num = share_token_num,
                            num_heads = num_heads,
                            mlp_ratio = mlp_ratio,
                            out_chans = out_chans,
                            qkv_bias = qkv_bias,
                            norm_layer = norm_layer,
                            act_layer = act_layer,
                            use_rel_pos = use_rel_pos,
                            rel_pos_zero_init = rel_pos_zero_init,
                            window_size = window_size,
                            global_attn_indexes = global_attn_indexes,
                            num_model = num_model,
                            require_atten=self.cfg.component.require_atten
                            )
        # elif parallel_type == 'token_dict':
        #     self.para_fuser = FusePro_Parallel_TokenDict(
        #                     img_size=img_size,
        #                     patch_size=patch_size,
        #                     in_chans=in_chans,
        #                     embed_dim=embed_dim,
        #                     depth=depth,
        #                     share_token_num=share_token_num,
        #                     num_heads=num_heads,
        #                     mlp_ratio=mlp_ratio,
        #                     out_chans=out_chans,
        #                     qkv_bias=qkv_bias,
        #                     norm_layer=norm_layer,
        #                     act_layer=act_layer,
        #                     use_rel_pos=use_rel_pos,
        #                     rel_pos_zero_init=rel_pos_zero_init,
        #                     window_size=window_size,
        #                     global_attn_indexes=global_attn_indexes,
        #                     num_model=num_model,
        #                     )
        # elif parallel_type == 'act_token':
        #     self.para_fuser = FusePro_Parallel_ActToken(
        #                     img_size=img_size,
        #                     patch_size=patch_size,
        #                     in_chans=in_chans,
        #                     embed_dim=embed_dim,
        #                     depth=depth,
        #                     share_token_num=share_token_num,
        #                     num_heads=num_heads,
        #                     mlp_ratio=mlp_ratio,
        #                     out_chans=out_chans,
        #                     qkv_bias=qkv_bias,
        #                     norm_layer=norm_layer,
        #                     act_layer=act_layer,
        #                     use_rel_pos=use_rel_pos,
        #                     rel_pos_zero_init=rel_pos_zero_init,
        #                     window_size=window_size,
        #                     global_attn_indexes=global_attn_indexes,
        #                     num_model=num_model,
        #                     )
        # elif parallel_type == 'sum':
        #     self.para_fuser = FusePro_Abla_Sum(img_size=img_size)
        # elif parallel_type == 'cat':
        #     self.para_fuser = FusePro_Abla_Cat(num_model=num_model,
        #                                        embed_dim=embed_dim)
        # elif parallel_type == 'cat_little':
        #     self.para_fuser = FusePro_Abla_CatLittle(num_model=num_model,
        #                                        embed_dim=embed_dim)
        # elif parallel_type == 'ca':
        #     assert num_model == 2
        #     self.para_fuser = FusePro_Abla_CrossAttention(
        #                     img_size=img_size,
        #                     patch_size=patch_size,
        #                     in_chans=in_chans,
        #                     embed_dim=embed_dim,
        #                     depth=depth,
        #                     num_heads=num_heads,
        #                     mlp_ratio=mlp_ratio,
        #                     out_chans=out_chans,
        #                     qkv_bias=qkv_bias,
        #                     norm_layer=norm_layer,
        #                     act_layer=act_layer,
        #                     use_rel_pos=use_rel_pos,
        #                     rel_pos_zero_init=rel_pos_zero_init,
        #                     window_size=window_size,
        #                     global_attn_indexes=global_attn_indexes,
        #                     num_model=num_model,
        #                     )
        # elif parallel_type == 'sa':
        #     assert num_model == 2
        #     self.para_fuser = FusePro_Abla_SelfAttention(
        #                     img_size=img_size,
        #                     patch_size=patch_size,
        #                     in_chans=in_chans,
        #                     embed_dim=embed_dim,
        #                     depth=depth,
        #                     num_heads=num_heads,
        #                     mlp_ratio=mlp_ratio,
        #                     out_chans=out_chans,
        #                     qkv_bias=qkv_bias,
        #                     norm_layer=norm_layer,
        #                     act_layer=act_layer,
        #                     use_rel_pos=use_rel_pos,
        #                     rel_pos_zero_init=rel_pos_zero_init,
        #                     window_size=window_size,
        #                     global_attn_indexes=global_attn_indexes,
        #                     num_model=num_model,
        #                     )
        else:
            print(f'{parallel_type} not defined. Error.')
            raise ValueError


    def forward(self, xs: [torch.Tensor]) -> torch.Tensor:
        '''
        xs: List[x], x:[B,H,W,C]
        '''
        # x = self.patch_embed(x)  # pre embed: [1, 3, 1024, 1024], post embed: [1, 64, 64, 768]
        # if self.pos_embed is not None:
        #     x = x + self.pos_embed
        xs = self.para_fuser(xs)
        return xs


class ModelMetaInfo():
    def __init__(self, feat_scale: [Tuple], feat_chan: int, name:str=''):
        h, w = feat_scale
        self.meta = {'h': h, 'w': w, 'c': feat_chan, 'name':name}
    def obtain_info_list(self):
        h = self.meta['h']
        w = self.meta['w']
        c = self.meta['c']

        return h,w,c




class PreBlock(nn.Module):
    def __init__(self,meta: ModelMetaInfo,
                 out_chans: int,):
        super(PreBlock,self).__init__()
        self.meta = meta
        h_ori,w_ori,c_ori = self.meta.obtain_info_list()
        self.norm = LayerNorm2d(c_ori)
        self.proj = nn.Sequential(
            nn.Linear(c_ori, c_ori*2, bias=True),
            nn.GELU(),
            nn.Linear(c_ori*2, out_chans, bias=True),
        )

    def forward(self,x):
        B,_,h_ori,w_ori = x.shape
        x = self.norm(x)
        x = x.flatten(2).permute(0,2,1)
        x = self.proj(x)
        B = x.shape[0]
        x = x.permute(0,2,1).reshape(B,-1,h_ori,w_ori)

        return x

# This class and its supporting functions below lightly adapted from the ViTDet backbone available at: https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/vit.py # noqa
class FusePre(nn.Module):
    def __init__(
        self,
        model_meta_list: [ModelMetaInfo] = [],
        out_chans: int = 256,
    ) -> None:
        """
        Args:
        """
        super().__init__()
        self.out_chans = out_chans

        self.preblocks = nn.ModuleList()
        for idx in range(len(model_meta_list)):
            meta_ = model_meta_list[idx]
            preblock = PreBlock(meta=meta_,
                                out_chans = self.out_chans,)
            self.preblocks.append(preblock)
        self.num_model = len(model_meta_list)

    def forward(self, xs: [torch.Tensor]) -> torch.Tensor:
        for idx in range(self.num_model):
            preblk = self.preblocks[idx]
            xs[idx] = preblk(xs[idx])
        return xs

class FinBlock(nn.Module):
    def __init__(self,meta: ModelMetaInfo,
                 embed_dim: int,
                 out_chans: int,
                 out_size: int):
        super(FinBlock,self).__init__()
        self.meta = meta
        self.out_size = out_size
        h_ori,w_ori,c_ori = self.meta.obtain_info_list()
        self.embed_dim = embed_dim
        self.norm = LayerNorm2d(embed_dim)
        self.proj1 = MLPBlock(embedding_dim=embed_dim,mlp_dim=embed_dim//4)
        self.proj2 = nn.Linear(embed_dim, out_chans, bias=True)

    def forward(self,x):
        '''
        x: [B,H,W,C]
        out: [B,C,H,W]
        '''

        x = x.permute(0,3,1,2)
        h_ori,w_ori,_ = self.meta.obtain_info_list()
        x = F.interpolate(x,size=self.out_size,mode='bilinear')
        x = self.norm(x)
        x = x.flatten(2).permute(0,2,1)
        x = self.proj1(x)
        x = self.proj2(x)
        B = x.shape[0]
        h_out = w_out = self.out_size
        x = x.permute(0,2,1).reshape(B,-1,h_out,w_out)
        x = x.permute(0,2,3,1)

        return x

class FuseFin(nn.Module):
    def __init__(
        self,
        feat_size: int = 32,
        out_chans: int = 256,
        embed_dim: int = 768,
        model_meta_list: [ModelMetaInfo] = [],

    ) -> None:
        """
        Args:
            feat_size (int): feat size.
            out_chans (int): out channel.
            model_meta_list (list)
        """
        super().__init__()
        self.feat_size = feat_size
        self.out_chans = out_chans
        self.embed_dim = embed_dim
        self.blocks = nn.ModuleList()
        for idx in range(len(model_meta_list)):
            meta_ = model_meta_list[idx]
            block = FinBlock(meta=meta_,
                            embed_dim = self.embed_dim,
                            out_chans = self.out_chans,
                            out_size = self.feat_size)
            self.blocks.append(block)
        self.num_model = len(model_meta_list)

    def forward(self, xs: [torch.Tensor]) -> torch.Tensor:
        for idx in range(self.num_model):
            blk = self.blocks[idx]
            xs[idx] = blk(xs[idx])
        return xs

class FuseFin_Select(nn.Module):
    def __init__(self,
                 final_type: str,
                 embed_dim: int = 768,
                 out_chans: int = 256,
                 feat_size: int = 32,
                 model_meta_list: [ModelMetaInfo] = [],
                 ):
        super().__init__()
        assert final_type in ['sum','cat']
        if final_type == 'sum':
            self.fin_fuser = FuseFin_Sum(embed_dim=embed_dim,
                                         out_chans=out_chans,
                                         feat_size=feat_size,
                                         model_meta_list=model_meta_list)
        # elif final_type == 'cat':
        #     self.fin_fuser = FuseFin_Cat(embed_dim=embed_dim,
        #                                  out_chans=out_chans,
        #                                  feat_size=feat_size,
        #                                  model_meta_list=model_meta_list)
        else:
            print(f'fin type {final_type} not implemented.')
            raise ValueError

    def forward(self,xs: [torch.Tensor]) -> torch.Tensor:
        """
        xs: [tensor], all tensors in shape [b, h, w, c]
        out:
        x: [b, c, h, w]
        """
        x  = self.fin_fuser(xs)
        return x

class FuseFin_Sum(nn.Module):
    def __init__(self,
                 embed_dim: int = 768,
                 out_chans: int = 256,
                 feat_size: int = 32,
                 model_meta_list: [ModelMetaInfo] = [],
                 ):
        super().__init__()
        self.num_model = len(model_meta_list)
        self.fin_fuser1 = FuseFin(feat_size=feat_size,
                                  out_chans=out_chans,
                                  embed_dim=embed_dim,
                                  model_meta_list = model_meta_list)
        self.fin_fuser2 = nn.Sequential(
            nn.Conv2d(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GELU(),
            nn.Conv2d(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
        )
    def forward(self,xs: [torch.Tensor]) -> torch.Tensor:
        """
        xs: [tensor], all tensors in shape [b, h, w, c]
        out:
        x: [b, c, h, w]
        """
        N = len(xs)
        assert N == self.num_model
        xs = self.fin_fuser1(xs)
        x = torch.stack(xs,dim=0).sum(dim=0)
        x = x.permute(0,3,1,2)
        x = self.fin_fuser2(x)  # [b, c, h, w], [1, 256, 64, 64]

        return x

# class FuseFin_Cat(nn.Module):
#     def __init__(self,
#                  embed_dim: int = 768,
#                  out_chans: int = 256,
#                  feat_size: int = 32,
#                  model_meta_list: [ModelMetaInfo] = [],
#                  ):
#         super().__init__()
#         self.num_model = len(model_meta_list)
#         self.fin_fuser1 = FuseFin(feat_size=feat_size,
#                                   out_chans=out_chans,
#                                   embed_dim=embed_dim,
#                                   model_meta_list = model_meta_list)
#         self.proj = nn.Linear(out_chans * self.num_model, out_chans, bias=False)  # 从 c -> c/r
#         self.fin_fuser2 = nn.Sequential(
#             nn.Conv2d(
#                 out_chans,
#                 out_chans,
#                 kernel_size=3,
#                 padding=1,
#                 bias=False,
#             ),
#             nn.GELU(),
#             nn.Conv2d(
#                 out_chans,
#                 out_chans,
#                 kernel_size=3,
#                 padding=1,
#                 bias=False,
#             ),
#             LayerNorm2d(out_chans),
#         )
#     def forward(self,xs: [torch.Tensor]) -> torch.Tensor:
#         """
#         xs: [tensor], all tensors in shape [b, h, w, c]
#         out:
#         x: [b, c, h, w]
#         """
#         N = len(xs)
#         assert N == self.num_model
#         xs = self.fin_fuser1(xs)
#         x = torch.cat(xs,dim=3)
#         x = self.proj(x)
#         x = x.permute(0,3,1,2)
#         x = self.fin_fuser2(x)  # [b, c, h, w], [1, 256, 64, 64]
#
#         return x

