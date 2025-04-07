# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from icecream import ic

from typing import List, Tuple, Type

from .common import LayerNorm2d
from .transformer import Attention
from torchvision.transforms import Resize as torch_resize
import cv2


class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        tranformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        # Select the correct mask or masks for output
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]

        # Prepare output
        return masks, iou_pred

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0) #[1+3,C]
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1) #[B,1+3,C],sparse:[B,Nprompt,C]
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1) # [B,1+3+Nprompt,C]

        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0) # [B,C,H,W] due to no prompt, B of tokens = 1
        src = src + dense_prompt_embeddings# [B,C,H,W]
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)# [B,C,H,W]
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens) # hs:tokens,[B,C,Nprompt]; src: img_embedding,[B,C,HW]
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w) #[B,C,H,W]
        upscaled_embedding = self.output_upscaling(src)#[B,C,4H,4W]
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)  # [b, Nmask,  C]

        b, c, h, w = upscaled_embedding.shape  # [B, C, 4H, 4W]
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)  # [1, 4, 256, 256], 256 = 4 * 64, the size of image embeddings

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred

class MaskDecoderLQ(nn.Module):
    def __init__(
        self,
        transformer_dim: int,
        transformer: nn.Module,
        num_mask_tokens: int,
        activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        tranformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.num_mask_tokens = num_mask_tokens

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        prompt_enc: nn.Module,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        masks = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            prompt_enc=prompt_enc,
        )

        return masks

    def predict_masks(
        self,
        image_embeddings: torch.Tensor, # [B,C,H,W], guess
        image_pe: torch.Tensor,
        prompt_enc: torch.nn.Module,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        cls_embeddings = prompt_enc.obtain_cls_embedding_lq()
        scale_embedding = prompt_enc.lq_embed
        B = image_embeddings.shape[0]
        dense_embedding = prompt_enc.obtain_mask_embedding(None,B)
        # Concatenate output tokens
        output_tokens = scale_embedding.weight + cls_embeddings
        tokens = output_tokens.unsqueeze(0).expand(B,-1,-1) #[B,Ncls,C]

        # output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0) #[1+3,C]
        # output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1) #[B,1+3,C],sparse:[B,Nprompt,C]
        # tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1) # [B,1+3+Nprompt,C]

        # Expand per-image data in batch direction to be per-mask
        src = image_embeddings
        src = src + dense_embedding
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        mask_tokens_out = hs

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)  # [b, c, token_num]

        b, c, h, w = upscaled_embedding.shape  # [h, token_num, h, w]
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)  # [1, 4, 256, 256], 256 = 4 * 64, the size of image embeddings

        return masks

class MaskDecoderHQ(nn.Module):
    def __init__(
        self,
        transformer_dim: int,
        num_mask_tokens: int,
        encoder_embed_dim: int,
        num_iter: int,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        tranformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.num_mask_tokens = num_mask_tokens
        self.num_iter = num_iter
        vit_dim = encoder_embed_dim
        self.compress_vit_feat = nn.Sequential(
            nn.ConvTranspose2d(vit_dim, transformer_dim, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim),
            nn.GELU(),
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 8, kernel_size=2, stride=2))

        self.embedding_encoder = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            nn.GELU(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
        )
        self.embedding_maskfeature = nn.Sequential(
                                        nn.Conv2d(transformer_dim // 8, transformer_dim // 4, 3, 1, 1),
                                        LayerNorm2d(transformer_dim // 4),
                                        nn.GELU(),
                                        nn.Conv2d(transformer_dim // 4, transformer_dim // 8, 3, 1, 1))
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            nn.GELU(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            nn.GELU(),
        )

        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

    def forward_once(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        mask: Tuple[torch.Tensor, None],
        prompt_enc: nn.Module,
        interm_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        mask = mask.detach()
        vit_features = interm_embeddings[0].permute(0, 3, 1, 2)  # early-layer ViT feature, after 1st global attention block in ViT
        hq_features = self.embedding_encoder(image_embeddings) + self.compress_vit_feat(vit_features)
        # extract from prompt
        cls_embeddings = prompt_enc.obtain_cls_embedding_hq()
        scale_embedding = prompt_enc.hq_embed
        B = image_embeddings.shape[0]
        dense_embedding = prompt_enc.obtain_mask_embedding(mask,B)
        # Concatenate output tokens
        output_tokens = scale_embedding.weight + cls_embeddings
        tokens = output_tokens.unsqueeze(0).expand(B,-1,-1) #[B,Ncls,C]
        Ntoken = tokens.shape[1]

        src = image_embeddings + image_pe + dense_embedding
        upscaled_embedding_sam = self.output_upscaling(src)
        upscaled_embedding_ours = self.embedding_maskfeature(upscaled_embedding_sam) + hq_features

        hyper_in_list: List[torch.Tensor] = []
        for i in range(Ntoken):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](tokens[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)  # [b, c, token_num]

        b, c, h, w = upscaled_embedding_ours.shape  # [h, token_num, h, w]
        masks = (hyper_in @ upscaled_embedding_ours.view(b, c, h * w)).view(b, -1, h, w)  # [1, 4, 256, 256], 256 = 4 * 64, the size of image embeddings

        # Prepare output
        return masks

    def forward(self,
                image_embeddings: torch.Tensor,
                image_pe: torch.Tensor,
                masks: Tuple[torch.Tensor, None],
                prompt_enc: nn.Module,
                interm_embeddings: torch.Tensor,
                ) -> Tuple[torch.Tensor]:
        masks_list = []
        for i in range(self.num_iter):
            masks = self.forward_once(image_embeddings,
                    image_pe,
                    masks,
                    prompt_enc,
                    interm_embeddings)
            masks_list.append(masks)
        return masks_list

class MaskDecoderHQ_KNet(nn.Module):
    def __init__(
        self,
        transformer_dim: int,
        num_mask_tokens: int,
        encoder_embed_dim: int,
        num_iter: int,
        use_inconf_roi: bool = True,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        tranformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.num_mask_tokens = num_mask_tokens
        self.num_iter = num_iter
        vit_dim = encoder_embed_dim
        self.use_inconf_roi = use_inconf_roi
        self.inconf_range = [0.3, 0.7]
        self.compress_vit_feat = nn.Sequential(
            nn.ConvTranspose2d(vit_dim, transformer_dim, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim),
            nn.GELU(),
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 8, kernel_size=2, stride=2))

        self.embedding_encoder = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            nn.GELU(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
        )

        self.fuse_encoder = nn.Sequential(
                                        nn.Conv2d(transformer_dim // 8, transformer_dim // 4, 3, 1, 1),
                                        LayerNorm2d(transformer_dim // 4),
                                        nn.GELU(),
                                        nn.Conv2d(transformer_dim // 4, transformer_dim // 4, 3, 1, 1))

        self.embedding_maskfeature = nn.Sequential(
                                        nn.Conv2d(transformer_dim // 4, transformer_dim // 4, 3, 1, 1),
                                        LayerNorm2d(transformer_dim // 4),
                                        nn.GELU(),
                                        nn.Conv2d(transformer_dim // 4, transformer_dim // 4, 3, 1, 1))

        self.kernel_updator = KernelUpdator(in_channels=transformer_dim,
                             feat_channels=transformer_dim // 4,
                             gate_sigmoid=True,)
        self.proto_proj = nn.Linear(transformer_dim // 4, transformer_dim, 1)
        self.self_atten = Attention(embedding_dim=transformer_dim, num_heads=8, downsample_rate=2)

        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 4, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

    def token_update(self,
                      mask: torch.Tensor, # [B,Ncls,H,W]
                      tokens: torch.Tensor, # [B,Ncls,C]
                      hrsfeat: torch.Tensor, # [B,C,H,W]
                     ):
        '''
        similar process as KNet
        '''
        B,Ncls,H,W = mask.shape
        sigmoid_mask = torch.sigmoid(mask)
        proto_mask = torch.einsum('bnhw,bchw->bnc', sigmoid_mask, hrsfeat) #[B,Ncls,C]
        proto_mask = self.proto_proj(proto_mask)
        update_tokens = self.kernel_updator(proto_mask,tokens)
        return update_tokens

    def forward_once(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        mask: Tuple[torch.Tensor, None],
        prompt_enc: nn.Module,
        interm_embeddings: torch.Tensor,
        mask_upsample=False,
    ) -> Tuple[torch.Tensor,torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.
        if use_inconf: generated mask is refine_roi, else refine_roi = highres_mask

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        mask = mask.detach()
        if self.use_inconf_roi:
            mask_bool = (mask<self.inconf_range[-1]) & (mask>self.inconf_range[0])
            mask = mask * mask_bool
        vit_features = interm_embeddings[0].permute(0, 3, 1, 2)  # early-layer ViT feature, after 1st global attention block in ViT
        hq_features = self.embedding_encoder(image_embeddings+image_pe) + self.compress_vit_feat(vit_features)
        hq_features = self.fuse_encoder(hq_features) #->transpose upscale
        B = image_embeddings.shape[0]
        dense_embedding = prompt_enc.obtain_mask_embedding(mask, B)
        src = hq_features + dense_embedding
        upscaled_embedding_ours = self.embedding_maskfeature(src) + src # back to origin size

        # extract from prompt
        cls_embeddings = prompt_enc.obtain_cls_embedding_hq()
        scale_embedding = prompt_enc.hq_embed
        # Concatenate output tokens
        output_tokens = scale_embedding.weight + cls_embeddings
        tokens = output_tokens.unsqueeze(0).expand(B,-1,-1) #[B,Ncls,C]
        Ntoken = tokens.shape[1]

        tokens = self.token_update(mask=mask,
                          tokens=tokens,
                          hrsfeat=upscaled_embedding_ours)

        tokens = self.self_atten(tokens,tokens,tokens)

        hyper_in_list: List[torch.Tensor] = []
        for i in range(Ntoken):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](tokens[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)  # [b, c, token_num]

        b, c, h, w = upscaled_embedding_ours.shape  # [h, token_num, h, w]
        masks = (hyper_in @ upscaled_embedding_ours.view(b, c, h * w)).view(b, -1, h, w)  # [1, 4, 256, 256], 256 = 4 * 64, the size of image embeddings

        # Prepare output
        if self.use_inconf_roi:
            return masks, mask_bool
        else:
            return masks, torch.ones_like(mask,dtype=torch.bool)

    def obtain_highres_logits(self, masks: torch.Tensor,
                              hq_refine_logits: torch.Tensor,
                              masks_roi: torch.Tensor):
        # method1: replace
        masks1 = hq_refine_logits*masks_roi + masks*torch.logical_not(masks_roi)
        # method2: max
        # masks2 = torch.max(torch.stack([masks,hq_refine_logits],dim=-1),dim=-1)[0]
        highres_logits = masks1

        return highres_logits

    def forward(self,
                image_embeddings: torch.Tensor,
                image_pe: torch.Tensor,
                masks: Tuple[torch.Tensor, None],
                prompt_enc: nn.Module,
                interm_embeddings: torch.Tensor,
                ) -> Tuple[List,List]:
        masks_roi_list = []
        hq_refine_logits_list = []
        high_res_logits_list = []
        for i in range(self.num_iter):
            hq_refine_logits, masks_roi = self.forward_once(image_embeddings,
                                      image_pe,
                                      masks, # the whole segmentation results
                                      prompt_enc,
                                      interm_embeddings)
            hres_logits = self.obtain_highres_logits(masks=masks,
                                       hq_refine_logits=hq_refine_logits,
                                       masks_roi=masks_roi)
            masks = hres_logits
            hq_refine_logits_list.append(hq_refine_logits)
            high_res_logits_list.append(hres_logits)
            masks_roi_list.append(masks_roi)
        hq_dict = {'hq_refine_logits_list':hq_refine_logits_list,
                   'masks_roi_list':masks_roi_list,
                   'high_res_logits_list':high_res_logits_list}
        return hq_dict

class MaskDecoderHQ_KNet_hres(nn.Module):
    def __init__(
        self,
        transformer_dim: int,
        num_mask_tokens: int,
        encoder_embed_dim: int,
        num_iter: int,
        use_inconf_roi: bool = True,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        tranformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.num_mask_tokens = num_mask_tokens
        self.num_iter = num_iter
        vit_dim = encoder_embed_dim
        self.use_inconf_roi = use_inconf_roi
        self.inconf_range = [0.3, 0.7]
        self.compress_vit_feat = nn.Sequential(
            nn.ConvTranspose2d(vit_dim, transformer_dim, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim),
            nn.GELU(),
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 8, kernel_size=2, stride=2))

        self.embedding_encoder = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            nn.GELU(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
        )

        self.fuse_encoder = nn.Sequential(
                                        nn.Conv2d(transformer_dim // 8, transformer_dim // 4, 3, 1, 1),
                                        LayerNorm2d(transformer_dim // 4),
                                        nn.GELU(),
                                        nn.Conv2d(transformer_dim // 4, transformer_dim // 4, 3, 1, 1))

        self.mask_donwsample_conv = nn.Sequential(
                                        nn.Conv2d(transformer_dim // 4, transformer_dim // 4, 3, 2, 1),
                                        LayerNorm2d(transformer_dim // 4),
                                        nn.GELU(),
                                        nn.Conv2d(transformer_dim // 4, transformer_dim // 4, 3, 2, 1))

        self.embedding_maskfeature = nn.Sequential(
                                        nn.Conv2d(transformer_dim // 4, transformer_dim // 4, 3, 1, 1),
                                        LayerNorm2d(transformer_dim // 4),
                                        nn.GELU(),
                                        nn.Conv2d(transformer_dim // 4, transformer_dim // 4, 3, 1, 1))

        self.src_upsample_conv = nn.Sequential(
                                        nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 4, kernel_size=2, stride=2),
                                        LayerNorm2d(transformer_dim // 4),
                                        nn.GELU(),
                                        nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 4, kernel_size=2, stride=2))

        self.kernel_updator = KernelUpdator(in_channels=transformer_dim,
                             feat_channels=transformer_dim // 4,
                             gate_sigmoid=True,)
        self.proto_proj = nn.Linear(transformer_dim // 4, transformer_dim, 1)
        self.self_atten = Attention(embedding_dim=transformer_dim, num_heads=8, downsample_rate=2)

        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 4, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

    def token_update(self,
                      mask: torch.Tensor, # [B,Ncls,H,W]
                      tokens: torch.Tensor, # [B,Ncls,C]
                      hrsfeat: torch.Tensor, # [B,C,H,W]
                     ):
        '''
        similar process as KNet
        '''
        B,Ncls,H,W = mask.shape
        sigmoid_mask = torch.sigmoid(mask)
        proto_mask = torch.einsum('bnhw,bchw->bnc', sigmoid_mask, hrsfeat) #[B,Ncls,C]
        proto_mask = self.proto_proj(proto_mask)
        update_tokens = self.kernel_updator(proto_mask,tokens)
        return update_tokens

    def forward_once(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        mask: Tuple[torch.Tensor, None],
        prompt_enc: nn.Module,
        interm_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor,torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.
        if use_inconf: generated mask is refine_roi, else refine_roi = highres_mask

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        mask = mask.detach()
        if self.use_inconf_roi:
            mask_bool = (mask<self.inconf_range[-1]) & (mask>self.inconf_range[0])
            mask = mask * mask_bool
        vit_features = interm_embeddings[0].permute(0, 3, 1, 2)  # early-layer ViT feature, after 1st global attention block in ViT
        hq_features = self.embedding_encoder(image_embeddings+image_pe) + self.compress_vit_feat(vit_features)
        hq_features = self.fuse_encoder(hq_features) #->transpose upscale

        B = image_embeddings.shape[0]
        dense_embedding = prompt_enc.obtain_mask_embedding(mask, B)
        dense_embedding = self.mask_donwsample_conv(dense_embedding)
        src = hq_features + dense_embedding
        # src = self.embedding_maskfeature(src) + src # back to origin size
        upscaled_embedding_ours = self.src_upsample_conv(src)

        # extract from prompt
        cls_embeddings = prompt_enc.obtain_cls_embedding_hq()
        scale_embedding = prompt_enc.hq_embed
        # Concatenate output tokens
        output_tokens = scale_embedding.weight + cls_embeddings
        tokens = output_tokens.unsqueeze(0).expand(B,-1,-1) #[B,Ncls,C]
        Ntoken = tokens.shape[1]

        tokens = self.token_update(mask=mask,
                             tokens=tokens,
                          hrsfeat=upscaled_embedding_ours)

        tokens = self.self_atten(tokens,tokens,tokens)

        hyper_in_list: List[torch.Tensor] = []
        for i in range(Ntoken):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](tokens[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)  # [b, c, token_num]

        b, c, h, w = upscaled_embedding_ours.shape  # [h, token_num, h, w]
        masks = (hyper_in @ upscaled_embedding_ours.view(b, c, h * w)).view(b, -1, h, w)  # [1, 4, 256, 256], 256 = 4 * 64, the size of image embeddings

        # Prepare output
        if self.use_inconf_roi:
            return masks, mask_bool
        else:
            return masks, torch.ones_like(mask,dtype=torch.bool)

    def obtain_highres_logits(self, masks: torch.Tensor,
                              hq_refine_logits: torch.Tensor,
                              masks_roi: torch.Tensor):
        # method1: replace
        masks1 = hq_refine_logits*masks_roi + masks*torch.logical_not(masks_roi)
        # method2: max
        # masks2 = torch.max(torch.stack([masks,hq_refine_logits],dim=-1),dim=-1)[0]
        highres_logits = masks1

        return highres_logits

    def forward(self,
                image_embeddings: torch.Tensor,
                image_pe: torch.Tensor,
                masks: Tuple[torch.Tensor, None],
                prompt_enc: nn.Module,
                interm_embeddings: torch.Tensor,
                ) -> Tuple[List,List]:
        masks_roi_list = []
        hq_refine_logits_list = []
        high_res_logits_list = []
        h,w = masks.shape[-2:]
        upfactor = 4
        masks = F.interpolate(masks,
            (h*upfactor, w*upfactor),
            mode="bilinear",
            align_corners=False,
        )
        for i in range(self.num_iter):
            hq_refine_logits, masks_roi = self.forward_once(image_embeddings,
                                                image_pe,
                                                masks, # the whole segmentation results
                                                prompt_enc,
                                                interm_embeddings,)
            hres_logits = self.obtain_highres_logits(masks=masks,
                                       hq_refine_logits=hq_refine_logits,
                                       masks_roi=masks_roi)
            masks = hres_logits
            hq_refine_logits_list.append(hq_refine_logits)
            high_res_logits_list.append(hres_logits)
            masks_roi_list.append(masks_roi)
        hq_dict = {'hq_refine_logits_list':hq_refine_logits_list,
                   'masks_roi_list':masks_roi_list,
                   'high_res_logits_list':high_res_logits_list}
        return hq_dict

# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x

class KernelUpdator(nn.Module):

    def __init__(self,
                 in_channels=256,
                 feat_channels=64,
                 gate_sigmoid=True,
                 gate_norm_act=False,
                 activate_out=False,):
        super(KernelUpdator, self).__init__()
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.gate_sigmoid = gate_sigmoid
        self.gate_norm_act = gate_norm_act
        self.activate_out = activate_out
        norm_func = nn.LayerNorm
        self.out_channels = in_channels

        self.num_params_in = self.feat_channels
        self.num_params_out = self.feat_channels
        self.dynamic_layer = nn.Linear(
            self.in_channels, self.num_params_in + self.num_params_out)
        self.input_layer = nn.Linear(self.in_channels,
                                     self.num_params_in + self.num_params_out,
                                     1)
        self.input_gate = nn.Linear(self.feat_channels, self.feat_channels, 1)
        self.update_gate = nn.Linear(self.feat_channels, self.feat_channels, 1)
        if self.gate_norm_act:
            self.gate_norm = norm_func(self.feat_channels)

        self.norm_in = norm_func(self.feat_channels)
        self.norm_out = norm_func(self.feat_channels)
        self.input_norm_in = norm_func(self.feat_channels)
        self.input_norm_out = norm_func(self.feat_channels)

        self.fc_layer = nn.Linear(self.feat_channels, self.out_channels, 1)
        self.fc_norm = norm_func(self.out_channels)
        self.activation = nn.ReLU(inplace=True)


    def forward(self, update_feature, input_feature):
        '''
        update_feature: from mask, [B,Ncls,C]
        input_feature: origin prototype, [B,Ncls,C]
        '''
        B, Ncls = update_feature.shape[:2]
        parameters = self.dynamic_layer(update_feature)
        param_in = parameters[..., :self.num_params_in]
        param_out = parameters[..., -self.num_params_out:]

        input_feats = self.input_layer(input_feature)
        input_in = input_feats[..., :self.num_params_in]
        input_out = input_feats[..., -self.num_params_out:]
        # generate gate
        gate_feats = input_in * param_in
        if self.gate_norm_act: # default False
            gate_feats = self.activation(self.gate_norm(gate_feats))

        input_gate = self.input_norm_in(self.input_gate(gate_feats))
        update_gate = self.norm_in(self.update_gate(gate_feats))
        if self.gate_sigmoid:
            input_gate = input_gate.sigmoid()
            update_gate = update_gate.sigmoid()
        # generate out
        param_out = self.norm_out(param_out)
        input_out = self.input_norm_out(input_out)

        if self.activate_out: # default False
            param_out = self.activation(param_out)
            input_out = self.activation(input_out)

        features = update_gate * param_out + input_gate * input_out

        features = self.fc_layer(features)
        features = self.fc_norm(features)
        features = self.activation(features)

        return features

class MaskDecoder_Iter(nn.Module):
    def __init__(
        self,
        transformer_dim: int,
        transformer: nn.Module,
        num_mask_tokens: int,
        activation: Type[nn.Module] = nn.GELU,
        iter = 3,
        cfg = None
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        tranformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.num_mask_tokens = num_mask_tokens
        self.iter = iter
        self.cfg = cfg

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )
        self.proto_proj = nn.Linear(transformer_dim // 8, transformer_dim, 1)
        # self.self_atten = Attention(embedding_dim=transformer_dim, num_heads=8, downsample_rate=2)
        self.kernel_updator = KernelUpdator(in_channels=transformer_dim,
                                            feat_channels=transformer_dim // 4,
                                            gate_sigmoid=True, )
        self.init_upsampler = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim, kernel_size=2, stride=2),
            activation(),
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        prompt_enc: nn.Module,
        batch_data: dict,
        mode = 'train',
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        out_dict = self.predict_masks_iter(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            prompt_enc=prompt_enc,
            batch_data = batch_data,
            mode = mode,
        )

        return out_dict

    def token_update(self,
                      mask: torch.Tensor, # [B,Ncls,H1,W1]
                      tokens: torch.Tensor, # [B,Ncls,C]
                      hrsfeat: torch.Tensor, # [B,C,H2,W2]
                     ):
        '''
        similar process as KNet
        '''
        B,Ncls,H,W = mask.shape
        act_mask = torch.sigmoid(mask)
        # act_mask = torch.softmax(mask, dim=1)
        proto_mask = torch.einsum('bnhw,bchw->bnc', act_mask, hrsfeat) #[B,Ncls,C]
        proto_mask = self.proto_proj(proto_mask)
        update_tokens = self.kernel_updator(proto_mask,tokens)
        return update_tokens

    def inner_loop(self, masks, tokens, feats, image_pe, prompt_enc, batch_data, mode='train'):
        '''
        in:
        masks: [B,Ncls,H,W], H,W=128, sigmoid
        tokens: [B,Ncls,C1], C1=256
        feats: [B,C1,H//4,W//4], C1=256
        return:
        the same shape
        '''
        if mode == 'train':
            gt_masks = batch_data['sem_mask'].cuda()
            masks = self.mask_logits_variant(masks, gt_masks)# not use actually
        B = feats.shape[0]
        dense_embedding = prompt_enc.obtain_mask_embedding(masks, B)

        # Expand per-image data in batch direction to be per-mask
        src = feats + dense_embedding
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        mask_tokens_out = hs
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        # kernel updator
        update_tokens = self.token_update(mask=masks, tokens=mask_tokens_out, hrsfeat=upscaled_embedding)

        # Upscale mask embeddings and predict masks using the mask tokens
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](update_tokens[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)  # [b, c, token_num]

        b, c, h, w = upscaled_embedding.shape  # [h, token_num, h, w]
        updated_masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h,
                                                                       w)
        updated_masks += masks
        updated_feats = src

        return updated_masks, update_tokens, updated_feats

    def init_pred(self, feats, prompt_enc):
        feats2 = self.init_upsampler(feats)
        # masks = prompt_enc.init_conv(feats.detach()) # dont require feats grad
        masks = prompt_enc.init_conv(feats2) # dont require feats grad
        # b,c,h,w = feats.shape
        # original_size = (4*h,4*w) #32->128
        # masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def predict_masks_iter(
        self,
        image_embeddings: torch.Tensor, # [B,C,H,W], guess
        image_pe: torch.Tensor,
        prompt_enc: torch.nn.Module,
        batch_data: dict,
        mode: str='train',
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        out_dict = {}
        high_res_logits_list = []

        cls_embeddings = prompt_enc.obtain_cls_embedding_hq()
        B = image_embeddings.shape[0]
        masks = self.init_pred(feats=image_embeddings,prompt_enc=prompt_enc)
        out_dict.update({'low_res_logits':masks})
        tokens = cls_embeddings.unsqueeze(0).expand(B, -1, -1)  # [B,Ncls,C]
        feats = image_embeddings
        for idx_iter in range(self.iter):
            # masks_sigm = torch.softmax(masks,dim=1)
            masks, tokens, feats = self.inner_loop(masks, tokens, feats,
                                                   prompt_enc=prompt_enc, image_pe=image_pe,
                                                   batch_data=batch_data,mode=mode)
            high_res_logits_list.append(masks)
        out_dict.update({'high_res_logits_list':high_res_logits_list})
        return out_dict

    def mask_logits_variant(self, masks, gt_masks):
        b,Ncls,h,w = masks.shape
        msk_feat = torch.nn.Dropout(p=0.1, inplace=True)(masks)
        gt_feat = torch_resize([h,w])(gt_masks).int()  # [B,C,H,W] #index as target class
        gt_feat = gt_feat.view(b, Ncls, h, w)#[B,5,H,W]
        # lab, cnts = torch.unique(gt_feat, sorted=True, return_counts=True)
        # unique = torch.stack((lab, cnts), dim=1)
        # unique_sorted, unique_ind = torch.sort(unique, dim=0)
        # more noise for tail
        # noise_list = [0.35,0.3,0.3,0.2,0.2,0.1,0.1,0,0]
        noise_list = self.cfg.component.noise_list
        for idx in range(Ncls):
            var = noise_list[idx]
            noise_mean = torch.mean(msk_feat[:,idx]).cuda()
            noise = torch.randn((msk_feat[:,idx].size())) * var
            noise = noise.cuda() + noise_mean
            msk_feat[:,idx,...][gt_feat[:,idx] == idx] = msk_feat[:,idx][gt_feat[:,idx] == idx] + \
                                                     noise[gt_feat[:,idx] == idx]
        return msk_feat

class MaskDecoder_IterSimple(nn.Module):
    def __init__(
        self,
        transformer_dim: int,
        transformer: nn.Module,
        num_mask_tokens: int,
        activation: Type[nn.Module] = nn.GELU,
        cfg = None
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        tranformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.num_mask_tokens = num_mask_tokens
        self.cfg = cfg

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

    def obtain_iter_num(self,mode):
        if mode == 'train':
            self.iter = self.cfg.component.train_iter
        else:
            self.iter = self.cfg.component.test_iter

        return self.iter


    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        prompt_enc: nn.Module,
        batch_data: dict,
        mode = 'train',
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """

        self.obtain_iter_num(mode)
        out_dict = self.predict_masks_iter(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            prompt_enc=prompt_enc,
            batch_data = batch_data,
            mode = mode,
        )

        return out_dict


    def inner_loop(self, masks, tokens, feats, image_pe, prompt_enc, batch_data, mode='train', iter_idx=0):
        '''
        in:
        masks: [B,Ncls,H,W], H,W=128, sigmoid
        tokens: [B,Ncls,C1], C1=256
        feats: [B,C1,H//4,W//4], C1=256
        return:
        the same shape
        '''
        #TODO: rewrite
        if mode == 'train':
            gt_masks = batch_data['sem_mask'].cuda()
            # if self.cfg.component.noise_version == 'v1':
            #     if iter_idx != 0:  # dont change init masks
            #         masks = self.mask_logits_variant_v1(masks, gt_masks)
            # elif self.cfg.component.noise_version == 'v2':
            #     if iter_idx != 0:  # dont change init masks
            #         masks = self.mask_logits_variant_v2(masks, gt_masks)
            # else:
            masks = masks
        # if self.cfg.component.noise_version == 'gt':
        #     gt_masks = batch_data['sem_mask'].cuda()
        #     if iter_idx == 0:  # only change init masks
        #         h,w = masks.shape[-2:]
        #         masks = torch_resize([h,w])(gt_masks)
        #         scale=3
        #         masks = (2*masks-1)*scale
            # for i in range(5):
            #     if i != 0:
            #         continue
            #     mask = masks[0,i,...]
            #     name = batch_data['case_name'][0]
            #     vis_variant_mask(mask,suffix=f'{name}_CLS{i}_iter{iter_idx}')
        # if mode=='train' and self.cfg.component.maskaug == True:
        #     masks_ori = masks.detach()
        #     masks = self.do_mask_aug(masks_ori)
        #     for idx_cls in range(5):
        #         # if idx_cls != 0:
        #         #     continue
        #         mask = masks[0,idx_cls,...]
        #         name = batch_data['case_name'][0]
        #         vis_variant_mask(mask,suffix=f'{name}_CLS{idx_cls}_aug_iter{iter_idx}')
        #         mask_ori = masks_ori[0,idx_cls,...]
        #         vis_variant_mask(mask_ori,suffix=f'{name}_CLS{idx_cls}_ori_iter{iter_idx}')

        B = feats.shape[0]
        masks = masks.detach()
        dense_embedding = prompt_enc.obtain_mask_embedding(masks, B)

        # Expand per-image data in batch direction to be per-mask
        src = feats + dense_embedding
        # src = feats
        pos_src = torch.repeat_interleave(image_pe, int(tokens.shape[0]), dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        mask_tokens_out = hs
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)

        # Upscale mask embeddings and predict masks using the mask tokens
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)  # [b, c, token_num]

        b, c, h, w = upscaled_embedding.shape  # [h, token_num, h, w]
        updated_masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h,w)
        # vis_cls_idx = [0,2,3,4]
        # for idx_cls in iter(vis_cls_idx):
        #     vis_mask = updated_masks[0,idx_cls,...]
        #     sample_name = batch_data['case_name'][0]
        #     vis_variant_mask(vis_mask,suffix=f'{sample_name}_cls{idx_cls}_iter{iter_idx}_1')
        # updated_masks = (2*torch.sigmoid(updated_masks)-1) * 1
        updated_masks += masks
        # vis_cls_idx = [0,2,3,4]
        # for idx_cls in iter(vis_cls_idx):
        #     vis_mask = updated_masks[0,idx_cls,...]
        #     sample_name = batch_data['case_name'][0]
        #     vis_variant_mask(vis_mask,suffix=f'{sample_name}_cls{idx_cls}_iter{iter_idx}_2')
        return updated_masks

    def init_masks(self, prompt_enc,batch_data, use_gt=False):
        mask_gt = batch_data['sem_mask']
        b,Ncls,h,w = mask_gt.shape
        masks = prompt_enc.obtain_init_masks(b,h//4,w//4)

        return masks

    def predict_masks_iter(
        self,
        image_embeddings: torch.Tensor, # [B,C,H,W], guess
        image_pe: torch.Tensor,
        prompt_enc: torch.nn.Module,
        batch_data: dict,
        mode: str='train',
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        out_dict = {}
        low_res_logits_list = []

        cls_embeddings = prompt_enc.obtain_cls_embedding()
        B = image_embeddings.shape[0]
        tokens = cls_embeddings.unsqueeze(0).expand(B, -1, -1)  # [B,Ncls,C]
        feats = image_embeddings
        masks = self.init_masks(prompt_enc,batch_data,use_gt=False)
        if mode == 'train':
            num_iter = self.cfg.component.train_iter
        else:
            num_iter = self.cfg.component.test_iter
        for idx_iter in range(num_iter):
            masks = self.inner_loop(masks, tokens, feats,
                                   prompt_enc=prompt_enc, image_pe=image_pe,
                                   batch_data=batch_data, mode=mode,
                                   iter_idx=idx_iter)
            low_res_logits_list.append(masks)
        out_dict.update({'low_res_logits_list':low_res_logits_list})
        return out_dict

    def mask_logits_variant_v1(self, masks, gt_masks):
        '''
        only add noise at target class layer
        gt_feat: [b,Ncls,h,w],[0,1]
        '''
        b,Ncls,h,w = masks.shape
        msk_feat = torch.nn.Dropout(p=0.0, inplace=True)(masks)
        gt_feat = torch_resize([h,w])(gt_masks).int()  # [B,C,H,W] #index as target class
        gt_feat = gt_feat.view(b, Ncls, h, w)#[B,5,H,W]

        noise_list = self.cfg.component.noise_list
        for idx in range(Ncls):
            var = noise_list[idx]
            noise_mean = torch.mean(msk_feat[:,idx]).cuda()
            noise = torch.randn((msk_feat[:,idx].size())) * var
            # noise = noise.cuda() + noise_mean
            noise = noise.cuda()
            mask_ = (gt_feat[:,idx] == 1)
            msk_feat[:,idx,...][mask_] = msk_feat[:,idx][mask_] + noise[mask_]
        return msk_feat

    def mask_logits_variant_v2(self, masks, gt_masks):
        '''
        add noise at all class layer
        gt_feat: [b,Ncls,h,w],[0,1]
        '''
        b,Ncls,h,w = masks.shape
        msk_feat = torch.nn.Dropout(p=0.0, inplace=True)(masks) #TODO:why True cannot use
        gt_feat = torch_resize([h,w])(gt_masks).int()  # [B,C,H,W] #index as target class
        gt_feat = gt_feat.view(b, Ncls, h, w)#[B,5,H,W]

        noise_list = self.cfg.component.noise_list
        for idx in range(Ncls):
            var = noise_list[idx]
            noise_mean = torch.mean(msk_feat[:,idx]).cuda()
            noise = torch.randn((msk_feat.size())) * var
            # noise = noise.cuda() + noise_mean
            noise = noise.cuda()
            mask_ = (gt_feat[:,idx,...]==1).unsqueeze(1).repeat([1,Ncls,1,1])
            msk_feat[mask_] = msk_feat[mask_] + noise[mask_]
        return msk_feat
    def do_mask_aug(self,masks):
        masks_ori = masks.clone()
        masks = masks.detach().cpu().numpy()
        B,C,H,W = masks.shape
        # form 1: random add or remove block
        canvas = np.zeros((B,C,H,W))
        v_list = [3,3,3,3,3] #v,s,t,b,c # TODO: equal mean
        box_size_list = [10,10,10,10,10] #v,s,t,b,c
        time1_max = 8 # aug time for each form
        def rand_my(min=0,max=0):
            '''
            return rand number by given max
            '''
            return float(min+np.random.rand()*(max-min))
        for idx_c in range(C):
            time1 = np.random.randint(0,time1_max)
            # time1 = 0
            for idx_t1 in range(time1):
                size_aug = rand_my(max=v_list[idx_c],min=-v_list[idx_c])
                # if np.random.random() > 0.5:
                #     size_aug = 5
                # else:
                #     size_aug = -5
                box_size = 2*int(rand_my(box_size_list[idx_c]))
                # box_size = box_size_list[idx_c]
                h_c = int(rand_my(max=H-1))
                w_c = int(rand_my(max=W-1))
                h_min = max(0,(h_c - box_size//2))
                h_max = min(H-1,(h_c + box_size//2))
                w_min = max(0, (w_c - box_size // 2))
                w_max = min(W - 1, (w_c + box_size // 2))
                p1 = (h_min,w_min)
                p2 = (h_max,w_max)
                for idx_B in range(B):
                    canvas_ = canvas[idx_B,idx_c,...]
                    canvas_ = draw_blocks(canvas_,size_aug,p1,p2)
                    canvas[idx_B, idx_c, ...] = canvas_
        canvas1 = canvas
        # form 2: random dilate or erode
        canvas = np.zeros((B,C,H,W))
        masks_bin = torch.sigmoid(masks_ori).detach().cpu().numpy()
        masks_bin = masks_bin > 0.5
        v_list = [3, 3, 3, 3, 3]  # v,s,t,b,c
        aug_size_list = [8,8,6,5,4] #v,s,t,b,c
        time1_max = 1# aug time for each form
        for idx_c in range(C):
            # time1 = np.random.randint(0,time1_max)
            time1 = 1
            for idx_t1 in range(time1):
                size_aug = rand_my(max=aug_size_list[idx_c],min=-aug_size_list[idx_c]) # pos for dilate, neg for erode
                # if np.random.random() > 0.5:
                #     size_aug = 10 # pos for dilate, neg for erode
                # else:
                #     size_aug = -10
                v_aug = rand_my(max=v_list[idx_c])
                # v_aug = 5
                for idx_B in range(B):
                    mask_bin_ = masks_bin[idx_B,idx_c,...]
                    canvas_ = canvas[idx_B,idx_c,...]
                    canvas_ = draw_edge_aug(canvas_,size_aug,mask_bin_,v_aug)
                    canvas[idx_B, idx_c, ...] = canvas_
        canvas2 = canvas
        # fues masks with all canvas
        masks = masks + canvas1 + canvas2
        masks = torch.Tensor(masks).type_as(masks_ori)
        return masks

def draw_blocks(canvas,value,p1,p2):
    '''
    canvas: numpy array, dtype: float, [H,W]
    p1: tuple, (h_min,w_min)
    p2: tuple, (h_max,w_max)
    '''
    import cv2
    cv2.rectangle(canvas,p1,p2,color=value,thickness=-1)
    return canvas

def draw_edge_aug(canvas, size_aug,mask,value):
    '''
    canvas: numpy array, dtype: float, [H,W]
    mask: numpy array, dtype: bool, [H,W]
    size_aug: abs for filter time, pos for dilate, neg for erode
    '''
    assert value >= 0
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,ksize=(3,3))
    num_iter = max(0,int(np.absolute(size_aug)))
    mask = mask.astype(np.uint8)*255
    if size_aug > 0:
        maskf = cv2.dilate(mask, k, iterations=num_iter)
    else:
        maskf = cv2.erode(mask, k, iterations=num_iter)
        value = -value
    mask_diff = np.logical_xor(mask>128,maskf>128)
    canvas += mask_diff * value

    return canvas

def vis_variant_mask(mask,suffix):
    mask = torch.sigmoid(mask)
    mask = mask.detach().cpu().numpy()
    out_dir = '/home/psilym/Downloads/temp_vis'
    img_name = f'img_{suffix}.jpg'
    import os.path as osp
    out_path = osp.join(out_dir,img_name)
    import PIL.Image as Im
    import numpy as np
    Im.fromarray((mask*255).astype(np.uint8)).save(out_path)
    print(f'Save image at {out_path}')

    return