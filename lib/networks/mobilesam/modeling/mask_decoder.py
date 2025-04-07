# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from typing import List, Tuple, Type

from .common import LayerNorm2d



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
        self.num_cls = cfg.num_classes

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
        if mode == 'train':
            masks = masks

        B = feats.shape[0]
        masks = masks.detach()
        dense_embedding = prompt_enc.obtain_mask_embedding(masks, B)

        # Expand per-image data in batch direction to be per-mask
        src = feats + dense_embedding
        pos_src = image_pe.repeat_interleave(int(tokens.shape[0]), dim=0)
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
        updated_masks += masks
        return updated_masks

    def init_masks(self, prompt_enc,batch_data, use_gt=False):
        b,_,h,w = batch_data['image'].shape
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

