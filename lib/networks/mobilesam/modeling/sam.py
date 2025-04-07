# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import Any, Dict, List, Tuple, Union

from .image_encoder import ImageEncoderViT

from .image_encoder import ImageEncoderViTMS
from .mask_decoder import MaskDecoder_IterSimple
from .prompt_encoder import PromptEncoderIter
from .tiny_vit_sam import TinyViT
class SAM_Iter_Simple(nn.Module):
    # mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: Union[ImageEncoderViTMS, TinyViT],
        prompt_encoder: PromptEncoderIter,
        mask_decoder_iter: MaskDecoder_IterSimple,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
        cfg = None,
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder_iter
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        self.image_size = cfg.img_size

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    def forward(self, batched_input, mode = 'train', require_embed=False, ext_embed=None):
        if mode == 'vis':
            outputs = self.forward_train(batched_input, mode=mode, require_embed=require_embed, ext_embed=ext_embed)
        else:
            outputs = self.forward_train(batched_input, mode=mode, require_embed=require_embed, ext_embed=ext_embed)

        return outputs

    def forward_train(self, batched_input, mode='train', require_embed=False, ext_embed=None):
        image_batch = batched_input['image'].cuda()
        input_images = self.preprocess(image_batch)
        image_embeddings = self.image_encoder(input_images)
        if ext_embed is not None:
            image_embeddings = ext_embed
        out_dict = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            prompt_enc=self.prompt_encoder,
            batch_data = batched_input,
            mode = mode,
        )
        out_dict = self.postprocess_out_dict(
            out_dict,
            tar_size=(self.image_size,self.image_size),
        )
        masks_use = out_dict['high_res_logits_list'][-1]
        outputs = {'masks': masks_use,
                   'iou_predictions': None,
                   'high_res_logits_list': out_dict['high_res_logits_list'],
                   }
        outputs.update({'sem_mask':outputs['masks']})

        if require_embed:
            outputs.update({'feats':image_embeddings})

        return outputs

    def postprocess_out_dict(self,out_dict:torch.Tensor,tar_size:Tuple[int, ...]):
        low_res_logits_list = out_dict['low_res_logits_list']
        new_list = []
        for idx in range(len(low_res_logits_list)):
            high_res_logits = self.postprocess_masks(low_res_logits_list[idx],tar_size=tar_size)
            new_list.append(high_res_logits)
        out_dict.update({'high_res_logits_list':new_list})
        return out_dict

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        tar_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(masks, tar_size, mode="bilinear", align_corners=False)
        return masks


    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x