# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from typing import List
from .modeling import TwoWayTransformer, TinyViT
from .modeling import SAM_Iter_Simple, PromptEncoderIter, MaskDecoder_IterSimple

def build_sam_vit_t(image_size, num_classes, pixel_mean=[123.675, 116.28, 103.53], pixel_std=[58.395, 57.12, 57.375],checkpoint=None,
                    cfg=None,):
    prompt_embed_dim = 256
    image_size = image_size
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    mobile_sam = SAM_Iter_Simple(
            image_encoder=TinyViT(img_size=image_size, image_embedding_size=image_embedding_size, in_chans=3, num_classes=num_classes,
                embed_dims=[64, 128, 160, 320],
                depths=[2, 2, 6, 2],
                num_heads=[2, 4, 5, 10],
                window_sizes=[7, 7, 14, 7],
                mlp_ratio=4.,
                drop_rate=0.,
                drop_path_rate=0.0,
                use_checkpoint=False,
                mbconv_expand_ratio=4.0,
                local_conv_size=3,
                layer_lr_decay=0.8
            ),
            prompt_encoder=PromptEncoderIter(
                embed_dim=prompt_embed_dim,
                image_embedding_size=(image_embedding_size, image_embedding_size),
                input_image_size=(image_size, image_size),
                num_class=num_classes,
                mask_in_chans=16,
                cfg=cfg,
            ),
            mask_decoder_iter=MaskDecoder_IterSimple(
                num_mask_tokens=num_classes,
                transformer=TwoWayTransformer(
                    depth=2,
                    embedding_dim=prompt_embed_dim,
                    mlp_dim=2048,
                    num_heads=8,
                ),
                transformer_dim=prompt_embed_dim,
                cfg=cfg,
            ),
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
            cfg = cfg,
        )

    mobile_sam.eval()
    if checkpoint is not None and checkpoint.lower() != 'none':
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        mobile_sam.load_state_dict(state_dict)
    return mobile_sam


sam_model_registry = {
    "vit_t": build_sam_vit_t,
}


def check_key_part_in_list(whole:str,part_list:List[str])->bool:
    '''
    whole: eg. prompt_encoder.pe_layer.4
    want_list: list of names as part of whole, eg. ['pe_layer','upscaling']
    return: bool,True if want_list has at least one element in whole
    '''
    for item in part_list:
        if item in whole:
            return True
    return False
def get_network(*args,**kwargs):
    vit_name = kwargs['vit_name']
    img_size = kwargs['img_size']
    num_classes = kwargs['num_classes']
    ckpt = kwargs['ckpt']
    sam, img_embedding_size = sam_model_registry[vit_name](image_size=img_size,
                                                                num_classes=num_classes,
                                                                checkpoint=ckpt, pixel_mean=[0, 0, 0],
                                                                pixel_std=[1, 1, 1])
    network = sam
    return network
