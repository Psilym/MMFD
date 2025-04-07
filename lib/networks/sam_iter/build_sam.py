# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.nn import functional as F
from icecream import ic
from typing import List

from functools import partial

from .modeling import Sam,ImageEncoderViT, MaskDecoder, PromptEncoder,  TwoWayTransformer
from .modeling import SAM_MS, ImageEncoderViTMS, PromptEncoderMS

def build_sam_vit_h(image_size, num_classes, pixel_mean=[123.675, 116.28, 103.53], pixel_std=[58.395, 57.12, 57.375],
                    checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
        num_classes=num_classes,
        image_size=image_size,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std
    )


build_sam = build_sam_vit_h


def build_sam_vit_l(image_size, num_classes, pixel_mean=[123.675, 116.28, 103.53], pixel_std=[58.395, 57.12, 57.375],
                    checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
        num_classes=num_classes,
        image_size=image_size,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std
    )


def build_sam_vit_b(image_size, num_classes, pixel_mean=[123.675, 116.28, 103.53], pixel_std=[58.395, 57.12, 57.375],
                    checkpoint=None):
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        # adopt global attention at [3, 6, 9, 12] transform layer, else window attention layer
        checkpoint=checkpoint,
        num_classes=num_classes,
        image_size=image_size,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std
    )


def build_sam_vit_b_iter(image_size, num_classes, pixel_mean=[123.675, 116.28, 103.53], pixel_std=[58.395, 57.12, 57.375],
                    checkpoint=None,checkpoint_HQ=None,cfg=None):
    return _build_sam_iter(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        # adopt global attention at [3, 6, 9, 12] transform layer, else window attention layer
        checkpoint=checkpoint,
        checkpoint_HQ=checkpoint_HQ,
        num_classes=num_classes,
        image_size=image_size,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
        cfg=cfg,
    )


sam_model_registry = {
    "default": build_sam_vit_h,
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
    "vit_b_iter": build_sam_vit_b_iter,
}


def _build_sam(
        encoder_embed_dim,
        encoder_depth,
        encoder_num_heads,
        encoder_global_attn_indexes,
        num_classes,
        image_size,
        pixel_mean,
        pixel_std,
        checkpoint=None,
):
    prompt_embed_dim = 256
    image_size = image_size
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size  # Divide by 16 here
    sam = Sam(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            # num_multimask_outputs=3,
            num_multimask_outputs=num_classes,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        # pixel_mean=[123.675, 116.28, 103.53],
        # pixel_std=[58.395, 57.12, 57.375],
        pixel_mean=pixel_mean,
        pixel_std=pixel_std
    )
    # sam.eval()
    sam.train()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            # state_dict = torch.load(f)
            state_dict = torch.load(f,map_location='cuda:0')
        try:
            sam.load_state_dict(state_dict)
        except:
            new_state_dict = load_from(sam, state_dict, image_size, vit_patch_size)
            sam.load_state_dict(new_state_dict)
    return sam, image_embedding_size

from .modeling import SAM_Iter_Simple,ImageEncoderViTMS, PromptEncoderIter, MaskDecoder_IterSimple
def _build_sam_iter(
        encoder_embed_dim,
        encoder_depth,
        encoder_num_heads,
        encoder_global_attn_indexes,
        num_classes,
        image_size,
        pixel_mean,
        pixel_std,
        checkpoint=None,
        checkpoint_HQ=None,
        cfg=None,
):
    prompt_embed_dim = 256
    image_size = image_size
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size  # Divide by 16 here
    sam = SAM_Iter_Simple(
        image_encoder=ImageEncoderViTMS(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
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
            cfg = cfg,
        ),
        # pixel_mean=[123.675, 116.28, 103.53],
        # pixel_std=[58.395, 57.12, 57.375],
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
        cfg = cfg,
    )
    # sam.eval()
    sam.train()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            # state_dict = torch.load(f)
            state_dict_sam = torch.load(f,map_location='cuda:0')
        new_state_dict = load_from_samed_my(sam, state_dict_sam, image_size, vit_patch_size)
        sam.load_state_dict(new_state_dict)
    return sam, image_embedding_size

def load_from(sam, state_dict, image_size, vit_patch_size):
    sam_dict = sam.state_dict()
    except_keys = ['mask_tokens', 'output_hypernetworks_mlps', 'iou_prediction_head']
    new_state_dict = {k: v for k, v in state_dict.items() if
                      k in sam_dict.keys() and except_keys[0] not in k and except_keys[1] not in k and except_keys[2] not in k}
    pos_embed = new_state_dict['image_encoder.pos_embed']
    token_size = int(image_size // vit_patch_size)
    if pos_embed.shape[1] != token_size:
        # resize pos embedding, which may sacrifice the performance, but I have no better idea
        pos_embed = pos_embed.permute(0, 3, 1, 2)  # [b, c, h, w]
        pos_embed = F.interpolate(pos_embed, (token_size, token_size), mode='bilinear', align_corners=False)
        pos_embed = pos_embed.permute(0, 2, 3, 1)  # [b, h, w, c]
        new_state_dict['image_encoder.pos_embed'] = pos_embed
        rel_pos_keys = [k for k in sam_dict.keys() if 'rel_pos' in k]
        global_rel_pos_keys = [k for k in rel_pos_keys if '2' in k or '5' in  k or '8' in k or '11' in k]
        for k in global_rel_pos_keys:
            rel_pos_params = new_state_dict[k]
            h, w = rel_pos_params.shape
            rel_pos_params = rel_pos_params.unsqueeze(0).unsqueeze(0)
            rel_pos_params = F.interpolate(rel_pos_params, (token_size * 2 - 1, w), mode='bilinear', align_corners=False)
            new_state_dict[k] = rel_pos_params[0, 0, ...]
    sam_dict.update(new_state_dict)
    return sam_dict

def load_from_my(sam, state_dict_sam, state_dict_hq, image_size, vit_patch_size):
    '''
    state_dict: params from loaded checkpoint
    '''
    sam_dict = sam.state_dict() # cur sam's param
    except_keys = ['mask_tokens', 'output_hypernetworks_mlps', 'iou_prediction_head']
    new_state_dict = {k: v for k, v in state_dict_sam.items() if
                      k in sam_dict.keys() and except_keys[0] not in k and except_keys[1] not in k and except_keys[2] not in k}
    pos_embed = new_state_dict['image_encoder.pos_embed']
    token_size = int(image_size // vit_patch_size)
    if pos_embed.shape[1] != token_size:
        # resize pos embedding, which may sacrifice the performance, but I have no better idea
        pos_embed = pos_embed.permute(0, 3, 1, 2)  # [b, c, h, w]
        pos_embed = F.interpolate(pos_embed, (token_size, token_size), mode='bilinear', align_corners=False)
        pos_embed = pos_embed.permute(0, 2, 3, 1)  # [b, h, w, c]
        new_state_dict['image_encoder.pos_embed'] = pos_embed
        rel_pos_keys = [k for k in sam_dict.keys() if 'rel_pos' in k]
        global_rel_pos_keys = [k for k in rel_pos_keys if '2' in k or '5' in  k or '8' in k or '11' in k]
        for k in global_rel_pos_keys:
            rel_pos_params = new_state_dict[k]
            h, w = rel_pos_params.shape
            rel_pos_params = rel_pos_params.unsqueeze(0).unsqueeze(0)
            rel_pos_params = F.interpolate(rel_pos_params, (token_size * 2 - 1, w), mode='bilinear', align_corners=False)
            new_state_dict[k] = rel_pos_params[0, 0, ...]
    # prompt encoder
    state_dict_prompt = {k: v for k, v in state_dict_sam.items() if 'prompt' in k}
    want_list = ['no_mask_embed','mask_downscaling','pe_layer']
    state_dict_prompt = {k: v for k, v in state_dict_prompt.items() if check_key_part_in_list(k,want_list)}
    num_cls = sam.prompt_encoder.num_cls_embeddings
    mask_down = state_dict_prompt['prompt_encoder.mask_downscaling.0.weight']
    mask_down = torch.repeat_interleave(mask_down,num_cls,dim=1)
    state_dict_prompt.update({'prompt_encoder.mask_downscaling.0.weight':mask_down})
    new_state_dict.update(state_dict_prompt)
    # mask decoder
    state_dict_maskdec = {k: v for k, v in state_dict_sam.items() if 'mask_decoder' in k}
    want_list = ['transformer','output_upscaling']
    state_dict_maskdec = {k: v for k, v in state_dict_maskdec.items() if check_key_part_in_list(k,want_list)}
    state_dict_maskdec = {k.replace('mask_decoder','mask_decoder_lq'): v for k, v in state_dict_maskdec.items()}
    new_state_dict.update(state_dict_maskdec)
    # hq
    state_dict_maskhq = {k: v for k, v in state_dict_hq.items() if 'mask_decoder' in k}
    want_list = ['compress_vit_feat', 'embedding_encoder', 'embedding_maskfeature', 'output_upscaling']
    state_dict_maskhq = {k: v for k, v in state_dict_maskhq.items() if check_key_part_in_list(k, want_list)}
    state_dict_maskhq = {k.replace('mask_decoder', 'mask_decoder_hq'): v for k, v in state_dict_maskhq.items()}
    new_state_dict.update(state_dict_maskhq)
    sam_dict.update(new_state_dict) # cur sam's param + params from checkpoint
    return sam_dict

def load_from_samed_my(sam, state_dict_sam, image_size, vit_patch_size):
    '''
    state_dict: params from loaded checkpoint
    '''
    sam_dict = sam.state_dict() # cur sam's param
    new_state_dict = {}
    # image encoder
    state_dict_imgenc = {k: v for k, v in state_dict_sam.items() if 'image_encoder' in k}
    except_keys = ['mask_tokens', 'iou_prediction_head']
    state_dict_imgenc = {k: v for k, v in state_dict_imgenc.items() if
                      k in sam_dict.keys() and not check_key_part_in_list(k,except_keys)}
    pos_embed = state_dict_imgenc['image_encoder.pos_embed']
    token_size = int(image_size // vit_patch_size)
    if pos_embed.shape[1] != token_size:
        # resize pos embedding, which may sacrifice the performance, but I have no better idea
        pos_embed = pos_embed.permute(0, 3, 1, 2)  # [b, c, h, w]
        pos_embed = F.interpolate(pos_embed, (token_size, token_size), mode='bilinear', align_corners=False)
        pos_embed = pos_embed.permute(0, 2, 3, 1)  # [b, h, w, c]
        state_dict_imgenc['image_encoder.pos_embed'] = pos_embed
        rel_pos_keys = [k for k in sam_dict.keys() if 'rel_pos' in k]
        global_rel_pos_keys = [k for k in rel_pos_keys if '2' in k or '5' in  k or '8' in k or '11' in k]
        for k in global_rel_pos_keys:
            rel_pos_params = state_dict_imgenc[k]
            h, w = rel_pos_params.shape
            rel_pos_params = rel_pos_params.unsqueeze(0).unsqueeze(0)
            rel_pos_params = F.interpolate(rel_pos_params, (token_size * 2 - 1, w), mode='bilinear', align_corners=False)
            state_dict_imgenc[k] = rel_pos_params[0, 0, ...]
    new_state_dict.update(state_dict_imgenc)
    # prompt encoder
    state_dict_prompt = {k: v for k, v in state_dict_sam.items() if 'prompt' in k}
    want_list = ['pe_layer','mask_downscaling']
    state_dict_prompt = {k: v for k, v in state_dict_prompt.items() if check_key_part_in_list(k,want_list)}
    spec_name = 'prompt_encoder.mask_downscaling.0.weight'
    shape1 = sam_dict[spec_name].shape[1]
    spec_value = state_dict_prompt[spec_name].repeat(1,shape1,1,1)
    state_dict_prompt[spec_name] = spec_value
    new_state_dict.update(state_dict_prompt)
    # mask decoder
    state_dict_maskdec = {k: v for k, v in state_dict_sam.items() if 'mask_decoder' in k}
    want_list = ['transformer','output_upscaling','output_hypernetworks_mlps']
    state_dict_maskdec = {k: v for k, v in state_dict_maskdec.items() if check_key_part_in_list(k,want_list)}
    new_state_dict.update(state_dict_maskdec)
    sam_dict.update(new_state_dict)

    return sam_dict

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
