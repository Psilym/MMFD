# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.nn import functional as F
from icecream import ic
from typing import List

from functools import partial
from .whole_arch_seg import MultiModelSeg
from .whole_arch import MultiModelArch
from ..sam_iter.modeling import TwoWayTransformer, MaskDecoder_IterSimple, PromptEncoderIter


def build_multimodel2s(image_size,
                      num_classes,
                      config_basic=None,
                      cfg=None):
    return _build_multimodel2s(
        config_basic,
        num_classes=num_classes,
        image_size=image_size,
        cfg=cfg,
    )


def _build_multimodel2s(
        config_basic,
        num_classes: int,
        image_size: int,
        cfg=None,
):
    prompt_embed_dim = 256
    image_size = image_size
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size  # Divide by 16 here
    from .utils import config_to_model_dict_onebyone
    model_dict = config_to_model_dict_onebyone(config_basic,cfg)
    model_dict.update({'cfg':cfg})
    sam = MultiModelSeg(
        image_encoder=MultiModelArch(model_dict),
        prompt_encoder=PromptEncoderIter(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            num_class=num_classes,
            mask_in_chans=16,
            cfg=cfg,
        ),
        mask_decoder=MaskDecoder_IterSimple(
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
        pixel_mean=[0,0,0],
        pixel_std=[1,1,1],
        cfg = cfg,
    )
    # sam.eval()
    sam.train()
    load_checkpoint_multimodel2s(sam,config_basic)
    load_from_sam_ckpt_my(sam, ckpt=cfg.sam_ckpt)
    return sam, image_embedding_size


def check_update_ckpt_param_number(model_dict, new_dict):
    count = 0
    for k_new in new_dict.keys():
        for k_ori in model_dict.keys():
            if k_ori == k_new:
                count += 1
    return count
def load_checkpoint_multimodel2s(model, config_basic):
    for idx, enc_name in enumerate(model.image_encoder.enc_names):
        if enc_name == 'medsam':
            model = load_ckpt_medsam(model,config_basic)
        elif enc_name == 'plip':
            model = load_ckpt_plip(model, config_basic)
        elif enc_name == 'pidinet':
            model = load_ckpt_pidinet(model,config_basic)
        else:
            print(f'Error. Loading enc name {enc_name} has not been implemented.')
            raise ValueError
    print("Finish loading checkpoint of MultiModel2s.")
    return model

def load_ckpt_medsam(model,config_basic):
    ckpt = config_basic.MODEL.MED_SAM_VIT.PRETRAINED_PATH
    if ckpt is not None:
        with open(ckpt, "rb") as f:
            ckpt_dict = torch.load(f, map_location='cuda:0')
    else:
        print(f'Cannot find ckpt from {ckpt}. Dont load.')
        return model
    enc_name = 'medsam'
    print(f"Start loading checkpoint of {enc_name} from path {ckpt}...")
    model_dict = model.state_dict()
    # MultiModel3Arch
    share_keys = [k.split('encoder.')[1:][0] for k in ckpt_dict.keys() if 'image_encoder' in k]
    # delete_keys = ['mask_token','fuse_pre','fuse_pro','fuse_fin']
    # share_keys = [k for k in share_keys if not check_key_part_in_list(k,delete_keys)]
    new_keys = [f'image_encoder.{enc_name}.{k}' for k in share_keys]
    new_values = [ckpt_dict[f'image_encoder.{k}'] for k in share_keys]
    new_dict = {k: v for k, v in zip(new_keys, new_values)}

    config = config_basic
    # enc: pos and rel_pos fitting
    image_size = config.DATA.IMG_SIZE
    patch_size = config.MODEL.MED_SAM_VIT.PATCH_SIZE
    token_size = image_size // patch_size
    pos_embed = new_dict[f'image_encoder.{enc_name}.pos_embed']
    if pos_embed.shape[1] != token_size:
        # resize pos embedding, which may sacrifice the performance, but I have no better idea
        pos_embed = pos_embed.permute(0, 3, 1, 2)  # [b, c, h, w]
        pos_embed = F.interpolate(pos_embed, (token_size, token_size), mode='bilinear', align_corners=False)
        pos_embed = pos_embed.permute(0, 2, 3, 1)  # [b, h, w, c]
        new_dict[f'image_encoder.{enc_name}.pos_embed'] = pos_embed
        rel_pos_keys = [k for k in new_dict.keys() if 'rel_pos' in k and enc_name in k]
        global_rel_pos_keys = [k for k in rel_pos_keys if '.2.' in k or '.5.' in  k or '.8.' in k or '.11.' in k]
        for k in global_rel_pos_keys:
            rel_pos_params = new_dict[k]
            h, w = rel_pos_params.shape
            rel_pos_params = rel_pos_params.unsqueeze(0).unsqueeze(0)
            rel_pos_params = F.interpolate(rel_pos_params, (token_size * 2 - 1, w), mode='bilinear', align_corners=False) #TODO:??seems strage
            new_dict[k] = rel_pos_params[0, 0, ...]
    # to model
    model_dict.update(new_dict)
    msg = model.load_state_dict(model_dict, strict=True)
    print(msg)
    del ckpt_dict
    torch.cuda.empty_cache()
    print(f"Finsh loading checkpoint of {enc_name} from path {ckpt}.")
    return model

def load_ckpt_vit_ssl(model,config_basic):
    ckpt = config_basic.MODEL.SSL_SAM_VIT.PRETRAINED_PATH
    if ckpt is not None:
        with open(ckpt, "rb") as f:
            ckpt_dict = torch.load(f, map_location='cuda:0')['model']
    else:
        print(f'Cannot find ckpt from {ckpt}. Dont load.')
        return model
    enc_name = 'vit_ssl'
    print(f"Start loading checkpoint of {enc_name} from path {ckpt}...")
    model_dict = model.state_dict()
    # MultiModel3Arch
    share_keys = ['.'.join(k.split('encoder.')[1:]) for k in ckpt_dict.keys() if 'encoder' in k]
    delete_keys = ['mask_token']
    share_keys = [k for k in share_keys if not check_key_part_in_list(k,delete_keys)]
    new_keys = [f'image_encoder.{enc_name}.{k}' for k in share_keys]
    new_values = [ckpt_dict[f'encoder.{k}'] for k in share_keys]
    new_dict = {k: v for k, v in zip(new_keys, new_values)}

    config = config_basic
    # enc: pos and rel_pos fitting
    image_size = config.DATA.IMG_SIZE
    patch_size = config.MODEL.SSL_SAM_VIT.PATCH_SIZE
    token_size = image_size // patch_size
    pos_embed = new_dict[f'image_encoder.{enc_name}.pos_embed']
    if pos_embed.shape[1] != token_size:
        # resize pos embedding, which may sacrifice the performance, but I have no better idea
        pos_embed = pos_embed.permute(0, 3, 1, 2)  # [b, c, h, w]
        pos_embed = F.interpolate(pos_embed, (token_size, token_size), mode='bilinear', align_corners=False)
        pos_embed = pos_embed.permute(0, 2, 3, 1)  # [b, h, w, c]
        new_dict[f'image_encoder.{enc_name}.pos_embed'] = pos_embed
        rel_pos_keys = [k for k in new_dict.keys() if 'rel_pos' in k and enc_name in k]
        global_rel_pos_keys = [k for k in rel_pos_keys if '.2.' in k or '.5.' in  k or '.8.' in k or '.11.' in k]
        for k in global_rel_pos_keys:
            rel_pos_params = new_dict[k]
            h, w = rel_pos_params.shape
            rel_pos_params = rel_pos_params.unsqueeze(0).unsqueeze(0)
            rel_pos_params = F.interpolate(rel_pos_params, (token_size * 2 - 1, w), mode='bilinear', align_corners=False)
            new_dict[k] = rel_pos_params[0, 0, ...]
    # to model
    model_dict.update(new_dict)
    msg = model.load_state_dict(model_dict, strict=True)
    print(msg)
    del ckpt_dict
    torch.cuda.empty_cache()
    print(f"Finsh loading checkpoint of {enc_name} from path {ckpt}.")
    return model

def load_ckpt_plip(model,config_basic):
    ckpt = config_basic.MODEL.PLIP.PRETRAINED_PATH
    if ckpt is not None:
        with open(ckpt, "rb") as f:
            ckpt_dict = torch.load(f, map_location='cuda:0')
    else:
        print(f'Cannot find ckpt from {ckpt}. Dont load.')
        return model
    enc_name = 'plip'
    print(f"Start loading checkpoint of {enc_name} from path {ckpt}...")
    model_dict = model.state_dict()
    # MultiModel3Arch
    share_keys = ['.'.join(k.split('vision_model.')[1:]) for k in ckpt_dict.keys() if 'vision_model' in k]
    delete_keys = ['position_ids']
    share_keys = [k for k in share_keys if not check_key_part_in_list(k,delete_keys)]
    new_keys = [f'image_encoder.{enc_name}.{k}' for k in share_keys]
    new_values = [ckpt_dict[f'vision_model.{k}'] for k in share_keys]
    new_dict = {k: v for k, v in zip(new_keys, new_values)}
    # m_keys = [k for k in model_dict.keys() if 'plip' in k]

    config = config_basic
    # enc: pos and rel_pos fitting
    image_size = config.DATA.IMG_SIZE
    patch_size = config.MODEL.PLIP.PATCH_SIZE
    token_size = image_size // patch_size
    pos_embed = new_dict[f'image_encoder.{enc_name}.embeddings.position_embedding.weight']
    Nemb,emb_channels = pos_embed.shape
    H_emb = W_emb = int((Nemb-1)**0.5)
    cls_embed = pos_embed[0,...].unsqueeze(0)
    pos_embed = pos_embed[1:,...].view(H_emb,W_emb,emb_channels).unsqueeze(0) #[1,H,W,C]
    if pos_embed.shape[1] != token_size:
        # resize pos embedding, which may sacrifice the performance, but I have no better idea
        pos_embed = pos_embed.permute(0, 3, 1, 2)  # [b, c, h, w]
        pos_embed = F.interpolate(pos_embed, (token_size, token_size), mode='bilinear', align_corners=False)
        pos_embed = pos_embed.permute(0, 2, 3, 1)  # [b, h, w, c]
        pos_embed = pos_embed[0].view(-1,emb_channels) #[h*w,c]
        pos_embed = torch.cat([cls_embed,pos_embed],dim=0)
        new_dict[f'image_encoder.{enc_name}.embeddings.position_embedding.weight'] = pos_embed
    # to model
    model_dict.update(new_dict)
    msg = model.load_state_dict(model_dict, strict=True)
    print(msg)
    del ckpt_dict
    torch.cuda.empty_cache()
    print(f"Finsh loading checkpoint of {enc_name} from path {ckpt}.")
    return model

def load_ckpt_pidinet(model,config_basic):
    ckpt = config_basic.MODEL.PIDINET.PRETRAINED_PATH
    if ckpt is not None:
        with open(ckpt, "rb") as f:
            ckpt_dict = torch.load(f, map_location='cuda:0')['state_dict']
    else:
        print(f'Cannot find ckpt from {ckpt}. Dont load.')
        return model
    enc_name = 'pidinet'
    print(f"Start loading checkpoint of {enc_name} from path {ckpt}...")
    model_dict = model.state_dict()
    # MultiModel3Arch
    share_keys = [k.split('module.')[-1] for k in ckpt_dict.keys()]
    delete_keys = ['classifier','conv_reduces']
    share_keys = [k for k in share_keys if not check_key_part_in_list(k,delete_keys)]
    new_keys = [f'image_encoder.{enc_name}.{k}' for k in share_keys]
    new_values = [ckpt_dict[f'module.{k}'] for k in share_keys]
    new_dict = {k: v for k, v in zip(new_keys, new_values)}

    # to model
    model_dict.update(new_dict)
    msg = model.load_state_dict(model_dict, strict=True)
    print(msg)
    del ckpt_dict
    torch.cuda.empty_cache()
    print(f"Finsh loading checkpoint of {enc_name} from path {ckpt}.")
    return model

def load_from_sam_ckpt_my(model, ckpt=None, enc_name = 'enc2'):
    '''
    state_dict: params from loaded checkpoint
    '''
    if ckpt is not None:
        with open(ckpt, "rb") as f:
            ckpt_dict = torch.load(f,map_location='cuda:0')
    else:
        return model
    print(f"Start loading checkpoint of SAM Prompt Encoder and Mask Decoder from path {ckpt}...")
    model_dict = model.state_dict() # cur model's param
    new_dict = {}
    # prompt encoder
    state_dict_prompt = {k: v for k, v in ckpt_dict.items() if 'prompt' in k}
    want_list = ['pe_layer','mask_downscaling']
    state_dict_prompt = {k: v for k, v in state_dict_prompt.items() if check_key_part_in_list(k,want_list)}
    spec_name = 'prompt_encoder.mask_downscaling.0.weight'
    shape1 = model_dict[spec_name].shape[1]
    spec_value = state_dict_prompt[spec_name].repeat(1,shape1,1,1)
    state_dict_prompt[spec_name] = spec_value
    new_dict.update(state_dict_prompt)
    # mask decoder
    state_dict_maskdec = {k: v for k, v in ckpt_dict.items() if 'mask_decoder' in k}
    want_list = ['transformer','output_upscaling','output_hypernetworks_mlps']
    state_dict_maskdec = {k: v for k, v in state_dict_maskdec.items() if check_key_part_in_list(k,want_list)}
    new_dict.update(state_dict_maskdec)

    model_dict.update(new_dict)
    msg = model.load_state_dict(model_dict, strict=True)
    print(msg)
    del ckpt_dict
    torch.cuda.empty_cache()
    return model


sam_model_registry = {
    "mmodel2s": build_multimodel2s,
}

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
