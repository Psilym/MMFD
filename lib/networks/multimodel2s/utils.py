import torch
from functools import partial
from .fuse_block import ModelMetaInfo
from transformers import CLIPVisionConfig
import json


def config_to_model_dict(config,cfg2):
    model_dict = {}
    # enc1
    enc1_name = 'ssl_sam_enc'
    enc1_params = {
        'depth': config.MODEL.SSL_SAM_VIT.ENCODER_DEPTH,
        'embed_dim' : config.MODEL.SSL_SAM_VIT.ENCODER_EMBED_DIM,
        'img_size' : config.DATA.IMG_SIZE,
        'mlp_ratio' : 4,
        'norm_layer' : partial(torch.nn.LayerNorm, eps=1e-6),
        'num_heads' : config.MODEL.SSL_SAM_VIT.ENCODER_NUM_HEADS,
        'patch_size' : config.MODEL.SSL_SAM_VIT.PATCH_SIZE,
        'qkv_bias' : True,
        'use_rel_pos' : True,
        'global_attn_indexes' : config.MODEL.SSL_SAM_VIT.ENCODER_GLOBAL_ATTN_INDEXES,
        'window_size' : config.MODEL.SSL_SAM_VIT.WINDOW_SIZE,
        'out_chans' : config.MODEL.SSL_SAM_VIT.OUT_DIM,}
    model_dict.update({enc1_name:enc1_params})
    # enc2
    enc2_name = 'med_sam_enc'
    enc2_params = {
        'depth': config.MODEL.MED_SAM_VIT.ENCODER_DEPTH,
        'embed_dim' : config.MODEL.MED_SAM_VIT.ENCODER_EMBED_DIM,
        'img_size' : config.DATA.IMG_SIZE,
        'mlp_ratio' : 4,
        'norm_layer' : partial(torch.nn.LayerNorm, eps=1e-6),
        'num_heads' : config.MODEL.MED_SAM_VIT.ENCODER_NUM_HEADS,
        'patch_size' : config.MODEL.MED_SAM_VIT.PATCH_SIZE,
        'qkv_bias' : True,
        'use_rel_pos' : True,
        'global_attn_indexes' : config.MODEL.MED_SAM_VIT.ENCODER_GLOBAL_ATTN_INDEXES,
        'window_size' : config.MODEL.MED_SAM_VIT.WINDOW_SIZE,
        'out_chans' : config.MODEL.MED_SAM_VIT.OUT_DIM,}
    model_dict.update({enc2_name:enc2_params})
    # enc3
    enc3_name = 'plip_enc'
    cfg_path = config.MODEL.PLIP.CONFIG_PATH
    with open(cfg_path, 'r') as f:
        cfg = json.load(f)
    vision_cfg = cfg['vision_config']
    clip_vcfger = CLIPVisionConfig(**vision_cfg)
    clip_vcfger.image_size = config.MODEL.PLIP.IMG_SIZE
    enc3_params = {'clip_vcfger':clip_vcfger,
                   'img_size': config.MODEL.PLIP.IMG_SIZE,
                   'patch_size': config.MODEL.PLIP.PATCH_SIZE,
                   'out_chans': config.MODEL.PLIP.OUT_DIM,
                   }
    model_dict.update({enc3_name:enc3_params})
    # meta generate
    scale = enc1_params['img_size'] // enc1_params['patch_size']
    enc1_meta = ModelMetaInfo(feat_scale=(scale,scale),feat_chan=enc1_params['out_chans'])
    scale = enc2_params['img_size'] // enc2_params['patch_size']
    enc2_meta = ModelMetaInfo(feat_scale=(scale, scale), feat_chan=enc2_params['out_chans'])
    scale = enc3_params['img_size'] // enc3_params['patch_size']
    enc3_meta = ModelMetaInfo(feat_scale=(scale, scale), feat_chan=enc3_params['out_chans'])
    model_meta_list = ([enc1_meta,enc2_meta,enc3_meta])
    # fuse pre arch
    fuse_pre_name = 'fuse_pre'
    fuse_pre_params = {
        'img_size': config.DATA.IMG_SIZE,
        'patch_size': config.MODEL.FUSE_PRE.PATCH_SIZE,
        'model_meta_list': model_meta_list,
        'out_chans': config.MODEL.FUSE_PRE.OUT_DIM,
    }
    model_dict.update({fuse_pre_name: fuse_pre_params})
    # fuse pro arch
    fuse_pro_name = 'fuse_pro'
    fuse_pro_params = {
        'parallel_type': cfg2.component.fuse_parallel_block,
        'final_type': cfg2.component.fuse_final_block,
        'num_model': len(model_meta_list),
        'img_size': config.DATA.IMG_SIZE,
        'patch_size': config.MODEL.FUSE_PRO.PATCH_SIZE,
        'in_chans': config.MODEL.FUSE_PRO.OUT_DIM,
        'embed_dim': config.MODEL.FUSE_PRO.ENCODER_EMBED_DIM,
        'depth': config.MODEL.FUSE_PRO.ENCODER_DEPTH,
        'num_heads': config.MODEL.FUSE_PRO.ENCODER_NUM_HEADS,
        'mlp_ratio': 4,
        'out_chans': config.MODEL.FUSE_PRO.OUT_DIM,
        'norm_layer': partial(torch.nn.LayerNorm, eps=1e-6),
        'qkv_bias': True,
        'use_rel_pos': True,
        'global_attn_indexes': config.MODEL.FUSE_PRO.ENCODER_GLOBAL_ATTN_INDEXES,
        'window_size': config.MODEL.FUSE_PRO.WINDOW_SIZE,
        'cfg': cfg2,
    }
    model_dict.update({fuse_pro_name: fuse_pro_params})

    # decoder
    return model_dict

def config_to_model_dict_onebyone(config,cfg2):
    model_dict = {}
    model_list = cfg2.component.model_list
    model_meta_list = []
    for model_name in model_list:
        func = f'add_model_dict_{model_name}'
        if model_name == 'medsam':
            model_dict_, meta = eval(func)(config, cfg2)
            model_dict.update(model_dict_)
            model_meta_list.append(meta)
        elif model_name == 'vit_ssl':
            model_dict_, meta = eval(func)(config, cfg2)
            model_dict.update(model_dict_)
            model_meta_list.append(meta)
        elif model_name == 'plip':
            model_dict_, meta = eval(func)(config, cfg2)
            model_dict.update(model_dict_)
            model_meta_list.append(meta)
        elif model_name == 'pidinet':
            model_dict_, meta = eval(func)(config, cfg2)
            model_dict.update(model_dict_)
            model_meta_list.append(meta)
        else:
            print(f'Error. {model_name} is not available. Please check the model list.')
            raise ValueError
    # fuse pre (temporary)
    fuse_pre_name = 'fuse_pre'
    fuse_pre_params = {
        'model_meta_list': model_meta_list,
        'out_chans': config.MODEL.FUSE_PRE.OUT_DIM,
    }
    model_dict.update({fuse_pre_name: fuse_pre_params})
    # fuse pro arch
    fuse_pro_name = 'fuse_pro'
    fuse_depth = cfg2.component.fuse_depth
    if fuse_depth == 'none':
        fuse_depth = config.MODEL.FUSE_PRO.ENCODER_DEPTH
    share_token_num = cfg2.component.fuse_token_num
    if share_token_num == 'none':
        share_token_num = config.MODEL.FUSE_PRO.SHARE_TOKEN_NUM
    fuse_pro_params = {
        'parallel_type': cfg2.component.fuse_parallel_block,
        'num_model': len(model_meta_list),
        'img_size': config.DATA.IMG_SIZE,
        'patch_size': config.MODEL.FUSE_PRO.PATCH_SIZE,
        'in_chans': config.MODEL.FUSE_PRO.OUT_DIM,
        'embed_dim': config.MODEL.FUSE_PRO.ENCODER_EMBED_DIM,
        'depth': fuse_depth,
        'share_token_num': share_token_num,
        'num_heads': config.MODEL.FUSE_PRO.ENCODER_NUM_HEADS,
        'mlp_ratio': 4,
        'out_chans': config.MODEL.FUSE_PRO.OUT_DIM,
        'norm_layer': partial(torch.nn.LayerNorm, eps=1e-6),
        'qkv_bias': True,
        'use_rel_pos': False,
        'global_attn_indexes': config.MODEL.FUSE_PRO.ENCODER_GLOBAL_ATTN_INDEXES,
        'window_size': config.MODEL.FUSE_PRO.WINDOW_SIZE,
        'cfg': cfg2,
    }
    model_dict.update({fuse_pro_name: fuse_pro_params})
    # fuse fin arch
    fuse_fin_name = 'fuse_fin'
    def obtain_fin_feat_hwc(model_meta_list):
        '''
        obtain h,w,c from hw largest feature map
        '''
        hw_list = []
        for meta_ in model_meta_list:
            h, w, c = meta_.obtain_info_list()
            hw_list.append(h * w)
        idx = hw_list.index(max(hw_list))
        h_tar, w_tar, c_tar = model_meta_list[idx].obtain_info_list()
        return h_tar,w_tar,c_tar

    # h_tar,w_tar,_ = obtain_fin_feat_hwc(model_meta_list)
    meta_ = [ meta for meta in model_meta_list if meta.meta['name'] == 'medsam'][0]
    h_tar = meta_.meta['h']
    w_tar = meta_.meta['w']
    assert h_tar == w_tar
    fuse_fin_params = {
        'final_type': cfg2.component.fuse_final_block,
        'embed_dim': config.MODEL.FUSE_FIN.EMBED_DIM,
        'feat_size': h_tar,
        'model_meta_list': model_meta_list,
        'out_chans': config.MODEL.FUSE_FIN.OUT_DIM,
    }
    model_dict.update({fuse_fin_name: fuse_fin_params})
    # decoder
    return model_dict

def add_model_dict_medsam(config, cfg2):
    model_dict = {}
    enc_name = 'medsam'
    enc_params = {
        'depth': config.MODEL.MED_SAM_VIT.ENCODER_DEPTH,
        'embed_dim' : config.MODEL.MED_SAM_VIT.ENCODER_EMBED_DIM,
        'img_size' : config.DATA.IMG_SIZE,
        'mlp_ratio' : 4,
        'norm_layer' : partial(torch.nn.LayerNorm, eps=1e-6),
        'num_heads' : config.MODEL.MED_SAM_VIT.ENCODER_NUM_HEADS,
        'patch_size' : config.MODEL.MED_SAM_VIT.PATCH_SIZE,
        'qkv_bias' : True,
        'use_rel_pos' : True,
        'global_attn_indexes' : config.MODEL.MED_SAM_VIT.ENCODER_GLOBAL_ATTN_INDEXES,
        'window_size' : config.MODEL.MED_SAM_VIT.WINDOW_SIZE,
        'out_chans' : config.MODEL.MED_SAM_VIT.OUT_DIM,
        'norm_mean': config.MODEL.MED_SAM_VIT.NORM_MEAN,
        'norm_std': config.MODEL.MED_SAM_VIT.NORM_STD,
        'norm_pre_scale': config.MODEL.MED_SAM_VIT.NORM_PRE_SCALE,
    }
    model_dict.update({enc_name:enc_params})

    scale = enc_params['img_size'] // enc_params['patch_size']
    meta = ModelMetaInfo(feat_scale=(scale,scale),feat_chan=enc_params['out_chans'],name='medsam')
    return model_dict, meta

def add_model_dict_vit_ssl(config, cfg2):
    model_dict = {}
    enc_name = 'vit_ssl'
    enc_params = {
        'depth': config.MODEL.SSL_SAM_VIT.ENCODER_DEPTH,
        'embed_dim' : config.MODEL.SSL_SAM_VIT.ENCODER_EMBED_DIM,
        'img_size' : config.DATA.IMG_SIZE,
        'mlp_ratio' : 4,
        'norm_layer' : partial(torch.nn.LayerNorm, eps=1e-6),
        'num_heads' : config.MODEL.SSL_SAM_VIT.ENCODER_NUM_HEADS,
        'patch_size' : config.MODEL.SSL_SAM_VIT.PATCH_SIZE,
        'qkv_bias' : True,
        'use_rel_pos' : True,
        'global_attn_indexes' : config.MODEL.SSL_SAM_VIT.ENCODER_GLOBAL_ATTN_INDEXES,
        'window_size' : config.MODEL.SSL_SAM_VIT.WINDOW_SIZE,
        'out_chans' : config.MODEL.SSL_SAM_VIT.OUT_DIM,
        'norm_mean': config.MODEL.SSL_SAM_VIT.NORM_MEAN,
        'norm_std': config.MODEL.SSL_SAM_VIT.NORM_STD,
        'norm_pre_scale': config.MODEL.SSL_SAM_VIT.NORM_PRE_SCALE,
    }
    model_dict.update({enc_name:enc_params})
    scale = enc_params['img_size'] // enc_params['patch_size']
    meta = ModelMetaInfo(feat_scale=(scale,scale),feat_chan=enc_params['out_chans'],name='vit_ssl')
    return model_dict, meta

def add_model_dict_plip(config, cfg2):
    model_dict = {}
    enc_name = 'plip'
    cfg_path = config.MODEL.PLIP.CONFIG_PATH
    with open(cfg_path, 'r') as f:
        cfg = json.load(f)
    vision_cfg = cfg['vision_config']
    clip_vcfger = CLIPVisionConfig(**vision_cfg)
    clip_vcfger.image_size = config.MODEL.PLIP.IMG_SIZE
    enc_params = {'clip_vcfger':clip_vcfger,
                   'img_size': config.MODEL.PLIP.IMG_SIZE,
                   'patch_size': config.MODEL.PLIP.PATCH_SIZE,
                   'out_chans': config.MODEL.PLIP.OUT_DIM,
                   }
    model_dict.update({enc_name:enc_params})
    scale = enc_params['img_size'] // enc_params['patch_size']
    meta = ModelMetaInfo(feat_scale=(scale,scale),feat_chan=enc_params['out_chans'],name='plip')
    return model_dict, meta

def add_model_dict_pidinet(config, cfg2):
    model_dict = {}
    enc_name = 'pidinet'
    from lib.networks.pidinet.config import config_model
    pdcs = config_model('carv4')
    enc_params = {'inplane':60,
                   'pdcs': pdcs,
                   'dil': 24,
                   'sa': True,
                  'norm_mean': config.MODEL.PIDINET.NORM_MEAN,
                  'norm_std': config.MODEL.PIDINET.NORM_STD,
                  'norm_pre_scale': config.MODEL.PIDINET.NORM_PRE_SCALE,
                   }
    model_dict.update({enc_name:enc_params})
    img_size = config.MODEL.PIDINET.IMG_SIZE
    patch_size = config.MODEL.PIDINET.PATCH_SIZE
    out_chans = config.MODEL.PIDINET.OUT_DIM
    scale = img_size // patch_size
    meta = ModelMetaInfo(feat_scale=(scale,scale),feat_chan=out_chans,name='pidinet')
    return model_dict, meta