import torch
import torch.nn as nn
import torch.nn.functional as F



# model_dict:
#     num_model
#     model1_name
#         params
#     model2_name
class MultiModelArch_old(nn.Module):
    def __init__(self, model_dict):
        super(MultiModelArch_old,self).__init__()
        # num_model = model_dict['num_model']
        self.model_dict = model_dict
        # encoder1
        from .image_encoder import ImageEncoderViTMS
        enc1_params = model_dict['ssl_sam_enc']
        self.enc1 = ImageEncoderViTMS(**enc1_params)
        # encoder2
        from .image_encoder import ImageEncoderViTMS
        enc2_params = model_dict['med_sam_enc']
        self.enc2 = ImageEncoderViTMS(**enc2_params)
        # encoder3
        from ..plip.modeling_clip import CLIPVisionTransformer_my
        enc3_params = model_dict['plip_enc']
        enc3_clip_vcfger = enc3_params['clip_vcfger']
        self.enc3 = CLIPVisionTransformer_my(enc3_clip_vcfger)
        # fuse architecture
        from . import FusePre, FusePro, FusePro_Select
        fuse_pre_params = model_dict['fuse_pre']
        self.fuse_pre = FusePre(**fuse_pre_params)
        fuse_pro_params = model_dict['fuse_pro']
        self.fuse_pro = FusePro_Select(**fuse_pro_params)
        # decoder
        # not use now
        # other task
        cfg = self.model_dict['cfg']
        cfg_component = cfg.component
        self.single_enc_type = 'none'
    def permute_feat(self,xs: [torch.Tensor]) -> [torch.Tensor]:
        """
        xs: list of x, x [B,C,H,W]
        out: xs: list of x, x [B,H,W,C]
        """
        for idx in range(len(xs)):
            xs[idx] = xs[idx].permute(0,2,3,1)
        return xs

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x)
        x3 = self.enc3(x)
        xs = [x1,x2,x3]
        xs = self.fuse_pre(xs)
        xs = self.permute_feat(xs)
        # if self.single_enc_type == 'addnull':
        #     x2 = xs[1]
        #     x1_null = torch.zeros_like(xs[0])
        #     x3_null = torch.zeros_like(xs[2])
        #     xs = [x1_null,x2,x3_null]
        # elif self.single_enc_type == 'addsame':
        #     x2 = xs[1]
        #     xs = [x2,x2,x2]
        x = self.fuse_pro(xs)
        # x = self.dec(x)
        return x
def check_model_dict(model_dict,available_enc_list = ['medsam']):
    for key in model_dict.keys():
        if 'fuse' in key:
            continue
        if key not in available_enc_list:
            print(f'{key} not in available enc list.')
            raise ValueError
    return 0

AVAILABLE_ENC_MODEL_LIST = ['medsam','vit_ssl','plip']

class MultiModelArch(nn.Module):
    def __init__(self, model_dict):
        super(MultiModelArch,self).__init__()
        # num_model = model_dict['num_model']
        self.model_dict = model_dict
        cfg = model_dict['cfg']
        enc_names = cfg.component.model_list
        self.enc_names = enc_names
        for enc_name in enc_names:
            enc_params = model_dict[enc_name]
            if enc_name == 'medsam':
                from .image_encoder import ImageEncoderViTMS
                enc = ImageEncoderViTMS(**enc_params)
                self.__setattr__(enc_name, enc)
            elif enc_name == 'vit_ssl':
                from .image_encoder import ImageEncoderViTMS
                enc = ImageEncoderViTMS(**enc_params)
                self.__setattr__(enc_name, enc)
            elif enc_name == 'plip':
                from ..plip.modeling_clip import CLIPVisionTransformer_my
                clip_vcfger = enc_params['clip_vcfger']
                enc = CLIPVisionTransformer_my(clip_vcfger)
                self.__setattr__(enc_name, enc)
            elif enc_name == 'pidinet':
                from ..pidinet.pidinet import PiDiNet
                enc = PiDiNet(**enc_params)
                self.__setattr__(enc_name, enc)
            else:
                print(f'Error. {enc_name} has not been implemented.')
                raise ValueError
        # fuse architecture
        from . import FusePre, FusePro, FusePro_Select, FuseFin_Select
        fuse_pre_params = model_dict['fuse_pre']
        self.fuse_pre = FusePre(**fuse_pre_params)
        fuse_pro_params = model_dict['fuse_pro']
        self.fuse_pro = FusePro_Select(**fuse_pro_params)
        fuse_fin_params = model_dict['fuse_fin']
        self.fuse_fin = FuseFin_Select(**fuse_fin_params)
        # decoder
        # not use now
        # other task
        cfg = self.model_dict['cfg']
        cfg_component = cfg.component
        self.single_enc_type = 'none'
    def permute_feat(self,xs: [torch.Tensor]) -> [torch.Tensor]:
        """
        xs: list of x, x [B,C,H,W]
        out: xs: list of x, x [B,H,W,C]
        """
        for idx in range(len(xs)):
            xs[idx] = xs[idx].permute(0,2,3,1)
        return xs

    # def xs_addnull(self, xs):
    #     for idx, enc_name in enumerate(self.enc_names):
    #         if enc_name == 'medsam':
    #             x_ = xs[idx]
    #         else:
    #             x_ = torch.zeros_like(xs[idx])
    #         xs[idx] = x_
    #     return xs

    def forward(self, x):
        xs = []
        for enc_name in self.enc_names:
            enc_ = self.__getattr__(enc_name)
            x_ = enc_(x)
            xs.append(x_)
        # if self.single_enc_type == 'addnull':
        #     xs = self.xs_addnull(xs)
        # elif self.single_enc_type == 'addsame':
        #     for idx, enc_name in enumerate(self.enc_names):
        #         enc_names = tuple(self.enc_names)
        #         idx_medsam = enc_names.index('medsam')
        #         x_ = xs[idx_medsam]
        #         xs[idx] = x_

        xs = self.fuse_pre(xs)

        # if self.single_enc_type == 'addnull':
        #     xs = self.xs_addnull(xs)

        xs = self.permute_feat(xs)
        xs = self.fuse_pro(xs)

        # if self.single_enc_type == 'addnull':
        #     xs = self.xs_addnull(xs)

        x = self.fuse_fin(xs)
        # x = self.dec(x)
        return x


def build(config):
    # model_dict = {}
    # enc1_name = config.ENCODER1.NAME
    # enc1_param = config.ENCODER1
    # model_dict.update({f'{enc1_name}':enc1_param})
    # enc2_name = config.ENCODER2.NAME
    # enc2_param = config.ENCODER2
    # model_dict.update({f'{enc2_name}':enc2_param})
    # enc3_name = config.ENCODER3.NAME
    # enc3_param = config.ENCODER3
    # model_dict.update({f'{enc3_name}':enc3_param})
    # fuse_name = config.FUSEARCH.NAME
    # fuse_param = config.FUSEARCH
    # model_dict.update({f'{fuse_name}':fuse_param})
    # dec_name = config.SAM_DECODER.NAME
    # dec_param = config.SAM_DECODER
    # model_dict.update({f'{dec_name}':dec_param})
    from .utils import config_to_model_dict
    model_dict = config_to_model_dict(config)
    model = MultiModelArch(model_dict)

    return model

def build_simmim_multimodel(config, logger):
    # model_type = config.MODEL.TYPE
    image_size = config.DATA.IMG_SIZE
    # image_embedding_size = image_size // vit_patch_size  # Divide by 16 here
    # prompt_embed_dim = 256
    from .utils import config_to_model_dict
    model_dict = config_to_model_dict(config)
    encoder = MultiModelArchForSimMIM(model_dict = model_dict)
    encoder_stride = config.MODEL.FUSE_PRO.PATCH_SIZE
    model = SimMIM_forMultiModelArch(encoder=encoder, encoder_stride=encoder_stride)
    model = load_checkpoint_sam_enc(config, model, logger)
    return model

def check_update_ckpt_param_number(model_dict, new_dict):
    count = 0
    for k_new in new_dict.keys():
        for k_ori in model_dict.keys():
            if k_ori == k_new:
                count += 1
    return count

def load_checkpoint_sam_enc(config, model, logger, freeze=True):
    logger.info("Start loading checkpoint of SSL_SAM_VIT and MED_SAM_VIT...")
    model_dict = model.state_dict()
    model_keys = model_dict.keys()
    # SSL SAM
    ckpt_path = config.MODEL.SSL_SAM_VIT.PRETRAINED_PATH
    logger.info(f"Checkpoint path of SSL_SAM_VIT: {ckpt_path} ")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    ckpt_dict = checkpoint['model']
    enc1_keys = [k.split('encoder.enc1.')[-1] for k in model_keys if 'enc1' in k]
    ckpt_keys = [f'encoder.enc1.{k}' for k in enc1_keys]
    ckpt_values = [ckpt_dict[f'encoder.{k}'] for k in enc1_keys]
    enc1_new_dict = {k: v for k, v in zip(ckpt_keys, ckpt_values)}
    model_dict.update(enc1_new_dict)

    # MED SAM
    ckpt_path = config.MODEL.MED_SAM_VIT.PRETRAINED_PATH
    logger.info(f"Checkpoint path of MED_SAM_VIT: {ckpt_path} ")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    ckpt_dict = checkpoint
    enc2_keys = [k.split('encoder.enc2.')[-1] for k in model_keys if 'enc2' in k]
    ckpt_keys = [f'encoder.enc2.{k}' for k in enc2_keys]

    ckpt_values = [ckpt_dict[f'image_encoder.{k}'] for k in enc2_keys]
    enc2_new_dict = {k: v for k, v in zip(ckpt_keys, ckpt_values)}

    # PILP
    ckpt_path = config.MODEL.PLIP.PRETRAINED_PATH
    logger.info(f"Checkpoint path of PILP: {ckpt_path} ")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    ckpt_dict = checkpoint
    enc3_keys = [k.split('encoder.enc3.')[-1] for k in model_keys if 'enc3' in k]
    ckpt_keys = [f'{k}' for k in enc3_keys]
    ckpt_values = [ckpt_dict[f'vision_model.{k}'] for k in ckpt_keys]
    enc3_new_keys = [f'encoder.enc3.{k}' for k in ckpt_keys]
    enc3_new_dict = {k: v for k, v in zip(enc3_new_keys, ckpt_values)}
    assert len(enc3_new_keys) == len(ckpt_values)
    model_dict.update(enc3_new_dict)

    image_size = config.DATA.IMG_SIZE
    patch_size = config.MODEL.MED_SAM_VIT.PATCH_SIZE
    token_size = image_size // patch_size
    pos_embed = enc2_new_dict['encoder.enc2.pos_embed']
    if pos_embed.shape[1] != token_size:
        # resize pos embedding, which may sacrifice the performance, but I have no better idea
        pos_embed = pos_embed.permute(0, 3, 1, 2)  # [b, c, h, w]
        pos_embed = F.interpolate(pos_embed, (token_size, token_size), mode='bilinear', align_corners=False)
        pos_embed = pos_embed.permute(0, 2, 3, 1)  # [b, h, w, c]
        enc2_new_dict['encoder.enc2.pos_embed'] = pos_embed
        rel_pos_keys = [k for k in enc2_new_dict.keys() if 'rel_pos' in k]
        global_rel_pos_keys = [k for k in rel_pos_keys if '2' in k or '5' in  k or '8' in k or '11' in k]
        for k in global_rel_pos_keys:
            rel_pos_params = enc2_new_dict[k]
            h, w = rel_pos_params.shape
            rel_pos_params = rel_pos_params.unsqueeze(0).unsqueeze(0)
            rel_pos_params = F.interpolate(rel_pos_params, (token_size * 2 - 1, w), mode='bilinear', align_corners=False)
            enc2_new_dict[k] = rel_pos_params[0, 0, ...]
    model_dict.update(enc2_new_dict)

    msg = model.load_state_dict(model_dict, strict=True)
    logger.info(msg)
    del checkpoint
    torch.cuda.empty_cache()
    logger.info("Finish loading checkpoint of SSL_SAM_VIT, MED_SAM_VIT and PLIP.")

    # lets freeze first
    if freeze:
        logger.info("Freeze Encoder1&2&3 parameters.")
        for param in model.encoder.enc1.parameters():
            param.requires_grad = False
        for param in model.encoder.enc2.parameters():
            param.requires_grad = False
        for param in model.encoder.enc3.parameters():
            param.requires_grad = False
    else:
        logger.info("Do not Freeze Encoder1&2 parameters.")

    return model


from timm.models.layers import trunc_normal_
class MultiModelArchForSimMIM(MultiModelArch):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # assert self.num_classes == 0
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.fuse_pre.out_chans))
        self._trunc_normal_(self.mask_token, std=.02)

    def _trunc_normal_(self, tensor, mean=0., std=1.):
        trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)

    def mask_src_feat(self,x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        '''
        x: [B,C,H,W]
        out: x:[B,H,W,C]
        '''
        x = x.flatten(2).permute(0,2,1)
        assert mask is not None
        B, L, C = x.shape

        mask_token = self.mask_token.expand(B, L, -1)
        w = mask.flatten(1).unsqueeze(-1).type_as(mask_token)
        x = x * (1 - w) + mask_token * w
        h = w = int(L**0.5)
        x = x.reshape((B,h,w,C))
        return x

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x = self.patch_embed(x)  # pre embed: [1, 3, 1024, 1024], post embed: [1, 64, 64, 768]
        # x: [B,C,H,W]
        x1 = self.enc1(x)
        x2 = self.enc2(x)
        x1,x2 = self.fuse_pre([x1,x2])
        x1 = self.mask_src_feat(x1,mask)
        x2 = self.mask_src_feat(x2,mask)
        x = self.fuse_pro([x1, x2])
        return x

class SimMIM_forMultiModelArch(nn.Module):
    def __init__(self, encoder, encoder_stride):
        super().__init__()
        self.encoder = encoder
        self.encoder_stride = encoder_stride

        self.decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=self.encoder.fuse_pro.out_chans,
                out_channels=self.encoder_stride ** 2 * 3, kernel_size=1),
            nn.PixelShuffle(self.encoder_stride),
        )

        self.in_chans = self.encoder.fuse_pro.out_chans
        self.patch_size = self.encoder.fuse_pro.patch_size

    def forward(self, x, mask):
        z = self.encoder(x, mask)
        x_rec = self.decoder(z)

        mask = mask.repeat_interleave(self.patch_size, 1).repeat_interleave(self.patch_size, 2).unsqueeze(1).contiguous()
        loss_recon = F.l1_loss(x, x_rec, reduction='none')
        loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / self.in_chans
        return loss

    @torch.jit.ignore
    def no_weight_decay(self):
        if hasattr(self.encoder, 'no_weight_decay'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay()}
        return {}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        if hasattr(self.encoder, 'no_weight_decay_keywords'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay_keywords()}
        return {}