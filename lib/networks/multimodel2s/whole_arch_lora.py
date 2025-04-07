import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
from .whole_arch import MultiModelArch
from .whole_arch_seg import MultiModelSeg
from safetensors import safe_open
from safetensors.torch import save_file

from icecream import ic


class _LoRA_qkv(nn.Module):
    """In Sam it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    """

    def __init__(
            self,
            qkv: nn.Module,
            linear_a_q: nn.Module,
            linear_b_q: nn.Module,
            linear_a_v: nn.Module,
            linear_b_v: nn.Module,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)

    def forward(self, x):
        qkv = self.qkv(x)  # B,N,N,3*org_C
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        qkv[:, :, :, : self.dim] += new_q
        qkv[:, :, :, -self.dim:] += new_v
        return qkv

class _LoRA_cross_q(nn.Module):
    """In FusePro, use CrossAtten it is implemented as
            B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        kv = self.kv_lin(x).reshape(B, H * W, 2, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q = q.unsqueeze(0).repeat((B,1,1,1))
        q = self.q_lin(q).reshape(B, 1, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # print(f'1 q:{q.shape}')
        # q, k, v with shape (B * nHead, H * W, C)
        k, v = kv.reshape(2, B * self.num_heads, H * W, -1).unbind(0)
        q = q.reshape(1, B * self.num_heads, 1, -1)[0]
    """

    def __init__(
            self,
            q_lin: nn.Module,
            linear_a_q: nn.Module,
            linear_b_q: nn.Module,
    ):
        super().__init__()
        self.q_lin = q_lin
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.dim = q_lin.in_features

    def forward(self, x):
        '''
        x: [B,N,N,org_C]
        '''
        q = self.q_lin(x)  # B,N,N,1*org_C
        new_q = self.linear_b_q(self.linear_a_q(x))
        q += new_q
        return q

class _LoRA_cross_kv(nn.Module):
    """In FusePro, use CrossAtten it is implemented as
            B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        kv = self.kv_lin(x).reshape(B, H * W, 2, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q = q.unsqueeze(0).repeat((B,1,1,1))
        q = self.q_lin(q).reshape(B, 1, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # print(f'1 q:{q.shape}')
        # q, k, v with shape (B * nHead, H * W, C)
        k, v = kv.reshape(2, B * self.num_heads, H * W, -1).unbind(0)
        q = q.reshape(1, B * self.num_heads, 1, -1)[0]
    """

    def __init__(
            self,
            kv_lin: nn.Module,
            linear_a_v: nn.Module,
            linear_b_v: nn.Module,
    ):
        super().__init__()
        self.kv_lin = kv_lin
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = kv_lin.in_features

    def forward(self, x):
        '''
        x: [B,N,N,org_C]
        '''
        kv = self.kv_lin(x)  # B,N,N,1*org_C
        new_v = self.linear_b_v(self.linear_a_v(x))
        kv[:, :, :, -self.dim:] += new_v
        return kv

class LoRA_MultiModelSeg(nn.Module):
    """Applies low-rank adaptation to model's image encoder.

    Args:
        sam_model: a vision transformer model, see base_vit.py
        r: rank of LoRA
        num_classes: how many classes the model output, default to the vit model
        lora_layer: which layer we apply LoRA.

    Examples::
        >>> model = ViT('B_16_imagenet1k')
        >>> lora_model = LoRA_ViT(model, r=4)
        >>> preds = lora_model(img)
        >>> print(preds.shape)
        torch.Size([1, 1000])
    """

    def __init__(self, model: MultiModelSeg, r: int, lora_layer=None):
        super().__init__()

        assert r > 0
        # base_vit_dim = sam_model.image_encoder.patch_embed.proj.out_channels
        # dim = base_vit_dim
        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(
                range(len(model.image_encoder.fuse_pro.blocks)))  # Only apply lora to the image encoder by default
        # create for storage, then we can init them or load weights
        self.w_As = []  # These are linear layers
        self.w_Bs = []

        # lets freeze first
        for param in model.image_encoder.enc1.parameters():
            param.requires_grad = False
        for param in model.image_encoder.enc2.parameters():
            param.requires_grad = False
        for param in model.image_encoder.fuse_pre.parameters():
            param.requires_grad = False
        for param in model.image_encoder.fuse_pro.parameters():
            param.requires_grad = False

        self.create_fuse_lora(model)
        # Here, we do the surgery
        for t_layer_i, blk in enumerate(model.image_encoder.fuse_pro.blocks):
            # If we only want few lora layer instead of all
            if t_layer_i not in self.lora_layer:
                continue
            for ca_layer_i, cblk in enumerate(blk.cblocks):
                w_kv_linear = cblk.attn.kv_lin
                w_q_linear = cblk.attn.q_lin
                self.dim = w_kv_linear.in_features
                w_a_linear_q = nn.Linear(self.dim, r, bias=False)
                w_b_linear_q = nn.Linear(r, self.dim, bias=False)
                w_a_linear_v = nn.Linear(self.dim, r, bias=False)
                w_b_linear_v = nn.Linear(r, self.dim, bias=False)
                self.w_As.append(w_a_linear_q)
                self.w_Bs.append(w_b_linear_q)
                self.w_As.append(w_a_linear_v)
                self.w_Bs.append(w_b_linear_v)
                cblk.attn.q_lin = _LoRA_cross_q(
                    w_q_linear,
                    w_a_linear_q,
                    w_b_linear_q,
                )
                cblk.attn.kv_lin = _LoRA_cross_kv(
                    w_kv_linear,
                    w_a_linear_v,
                    w_b_linear_v,
                )

        self.reset_parameters()
        self.sam = model

    def save_lora_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.

        save both lora and fc parameters.
        """

        assert filename.endswith(".pt") or filename.endswith('.pth')

        num_layer = len(self.w_As)  # actually, it is half
        a_tensors = {f"w_a_{i:03d}": self.w_As[i].weight for i in range(num_layer)}
        b_tensors = {f"w_b_{i:03d}": self.w_Bs[i].weight for i in range(num_layer)}
        prompt_encoder_tensors = {}
        mask_decoder_tensors = {}
        # save prompt encoder, only `state_dict`, the `named_parameter` is not permitted
        if isinstance(self.sam, torch.nn.DataParallel) or isinstance(self.sam, torch.nn.parallel.DistributedDataParallel):
            state_dict = self.sam.module.state_dict()
        else:
            state_dict = self.sam.state_dict()
        for key, value in state_dict.items():
            if 'prompt_encoder' in key:
                prompt_encoder_tensors[key] = value
            if 'mask_decoder' in key:
                mask_decoder_tensors[key] = value

        merged_dict = {**a_tensors, **b_tensors, **prompt_encoder_tensors, **mask_decoder_tensors}
        torch.save(merged_dict, filename)

    def load_lora_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.\

        load both lora and fc parameters.
        """

        assert filename.endswith(".pt") or filename.endswith('.pth')
        print(f'Load lora parameters from checkpoint file {filename}')
        state_dict = torch.load(filename)

        for i, w_A_linear in enumerate(self.w_As):
            saved_key = f"w_a_{i:03d}"
            saved_tensor = state_dict[saved_key]
            w_A_linear.weight = Parameter(saved_tensor)

        for i, w_B_linear in enumerate(self.w_Bs):
            saved_key = f"w_b_{i:03d}"
            saved_tensor = state_dict[saved_key]
            w_B_linear.weight = Parameter(saved_tensor)

        sam_dict = self.sam.state_dict()
        sam_keys = sam_dict.keys()

        # load prompt encoder
        prompt_encoder_keys = [k for k in sam_keys if 'prompt_encoder' in k]
        prompt_encoder_values = [state_dict[k] for k in prompt_encoder_keys]
        prompt_encoder_new_state_dict = {k: v for k, v in zip(prompt_encoder_keys, prompt_encoder_values)}
        sam_dict.update(prompt_encoder_new_state_dict)

        # load mask decoder
        mask_decoder_keys = [k for k in sam_keys if 'mask_decoder' in k]
        mask_decoder_values = [state_dict[k] for k in mask_decoder_keys]
        mask_decoder_new_state_dict = {k: v for k, v in zip(mask_decoder_keys, mask_decoder_values)}
        sam_dict.update(mask_decoder_new_state_dict)
        self.sam.load_state_dict(sam_dict)

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def forward(self, batched_input, mode='train', require_embed=False):
        return self.sam(batched_input, mode=mode, require_embed=require_embed)

class FullFT_MultiModelSeg(nn.Module):
    """Applies full finetuning to model.

    Args:
        sam_model: a vision transformer model, see base_vit.py
    """

    def __init__(self, model: MultiModelSeg,model_freeze_list:[]):
        super().__init__()

        # lets freeze first
        for param in model.parameters():
            param.requires_grad = True

        for enc_name in model_freeze_list:
            for param in model.image_encoder.__getattr__(f'{enc_name}').parameters():
                param.requires_grad = False

        self.sam = model

    def save_lora_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.

        save both lora and fc parameters.
        """

        assert filename.endswith(".pt") or filename.endswith('.pth')
        merged_dict = self.sam.state_dict()
        torch.save(merged_dict, filename)

    def load_lora_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.\

        load both lora and fc parameters.
        """

        assert filename.endswith(".pt") or filename.endswith('.pth')
        print(f'Load lora parameters from checkpoint file {filename}')
        state_dict = torch.load(filename)

        sam_dict = self.sam.state_dict()

        self.sam.load_state_dict(state_dict)


    def forward(self, batched_input, mode='train', require_embed=False):
        return self.sam(batched_input, mode=mode, require_embed=require_embed)



class LoRA_MultiModelSeg_Selective(nn.Module):
    """Applies low-rank adaptation to model's image encoder.

    Args:
        sam_model: a vision transformer model, see base_vit.py
        r: rank of LoRA
        num_classes: how many classes the model output, default to the vit model
        lora_layer: which layer we apply LoRA.

    Examples::
        >>> model = ViT('B_16_imagenet1k')
        >>> lora_model = LoRA_ViT(model, r=4)
        >>> preds = lora_model(img)
        >>> print(preds.shape)
        torch.Size([1, 1000])
    """

    def __init__(self, model: MultiModelSeg, r: int, lora_layer=None, model_for_lora=['fuse']):
        super().__init__()
        AVAILABLE_MODEL_for_LoRA = ['medsam']
        for model_name in model_for_lora:
            assert model_name in AVAILABLE_MODEL_for_LoRA
        self.model_for_lora_list = model_for_lora
        self.model_no_freeze_list = ['fuse_pre','fuse_pro','fuse_fin']
        # AVAILABLE_MODEL_for_NOFREEZE = ['fuse_pre','fuse_pro']
        # for model_name in model_no_freeze:
        #     assert model_name in AVAILABLE_MODEL_for_NOFREEZE
        # self.model_no_freeze_list = model_no_freeze
        assert r > 0
        self.r = r

        # base_vit_dim = sam_model.image_encoder.patch_embed.proj.out_channels
        # dim = base_vit_dim
        # create for storage, then we can init them or load weights
        self.w_As = []  # These are linear layers
        self.w_Bs = []

        # lets freeze first
        enc_names = model.image_encoder.enc_names
        for enc_name in enc_names:
            for param in model.image_encoder.__getattr__(f'{enc_name}').parameters():
                param.requires_grad = False

        if lora_layer:
            self.lora_layer = lora_layer
        else:
            Nlayer = 0
            for model_name in model_for_lora:
                assert model_name in AVAILABLE_MODEL_for_LoRA
                Nlayer += len(model.image_encoder.__getattr__(f'{model_name}').blocks)
            self.lora_layer = list(
                range(Nlayer))  # Only apply lora to the image encoder by default
        for model_name in model_for_lora:
            if model_name == 'medsam':
                self.create_medsam_lora(model)
            else:
                raise ValueError

        self.reset_parameters()
        self.sam = model

    def create_medsam_lora(self, model):
        for t_layer_i, blk in enumerate(model.image_encoder.medsam.blocks):
            # If we only want few lora layer instead of all
            if t_layer_i not in self.lora_layer:
                continue
            w_qkv_linear = blk.attn.qkv
            self.dim = w_qkv_linear.in_features
            w_a_linear_q = nn.Linear(self.dim, self.r, bias=False)
            w_b_linear_q = nn.Linear(self.r, self.dim, bias=False)
            w_a_linear_v = nn.Linear(self.dim, self.r, bias=False)
            w_b_linear_v = nn.Linear(self.r, self.dim, bias=False)
            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)
            blk.attn.qkv = _LoRA_qkv(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v,
            )

    def save_lora_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.

        save both lora and fc parameters.
        """

        assert filename.endswith(".pt") or filename.endswith('.pth')

        num_layer = len(self.w_As)  # actually, it is half
        a_tensors = {f"w_a_{i:03d}": self.w_As[i].weight for i in range(num_layer)}
        b_tensors = {f"w_b_{i:03d}": self.w_Bs[i].weight for i in range(num_layer)}
        prompt_encoder_tensors = {}
        mask_decoder_tensors = {}
        # save prompt encoder, only `state_dict`, the `named_parameter` is not permitted
        if isinstance(self.sam, torch.nn.DataParallel) or isinstance(self.sam, torch.nn.parallel.DistributedDataParallel):
            state_dict = self.sam.module.state_dict()
        else:
            state_dict = self.sam.state_dict()
        for key, value in state_dict.items():
            if 'prompt_encoder' in key:
                prompt_encoder_tensors[key] = value
            if 'mask_decoder' in key:
                mask_decoder_tensors[key] = value
        # nofreeze part saving
        fuse_tensors = {}
        if len(self.model_no_freeze_list) > 0:
            for model_name in self.model_no_freeze_list:
                if model_name == 'fuse_pro':
                    for key, value in state_dict.items():
                        if 'fuse_pro' in key:
                            fuse_tensors[key] = value
                elif model_name == 'fuse_pre':
                    for key, value in state_dict.items():
                        if 'fuse_pre' in key:
                            fuse_tensors[key] = value
                elif model_name == 'fuse_fin':
                    for key, value in state_dict.items():
                        if 'fuse_fin' in key:
                            fuse_tensors[key] = value
                else:
                    print('Not implemented yet.')
                    raise ValueError

        merged_dict = {**a_tensors, **b_tensors,
                       **prompt_encoder_tensors, **mask_decoder_tensors,
                       **fuse_tensors}
        torch.save(merged_dict, filename)

    def load_lora_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.\

        load both lora and fc parameters.
        """

        assert filename.endswith(".pt") or filename.endswith('.pth')
        print(f'Load lora parameters from checkpoint file {filename}')
        state_dict = torch.load(filename)

        for i, w_A_linear in enumerate(self.w_As):
            saved_key = f"w_a_{i:03d}"
            saved_tensor = state_dict[saved_key]
            w_A_linear.weight = Parameter(saved_tensor)

        for i, w_B_linear in enumerate(self.w_Bs):
            saved_key = f"w_b_{i:03d}"
            saved_tensor = state_dict[saved_key]
            w_B_linear.weight = Parameter(saved_tensor)

        sam_dict = self.sam.state_dict()
        sam_keys = sam_dict.keys()

        # load prompt encoder
        prompt_encoder_keys = [k for k in sam_keys if 'prompt_encoder' in k]
        prompt_encoder_values = [state_dict[k] for k in prompt_encoder_keys]
        prompt_encoder_new_state_dict = {k: v for k, v in zip(prompt_encoder_keys, prompt_encoder_values)}
        sam_dict.update(prompt_encoder_new_state_dict)

        # load mask decoder
        mask_decoder_keys = [k for k in sam_keys if 'mask_decoder' in k]
        mask_decoder_values = [state_dict[k] for k in mask_decoder_keys]
        mask_decoder_new_state_dict = {k: v for k, v in zip(mask_decoder_keys, mask_decoder_values)}
        sam_dict.update(mask_decoder_new_state_dict)

        # load parts of unfreeze
        if len(self.model_no_freeze_list) > 0:
            for model_name in self.model_no_freeze_list:
                if model_name == 'fuse_pro':
                    fuse_keys = [k for k in sam_keys if 'fuse_pro' in k]
                    fuse_values = [state_dict[k] for k in fuse_keys]
                    fuse_new_state_dict = {k: v for k, v in zip(fuse_keys, fuse_values)}
                    sam_dict.update(fuse_new_state_dict)
                elif model_name == 'fuse_pre':
                    fuse_keys = [k for k in sam_keys if 'fuse_pre' in k]
                    fuse_values = [state_dict[k] for k in fuse_keys]
                    fuse_new_state_dict = {k: v for k, v in zip(fuse_keys, fuse_values)}
                    sam_dict.update(fuse_new_state_dict)
                elif model_name == 'fuse_fin':
                    fuse_keys = [k for k in sam_keys if 'fuse_fin' in k]
                    fuse_values = [state_dict[k] for k in fuse_keys]
                    fuse_new_state_dict = {k: v for k, v in zip(fuse_keys, fuse_values)}
                    sam_dict.update(fuse_new_state_dict)
                else:
                    print('Not implemented yet.')
                    raise ValueError
        self.sam.load_state_dict(sam_dict,strict=True)

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def forward(self, batched_input, mode='train', require_embed=False):
        return self.sam(batched_input, mode=mode, require_embed=require_embed)
