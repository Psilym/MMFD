import numpy as np
import PIL
import cv2
from typing import Any, Callable, List, Optional, Tuple
import random
import albumentations as A
from lib.utils import img_utils

AUGMENT_METHOD_LIST = ['Resize', 'Flip', 'RandomScale','RandomBrightnessContrast',
                       'AddMultipleNoise','GridDistortion','ElasticTransform','RandomGamma',
                       'Rotate','Affine','HEBDecompose']
def obtain_class_object(class_name: str):
    return eval(class_name)

class MultiTransform():
    """
    A class for collecting multiple transforms.
    Note: All transforms should be a callable class.
    """
    def __init__(self):
        self.trans_seq = []

        self.available_aug_list = AUGMENT_METHOD_LIST

    def append(self, aug_name: str, **kwargs):
        if aug_name in self.available_aug_list:
            trans_Cls = eval(aug_name)
            trans_cls = trans_Cls(**kwargs)
            trans_func = trans_cls # callable
            self.trans_seq.append(trans_func)
        else:
            print(f'Not find {aug_name} in list.')
            raise ValueError

    def __call__(self,img, sem_mask=None, ins_mask=None):
        img_ = img
        sem_mask_ = sem_mask
        ins_mask_ = ins_mask
        for idx_aug in range(len(self.trans_seq)):
            trans_func = self.trans_seq[idx_aug]
            img_, sem_mask_, ins_mask_ = trans_func(img_,sem_mask_,ins_mask_)

        return img_, sem_mask_, ins_mask_



class Resize():
    """Resize"""
    def __init__(self, size: Tuple = (1024,1024)):
        self.size = size
    def __call__(self, img, sem_mask=None, ins_mask=None):
        out_trans = []
        # img
        img_trans = img_utils.resize_my(img,self.size)
        out_trans.append(img_trans)
        # sem_trans
        if sem_mask is not None:
            mask_trans = img_utils.resize_my(sem_mask,self.size)
        out_trans.append(mask_trans)
        # ins_trans
        if ins_mask is not None:
            mask_trans = img_utils.resize_my(ins_mask,self.size)
        out_trans.append(mask_trans)
        return out_trans

class Flip():
    """Flip"""
    def __init__(self, use_horizontal=True, use_vertical=True, prob_th = 0.5):
        self.use_horizontal = use_horizontal
        self.use_vertical = use_vertical
        self.prob_th = 0.5
    def __call__(self, img, sem_mask, ins_mask):
        """
        img: ndarray, [C,H,W]
        sem_mask: ndarray, [C,H,W]
        """
        out_trans = []
        use_h = random.random() > self.prob_th
        use_v = random.random() > self.prob_th
        # img
        img_trans = img
        img_trans = img_trans[:,::-1,:] if use_h else img_trans
        img_trans = img_trans[:,:,::-1] if use_v else img_trans
        out_trans.append(img_trans)
        # sem_trans
        if sem_mask is not None:
            sem_mask = sem_mask
            sem_mask = sem_mask[:,::-1,:] if use_h else sem_mask
            sem_mask = sem_mask[:,:,::-1] if use_v else sem_mask
        out_trans.append(sem_mask)
        # ins_trans
        if ins_mask is not None:
            ins_mask = ins_mask[:, ::-1, :] if use_h else ins_mask
            ins_mask = ins_mask[:, :, ::-1] if use_v else ins_mask
        out_trans.append(ins_mask)
        return out_trans

class RandomScale():
    """RandomScale"""
    def __init__(self,scale_range=(1.0,1.0)):
        assert len(scale_range) == 2
        self.scale_range = scale_range
        self.random_scale = A.RandomScale(scale_limit=(scale_range[0]-1,scale_range[1]-1),
                                          interpolation=cv2.INTER_LINEAR,
                                          always_apply=True,
                                          p=1.0)

    def __call__(self, img, sem_mask, ins_mask):
        out_trans = []
        # img
        if sem_mask is None and ins_mask is None:
            img = img.transpose(1,2,0)
            augmented = self.random_scale(image=img)
            img_trans = augmented['image'].transpose(2, 0, 1)
            out_trans.append(img_trans)
            out_trans.append(sem_mask)
            out_trans.append(ins_mask)
        else:
            img = img.transpose(1,2,0)
            Nsem = sem_mask.shape[0]
            masks = np.concatenate([sem_mask,ins_mask],axis=0)
            masks = masks.transpose(1, 2, 0)
            augmented = self.random_scale(image=img, mask=masks.astype(float))
            img_trans = augmented['image'].transpose(2, 0, 1)
            masks = augmented['mask'].astype(masks.dtype).transpose(2, 0, 1)
            sem_mask = masks[:Nsem,...]
            ins_mask = masks[Nsem:,...]
            out_trans.append(img_trans)
            out_trans.append(sem_mask)
            out_trans.append(ins_mask)

        return out_trans

class HEBDecompose():
    """HEBDecompose"""
    def __init__(self,use=True):
        self.use = use
        from skimage.color import rgb2hed
        self.rgb2hed = rgb2hed
    def __call__(self, img, sem_mask=None, ins_mask=None):
        '''
        img: range from [0,1]
        '''
        out_trans = []
        # img
        if self.use:
            img = img.transpose(1,2,0)
            hed = self.rgb2hed(img)
            img_trans = hed.transpose(2,0,1)
        else:
            img_trans = img
        out_trans.append(img_trans)
        out_trans.append(sem_mask)
        out_trans.append(ins_mask)

        return out_trans

def apply_Atrans(A_obj,img,sem_mask=None,ins_mask=None):
    """
    img:shape [C,H,W]
    sem_mask:shape [Nsem,H,W]
    ins_mask:shape [Nins,H,W]
    """
    out_trans = []
    img = img.transpose(1,2,0)
    if sem_mask is None and ins_mask is None:
        out_trans.append(A_obj(image=img)['image'])
        return out_trans
    if sem_mask is not None or ins_mask is not None:
        Nsem = len(sem_mask) if sem_mask is not None else 0
        # Nins = len(ins_mask) if ins_mask is not None else 0
        # Ntot = Nsem + Nins
        tot_mask_list = []
        if sem_mask is not None:
            tot_mask_list.append(sem_mask)
        if ins_mask is not None:
            tot_mask_list.append(ins_mask)
        tot_mask = np.concatenate(tot_mask_list, axis=0).transpose(1,2,0)
        augment = A_obj(image=img,mask=tot_mask)
        imga = augment['image'].transpose(2,0,1)
        out_trans.append(imga)
        maska = augment['mask'].transpose(2,0,1)
        sem_maska = maska[:Nsem,...] if sem_mask is not None else None #TO-DO: test N=zero condition
        ins_maska = maska[Nsem:,...] if ins_mask is not None else None
        out_trans.append(sem_maska)
        out_trans.append(ins_maska)
        return out_trans

class RandomBrightnessContrast():
    """RandomBrightnessContrast"""
    def __init__(self,p=1.0):
        self.p = p
        self.a_obj = A.RandomBrightnessContrast(p=self.p)

    def __call__(self, img, sem_mask=None, ins_mask=None):
        out_trans = apply_Atrans(self.a_obj,img,sem_mask=sem_mask,ins_mask=ins_mask)
        return out_trans


class AddMultipleNoise():
    def __init__(self,p):
        self.p = p
        self.a_obj_gn = A.GaussNoise(var_limit=5. / 255., p=self.p)
        self.a_obj_mn = A.MultiplicativeNoise(p=self.p)

    def __call__(self, img, sem_mask=None, ins_mask=None):
        r = np.random.uniform()
        if r < 0.50:
            out_trans = apply_Atrans(self.a_obj_gn,img,sem_mask=sem_mask,ins_mask=ins_mask)
        else:
            out_trans = apply_Atrans(self.a_obj_mn,img,sem_mask=sem_mask,ins_mask=ins_mask)

        return out_trans

class GridDistortion():
    """GridDistortion"""
    def __init__(self,p=1.0):
        self.p = p
        self.a_obj = A.GridDistortion(p=self.p)

    def __call__(self, img, sem_mask=None, ins_mask=None):
        out_trans = apply_Atrans(self.a_obj,img,sem_mask=sem_mask,ins_mask=ins_mask)
        return out_trans

class ElasticTransform():
    """ElasticTransform"""
    def __init__(self,p=1.0):
        self.p = p
        self.a_obj = A.ElasticTransform(sigma=50, alpha=1,  p=self.p)

    def __call__(self, img, sem_mask=None, ins_mask=None):
        out_trans = apply_Atrans(self.a_obj,img,sem_mask=sem_mask,ins_mask=ins_mask)
        return out_trans

class RandomGamma():
    """RandomGamma"""
    def __init__(self,p=1.0):
        self.p = p
        self.a_obj = A.RandomGamma((70, 150), p=self.p)

    def __call__(self, img, sem_mask=None, ins_mask=None):
        out_trans = apply_Atrans(self.a_obj,img,sem_mask=sem_mask,ins_mask=ins_mask)
        return out_trans

class Rotate():
    """Rotate"""
    def __init__(self,p=1.0):
        self.p = p
        self.a_obj = A.Rotate(p=self.p,crop_border=True)

    def __call__(self, img, sem_mask=None, ins_mask=None):
        out_trans = apply_Atrans(self.a_obj,img,sem_mask=sem_mask,ins_mask=ins_mask)
        return out_trans

class Affine():
    """Affine: including scale, translate, rotate, shear"""
    def __init__(self,scale_range=(0.7,1.1),translate_percent=(-0.3,0.3),
                 rotate=(-360,360),shear=(-30,30),p=1.0):
        self.p = p
        self.a_obj = A.Affine(scale=scale_range,translate_percent=translate_percent,
                              rotate=rotate,shear=shear,p=self.p)

    def __call__(self, img, sem_mask=None, ins_mask=None):
        out_trans = apply_Atrans(self.a_obj,img,sem_mask=sem_mask,ins_mask=ins_mask)
        return out_trans

# class LargeScaleJitter(object):
#     """
#         implementation of large scale jitter from copy_paste
# https://github.com/gaopengcuhk/Pretrained-Pix2Seq/blob/7d908d499212bfabd33aeaa838778a6bfb7b84cc/datasets/transforms.py
#     """
#
#     def __init__(self, output_size=1024, aug_scale_min=0.1, aug_scale_max=2.0):
#         self.desired_size = torch.tensor(output_size)
#         self.aug_scale_min = aug_scale_min
#         self.aug_scale_max = aug_scale_max
#
#     def pad_target(self, padding, target):
#         target = target.copy()
#         if "masks" in target:
#             target['masks'] = torch.nn.functional.pad(target['masks'], (0, padding[1], 0, padding[0]))
#         return target
#
#     def __call__(self, sample):
#         imidx, image, label, image_size =  sample['imidx'], sample['image'], sample['label'], sample['shape']
#
#         #resize keep ratio
#         out_desired_size = (self.desired_size * image_size / max(image_size)).round().int()
#
#         random_scale = torch.rand(1) * (self.aug_scale_max - self.aug_scale_min) + self.aug_scale_min
#         scaled_size = (random_scale * self.desired_size).round()
#
#         scale = torch.minimum(scaled_size / image_size[0], scaled_size / image_size[1])
#         scaled_size = (image_size * scale).round().long()
#
#         scaled_image = torch.squeeze(F.interpolate(torch.unsqueeze(image,0),scaled_size.tolist(),mode='bilinear'),dim=0)
#         scaled_label = torch.squeeze(F.interpolate(torch.unsqueeze(label,0),scaled_size.tolist(),mode='bilinear'),dim=0)
#
#         # random crop
#         crop_size = (min(self.desired_size, scaled_size[0]), min(self.desired_size, scaled_size[1]))
#
#         margin_h = max(scaled_size[0] - crop_size[0], 0).item()
#         margin_w = max(scaled_size[1] - crop_size[1], 0).item()
#         offset_h = np.random.randint(0, margin_h + 1)
#         offset_w = np.random.randint(0, margin_w + 1)
#         crop_y1, crop_y2 = offset_h, offset_h + crop_size[0].item()
#         crop_x1, crop_x2 = offset_w, offset_w + crop_size[1].item()
#
#         scaled_image = scaled_image[:,crop_y1:crop_y2, crop_x1:crop_x2]
#         scaled_label = scaled_label[:,crop_y1:crop_y2, crop_x1:crop_x2]
#
#         # pad
#         padding_h = max(self.desired_size - scaled_image.size(1), 0).item()
#         padding_w = max(self.desired_size - scaled_image.size(2), 0).item()
#         image = F.pad(scaled_image, [0,padding_w, 0,padding_h],value=128)
#         label = F.pad(scaled_label, [0,padding_w, 0,padding_h],value=0)
#
#         return {'imidx':imidx,'image':image, 'label':label, 'shape':torch.tensor(image.shape[-2:])}