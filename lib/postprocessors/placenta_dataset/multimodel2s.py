import numpy as np
import torch
from . import utils
from torch.nn import functional as F

class Postprocessor:
    def __init__(self, adapt_cfg):
        self.cfg = adapt_cfg
        self.post_types = self.cfg.post.post_types

    @staticmethod
    def argmax(masks):
        '''
        can only used for pred with background class
        '''
        sem_masks = masks
        B,Ncls,H,W = sem_masks.shape
        assert Ncls > 1
        arg_map = sem_masks.argmax(dim=1) # [B,H,W]
        masks_post = F.one_hot(arg_map,num_classes=-1) #[B,H,W,Ncls]
        masks_post = masks_post.permute(0,-1,1,2)
        return masks_post

    @staticmethod
    def heuristic(masks):
        sem_masks = masks
        sem_masks = sem_masks > 0.5
        B,Ncls,H,W = sem_masks.shape
        assert Ncls > 1
        sem_Mask = torch.zeros((B,H,W),dtype=torch.long,device=masks.device)
        for i in range(Ncls):
            sem_Mask[sem_masks[:, i, :, :]] = i + 1
        masks_post = F.one_hot(sem_Mask,num_classes=-1) #[B,H,W,Ncls]
        masks_post = masks_post.permute(0,-1,1,2)
        masks_post = masks_post[:,1:,...]
        return masks_post

    @staticmethod
    def equal(masks):
        sem_masks = (masks > 0.5).type(torch.float)
        B,Ncls,H,W = sem_masks.shape
        assert Ncls > 1
        masks_post = sem_masks
        return masks_post

    def post_processing(self, out_ori):
        '''
        do post processing step
        note: masks should be non-sigmoid, ranging from large variance
        '''
        # 1
        post_required = self.cfg.post.required
        if not post_required:
            out_ori.update({'masks_post':out_ori['masks']})
            return
        # start
        available_methods = [method for method in dir(self) if callable(getattr(self,method))]
        masks = out_ori['masks']
        for _type,_required in iter(self.post_types.items()):
            assert _type in available_methods
            if  _required==True:
                masks = self.__getattribute__(_type)(masks)
        out_ori.update({'masks_post':masks})

        return

    def post_processing_direct(self, masks):
        '''
        do post processing step, this version is for direct use by inputing masks
        note: masks should be probabilities, ranging from [0,1]
        '''
        # 1
        B,C,H,W = masks.shape
        post_required = self.cfg.post.required
        if not post_required:
            return masks
        # start
        masks_post = masks
        available_methods = [method for method in dir(self) if callable(getattr(self,method))]
        for _type,_required in iter(self.post_types.items()):
            assert _type in available_methods
            if _required==True:
                masks_post = self.__getattribute__(_type)(masks)
        return masks_post