import numpy as np
import torch
import os
import os.path as osp
import torch.nn.functional as F
import importlib
import PIL.Image as Im

class Visualizer:
    def __init__(self,result_dir, cfg=None, postprocessor=None):
        self.ori_scale = (1024, 1024)
        self.use_postprocess = False
        self.mask_th = 0.5
        self.sem_alpha = 0.3 # transparent factor: 1 for only origin image
        self.ct_threshold = 0.3
        self.color_map = np.array([[0, 0, 0], [128, 0, 0], [0, 128, 0], [0, 224, 224], [0, 0, 128]])
        self.result_dir = result_dir
        self.cfg=cfg
        self.postprocessor= postprocessor
        # self.edge_th = 10
        # self.inp_zero_bias = 16

    def draw_sem_overlapped(self, canvas, inp_ori, sem_mask, color, alpha=0.8):
        '''
        canvas: [H,W,3]
        inp_ori: [H,W,3]
        sem_mask: one class mask, [H,W], range from 0,1
        color: 3len,[r,g,b]
        '''

        assert isinstance(color, (list)) and len(color) == 3
        assert len(sem_mask.shape) == 2
        color_sem = np.array(color)
        mask_th = self.mask_th
        sem_mask = sem_mask[..., np.newaxis] > mask_th
        alpha = alpha  # the transmission of sem_mask
        canvas = canvas * (1 - sem_mask) + sem_mask * ((1 - alpha) * color_sem + alpha * inp_ori)
        # canvas = canvas * (1 - sem_mask) + sem_mask * (alpha * inp_ori)
        # inp = inp * (1 - sem_mask) + sem_mask * ((1 - alpha) * color_sem + alpha * inp)
        return canvas

    def visualize_semantic_mask_overlapped_cv2_once(self, img, mask, case_name):
        '''
        for vis paper
        pred_mask: should be non-act logits, shape[1,C,H,W]
        img: [1,3,H,W]
        '''
        self.colors = [[128, 0, 0], [0, 128, 0], [0, 224, 224], [0, 0, 128]]
        alpha = 0.5
        _,_,h,w = img.shape
        # pred_mask = out_dict['masks']
        pred_mask = torch.sigmoid(mask)
        pred_mask = F.interpolate(
            pred_mask.type(torch.float),
            (h, w),
            mode="bilinear",
            align_corners=False,
        )
        img = img.detach().cpu().numpy()[0,...]*255
        img = np.transpose(img,(1,2,0))
        dataset_pack = importlib.import_module(f'lib.datasets.{self.cfg.dataset}.{self.cfg.task}')
        SEM_CLS_FIN_LIST = dataset_pack.SEM_CLS_FIN_LIST
        if self.postprocessor is not None:
            pred_mask = self.postprocessor.post_processing_direct(pred_mask)
        pred_masks = pred_mask[0,...].detach().cpu().numpy()
        img_ori = img.copy()
        sem_mask = (pred_masks>self.mask_th).transpose(1,2,0).astype(bool)
        # sem_mask = (pred_masks>0).transpose(1,2,0).astype(bool)
        # print(color.shape
        canvas_fin = np.asarray(img_ori)
        Ncls = len(SEM_CLS_FIN_LIST)
        for idx_cls in range(Ncls):
            mask_ = sem_mask[..., idx_cls]  # 0-1 probs
            color = self.colors[idx_cls]
            canvas_fin = self.draw_sem_overlapped(canvas_fin, img_ori, mask_, color=color, alpha=alpha)
        # save
        root_path = osp.join(self.result_dir, 'vis')
        if not os.path.exists(root_path):
            os.makedirs(root_path)
        img_name = case_name
        img_path = osp.join(root_path, img_name + '.jpg')
        Im.fromarray(canvas_fin.astype(np.uint8)).save(img_path)
        return 0

    def vis_iter(self,img, out_dict, case_name):
        Niter = self.cfg.component.test_iter
        for idx in range(Niter+1):
            # if idx == 0:
            #     mask = out_dict['low_res_logits']
            if idx < Niter:
                mask = out_dict['high_res_logits_list'][idx]
            else:
                mask = out_dict['masks']
            # if idx < Niter+1:
            #     continue
            case_name2 = case_name + f'_iter{idx}'
            self.visualize_semantic_mask_overlapped_cv2_once(img,mask,case_name2)
            break # only first iter
        return 0

    def visualize(self, img, out_dict, case_name):
        self.vis_iter(img, out_dict, case_name)



