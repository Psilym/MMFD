import os
import cv2
import json
import numpy as np
import torch.nn.functional

import os.path as osp
from lib.utils.pan_seg.eval_utils import multi_iou, multi_scores, multi_hd95, multi_softdice
from typing import List

class Evaluator:
    def __init__(self, cfg,stage):
        self.eval_cfg = cfg
        self.metrics = {}
        self.result_dir = cfg.result_dir
        os.system('mkdir -p {}'.format(self.result_dir))
        self.stage = stage
        self.cfg_component = cfg['component'] if 'component' in cfg.keys() else None

        # semantic
        self.sem_cls_num = cfg.num_classes
        self.MIoU = multi_iou(self.sem_cls_num)
        self.MIoU_list = []
        self.num_images = 0
        if self.stage == 'test':
            self.score_calculor = multi_scores(self.sem_cls_num)
            self.dice_calculor  = multi_softdice(self.sem_cls_num)
            self.dice = []
            self.acc = []
            self.precision = []
            self.recall = []
            self.fscore = []
            self.hd_calculor = multi_hd95(self.sem_cls_num)
            self.hd95 = []

    def clear_cache(self):
        # semantic
        self.MIoU_list = []
        print('Clear evaluator cache.')
        return 0

    def evaluate(self, output, batch):
        pred_mask = torch.sigmoid(output['sem_mask']).cpu()
        sample_name = batch['meta_info']['sample_name']
        mask_gt = batch['sem_mask'].cpu()

        b = mask_gt.shape[0]
        if b != 1:
            print('Error. During evaluation, batch size is not 1.')
            raise ValueError
        iou = self.MIoU.get_multi_iou(pred_mask,mask_gt)
        self.MIoU_list.append(iou)
        if self.stage == 'test':
            output_mask = output['sem_mask']
            output_mask = torch.sigmoid(output_mask).cpu()
            dice = self.dice_calculor.get_multi_dice(output_mask,mask_gt)
            self.dice.append(dice)
            mask_gt = batch['sem_mask'].cpu()
            acc = self.score_calculor.get_multi_acc(output_mask,mask_gt)
            self.acc.append(acc)
            precision = self.score_calculor.get_multi_precision(output_mask,mask_gt)
            self.precision.append(precision)
            recall = self.score_calculor.get_multi_recall(output_mask,mask_gt)
            self.recall.append(recall)
            fscore = self.score_calculor.get_multi_fscore(output_mask,mask_gt)
            self.fscore.append(fscore)
            hd95 = self.hd_calculor.get_multi_hd95(output_mask,mask_gt)
            self.hd95.append(hd95)

    def summarize(self):
        # semantic
        MIoU = sum(self.MIoU_list)/(len(self.MIoU_list)+0.001)
        MIoU = MIoU.tolist()
        for idx in range(len(MIoU)):
            self.metrics.update({f'MIoU_{idx}': MIoU[idx]})
        mean_MIoU = sum(MIoU) / len(MIoU)
        self.metrics.update({f'MIoU_mean': mean_MIoU})
        if self.stage == 'test':
            dice = sum(self.dice)/(len(self.dice)+0.001)
            acc = sum(self.acc)/(len(self.acc)+0.001)
            precision = sum(self.precision)/(len(self.precision)+0.001)
            recall = sum(self.recall)/(len(self.recall)+0.001)
            fscore = sum(self.fscore)/(len(self.fscore)+0.001)
            def summarize_95hds(hd95s: List[np.ndarray]):
                hd95s2 = [hd for hd in hd95s if not np.isnan(hd.sum())]
                mhd95 = sum(hd95s2) / (len(hd95s2) + 0.001)
                return mhd95
            hd95 = summarize_95hds(self.hd95)
            self.metrics.update({'dice': list(dice),'dice_mean': sum(list(dice))/len(list(dice)),
                                 'acc': list(acc), 'acc_mean': sum(list(acc))/len(list(acc)),
                                 'precision': list(precision),'precision_mean': sum(list(precision))/len(list(precision)),
                                 'recall': list(recall), 'recall_mean': sum(list(recall))/len(list(recall)),
                                 'fscore': list(fscore),'fscore_mean': sum(list(fscore))/len(list(fscore)),
                                 'hd95': list(hd95),'hd95_mean': sum(list(hd95))/len(list(hd95)),
                                 })
        self.metrics.update({'num_imgs':len(self.MIoU_list)})
        self.clear_cache()

        return self.metrics

    def save_results(self):
        eval_json = osp.join(self.eval_cfg.model_dir, 'eval_results.json')
        dict_str = json.dumps(self.metrics, sort_keys=True, indent=4, separators=(',', ':'))
        with open(eval_json, 'w') as json_file:
            json_file.write(dict_str)
            print(f'Save eval results at {eval_json}')
        return 0
