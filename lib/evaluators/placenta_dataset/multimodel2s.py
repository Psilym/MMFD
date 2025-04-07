import os
import numpy as np
from lib.utils.pan_seg.eval_utils import multi_iou, multi_scores, multi_hd95, multi_softdice
import torch
from typing import List
class Evaluator_val:
    def __init__(self, stage, result_dir=None, postprocessor = None, cfg = None):
        # self.eval_cfg = cfg
        self.metrics = {}
        self.result_dir = result_dir
        os.system('mkdir -p {}'.format(self.result_dir))
        self.stage = stage
        # semantic
        self.sem_cls_num = cfg.num_classes
        self.MIoU = multi_iou(self.sem_cls_num)
        self.MIoU_list = []
        self.postprocessor = postprocessor


    def clear_cache(self):
        # semantic
        self.MIoU_list = []
        print('Clear evaluator cache.')
        return 0

    def evaluate(self, outputs, batch_mask):
        # semantic

        pred_mask = torch.sigmoid(outputs['masks'])
        if self.postprocessor != None:
            pred_mask = self.postprocessor.post_processing_direct(pred_mask)
        mask_gt = batch_mask

        b = mask_gt.shape[0]
        if b != 1:
            print('Error. During evaluation, batch size is not 1.')
            raise ValueError
        iou = self.MIoU.get_multi_iou(pred_mask,mask_gt)
        self.MIoU_list.append(iou)

    def summarize(self):
        # semantic
        MIoU = sum(self.MIoU_list)/(len(self.MIoU_list)+0.001)
        MIoU = MIoU.tolist()
        for idx in range(len(MIoU)):
            self.metrics.update({f'MIoU_{idx}': MIoU[idx]})
        mean_MIoU = sum(MIoU)/len(MIoU)
        self.metrics.update({'MIoU_mean':mean_MIoU})

        num_imgs = len(self.MIoU_list)
        self.metrics.update({'num_imgs':num_imgs})
        self.clear_cache()

        return self.metrics

class Evaluator_ms:
    def __init__(self, stage, result_dir=None, cfg=None, postprocessor=None):
        self.cfg = cfg
        self.num_level = cfg.component.test_iter
        self.metrics = {}
        self.result_dir = result_dir
        os.system('mkdir -p {}'.format(self.result_dir))
        self.stage = stage
        # semantic
        self.sem_cls_num = cfg.num_classes
        self.MIoU = multi_iou(self.sem_cls_num)
        self.MIoU_ms = [[] for _ in range(self.num_level)] # List[List[np.array]]
        self.postprocessor = postprocessor
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
        self.miou_list = []
        # instance
        # self.coco_evalor = coco_evaluator.CocoEvaluator(self.eval_cfg)
        print('Clear evaluator cache.')
        return 0

    def evaluate(self, output, batch_mask):
        mask_gt = batch_mask
        b,_,h,w = mask_gt.shape
        if b != 1:
            print('Error. During evaluation, batch size is not 1.')
            raise ValueError

        for idx_level in range(self.num_level):
            if idx_level == 0:
                output_mask = output['high_res_logits_list'][idx_level]
            else:
                output_mask = output['high_res_logits_list'][idx_level]
            output_mask = torch.sigmoid(output_mask)
            if self.postprocessor != None:
                output_mask = self.postprocessor.post_processing_direct(output_mask)
            # output_mask = F.interpolate(
            #     output_mask.type(torch.float),
            #     (h, w),
            #     mode="bilinear",
            #     align_corners=False,
            # )
            iou = self.MIoU.get_multi_iou(output_mask,mask_gt)

            self.MIoU_ms[idx_level].append(iou)

        if self.stage == 'test':
            output_mask = output['high_res_logits_list'][-1]
            output_mask = torch.sigmoid(output_mask)
            dice = self.dice_calculor.get_multi_dice(output_mask,mask_gt)
            self.dice.append(dice)
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
        return 0

    def summarize(self):
        # semantic
        for idx_level in range(self.num_level):
            miou_list = self.MIoU_ms[idx_level]
            MIoU = sum(miou_list)/(len(miou_list)+0.001)
            MIoU = MIoU.tolist()
            for idx in range(len(MIoU)):
                self.metrics.update({f'MIoU_{idx}_L{idx_level}': MIoU[idx]})
            mean_MIoU = sum(MIoU) / len(MIoU)
            self.metrics.update({f'MIoU_mean_L{idx_level}': mean_MIoU})
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

        num_imgs = len(self.MIoU_ms[-1])
        self.metrics.update({'num_imgs':num_imgs})
        self.clear_cache()

        return self.metrics
