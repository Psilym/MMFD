import torch.cuda

from lib.config.yacs import CfgNode as CN
import argparse
import os
import os.path as osp

# this file store the specific config of framework

cfg = CN()



def make_task_cfg(basic_cfg):
    ######## basic cfg related ###########
    cfg.model_dir = osp.join(basic_cfg.out_root,basic_cfg.model)
    ######## final merge #############
    basic_cfg.merge_from_other_cfg(cfg)

    return basic_cfg