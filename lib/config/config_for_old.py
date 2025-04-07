from .yacs import CfgNode as CN
import argparse
import os
import os.path as osp

cfg = CN()
# this file store the basic config
# model
cfg.model = 'hello'
cfg.out_dir = f'data/exp'
cfg.model_dir = osp.join(cfg.out_dir, cfg.model)

# network
cfg.network = CN()
# cfg.network.params = CN()

# task
cfg.task = ''

# gpus
cfg.gpus = [0]

# if load the pretrained network
cfg.resume = True

# split
cfg.idx_fold = 'no_split'

# -----------------------------------------------------------------------------
# train
# -----------------------------------------------------------------------------
cfg.train = CN()

cfg.train.dataset = 'CocoTrain'
cfg.train.epoch = 140
cfg.train.num_workers = 8

# use adam as default
cfg.train.optim = 'adam'
cfg.train.lr = 1e-4
cfg.train.weight_decay = 5e-4

cfg.train.warmup = False
cfg.train.milestones = [80, 120, 200, 240]
cfg.train.gamma = 0.5

cfg.train.batch_size = 4

# val
cfg.val = CN()
cfg.val.dataset = 'CocoVal'
cfg.val.batch_size = 1
cfg.val.epoch = -1
# test
cfg.test = CN()
cfg.test.dataset = 'CocoTest'
cfg.test.batch_size = 1
cfg.test.epoch = -1

# augment
cfg.train.augment = CN()
cfg.val.augment = CN()

# recorder
cfg.record_dir = cfg.model_dir

# result
cfg.result_dir = cfg.model_dir

# evaluation
cfg.skip_eval = False

cfg.save_ep = 5
cfg.eval_ep = 5

cfg.use_gt_det = False

# # save the root
# cfg.model_dir_root = cfg.model_dir
# cfg.result_dir_root = cfg.result_dir
# cfg.record_dir_root = cfg.record_dir

# -----------------------------------------------------------------------------
# post process
# -----------------------------------------------------------------------------
cfg.post = CN()
cfg.post.required = False
cfg.post.th_toedge = 0.02
cfg.post.th_cnt = 0.2
cfg.post_types = []

# -----------------------------------------------------------------------------
# snake
# -----------------------------------------------------------------------------
cfg.ct_score = 0.05
cfg.demo_path = ''