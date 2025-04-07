import torch.cuda

from .yacs import CfgNode as CN
import argparse
import os
import os.path as osp


def obtain_basic_config(cfg):
    '''
    input: global cfg
    '''
    # this file store the basic config
    # model
    cfg.model = 'hello'
    cfg.out_dir = f'data/exp'
    cfg.model_dir = osp.join(cfg.out_dir,cfg.model)

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

    return cfg

def parse_cfg(cfg):
    if len(cfg.task) == 0:
        raise ValueError('task must be specified')

    # assign the gpus
    # Ngpu = torch.cuda.device_count()
    # gpu_list = list(cfg.gpus)[:Ngpu]
    # gpu_list = [min(gpu,Ngpu-1) for gpu in gpu_list]
    # gpu_list = list(set(gpu_list))
    # gpu_list = cfg.gpus
    # command = ', '.join([str(gpu) for gpu in gpu_list])
    # os.environ['CUDA_VISIBLE_DEVICES'] = command
    # print(f'Define CUDA_VISIBLE_DEVICES = {command}')
    # cfg.model_dir = f'data/exp/{cfg.model}'
    cfg.result_dir = cfg.model_dir
    cfg.record_dir = cfg.model_dir



def get_task_cfg(cfg):
    import imp
    task = cfg.task
    path = os.path.join('lib/config', f'{task}.py')
    task_cfg = imp.load_source('module_name', path).make_task_cfg(cfg)
    return task_cfg

def make_cfg(args):
    # global cfg
    cfg = CN()
    cfg = obtain_basic_config(cfg)
    cfg.merge_from_file(args.cfg_file)
    cfg = get_task_cfg(cfg)
    parse_cfg(cfg)
    return cfg

def obtain_cfg_distill(cfg_tot, teacher=True):
    # global cfg
    cfg = CN()
    cfg = obtain_basic_config(cfg)
    if teacher:
        cfg.merge_from_file(cfg_tot.network_t.cfg_file)
    else:
        cfg.merge_from_file(cfg_tot.network_s.cfg_file)

    cfg = get_task_cfg(cfg)
    parse_cfg(cfg)
    return cfg

