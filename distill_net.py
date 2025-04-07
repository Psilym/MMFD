import argparse
import imp
import os
import torch.multiprocessing
import random
import numpy as np
import os.path as osp

def seed_torch(seed=1234):
	random.seed(seed) # Python random
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed) # numpy random
	torch.manual_seed(seed) # torch cpu random
	torch.cuda.manual_seed(seed) # torch gpu random
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	# torch.backends.cudnn.benchmark = False
	# torch.backends.cudnn.deterministic = True

seed_torch(1234)



def adapt_split_config(cfg_ori,fewshot_perc):
    cfg = cfg_ori.clone()
    cfg_ori.freeze() # first clone, then freeze
    cfg.fewshot_perc = fewshot_perc
    cfg.model_dir = osp.join(cfg.out_dir,cfg.task,cfg.model,f'fewshot_perc{fewshot_perc}')
    cfg.result_dir =cfg.model_dir
    cfg.record_dir =cfg.model_dir
    cfg_ori.defrost()

    return cfg

def main(cfg_tot,cfg_t,cfg_s,args):
    task = cfg_tot.task
    path = osp.join('lib/train',f'{task}','general_process.py')
    gp = imp.load_source('module_name',path)
    if args.test:
        perc_list = cfg_tot.fewshot_perc_list
        for fewshot_perc in iter(perc_list):
            cfg_fold = adapt_split_config(cfg_tot, fewshot_perc)
            cfg_t = adapt_split_config(cfg_t,fewshot_perc)
            cfg_s = adapt_split_config(cfg_s,fewshot_perc)
            print(f'Start fewshot percent {fewshot_perc}...')
            metrics = gp.test_process(cfg_fold,cfg_t,cfg_s)
    else:
        perc_list = cfg_tot.fewshot_perc_list
        for fewshot_perc in iter(perc_list):
            cfg_fold = adapt_split_config(cfg_tot,fewshot_perc)
            cfg_fold.save(osp.join(cfg_fold.model_dir,f'config_perc{fewshot_perc}.yaml'))
            print(f'Start fewshot percent {fewshot_perc}...')
            cfg_t = adapt_split_config(cfg_t,fewshot_perc)
            cfg_s = adapt_split_config(cfg_s,fewshot_perc)
            gp.train_process(cfg_fold,cfg_t,cfg_s)
            break



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_file", default="configs/default.yaml", type=str)
    parser.add_argument('--test', action='store_true', dest='test', default=False)
    args = parser.parse_args()

    from lib.config.config import make_cfg,obtain_cfg_distill
    cfg_tot = make_cfg(args)
    cfg_t = obtain_cfg_distill(cfg_tot,teacher=True)
    cfg_s = obtain_cfg_distill(cfg_tot,teacher=False)
    main(cfg_tot,cfg_t,cfg_s,args)



