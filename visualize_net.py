from lib.config import cfg
from lib.postprocessors import make_postprocessor
import os.path as osp
import argparse
import os
import torch.backends.cudnn as cudnn
import logging
import numpy as np
import torch
import importlib
import sys

def visualize_dataset(img_path,ckpt_path,out_dir):

    from lib.config.config import make_cfg,obtain_cfg_distill
    cfg_tot = make_cfg(args)
    cfg_t = obtain_cfg_distill(cfg_tot,teacher=True)
    cfg_s = obtain_cfg_distill(cfg_tot,teacher=False)
    fewshot_perc = 100
    from distill_net import adapt_split_config
    cfg_tot = adapt_split_config(cfg_tot, fewshot_perc)
    cfg_t = adapt_split_config(cfg_t, fewshot_perc)
    cfg_s = adapt_split_config(cfg_s, fewshot_perc)

    cfg_tot.exp_dir = out_dir

    if not cfg_tot.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    if not os.path.exists(cfg_tot.exp_dir):
        os.makedirs(cfg_tot.exp_dir)

    # student
    from lib.networks.mobilesam import get_network
    net_s = get_network(cfg_s)
    net_s = net_s.cuda()
    from lib.train.multimodel2s_distill.general_process import load_resume_model_student_from_ckpt
    load_resume_model_student_from_ckpt(net_s, ckpt_path)

    log_folder = os.path.join(cfg_tot.exp_dir, 'test_log')
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/' + 'log.txt', level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(cfg_tot))
    net_t = None
    # low_res = cfg_tot.img_size // 4 #for input


    # from lib.datasets.dataset_catalog import DatasetCatalog
    # dataset_name = cfg_tot.dataset
    # dataset_args = DatasetCatalog.get(dataset_name)

    # low_res = cfg_tot.img_size//4

    def obtain_batch_from_single_image(img_path):

        import cv2
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        img = img/255.0
        img = np.transpose(img, (2, 0, 1))
        from lib.utils import img_utils
        tar_size = (512,512)
        img = img_utils.resize_my(img,tar_size)
        batch = {'image': img}
        # del img
        img_name = osp.split(img_path)[-1].split('.')[0]
        meta = {'img_name': img_name}
        batch.update({'meta':meta})
        # collate
        from torch.utils.data.dataloader import default_collate
        ret = {'image': default_collate([b['image'] for b in [batch]])}
        meta = default_collate([b['meta'] for b in [batch]])
        ret.update({'meta': meta})
        batch = ret

        return batch
    batch = obtain_batch_from_single_image(img_path)

    for k in batch:
        if k != 'meta' and k != 'py':
            batch[k] = batch[k].cuda()

    result_dir = osp.join(cfg_tot.exp_dir,'result_dir')
    pack = importlib.import_module(f'lib.visualizers.{cfg_tot.task}')
    visualizer = pack.Visualizer(result_dir, cfg=cfg_tot, postprocessor=None)

    net_s.eval()
    with torch.no_grad():
        outputs = net_s(batch, mode='test')
    if visualizer is not None:
        case_name = batch['meta']['img_name'][0]
        visualizer.visualize(batch['image'], outputs, case_name)
    del net_s, visualizer, outputs, batch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_file", default="configs/default.yaml", type=str)
    parser.add_argument('--test', action='store_true', dest='test', default=False)
    parser.add_argument('--img_path', type=str, default='data/example.jpg')
    parser.add_argument('--ckpt_path', type=str, default='data/ckpt.pth')
    parser.add_argument('--out_dir', type=str, default='data/output')
    args = parser.parse_args()
    img_path = args.img_path
    ckpt_path = args.ckpt_path
    out_dir = args.out_dir
    if not osp.exists(img_path):
        print(f'Image does not exists in path {img_path}')
    else:
        visualize_dataset(img_path,ckpt_path,out_dir)
        print(f'Finish visualizing img {img_path}')
