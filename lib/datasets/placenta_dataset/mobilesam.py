from torch.utils.data.dataset import Dataset
import numpy as np
import warnings

warnings.filterwarnings('ignore')
import pandas as pd
import os.path as osp
import cv2
import json
from lib.utils import img_utils

SEM_CLS_LIST = ['villi', 'villiu', 'stem', 'thrombus', 'bloodvessel', 'cellgroup', 'other'] # related with origin anno
SEM_CLS_FIN_LIST = ['villit', 'thrombus', 'bloodvessel', 'cellgroup']

class Generic_Seg_Dataset(Dataset):
    def __init__(self, dataset_dir='data/dataset/', column_label='sample_name', Transforms=None):
        self.dataset_dir = dataset_dir
        self.dataset_csv_file = osp.join(self.dataset_dir, 'sample_list.csv')
        self.meta_file = osp.join(self.dataset_dir, 'meta_info.json')
        self.column_label = column_label

        self.dataset_df = pd.read_csv(self.dataset_csv_file)[column_label]
        self.num_samples = len(self.dataset_df)
        self.split_csv_file = None

        with open(self.meta_file, 'r') as meta_f:
            json_data = json.load(meta_f)
        self.meta_info = json_data
        self.cls2clr = self.meta_info['cls2clr']
        self.ann_dir = osp.join(self.dataset_dir, 'ann')
        self.hsi_dir = osp.join(self.dataset_dir, 'hsi')
        self.sem_cls_list = SEM_CLS_LIST
        self.sem_cls_fin_list = SEM_CLS_FIN_LIST
        self.Transforms = Transforms
        self.tar_size = (512,512)

    def generate_sem_mask(self, seml0_file, seml1_file):
        seml0_img_rgb = cv2.imread(seml0_file)[:, :, ::-1]
        seml0_img_tot = seml0_img_rgb[:, :, 0] * 256 ** 2 + seml0_img_rgb[:, :, 1] * 256 + seml0_img_rgb[:, :, 2]
        seml1_img_rgb = cv2.imread(seml1_file)[:, :, ::-1]
        seml1_img_tot = seml1_img_rgb[:, :, 0] * 256 ** 2 + seml1_img_rgb[:, :, 1] * 256 + seml1_img_rgb[:, :, 2]
        h, w = seml0_img_rgb.shape[:2]
        num_sem_cls = len(self.sem_cls_list)
        sem_mask = np.zeros((num_sem_cls, h, w), dtype=bool)
        from lib.utils.pan_seg.configs import seml1_class_list, seml0_class_list
        for idx_cls, name_cls in enumerate(self.sem_cls_list[:-1]):
            if name_cls in seml0_class_list:
                select_img_tot = seml0_img_tot
            elif name_cls in seml1_class_list:
                select_img_tot = seml1_img_tot
            else:
                print(f'{name_cls} not in the list.')
            tot_color = self.cls2clr[name_cls] * np.ones(select_img_tot.shape, dtype=int)
            sem_mask[idx_cls, :, :] = np.equal(select_img_tot, tot_color)
        sem_mask[-1, :, :] = (sem_mask[:-1, :, :].sum(axis=0) == 0)
        sem_mask = sem_mask.astype(np.float32)
        sem_mask = img_utils.resize_my(sem_mask,self.tar_size)
        sem_mask_ori = img_utils.resize_my(sem_mask,tar_size=self.tar_size)
        mask_labels_std = self.sem_cls_fin_list
        # fuse villi and villiu and stem as villit
        idx_v = self.sem_cls_list.index('villi')
        idx_vu = self.sem_cls_list.index('villiu')
        idx_s = self.sem_cls_list.index('stem')
        fuse_mask = sem_mask[idx_v] + sem_mask[idx_vu] + sem_mask[idx_s]
        fuse_mask = np.clip(fuse_mask,a_min=0,a_max=1)
        idx_t = self.sem_cls_list.index('thrombus')
        idx_b = self.sem_cls_list.index('bloodvessel')
        idx_c = self.sem_cls_list.index('cellgroup')
        save_sem_mask = np.stack([sem_mask[i] for i in [idx_t,idx_b,idx_c]],axis=0)
        sem_mask_std = np.concatenate([fuse_mask[np.newaxis,...],save_sem_mask],axis=0)

        return sem_mask_std, mask_labels_std


    def __getitem__(self, idx):
        sample_name = self.dataset_df[idx]

        clr_img_file = osp.join(self.hsi_dir, sample_name + '.jpg')
        img = cv2.imread(clr_img_file)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        img = img/255.0
        # img = (img - snake_config.mean) / snake_config.std
        img = np.transpose(img, (2, 0, 1))
        img = img_utils.resize_my(img,self.tar_size)

        # sem map
        seml0_file = osp.join(self.ann_dir, sample_name + '_seml0.png')
        seml1_file = osp.join(self.ann_dir, sample_name + '_seml1.png')
        sem_mask, mask_labels = self.generate_sem_mask(seml0_file, seml1_file)

        img, sem_mask, _ = self.Transforms(img, sem_mask=sem_mask) # out in scale 512

        meta_info = {}
        meta_info.update({'sample_name':sample_name,'img_idx':idx,
                          'hsi_dir':self.hsi_dir,'ann_dir':self.ann_dir,
                          })

        data_dict = {}
        data_dict.update({'img': img,
                          'sem_mask': sem_mask,
                          'meta_info': meta_info})

        return data_dict

    def __len__(self):
        return self.num_samples
        # return 10

class Split_Dataset(Generic_Seg_Dataset):
    def __init__(self, split_dir, fewshot_perc, column_label='train',
                 dataset_dir='data/dataset/', Transforms = None):
        super(Split_Dataset,self).__init__(dataset_dir=dataset_dir, Transforms=Transforms)
        self.column_label = column_label
        csv_file_name = f'dataset_split.csv'
        csv_file_path = osp.join(split_dir, f'fewer_test30_fewshot{fewshot_perc}', csv_file_name)
        self.dataset_df = pd.read_csv(csv_file_path)[self.column_label]
        self.dataset_df.dropna(axis=0, how='any', inplace=True)
        self.num_samples = len(self.dataset_df)
        self.fewshot_perc = fewshot_perc

    def __len__(self):
        return self.num_samples
        # return 5
