import torch
import os
from torch import nn
import numpy as np
import torch.nn.functional as F
from termcolor import colored
import glob
import os.path as osp


def load_model(net, optim, scheduler, recorder, model_dir, resume=True, epoch=-1):
    if not resume:
        os.system('rm -rf {}'.format(model_dir))
        return 0

    if not os.path.exists(model_dir):
        print(colored('WARNING: NO MODEL LOADED !!!', 'red'))
        return 0
    # pths = [pth.split('.')[0] for pth in os.listdir(model_dir)]
    ckpts = list(glob.glob(osp.join(model_dir,'*.pth')))
    pths = []
    for ckpt_file in iter(ckpts):
        ckpt_name = osp.split(ckpt_file)[-1].split('.')[0]
        if ckpt_name.isdigit():
            pths.append(int(ckpt_name))
    if len(pths) == 0:
        print(colored('WARNING: NO MODEL LOADED !!!', 'red'))
        return 0
    if epoch == -1:
        pth = max(pths)
    else:
        pth = epoch
    print('load model: {}'.format(os.path.join(model_dir, '{}.pth'.format(pth))))
    pretrained_model = torch.load(os.path.join(model_dir, '{}.pth'.format(pth)))
    net.load_state_dict(pretrained_model['net'])
    optim.load_state_dict(pretrained_model['optim'])
    scheduler.load_state_dict(pretrained_model['scheduler'])
    recorder.load_state_dict(pretrained_model['recorder'])
    return pretrained_model['epoch'] + 1


def save_model(net, optim, scheduler, recorder, epoch, model_dir):
    os.system('mkdir -p {}'.format(model_dir))
    torch.save({
        'net': net.state_dict(),
        'optim': optim.state_dict(),
        'scheduler': scheduler.state_dict(),
        'recorder': recorder.state_dict(),
        'epoch': epoch
    }, os.path.join(model_dir, '{}.pth'.format(epoch)))
    print(f'Save model of epoch{epoch}.')
    # # remove previous pretrained model if the number of models is too big
    # pths = [int(pth.split('.')[0]) for pth in os.listdir(model_dir)]
    # if len(pths) <= 200:
    #     return
    # os.system('rm {}'.format(os.path.join(model_dir, '{}.pth'.format(min(pths)))))


def load_network(net, model_dir, resume=True, epoch=-1, strict=True):
    if not resume:
        return 0

    if not os.path.exists(model_dir):
        print(colored('WARNING: NO MODEL LOADED !!!', 'red'))
        return 0

    # pths = [int(pth.split('.')[0]) for pth in os.listdir(model_dir) if 'pth' in pth]
    # pths = [pth[:-4] for pth in os.listdir(model_dir) if 'pth' in pth]
    pths = []
    pths_temp = glob.glob(osp.join(model_dir,'*.pth'))
    for pth in iter(pths_temp):
        pth = osp.split(pth)[-1]
        if pth.split('.')[0].isdigit():
            pths.append(int(pth.split('.')[0]))
    if len(pths) == 0:
        print(colored('WARNING: NO MODEL LOADED !!!', 'red'))
        return 0

    if epoch == -1:
        pth = max(pths)
    else:
        pth = epoch
    print('load model: {}'.format(os.path.join(model_dir, '{}.pth'.format(pth))))
    pretrained_model = torch.load(os.path.join(model_dir, '{}.pth'.format(pth)))
    net.load_state_dict(pretrained_model['net'], strict=strict)
    return pretrained_model['epoch'] + 1


class MulticlassDiceLoss(nn.Module):
    """
    requires one hot encoded target. Applies DiceLoss on each class iteratively.
    requires input.shape[0:1] and target.shape[0:1] to be (N, C) where N is
    batch size and C is number of classes
    """
    def __init__(self,weight:list=None):
        super(MulticlassDiceLoss, self).__init__()
        self.weights = weight
        from segmentation_models_pytorch import losses
        self.losser = losses.DiceLoss(mode='binary',smooth=1,from_logits=False)#need input normalized scores
        self.loss_default_weight = 1


    def forward(self, input, target):
        '''
        input,target: [B,C,H,W]
        #mask_ignore: [B,1,H,W], range from 0-1, 1 for ignore, 0 for use
        note: input should be non-activated, which means non-normalized
        '''

        C = input.shape[1]

        # if weights is None:
        # 	weights = torch.ones(C) #uniform weights for all classes

        dice_crit = self.losser.forward
        totalLoss = 0
        input = F.sigmoid(input)
        for i in range(C):
            pred_ = input[:,i,...].unsqueeze(1).contiguous() #[N,1,H,W]
            true_ = target[:,i,...].unsqueeze(1).contiguous()
            diceLoss = dice_crit(pred_, true_)
            if self.weights is not None:
                diceLoss *= self.weights[i]
            totalLoss += diceLoss
        totalLoss *= self.loss_default_weight

        return totalLoss

class MulticlassBCELoss(nn.Module):
    """
    requires one hot encoded target. Applies BCELoss on each class iteratively.
    requires input.shape[0:1] and target.shape[0:1] to be (N, C) where N is
    batch size and C is number of classes
    """
    def __init__(self,weight:list=None):
        super(MulticlassBCELoss, self).__init__()
        self.weights = weight
        # from segmentation_models_pytorch import losses
        # self.losser = losses.SoftBCEWithLogitsLoss() # not very good for cellgroup
        self.losser = nn.BCEWithLogitsLoss(reduction='mean')#sigmoid in this func

    def forward(self, input, target):
        '''
        Note: input should be non-activated, which means non-normalized
        '''

        C = input.shape[1]

        # if weights is None:
        # 	weights = torch.ones(C) #uniform weights for all classes
        totalLoss = 0
        bce_func = self.losser.forward
        for i in range(C):
            pred_ = input[:,i,...].unsqueeze(1).contiguous() #[N,1,H,W]
            true_ = target[:,i,...].unsqueeze(1).contiguous()
            loss_ = bce_func(pred_, true_)
            if self.weights is not None:
                loss_ *= self.weights[i]
            totalLoss += loss_

        return totalLoss

class MulticlassFocalLoss(nn.Module):
    """
    requires one hot encoded target. Applies BCELoss on each class iteratively.
    requires input.shape[0:1] and target.shape[0:1] to be (N, C) where N is
    batch size and C is number of classes
    """
    def __init__(self,weight:list=None):
        super(MulticlassFocalLoss, self).__init__()
        self.weights = weight
        self.loss_default_weight = 10
        from segmentation_models_pytorch import losses
        self.losser = losses.FocalLoss(mode='binary')

    def forward(self, input, target):

        C = input.shape[1]

        # if weights is None:
        # 	weights = torch.ones(C) #uniform weights for all classes
        totalLoss = 0
        focal_func = self.losser.forward
        for i in range(C):
            pred_ = input[:,i,...].unsqueeze(1).contiguous() #[N,1,H,W]
            true_ = target[:,i,...].unsqueeze(1).contiguous()
            loss_ = focal_func(pred_, true_)
            if self.weights is not None:
                loss_ *= self.weights[i]
            totalLoss += loss_
        totalLoss *= self.loss_default_weight

        return totalLoss

class MSEDistillLoss(nn.Module):
    """
    requires one hot encoded target. Applies DiceLoss on each class iteratively.
    requires input.shape[0:1] and target.shape[0:1] to be (N, C) where N is
    batch size and C is number of classes
    """
    def __init__(self,weight:list=None):
        super(MSEDistillLoss, self).__init__()
        self.weights = weight
        self.losser = torch.nn.functional.mse_loss
        self.loss_default_weight = 1


    def forward(self, feat_t, feat_s):
        '''
        feat: [B,C,H,W]
        '''

        # if weights is None:
        # 	weights = torch.ones(C) #uniform weights for all classes
        crit = self.losser
        totalLoss = 0
        loss = crit(feat_t.detach(),feat_s)
        if self.weights is not None:
            loss *= self.weights
        totalLoss += loss
        totalLoss *= self.loss_default_weight

        return totalLoss

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, optim, scheduler, recorder, model_dir,
                 patience=7, verbose=True, delta=0, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        # self.path = path # dont use
        self.trace_func = trace_func
        self.optim = optim
        self.scheduler = scheduler
        self.recorder = recorder
        self.model_dir = model_dir

    def __call__(self, val_loss, model, epoch):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch)
            self.counter = 0

    def save_model(self, net, epoch):
        '''follow above save_model function'''
        os.system('mkdir -p {}'.format(self.model_dir))
        torch.save({
            'net': net.state_dict(),
            'optim': self.optim.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'recorder': self.recorder.state_dict(),
            'epoch': epoch
        }, os.path.join(self.model_dir, 'EarlyStop.pth'))

    def save_checkpoint(self, val_loss, model,epoch):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}.')
        self.save_model(model, epoch)
        self.val_loss_min = val_loss

