import segmentation_models_pytorch as smp
from segmentation_models_pytorch import utils
import numpy as np
from skimage.metrics import hausdorff_distance

def calc_dice(y_pred,y_true, reduce_axes, beta=1., epsilon=1e-6):
    """
    Compute soft dice coefficient according to V-Net paper.
    (https://arxiv.org/abs/1606.04797)
    Unlike F-beta score, dice uses squared probabilities
    instead of probabilities themselves for denominator.
    Due to the squared entries, gradients will be y_true and y_pred instead of 1.
    in params:
    y_pred, y_true: [B,H,W]
    """
    beta2 = beta ** 2   # beta squared
    numerator = (1 + beta2) * (y_true * y_pred).sum(reduce_axes)
    denominator = beta2 * y_true.square().sum(reduce_axes) + y_pred.square().sum(reduce_axes)
    denominator = denominator.clamp(min=epsilon)
    return (numerator) / denominator

class multi_softdice():
    def __init__(self,input_channel):
        self.dice = calc_dice
        self.input_channel = input_channel
    def get_multi_dice(self,pred,label):
        '''
        note: pred should be score instead of bool
        '''
        multi_score = np.zeros(self.input_channel)
        for i in range(self.input_channel):
            multi_score[i] = (self.dice(pred[:,i],label[:,i],reduce_axes=(0,1,2)))
        return multi_score

class multi_iou():
    def __init__(self,input_channel):
        self.iou = smp.utils.metrics.IoU()
        self.input_channel = input_channel
    def get_multi_iou(self,pred,label):
        multi_score = np.zeros(self.input_channel)
        for i in range(self.input_channel):
            # if label[:,i].sum() == 0:
            #     print('label is all false')
            multi_score[i] = (self.iou(pred[:,i],label[:,i]))
        return multi_score
    def get_multi_dice(self,pred,label):
        multi_score = np.zeros(self.input_channel)
        for i in range(self.input_channel):
            multi_score[i] = (self.iou(pred[:,i],label[:,i]))
        dice = (2*multi_score) / (multi_score+1)
        return dice

class multi_scores():
    def __init__(self,input_channel):
        self.accer = smp.utils.metrics.Accuracy()
        self.precisioner = smp.utils.metrics.Precision()
        self.recaller = smp.utils.metrics.Recall()
        self.fscorer = smp.utils.metrics.Fscore()
        self.input_channel = input_channel
    def get_multi_acc(self,pred,label):
        multi_score = np.zeros(self.input_channel)
        for i in range(self.input_channel):
            multi_score[i] = (self.accer(pred[:,i],label[:,i]))
        return multi_score
    def get_multi_precision(self,pred,label):
        multi_score = np.zeros(self.input_channel)
        for i in range(self.input_channel):
            multi_score[i] = (self.precisioner(pred[:,i],label[:,i]))
        return multi_score
    def get_multi_recall(self,pred,label):
        multi_score = np.zeros(self.input_channel)
        for i in range(self.input_channel):
            multi_score[i] = (self.recaller(pred[:,i],label[:,i]))
        return multi_score
    def get_multi_fscore(self,pred,label):
        multi_score = np.zeros(self.input_channel)
        for i in range(self.input_channel):
            multi_score[i] = (self.fscorer(pred[:,i],label[:,i]))
        return multi_score

class multi_hd95:
    def __init__(self,input_channel):
        self.input_channel = input_channel
        self.th = 0.5
    def get_multi_hd95(self,pred,label):
        '''
        pred, label: should be [C,H,W] cuda tensor
        '''
        pred = pred.detach().cpu().numpy()
        label = label.detach().cpu().numpy()
        pred = pred > self.th
        label = label > self.th
        multi_score = np.zeros(self.input_channel)
        for i in range(self.input_channel):
            sur_dis = hausdorff_distance(pred[:,i],label[:,i])
            hd95 = np.percentile(sur_dis,95)
            multi_score[i] = hd95
        return multi_score