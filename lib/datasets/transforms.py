class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, kpts=None):
        for t in self.transforms:
            img, kpts = t(img, kpts)
        if kpts is None:
            return img
        else:
            return img, kpts

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class ToTensor(object):
    def __call__(self, img, kpts):
        return img / 255., kpts


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img, kpts):
        img -= self.mean
        img /= self.std
        return img, kpts


def make_transforms(cfg, is_train):
    if is_train is True:
        transform = Compose(
            [
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    else:
        transform = Compose(
            [
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    return transform
import imp
import os

def make_transformsmy(cfg, is_train):
    if is_train:
        augment_dict = cfg.train.augment
    else:
        augment_dict = cfg.val.augment

    data_source = cfg.dataset
    module = '.'.join(['lib.datasets', data_source, 'transform'])
    path = os.path.join('lib/datasets', data_source, 'transform.py')
    MultiTranser = imp.load_source(module, path).MultiTransform()
    for key in augment_dict.keys():
        augment_item = key
        if key not in augment_dict.keys():
            print(f'{key} is not in augment dict.')
        augment_param = augment_dict[key]
        MultiTranser.append(augment_item,**augment_param)
    return MultiTranser


