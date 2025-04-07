from torch.utils.data.dataloader import default_collate


def sem_collator(batch):

    ret = {'img': default_collate([b['img'] for b in batch])}

    meta = default_collate([b['meta_info'] for b in batch])
    ret.update({'meta_info': meta})
    # semantic mask
    sem_mask = default_collate([b['sem_mask'] for b in batch])
    ret.update({'sem_mask': sem_mask})

    return ret

def mobilesam_collator(batch):

    ret = {'img': default_collate([b['img'] for b in batch])}

    meta = default_collate([b['meta_info'] for b in batch])
    ret.update({'meta_info': meta})
    # semantic mask
    sem_mask = default_collate([b['sem_mask'] for b in batch])
    ret.update({'sem_mask': sem_mask})

    ret.update({
        'image': ret['img'],
        'label': ret['sem_mask'],
        'case_name': ret['meta_info']['sample_name'],
    })

    return ret


_collators = {
    'mobilesam': mobilesam_collator,


}


def make_collator(cfg):
    if cfg.task in _collators:
        return _collators[cfg.task]
    else:
        return default_collate

