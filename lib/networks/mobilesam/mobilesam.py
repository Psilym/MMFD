from .build_sam import build_sam_vit_t

def get_network(*args,**kwargs):
    cfg = kwargs['cfg']
    network = build_sam_vit_t(image_size=cfg.img_size,
                                                num_classes=cfg.num_classes,
                                                checkpoint=cfg.ckpt,
                                                pixel_mean=[0, 0, 0],
                                                pixel_std=[1, 1, 1],
                                                cfg=cfg)
    return network