# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .build_sam import (
    build_sam,
    build_sam_vit_h,
    build_sam_vit_l,
    build_sam_vit_b,
    sam_model_registry,
)
from .predictor import SamPredictor
from .automatic_mask_generator import SamAutomaticMaskGenerator

from .build_sam import  get_network as sam_get_network
_network_factory = {
    'samed': sam_get_network,
}


def get_network(cfg):
    arch = cfg.network.name
    params = cfg.network.params
    get_model = _network_factory[arch]
    if 'component' in list(cfg.keys()):
        cfg_component = cfg['component']
    else:
        cfg_component = None
    params.update({'cfg_component':cfg_component})
    network = get_model(**params)
    return network
