# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from . import mobilesam
_network_factory = {
    'mobilesam': mobilesam.get_network,
}

def get_network(cfg):
    arch = cfg.network.name
    params = cfg.network.params
    get_model = _network_factory[arch]
    params.update({'cfg':cfg})
    network = get_model(**params)
    return network

