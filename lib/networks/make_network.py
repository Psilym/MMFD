import os
import imp



def make_network(cfg):
    module = '.'.join(['lib.networks', cfg.task])
    path = os.path.join('lib/networks', cfg.task, '__init__.py')
    return imp.load_source(module, path).get_network(cfg,)
