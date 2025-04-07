import os
import imp


def make_visualizer(cfg):
    if cfg.visualizer == 'base':
        module = '.'.join(['lib.visualizers', 'base'])
        path = os.path.join('lib/visualizers', 'base.py')
        visualizer = imp.load_source(module, path).Visualizer(cfg)
    else:
        module = '.'.join(['lib.visualizers', cfg.task])
        path = os.path.join('lib/visualizers', cfg.task+'.py')
        visualizer = imp.load_source(module, path).Visualizer(cfg)
    return visualizer
