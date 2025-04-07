import imp
import os
from lib.datasets.dataset_catalog import DatasetCatalog


def _postprocessor_factory(cfg):
    task = cfg.task
    dataset = cfg.dataset
    module = '.'.join(['lib.postprocessors', dataset, task])
    path = os.path.join('lib/postprocessors', dataset, task+'.py')
    postprocessor = imp.load_source(module, path).Postprocessor(cfg)
    return postprocessor


def make_postprocessor(cfg):
    return _postprocessor_factory(cfg)
