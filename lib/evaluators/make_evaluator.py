import imp
import os
from lib.datasets.dataset_catalog import DatasetCatalog


def _evaluator_factory(cfg,stage):
    task = cfg.task
    dataset = cfg.dataset
    assert stage in ['val','test','perimg_test']
    module = '.'.join(['lib.evaluators', dataset, task])
    path = os.path.join('lib/evaluators', dataset, task+'.py')
    evaluator = imp.load_source(module, path).Evaluator(cfg,stage)
    return evaluator


def make_evaluator(cfg,stage='val'):
    if cfg.skip_eval:
        return None
    else:
        return _evaluator_factory(cfg,stage)
