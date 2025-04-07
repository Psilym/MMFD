from lib.networks import make_network
from lib.train import make_trainer, make_optimizer, make_lr_scheduler, make_recorder
from lib.datasets import make_split_data_loader
from lib.utils.net_utils import load_model, save_model, load_network
from lib.evaluators import make_evaluator
from lib.utils.net_utils import EarlyStopping

def train_process(cfg):
    network = make_network(cfg)
    trainer = make_trainer(cfg, network)
    optimizer = make_optimizer(cfg, network)
    scheduler = make_lr_scheduler(cfg, optimizer)

    evaluator = make_evaluator(cfg,stage='val')
    recorder = make_recorder(cfg)
    # postprocessor = make_postprocessor(cfg)
    earlystoper = EarlyStopping(optimizer,scheduler,recorder,cfg.model_dir,patience=cfg.earlystop.patience)
    begin_epoch = load_model(network, optimizer, scheduler, recorder, cfg.model_dir, resume=cfg.resume)
    # set_lr_scheduler(cfg, scheduler)34

    train_loader = make_split_data_loader(cfg, is_train=True)
    val_loader = make_split_data_loader(cfg, is_train=False)
    max_iterations = cfg.train.max_iterations
    max_epoch = max_iterations // len(train_loader) + 1
    print("{} iterations per epoch. {} max iterations and {} max epochs ".format(len(train_loader), max_iterations,
                                                                               max_epoch))
    save_interval = max(cfg.save_iter // len(train_loader),1) # in epoch
    eval_interval = max(cfg.eval_iter // len(train_loader),1) # in epoch
    print("Save epochs are {}. Eval epochs are {}. ".format(save_interval, eval_interval))

    for epoch in range(begin_epoch, max_epoch):
        recorder.epoch = epoch
        trainer.train(epoch, train_loader, scheduler, optimizer, recorder)

        if (epoch + 1) % save_interval == 0\
                or (epoch + 1) == cfg.train.epoch:
            save_model(network, optimizer, scheduler, recorder, epoch, cfg.model_dir)

        if (epoch + 1) % eval_interval == 0:
            trainer.val(epoch, val_loader, evaluator, recorder, postprocessor=None)
            earlystoper(recorder.loss_stats['loss'].median,network,epoch)

    return network

def test_process(cfg):
    network = make_network(cfg)
    trainer = make_trainer(cfg, network)
    val_loader = make_split_data_loader(cfg, is_train=False)
    evaluator = make_evaluator(cfg,stage='test')
    epoch = load_network(network, cfg.model_dir, resume=cfg.resume, epoch=cfg.test.epoch)
    trainer.val(epoch, val_loader, evaluator, postprocessor=None)
    metrics = evaluator.metrics
    evaluator.save_results()
    return metrics
