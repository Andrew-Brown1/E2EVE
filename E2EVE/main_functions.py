# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# Taming Transformers: https://github.com/CompVis/taming-transformers
# --------------------------------------------------------
import importlib
import argparse
import os
import pathlib
import datetime
import sys
import yaml
import torch

from pytorch_lightning.trainer import Trainer
import signal
from argparse import ArgumentParser
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from omegaconf import OmegaConf, DictConfig, ListConfig
from E2EVE.callbacks.training_callbacks import SetupCallback, ImageLogger
from omegaconf._utils import get_yaml_loader
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger



def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)

    return getattr(importlib.import_module(module, package=None), cls)

def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if opt[k] != getattr(args, k))

def instantiate_from_config(config):
    if "target" not in config:
        raise KeyError("Expected key `target` to instantiate.")

    return get_obj_from_str(config["target"])(**config.get("params", {}))

class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, val_batch_size, train=None, validation=None, test=None,
                  num_workers=None,num_transform_workers=None, non_paired_validation=True):
        super().__init__()

        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.dataset_configs = {}
        self.num_workers = num_workers if num_workers is not None else batch_size*2
        
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader

        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = self._val_dataloader
            
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = self._test_dataloader


    def prepare_data(self):
        
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self, stage=None):

        self.datasets = {}
        for k in self.dataset_configs:

            self.datasets[k] = instantiate_from_config(self.dataset_configs[k])
        
        # sanity check 
        assert(self.datasets['train'].mask_method==self.datasets['validation'].mask_method)
                
    def _train_dataloader(self):
        return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                            num_workers=self.num_workers, shuffle=True)

    def _val_dataloader(self):
        return DataLoader(self.datasets["validation"],
                        batch_size=self.val_batch_size, num_workers=self.num_workers, shuffle=False)


    def _test_dataloader(self):
        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=self.num_workers)


def main_train_function(arguments):


    (opt, extra_name, loaded_config) = arguments
    
    # -------------------------------------------------------------------------------
    # pre-amble - sort out the loaded configs and arguments
    # -------------------------------------------------------------------------------
    
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    sys.path.append(os.getcwd())
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    opt2, _ = parser.parse_known_args()
    opt2 = vars(opt2)
    if not isinstance(opt, dict):
        opt1 = vars(opt)
    else:
        opt1 = opt
    opt = {**opt1, **opt2}
    config = loaded_config
    config = OmegaConf.merge(*[config])
    lightning_config = config.pop("lightning", OmegaConf.create())

    # -------------------------------------------------------------------------------
    # log directories - setup the directories that will be logged to
    # -------------------------------------------------------------------------------
    if extra_name == "local":
        # then the script is being run locally - ID it with the time
        nowname = extra_name + '_' + now
    else:
        # then the script is being run via FBLearner - ID it with the flow ID
        nowname = extra_name

    logdir = os.path.join(opt["log_dir"], nowname)
    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    sampledir = os.path.join(logdir, "val_samples")
    seed_everything(opt["seed"],workers=True)
    config['model']['params']['val_sample_dir'] = sampledir
    
    # -------------------------------------------------------------------------------
    # gpu management
    # -------------------------------------------------------------------------------

    trainer_config = lightning_config.get("trainer", OmegaConf.create())
    trainer_config["distributed_backend"] = opt['dist_backend']

    if "gpus" not in trainer_config:
        trainer_config["gpus"] = "0,"

    if opt['dist_backend'] == 'ddp' or opt['dist_backend'] == 'ddp2':
        trainer_config["gpus"] = -1 # this tells PL to use all available gpus

    for k in nondefault_trainer_args(opt):
        trainer_config[k] = opt[k]  # getattr(opt, k)

    if "gpus" not in trainer_config:
        del trainer_config["distributed_backend"]
        cpu = True
    else:
        gpuinfo = trainer_config["gpus"]
        print(f"Running on GPUs {gpuinfo}")
        cpu = False
    trainer_opt = argparse.Namespace(**trainer_config)
    lightning_config.trainer = trainer_config
    config['model']['params']['distributed_backend'] = trainer_config["distributed_backend"]
    
    if not cpu:
        ngpu = len(lightning_config.trainer.gpus.strip(",").split(','))
    else:
        ngpu = 1
        
    # -------------------------------------------------------------------------------
    # learning rate management
    # -------------------------------------------------------------------------------

    bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
    
    # accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches or 1
    accumulate_grad_batches = 1
    print(f"accumulate_grad_batches = {accumulate_grad_batches}")
    lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
    starting_lr = accumulate_grad_batches * ngpu * bs * base_lr
    if  "transformer_config" in config['model']['params'].keys():
            config['model']['params']['starting_lr'] = starting_lr
            
            
    # -------------------------------------------------------------------------------
    # data setup
    # -------------------------------------------------------------------------------
    data = DataModuleFromConfig(**config.data.get("params", {}))
    data.prepare_data()
    data.setup()
    
    if config.data.params.non_paired_validation==True and 'GPT' in config.model.target:
        # if training the transformer with the E2EVE non paired validation
        # send all the required data to the model
        config.model.params.non_paired_validation=config.data.params.validation.params.non_paired_validation
        config.model.params.non_paired_validate_every_n_epoch=config.data.params.validation.params.non_paired_validate_every_n_epoch
        config.model.params.filter_by_visual_cue=config.data.params.validation.params.filter_by_visual_cue
        config.model.params.num_val_samples_to_keep=config.data.params.validation.params.num_val_samples_to_keep
        config.model.params.num_validation_samples=config.data.params.validation.params.num_validation_samples
        config.model.params.non_paired_val_inception_features_path=config.data.params.validation.params.non_paired_val_inception_features_path
        config.model.params.non_paired_val_patch_inception_features_path=config.data.params.validation.params.non_paired_val_patch_inception_features_path
        config.model.params.non_paired_val_retrieval_dict_path=config.data.params.validation.params.non_paired_val_retrieval_dict_path
        # send the mask hyper-params to the model (maybe there is cleaner way of doing this...)
        config.model.params.mask_method=data.datasets['train'].mask_method
        
    # -------------------------------------------------------------------------------
    # model setup
    # ------------------------------------------------------------------------------- 
    
    model = instantiate_from_config(config.model)
    
    model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
    print("Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
        model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr))
    
    # -------------------------------------------------------------------------------
    # callback setup
    # -------------------------------------------------------------------------------
    callbacks_cfg = OmegaConf.create()
    calls = []
    
    # i.e. training the model
    default_callbacks_cfg = {
        "setup_callback": {
            "target": "main.SetupCallback",
            "params": {
                "now": now,
                "logdir": logdir,
                "ckptdir": ckptdir,
                "cfgdir": cfgdir,
                'sample_dir': sampledir,
                "config": config,
                "lightning_config": lightning_config,
                "print_log_dir": 200

            }
        },
        # this callback logs images
        "image_logger": {
            "target": "main.ImageLogger",
            "params": {
                "batch_frequency": 20,
            }
        },
        "learning_rate_logger": {
            "target": "main.LearningRateMonitor",
            "params": {
                "logging_interval": "step",
                #"log_momentum": True
            }
        },
    }

    callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
    calls = [SetupCallback(**callbacks_cfg["setup_callback"].get("params", {})),
            ImageLogger(**callbacks_cfg["image_logger"].get("params", {})),
            LearningRateMonitor(**callbacks_cfg["learning_rate_logger"].get("params", {}))]
    
    checkpoint_callback = ModelCheckpoint(dirpath=ckptdir,
            filename="{epoch}-{step}", save_last=True)
    
    calls.append(checkpoint_callback)
    
    # -------------------------------------------------------------------------------
    # trainer setup
    # -------------------------------------------------------------------------------

    trainer_kwargs = {}

    # -------------------------------------------------------------------------------
    # setup logging with weights and biases
    # -------------------------------------------------------------------------------
    if not os.path.isdir(os.path.join(logdir)):
        os.mkdir(os.path.join(logdir))
    if not os.path.isdir(os.path.join(logdir, "logs")):
        os.mkdir(os.path.join(logdir, "logs"))
    print('logging to '+os.path.join(logdir, "logs"))
    wandb_logger = WandbLogger(name=nowname,project='E2EVE',save_dir=os.path.join(logdir, "logs"))
    trainer_kwargs['log_every_n_steps']=20
    trainer_kwargs['logger']=wandb_logger

    # -------------------------------------------------------------------------------
    # checkpoint resume
    # -------------------------------------------------------------------------------
    # trainer_kwargs["checkpoint_callback"] = True
    # last_checkpoint = find_last_checkpoint_path(ckptdir)
    # print('RESUMING FROM CHECKPOINT FROM PREVIOUS FAILED RUN - ' + str(last_checkpoint))

    # ------------------------------------------
    trainer_kwargs["callbacks"] = calls
    trainer_kwargs['deterministic'] = True
    trainer_kwargs['default_root_dir'] = ckptdir
    trainer_kwargs['num_sanity_val_steps'] = 4

    trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)

   

    # -------------------------------------------------------------------------------
    #  train / validate 
    # -------------------------------------------------------------------------------

    if loaded_config['model']["run_only_validation_epoch"]:

        trainer.validate(model, dataloaders=data.val_dataloader())
        quit()
        
    print('starting_training')

    trainer.fit(model, data, ckpt_path=opt['resume'])


