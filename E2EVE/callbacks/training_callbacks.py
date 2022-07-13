# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

from pytorch_lightning.callbacks import Callback
import os
import numpy as np
import time
import pathlib
import torchvision
import wandb
import torch
from omegaconf import OmegaConf
import torch.distributed as dist
from pytorch_lightning.utilities.distributed import rank_zero_only

import pytorch_lightning as pl
from omegaconf._utils import is_dataclass, is_attr_class

from E2EVE.evaluation.eval_samples import log_evaluation_metrics

class SetupCallback(Callback):
    def __init__(self, now, logdir, ckptdir, cfgdir, sample_dir, config, lightning_config, print_log_dir):
        super().__init__()

        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.sample_dir = sample_dir
        self.config = config
        self.lightning_config = lightning_config
        self.print_log_dir = print_log_dir

    def setup(self,trainer, pl_module, stage=None):

        if pl_module.non_paired_validation:
            # then make the logs

            if not os.path.isdir(self.logdir):
                os.mkdir(self.logdir)
            if not os.path.isdir(self.sample_dir):
                os.mkdir(self.sample_dir)

            if not os.path.isdir(os.path.join(self.sample_dir,'samples0')):
                os.mkdir(os.path.join(self.sample_dir,'samples0'))

    def on_pretrain_routine_start(self, trainer, pl_module):

        if trainer.global_rank == 0:
            # Create logdirs and save configs


            if not os.path.isdir(self.logdir):
                os.mkdir(self.logdir)
            if not os.path.isdir(self.ckptdir):
                os.mkdir(self.ckptdir)
            if not os.path.isdir(self.cfgdir):
                os.mkdir(self.cfgdir)

            # create the inner log dir
            os.mkdir(os.path.join(self.logdir,self.now))

            self.save_omega_conf(self.config,
                           os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)))

            self.save_omega_conf(OmegaConf.create({"lightning": self.lightning_config}),
                           os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)))


    def on_validation_epoch_end(self, trainer, pl_module):
        # gather and compute metrics over all of the samples generated during the validation iterations (will work 
        # alongside distributed training)
        if pl_module.non_paired_validation and pl_module.run_eval_metrics and (pl_module.current_epoch % pl_module.non_paired_validate_every_n_epoch == 0):

            pl_module.current_validation_iteration += 1

            if pl_module.global_rank == 0:
                print('gathering all metrics')
                print('dist backend = '+pl_module.distributed_backend)

            # get all of the metrics/features that were computed from the samples at the sample level
            retrieval_objects_to_gather = [pl_module._EvalMetricsOnFly.r1_c,pl_module._EvalMetricsOnFly.r5_c,pl_module._EvalMetricsOnFly.r10_c,pl_module._EvalMetricsOnFly.r20_c]
            diversity_objects_to_gather = pl_module._EvalMetricsOnFly.diversity
            patch_diversity_objects_to_gather = pl_module._EvalMetricsOnFly.diversity_patches
            FID_object_to_gather = pl_module.val_features
            FID_object_to_gather_patches = pl_module.val_patch_features
            L1_real_objects_to_gather = pl_module._EvalMetricsOnFly.L1_real
            L1_reconstruct_objects_to_gather = pl_module._EvalMetricsOnFly.L1_reconstruct
            NLL_objects_to_gather = [pl_module._EvalMetricsOnFly.nll_paired]
            obects_to_gather = [retrieval_objects_to_gather, diversity_objects_to_gather, FID_object_to_gather, patch_diversity_objects_to_gather,
                                L1_real_objects_to_gather,L1_reconstruct_objects_to_gather,FID_object_to_gather_patches, NLL_objects_to_gather]
            
            # if using disributed data parralel, gather these data structures from accross nodes
            if 'ddp' in pl_module.distributed_backend:
                # then need to distributed gather everything
                object_output = [None for _ in range(dist.get_world_size())]
                dist.all_gather_object(object_output, obects_to_gather)
            else:
                object_output = [obects_to_gather]

            # compute metrics and log them
            if pl_module.global_rank == 0:
                print('total parts = '+str(len(object_output)))
                # then need to do the post_processing and evaluation
                start_metric_eval = time.time()
                metrics = pl_module._EvalMetricsOnFly.gather_all_metrics(object_output)
                results_file_name = os.path.join(self.logdir,self.now + '.txt')
                log_evaluation_metrics(pl_module, metrics, results_file_name)
                pl_module.LPIPS_model = pl_module.LPIPS_model.to(pl_module.device)
                print('total val time = '+str(time.time()-pl_module.val_time))
                print('total val metric time = '+str(time.time()-start_metric_eval))

            # reset all of the metrics for the next epoch
            pl_module.val_features = []
            pl_module.val_patch_features = []
            pl_module._EvalMetricsOnFly.initialise_metrics()


    def on_validation_epoch_start(self, trainer, pl_module):
        if pl_module.global_rank == 0 and pl_module.non_paired_validation:
            pl_module.val_time = time.time()
            # also verify that all of the metrics are zero'd
            if pl_module.run_eval_metrics:
                assert(len(pl_module._EvalMetricsOnFly.r1_c)==0 and len(pl_module._EvalMetricsOnFly.diversity)==0 and len(pl_module.val_features) == 0)

    def on_train_epoch_end(self, trainer, pl_module): 
        # log the validation overall loss (mean)
        if pl_module.global_rank == 0:
            # now set up the validation sample directories
            if pl_module.non_paired_validation and (pl_module.current_epoch % pl_module.non_paired_validate_every_n_epoch == 0):
                # then make sure that the next directory is ready
                next_sample_dir = os.path.join(self.sample_dir,'samples' + str(pl_module.current_epoch+1))
                if not os.path.isfile(next_sample_dir):
                    os.mkdir(next_sample_dir)


    def save_omega_conf(self, config, f, resolve = False):
        """this method is for saving config files"""

        if is_dataclass(config) or is_attr_class(config):
            config = OmegaConf.create(config)
        data = OmegaConf.to_yaml(config, resolve=resolve)

        if isinstance(f, (str, pathlib.Path)):
            with open(f, 'w') as file:
                file.write(data)
        elif hasattr(f, "write"):
            f.write(data)
            f.flush()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """this will be for manually running the validation (if using hive)"""

        # (1) step the LR scheduler (currently - only if transformer architecture)
        if 'transformer' in pl_module.id and pl_module.lr_schedulers() is not None:
            pl_module.lr_schedulers().step()
            pl_module.learning_rate = pl_module.lr_schedulers().get_lr()[0]


class ImageLogger(Callback):
    def __init__(self, batch_frequency, clamp=True, increase_log_steps=False):
        super().__init__()
        self.batch_freq = batch_frequency
        self.max_images = 4

        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp

    @rank_zero_only
    def log_local(self, save_dir, split, images,
                  global_step, current_epoch, batch_idx, global_batch_idx,pl_module, trainer):
        # this is for making a grid image of all of the logged images

        image_order = ['inputs','driver_0','masked_source_0','reconstructions','driver_rec_0','masked_source_rec_0',
                        'samples','samples_det']
        caption = []
        original_size = images['inputs'].shape
        for k in images:

            if not images[k].shape[2] == original_size[2]:
                images[k] = torch.nn.functional.interpolate(images[k],size=(original_size[2],original_size[2]))
            images[k] = torch.transpose(images[k],2,3)

            grid = torchvision.utils.make_grid(images[k], nrow=4)

            grid = (grid+1.0)/2.0

            images[k] = grid

        # transposing everything
        for k in images:
            images[k] = torch.transpose(images[k],1,2)

        inputs = []
        for key in image_order:
            if key in images:
                caption.append(key)
                inputs.append(images[key])

        total_grid = np.concatenate(inputs,axis=2)

        pl_module.logger.experiment.log({"sampled_images_"+split:[wandb.Image(np.transpose(total_grid, (1,2,0)), caption=' - '.join(caption))]})


    def log_img(self, pl_module, batch, batch_idx, trainer, split="train"):

        if (self.check_frequency(batch_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            # double check that the transformer is in eval() mode
            if not pl_module.id == 'vqgan':
                is_transformer_train = pl_module.transformer.training
                if is_transformer_train:
                    pl_module.transformer.eval()

            # sample the log images
            with torch.no_grad():
                images = pl_module.log_images(batch, split=split)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            global_batch_idx = pl_module.global_step

            self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx, global_batch_idx,pl_module, trainer)

            if is_train:
                pl_module.train()
            if not pl_module.id == 'vqgan':
                if is_transformer_train:
                    pl_module.transformer.train()

    def check_frequency(self, batch_idx):
        if (batch_idx % self.batch_freq) == 0 or (batch_idx in self.log_steps):
            try:
                self.log_steps.pop(0)
            except IndexError:
                pass
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):

        self.log_img(pl_module, batch, batch_idx,trainer, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):

        self.log_img(pl_module, batch, batch_idx,trainer, split="val")
