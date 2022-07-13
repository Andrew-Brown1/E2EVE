# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# Taming Transformers: https://github.com/CompVis/taming-transformers
# --------------------------------------------------------
import os
import numpy as np
import copy
import torch
import pickle
import torch.nn.functional as F
import pytorch_lightning as pl
from PIL import Image
import time
from torchvision import transforms as T

from utils.inception import InceptionModel

from E2EVE.model.lr_scheduler import LinearWarmupCosineAnnealingLR
from E2EVE.main_functions import instantiate_from_config
from E2EVE.evaluation.eval_samples import EvalMetricsOnFly
from E2EVE.modules.util import SOSProvider
from E2EVE.modules.losses.lpips import LPIPS
from utils.utils import auto_init_args


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class Cond_GPT(pl.LightningModule):
    def __init__(self,
                 transformer_config,
                 first_stage_config,
                 cond_stage_config,
                 cond_stage2_config = None,
                 ckpt_path=None,
                 first_stage_key="source_image",
                 cond_stage_key=None,
                 cond_stage2_key = None,
                 starting_lr = 0.0000001,
                 max_lr_cosine_steps = 300000,
                 number_of_conditions = 1,
                driver_image_size=64,
                mask_method='block',
                val_sample_dir=None,
                distributed_backend=None,
                log_validation_images = False,
                sample_top_k=100,
                sample_top_p=None,
                run_eval_metrics=True,
                extra_retrieval_search=False,
                non_paired_validation=True,
                non_paired_validate_every_n_epoch=1,
                filter_by_visual_cue=False,
                num_validation_samples=10,
                num_val_samples_to_keep=10,
                non_paired_val_inception_features_path='',
                non_paired_val_patch_inception_features_path='',
                non_paired_val_retrieval_dict_path=''):

        super().__init__()

        auto_init_args(self)
        print('checkpoint ' + str(ckpt_path))

        self.id = 'transformer'        
        
        # initialise the model 
        self.first_stage_model = self.init_stage_from_ckpt(first_stage_config)
        self.cond_stage_model = self.init_stage_from_ckpt(cond_stage_config)
        self.cond_stage2_model = self.init_stage_from_ckpt(cond_stage2_config)
        self.transformer = instantiate_from_config(config=transformer_config)

        # resume from checkpoint if specified
        if ckpt_path is not None:
            print('loading checkpoint '+ckpt_path)
            self.init_from_ckpt(ckpt_path)
            
        # initialising some augmentations - required when logging images

        self.crop_processor =  T.Compose([
            T.RandomResizedCrop(driver_image_size,
                                scale=(1, 1.),
                                ratio=(1., 1.)),
        ])

        # initialise all of the data required for non-paired validation

        if self.non_paired_validation:

            self.current_validation_iteration = 0
            # first, initialise the LPIPS model
            print('initialising LPIPS for the sampling validation')
            self.LPIPS_model = LPIPS().eval()
            print('initialising Inception for the sampling validation')
            self.InceptionModel = InceptionModel().eval()
            # last will be the CLIP model for the retrieval (although that is a very big model...)
            self.validation_samples_path = val_sample_dir
            self.val_time = 0
            self.val_features = []
            self.val_patch_features = []
            print('initialising eval metrics on fly object')
            if self.run_eval_metrics:
                self._EvalMetricsOnFly = EvalMetricsOnFly(self.non_paired_val_inception_features_path,
                                                          self.non_paired_val_patch_inception_features_path,
                                                          self.non_paired_val_retrieval_dict_path
                                                          )

            if self.filter_by_visual_cue:
                assert(self.num_validation_samples >= self.num_val_samples_to_keep)
            

    def init_from_ckpt(self, path):

        sd = torch.load(path, map_location="cpu")

        if 'state_dict' in sd:
            sd = sd['state_dict']

        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def init_stage_from_ckpt(self, config):
        model = instantiate_from_config(config)
        model = model.eval()
        model.train = disabled_train
        return model


    def forward(self, xx, cc, c2):
        # one step to produce the logits

        _, z_indices = self.encode_to_z(xx)
        _, c_indices, cond_token_inds = self.encode_to_c([cc, c2],xx)

        cz_indices = torch.cat((c_indices, z_indices), dim=1)

        # target includes all sequence elements (no need to handle first one
        # differently because we are conditioning)
        target = z_indices
        # make the prediction

        logits, _ = self.transformer(cz_indices[:, :-1], cond_indices=cond_token_inds)
        # cut off conditioning outputs - output i corresponds to p(z_i | z_{<i}, c)
        logits = logits[:, c_indices.shape[1]-1:]

        return logits, target

    def top_p_logits(self, logits, top_p=0.9):
        # "nucleus sampling" implementation

        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        out = logits.clone()
        out[:,indices_to_remove] = -float('Inf')

        return out

    def top_k_logits(self, logits, k):      
        # sampling from the top k indexes with highest probability

        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float('Inf')
        return out

    @torch.no_grad()
    def sample(self, xx, cc, steps, cond_indices=None, temperature=1.0, sample=False, top_k=None, top_p=None,
               callback=lambda k: None):

        xx = torch.cat((cc,xx),dim=1)
        block_size = self.transformer.get_block_size()
        assert not self.transformer.training

        for k in range(steps):

            callback(k)
            assert xx.size(1) <= block_size # make sure model can see conditioning
            x_cond = xx if xx.size(1) <= block_size else xx[:, -block_size:]  # crop context if needed

            logits, _ = self.transformer(x_cond,cond_indices=cond_indices)
            # pluck the logits at the final step and scale by temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop probabilities to only the top k options

            if top_k is not None:
                logits = self.top_k_logits(logits, top_k)
            elif top_p is not None:
                logits = self.top_p_logits(logits, top_p)


            # apply softmax to convert to probabilities
            probs = F.softmax(logits, dim=-1)

            # sample from the distribution or take the most likely
            if sample:
                ix = torch.multinomial(probs, num_samples=1)
            else:
                _, ix = torch.topk(probs, k=1, dim=-1)
            # append to the sequence and continue
            xx = torch.cat((xx, ix), dim=1)
        # cut off conditioning
        xx = xx[:, cc.shape[1]:]
        return xx

    @torch.no_grad()
    def encode_to_z(self, x):
        quant_z, _, info = self.first_stage_model.encode(x)
        indices = info[2].view(quant_z.shape[0], -1)

        return quant_z, indices

    @torch.no_grad()
    def encode_to_c(self, cc, xx):
        """
        encoding the conditioning information
        """

        if self.cond_stage_key != "image":
            # so if there is some conditioning

            condition_inds = []
            quants = []
            inds = []
            # for each of the different conditioning signals (i.e. driver and masked source)
            for cond_index, condition_c in enumerate(cc):
                if condition_c is not None:
                    # self.number_of_conditions is the number of each conditioning signal that are present
                    assert(condition_c.shape[1] == self.number_of_conditions)
                    for i in range(self.number_of_conditions):
                        # individually forward pass each of the conditioning samples within this modality

                        if cond_index == 0:
                            quant_c, _, [_,_,indices] = self.cond_stage_model.encode(condition_c[:,i])
                        else:
                            quant_c, _, [_,_,indices] = self.cond_stage2_model.encode(condition_c[:,i])

                        if len(indices.shape) > 2 or self.cond_stage_config is not None:
                            indices = indices.view(condition_c.shape[0], -1)

                        token_index = ((cond_index*self.number_of_conditions)+i)
                        condition_inds.append(torch.tensor([token_index]).repeat(indices.shape[1]).to(self.device))
                        quants.append(quant_c)
                        inds.append(indices)

            return quants, torch.cat(inds,dim=-1), torch.cat(condition_inds)


        else:
            # if no conditioning (then the tensor is shaped differently)
            quant_c, _, [_,_,indices] = self.cond_stage_model.encode(cc)
            return quant_c, indices, torch.tensor([0]).repeat(indices.shape[1]).to(self.device)

    @torch.no_grad()
    def decode_to_img(self, index, zshape):

        bhwc = (zshape[0],zshape[2],zshape[3],zshape[1])
        quant_z = self.first_stage_model.quantize.get_codebook_entry(
            index.reshape(-1), shape=bhwc)
        x = self.first_stage_model.decode(quant_z)
        return x



    def log_validation_sample_input_images(self, batch, sample_ID, save_path):
        # log the inputs for each validation sample
        input_im = (((torch.clamp(torch.squeeze(batch['source_image']).detach().cpu(),-1,1).numpy()+1.0)/2.0)*255).astype(np.uint8)
        self.save_image(input_im, os.path.join(save_path,sample_ID + '_source_image.jpg'))
        crop = (((torch.clamp(torch.squeeze(batch['driver_image']).detach().cpu(),-1,1).numpy()+1.0)/2.0)*255).astype(np.uint8)
        self.save_image(crop, os.path.join(save_path,sample_ID + '_driver_image.jpg'))
        input_im = (((torch.clamp(torch.squeeze(batch['masked_source']).detach().cpu(),-1,1).numpy()+1.0)/2.0)*255).astype(np.uint8)
        self.save_image(input_im, os.path.join(save_path,sample_ID + '_masked_source.jpg'))
        input_im = (((torch.clamp(torch.squeeze(batch['image_b_with_driver_removed']).detach().cpu(),-1,1).numpy()+1.0)/2.0)*255).astype(np.uint8)
        self.save_image(input_im, os.path.join(save_path,sample_ID + '_image_b_with_driver_removed.jpg'))
        input_im = (((torch.clamp(torch.squeeze(batch['paired_driver_image']).detach().cpu(),-1,1).numpy()+1.0)/2.0)*255).astype(np.uint8)
        self.save_image(input_im, os.path.join(save_path,sample_ID + '_paired_driver_image.jpg'))

    @torch.no_grad()
    def validation_sample(self, batch, save_path, num_samples=1, verbose=False, temperature=None, top_k=100, top_p=None, callback=None, lr_interface=False, do_not_remake=False, **kwargs):
        # this method will, for a given set of inputs, generate a number of samples, and then compute evaluation metrics on these samples
        
        
        # check that the perceptual model is on cpu
        if self.run_eval_metrics:
            if next(self.LPIPS_model.parameters()).is_cuda:
                # putting LPIPS model on CUDA (for evaluation metrics on the generated samples)
                self.LPIPS_model = self.LPIPS_model.cpu() 

        # ---------------------------------------------------------
        # prepare the inputs and data structs for storing results
        # ---------------------------------------------------------

        assert(batch['source_image'].shape[0]==1)
        N=1
        xx, cc, c2 = self.get_xc(batch, N)
        xx = xx.to(device=self.device)
        cc = cc.to(device=self.device)
        if c2 is not None:
            c2 = c2.to(device=self.device)

        quant_z, z_indices = self.encode_to_z(xx)
        quant_c, c_indices, cond_token_inds = self.encode_to_c([cc, c2],xx)

        sample_ID = batch['file_path_'][0]
 
        features_for_diversity_computation = []
        patch_features_for_diversity_computation = []

        # ---------------------------------------------------------
        # save the input images (optionally)
        # ---------------------------------------------------------

        if self.log_validation_images:
            self.log_validation_sample_input_images(batch, sample_ID, save_path)
            
        # ---------------------------------------------------------
        # generate the samples, and compute evaluation metrics
        # ---------------------------------------------------------

        if self.run_eval_metrics:
            self._EvalMetricsOnFly.similarities_to_visual_cue = []
            
        for sample_index in range(self.num_validation_samples):


            image_path_name = batch['file_path_'][0][:-4] + '_' + str(self.global_rank) + str(sample_index) + '.jpg'

            # compute the sample, and return the samples
            x_sample_nopix, x_sample_nopix_patch, x_rec, sample, input_im, crop, sample_patch = self._make_evaluation_sample(z_indices, c_indices, cond_token_inds, temperature, top_k, top_p, callback, quant_z, batch, xx, save_path, image_path_name)


            # ---------------------------------------------------------
            # compute the LPIPS features for diversity computation
            # ---------------------------------------------------------
            if self.run_eval_metrics:
                features_for_diversity_computation, patch_features_for_diversity_computation = self._run_eval_metrics(x_sample_nopix, features_for_diversity_computation, x_sample_nopix_patch, patch_features_for_diversity_computation, verbose, sample, sample_patch,input_im, crop,batch,x_rec, sample_ID, save_path, image_path_name)

        # ---------------------------------------------------------
        # run evaluation metrics for all samples
        # ---------------------------------------------------------
        self._run_eval_on_all_samples(features_for_diversity_computation, patch_features_for_diversity_computation, num_samples, batch)


    def _make_evaluation_sample(self, z_indices, c_indices, cond_token_inds, temperature, top_k, top_p, callback, quant_z, batch, xx, save_path, image_path_name):

        # ---------------------------------------------------------
        # make the sample
        # ---------------------------------------------------------

        z_start_indices = z_indices[:, :0]

        index_sample = self.sample(z_start_indices, c_indices, cond_indices=cond_token_inds,
                                steps=z_indices.shape[1],
                                temperature=temperature if temperature is not None else 1.0,
                                sample=True,
                                top_k=top_k,
                                top_p=top_p,
                                callback=callback if callback is not None else lambda k: None)

        x_sample_nopix = self.decode_to_img(index_sample, quant_z.shape)
        x_sample_nopix_patch = x_sample_nopix[:,:,batch['meta_coordinates'][0][0].item():batch['meta_coordinates'][0][1].item(),batch['meta_coordinates'][0][2].item():batch['meta_coordinates'][0][3].item()]

        # full_image reconstruction
        x_rec = self.decode_to_img(z_indices, quant_z.shape)

        if not self.run_eval_metrics and self.mask_method == 'random_mask':
            # then map the backgrounds
            x_sample_nopix = (((batch['masked_source'].squeeze(0)!=-1)*batch['masked_source'].squeeze(0)) + ((batch['masked_source'].squeeze(0)==-1)*x_sample_nopix.permute(0,2,3,1))).permute(0,3,1,2)

        # ---------------------------------------------------------
        # prepare the sampled images to be saved
        # ---------------------------------------------------------
        # save the image at this point (maybe save the crop and the inpaint and the image as well)
        sample = (((torch.clamp(torch.squeeze(x_sample_nopix).detach().cpu(),-1,1).transpose(0,1).transpose(1,2).numpy()+1.0)/2.0)*255).astype(np.uint8)
        input_im = (((torch.clamp(torch.squeeze(batch['source_image']).detach().cpu(),-1,1).numpy()+1.0)/2.0)*255).astype(np.uint8)
        crop = (((torch.clamp(torch.squeeze(batch['driver_image']).detach().cpu(),-1,1).numpy()+1.0)/2.0)*255).astype(np.uint8)
        sample_patch = sample[batch['meta_coordinates'][0][0].item():batch['meta_coordinates'][0][1].item(),batch['meta_coordinates'][0][2].item():batch['meta_coordinates'][0][3].item(),:]


        if self.log_validation_images:
            self.save_image(sample, os.path.join(save_path,image_path_name))

        return x_sample_nopix, x_sample_nopix_patch, x_rec, sample, input_im, crop, sample_patch


    def _run_eval_on_all_samples(self, features_for_diversity_computation, patch_features_for_diversity_computation, num_samples, batch):

        # ------------------------------------------------------------
        # only keep the most similar to the visual cue
        # ------------------------------------------------------------

        if self.filter_by_visual_cue and self.run_eval_metrics:
            # this basically takes all of the samples, and just keeps all of the outputs that correspond to the (num_samples_keep) most similar samples to the visual cue
            features_for_diversity_computation, patch_features_for_diversity_computation, self.val_features, self.val_patch_features =  self._EvalMetricsOnFly.filter_by_visual_cue(
                                    features_for_diversity_computation, patch_features_for_diversity_computation, self.num_validation_samples, self.num_val_samples_to_keep, self.val_features, self.val_patch_features)

        # ---------------------------------------------------------
        # compute/store the diversity metrics for these samples
        # ---------------------------------------------------------
        if num_samples > 0 and self.run_eval_metrics:
            self._EvalMetricsOnFly.compute_diversity_on_fly(self.LPIPS_model, features_for_diversity_computation)
            self._EvalMetricsOnFly.compute_diversity_on_fly(self.LPIPS_model, patch_features_for_diversity_computation, patches=True)

        # putting LPIPs model back on GPU
        if self.run_eval_metrics:
            self.LPIPS_model = self.LPIPS_model.to(self.device) # is this the issue?

            batch['driver_image'] = batch['paired_driver_image']
            paired_nll_loss = self.shared_step(batch, 0)
            self._EvalMetricsOnFly.log_nll_on_fly(paired_nll_loss.item())


    def _run_eval_metrics(self, x_sample_nopix, features_for_diversity_computation, x_sample_nopix_patch, patch_features_for_diversity_computation, verbose, sample, sample_patch,input_im, crop,batch,x_rec, sample_ID, save_path, image_path_name):
        # compute evaluation metrics on the samples
        
        # ---------------------------------------------------------
        # compute the Inception features for FID and retrieval
        # ---------------------------------------------------------
        perceptual_feature = self.LPIPS_model.forward_feature(x_sample_nopix.cpu())
        features_for_diversity_computation.append(perceptual_feature)

        # if self.evaluation_method == 'clothing_restyle':
        #     x_sample_nopix_patch = F.interpolate(x_sample_nopix_patch, size=(64,64))

        perceptual_feature_patch = self.LPIPS_model.forward_feature(x_sample_nopix_patch.cpu())
        patch_features_for_diversity_computation.append(perceptual_feature_patch)

        # ---------------------------------------------------------
        # compute the Inception features for FID and retrieval
        # ---------------------------------------------------------

        inception_feature_sample = self.InceptionModel(sample,transform=True, device=self.device)
        inception_feature_sample_patch = self.InceptionModel(sample_patch,transform=True, device=self.device)
        inception_feature_input = self.InceptionModel(input_im,transform=True, device=self.device)
        inception_feature_crop = self.InceptionModel(crop,transform=True, device=self.device)

        self.val_features.append(inception_feature_sample)
        self.val_patch_features.append(inception_feature_sample_patch)

        # ---------------------------------------------------------
        # compute the L1 reconstruction metrics
        # ---------------------------------------------------------
        self._EvalMetricsOnFly.compute_L1_on_fly(x_sample_nopix, batch['meta_coordinates'][0], x_rec , reconstruct=True, segmentor=(self.mask_method == 'random_mask'),batch_info=batch)
        self._EvalMetricsOnFly.compute_L1_on_fly(x_sample_nopix, batch['meta_coordinates'][0], batch['source_image'].permute(0,3,1,2) , reconstruct=False,  segmentor=(self.mask_method == 'random_mask'),batch_info=batch)


        # ---------------------------------------------------------
        # compute/store the retrieval metrics for this sample
        # ---------------------------------------------------------

        inception_feature_sample_patch = self.InceptionModel.sweep_crops(sample_patch,inception_feature_crop,transform=True, device=self.device, extra=self.extra_retrieval_search)

        self._EvalMetricsOnFly.compute_retrieval_metrics_crop_on_fly(sample_ID, inception_feature_sample_patch, inception_feature_crop,verbose=False,retrieval_set_size=100)

        return features_for_diversity_computation, patch_features_for_diversity_computation


    def save_image(self, image, image_path):

        Image.fromarray(image).save(image_path)

    def remake_inpaint_data(self, batch):
        # this function is used so that when logging images during training, drivers and sources come from different images 
        # but are aligned (i.e. the driver comes from the same region as the masked area in the source)

        extended_coors = batch['meta_coordinates'][0]

        # first, remake the inpaint images
        # -1 is the value that the zerod out regions are set to
        batch['masked_source'] = torch.unsqueeze(copy.deepcopy(batch['source_image']),dim=1)
        batch['masked_source'][:,:,extended_coors[2]:extended_coors[3],extended_coors[0]:extended_coors[1],:] = -1

        # need to do the transform here
        # second, make the crops
        batch['driver_image'] = torch.unsqueeze(self.crop_processor(copy.deepcopy(batch['source_image'])[:,extended_coors[6]:extended_coors[7],extended_coors[4]:extended_coors[5],:].permute(0,3,1,2)).permute(0,2,3,1),dim=1)

        # convert to numpy and then back to tensor

    @torch.no_grad()
    def log_images(self, batch, temperature=None, top_k=None, callback=None,do_not_remake=False, only_deterministic=False, **kwargs):

        if not do_not_remake:
            # here, the inputs are remade, so that the images are logged where the masked source 
            # and driver come from different images
            if not (self.mask_method == 'random_mask'):
                self.remake_inpaint_data(batch)
            self.swap_conditionings(batch,'driver_image')

        log = {}

        N = 4
                
        xx, cc, c2 = self.get_xc(batch, N)

        xx = xx.to(device=self.device)
        cc = cc.to(device=self.device)
        if c2 is not None:
            c2 = c2.to(device=self.device)

        quant_z, z_indices = self.encode_to_z(xx)
        quant_c, c_indices, cond_token_inds = self.encode_to_c([cc, c2],xx)
        
        # ---------------------------------------------------------
        # generate and log the samples
        # ---------------------------------------------------------

        # sample
        x_sample_nopix=None
        if not only_deterministic:

            z_start_indices = z_indices[:, :0]
            index_sample = self.sample(z_start_indices, c_indices, cond_indices=cond_token_inds,
                                    steps=z_indices.shape[1],
                                    temperature=temperature if temperature is not None else 1.0,
                                    sample=True,
                                    top_k=top_k if top_k is not None else 100,
                                    callback=callback if callback is not None else lambda k: None)
            x_sample_prob = self.decode_to_img(index_sample, quant_z.shape)

        # deterministic sample
        z_start_indices = z_indices[:, :0]
        index_sample = self.sample(z_start_indices, c_indices, cond_indices=cond_token_inds,
                                   steps=z_indices.shape[1],
                                   sample=False,
                                   callback=callback if callback is not None else lambda k: None)
        x_sample_det = self.decode_to_img(index_sample, quant_z.shape)

        # reconstruction
        x_rec = self.decode_to_img(z_indices, quant_z.shape)

        log["inputs"] = xx
        log["reconstructions"] = x_rec
        if not only_deterministic:
            log["samples"] = x_sample_prob
        log["samples_det"] = x_sample_det
        # ---------------------------------------------------------
        # log the conditioning information
        # ---------------------------------------------------------

        if self.cond_stage_key != "source_image":
            # repeat this for the number of conditioning patches
            for i in range(self.number_of_conditions):
                cond_rec = self.cond_stage_model.decode(quant_c[i])

                log["driver_rec_"+str(i)] = cond_rec
                log["driver_"+str(i)] = cc[:,i]


        if self.cond_stage2_key == 'masked_source':

            assert(self.number_of_conditions==1)
            for i in range(self.number_of_conditions):
                shifted_index = (self.number_of_conditions + i)

                cond_rec = self.cond_stage2_model.decode(quant_c[shifted_index])

                log["masked_source_rec_"+str(i)] = cond_rec
                log["masked_source_"+str(i)] = c2[:,i]

        return log


    def get_input(self, key, batch):
        x = batch[key]
        if len(x.shape) == 3:
            x = x[..., None]

        if key == 'driver_image' or key == 'source_image' or key == 'coord' or key == 'masked_source':
            # if it is image data
            if len(x.shape) == 4:
                x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
            # have to be really careful here with the coordinate conditionings
            if len(x.shape) == 5:
                # this is for when there are multiple conditionings
                x = x.permute(0, 1, 4, 2, 3).to(memory_format=torch.contiguous_format)
        if x.dtype == torch.double:
            x = x.float()
        return x

    def get_xc(self, batch, N=None):

        xx = self.get_input(self.first_stage_key, batch)
        c1 = self.get_input(self.cond_stage_key, batch)

        if self.cond_stage2_key is not None:
            c2 = self.get_input(self.cond_stage2_key, batch)
            if N is not None:
                c2 = c2[:N]
        else:
            c2 = None

        if N is not None:
            xx = xx[:N]
            c1 = c1[:N]
        return xx, c1, c2

    def shared_step(self, batch, batch_idx):

        xx, c1, c2 = self.get_xc(batch)

        logits, target = self(xx, c1, c2)

        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1),ignore_index=-1) # ignore index is for EDIBERT
        return loss

    def training_step(self, batch, batch_idx):
                
        loss = self.shared_step(batch, batch_idx)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("learning_rate", self.learning_rate, prog_bar=True,  on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):


        if self.non_paired_validation:
            # then take the E2EVE style validation step - computing evaluation metrics on 
            # non-paired inputs (i.e. without ground truth)

            # then do the sampling validation
            if self.current_epoch % self.non_paired_validate_every_n_epoch == 0:
                val_save_path = os.path.join(self.validation_samples_path,'samples' + str(self.current_validation_iteration))
                if self.log_validation_images or self.high_res_save:
                    if not os.path.isdir(val_save_path):
                        os.path.isdir(val_save_path)

                self.validation_sample(batch,save_path=val_save_path,num_samples=self.num_validation_samples,top_k=self.sample_top_k, top_p=self.sample_top_p)

        loss = self.shared_step(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True,  on_step=True, on_epoch=True)

    
        return loss

    def swap_conditionings(self, batch, key):
        # this method just changes around the crop conditionings
        first_conditions = torch.unsqueeze(batch[key][:,0],dim=1)
        # flip the first ones
        first_conditions = torch.flip(first_conditions,[0])
        # stitch back together with the first
        batch[key] = first_conditions

    def configure_optimizers(self):
        """
        Following minGPT:
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelis_weight_modules = (torch.nn.Linear, )
        blacklis_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.transformer.named_modules():
            for pn, _p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelis_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklis_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.transformer.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(decay)], "weight_decay": 0.01},
            {"params": [param_dict[pn] for pn in sorted(no_decay)], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=self.learning_rate, betas=(0.9, 0.95))

        scheduler = LinearWarmupCosineAnnealingLR(optimizer, base_lr = self.starting_lr, max_epochs=self.max_lr_cosine_steps)

        return {'optimizer':optimizer, 'lr_scheduler': scheduler}
