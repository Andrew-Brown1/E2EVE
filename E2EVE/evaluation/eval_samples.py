# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
"""
This file contains all the functionality for evaluating a set of metrics, give a set of sampled images
"""
import os
import sys
import pickle
import numpy as np
import json
import torch
from tqdm import tqdm

from utils.utils import calculate_frechet_distance

def log_evaluation_metrics(pl_module, metrics, results_file_name, tb_logger=True):

    print('perceptual diversity = ' + str(metrics['diversity']))
    print('patch perceptual diversity = ' + str(metrics['diversity_patch']))
    print('FID = ' + str(metrics['FID']))
    print('FID patch = ' + str(metrics['FID_patch']))
    print('crop: R1: ' + str(metrics['retrieval'][0])[:6] + ', R5: '+ str(metrics['retrieval'][1])[:6] + ', R10: '+ str(metrics['retrieval'][2])[:6] + ', R20: '+ str(metrics['retrieval'][3])[:6] )
    print('L1 real: '+str(metrics['L1_real'])[:5] + ' |     L1 reconstruct: '+str(metrics['L1_reconstruct'])[:5])
    print('Paired NLL: '+str(metrics['Paired_NLL'])[:5])


    pl_module.log('E2EVE_val/Diversity_NP',metrics['diversity'], on_epoch=True)
    pl_module.log('E2EVE_val/patch_diversity_NP',metrics['diversity_patch'], on_epoch=True)
    pl_module.log('E2EVE_val/FID',metrics['FID'], on_epoch=True)
    pl_module.log('E2EVE_val/FID_R',metrics['FID_patch'], on_epoch=True)
    pl_module.log('E2EVE_val/Retrieval_NP_CR1',metrics['retrieval'][0], on_epoch=True)
    pl_module.log('E2EVE_val/Retrieval_NP_CR5',metrics['retrieval'][1], on_epoch=True)
    pl_module.log('E2EVE_val/Retrieval_NP_CR10',metrics['retrieval'][2],  on_epoch=True)
    pl_module.log('E2EVE_val/Retrieval_NP_CR20',metrics['retrieval'][3], on_epoch=True)
    pl_module.log('E2EVE_val/L1_real',metrics['L1_real'],  on_epoch=True)
    pl_module.log('E2EVE_val/L1_reconstruct',metrics['L1_reconstruct'], on_epoch=True)
    pl_module.log('E2EVE_val/Paired_NLL',metrics['Paired_NLL'],  on_epoch=True)
    
    # write to the text file (because these tensorboard logs die a death for some reason)
    if not os.path.isfile(results_file_name):
        # make the file with the headings
        with open(results_file_name, 'w') as f:
            f.write('global-step: FID   FID-patch  cr1  cr5  cr10  cr20  div   div-patch  L1-real L1-recon   NLL-pair')
            f.write('\n')

    with open(results_file_name, 'a') as f:
            f.write(str(pl_module.global_step) + ':  ' + str(metrics['FID']) + '  ' + str(metrics['FID_patch'])+  '  '+ str(metrics['retrieval'][0])[:6]
                 + '  ' + str(metrics['retrieval'][1])[:6] + '  '+ str(metrics['retrieval'][2])[:6] + '  '+ str(metrics['retrieval'][3])[:6]
                +  '  ' +  str(metrics['diversity']) + '  '+ str(metrics['diversity_patch']) + '  ' + str(metrics['L1_real'])[:5] + '  '
                 + str(metrics['L1_reconstruct'])[:5]  + '  '+ str(metrics['Paired_NLL'])[:5]  )
            f.write('\n')

class EvalMetricsOnFly:

    def __init__(self,
                 non_paired_val_inception_features_path,
                 non_paired_val_patch_inception_features_path,
                 non_paired_val_retrieval_dict_path):
        # this version of the class is to avoid the saving of pickle files of features, which end up taking a long time
        # to load. Now, we have to save them for the FID computation though, so lets actually try that first

        # read the necessary features

        self.read_test_features(non_paired_val_inception_features_path,
                                non_paired_val_patch_inception_features_path,
                                non_paired_val_retrieval_dict_path)

        self.initialise_metrics()
        
    def read_test_features(self, inception_features_path,patch_inception_features_path,retrieval_dict_path):
        #  this loads all of the meta data required for deterministic evaluation. 
        # (1) inception_features_path - extracted inception features from whole real images (required for FID)
        # (2) patch_inception_features_path - extracted inception features from real image regions (required for edit-region FID)
        # (3) retrieval_dict_path - used for the retrieval metrics (determining the retrieval set for each sample)
        
        with open(inception_features_path,'rb') as f:
            self.inception_features = pickle.load(f)
        
        with open(patch_inception_features_path,'rb') as f:
            self.patch_inception_features = pickle.load(f)
            
        with open(retrieval_dict_path,'rb') as f:
            self.retrieval_dict = pickle.load(f)

    def filter_by_visual_cue(self, diversity_features1, diversity_features2, num_samples_old, num_samples_target, val_feats1, val_feats2):

        sorted_scores = np.argsort(self.similarities_to_visual_cue)

        # get the metrics that correspond to the most similar samples to the visual cue
        r1_c = np.array(self.r1_c[-num_samples_old:])[sorted_scores[:num_samples_target]].tolist()
        r5_c = np.array(self.r5_c[-num_samples_old:])[sorted_scores[:num_samples_target]].tolist()
        r10_c = np.array(self.r10_c[-num_samples_old:])[sorted_scores[:num_samples_target]].tolist()
        r20_c = np.array(self.r20_c[-num_samples_old:])[sorted_scores[:num_samples_target]].tolist()
        L1_reconstruct = np.array(self.L1_reconstruct[-num_samples_old:])[sorted_scores[:num_samples_target]].tolist()
        L1_real = np.array(self.L1_real[-num_samples_old:])[sorted_scores[:num_samples_target]].tolist()

        # reset the old stores of metrics and add the new ones
        self.r1_c = self.r1_c[:-num_samples_old] + r1_c
        self.r5_c = self.r5_c[:-num_samples_old] + r5_c
        self.r10_c = self.r10_c[:-num_samples_old] + r10_c
        self.r20_c = self.r20_c[:-num_samples_old] + r20_c
        self.L1_reconstruct = self.L1_reconstruct[:-num_samples_old] + L1_reconstruct
        self.L1_real = self.L1_real[:-num_samples_old] + L1_real

        # reset FID and the diversity_features
        output_div_feats_1 = []
        output_div_feats_2 = []
        val_feats_new1 = val_feats1[-num_samples_old:]
        val_feats_new2 = val_feats2[-num_samples_old:]
        val_feats1 = val_feats1[:-num_samples_old]
        val_feats2 = val_feats2[:-num_samples_old]

        for ii in range(len(sorted_scores[:num_samples_target])):
            output_div_feats_1.append(diversity_features1[sorted_scores[ii]])
            output_div_feats_2.append(diversity_features2[sorted_scores[ii]])
            val_feats1.append(val_feats_new1[sorted_scores[ii]])
            val_feats2.append(val_feats_new2[sorted_scores[ii]])

        return output_div_feats_1, output_div_feats_2, val_feats1, val_feats2

        # basically argsort the similarities - then remove the previous entries to all of the lists, and just keep the similar ones

    def initialise_metrics(self):

        # initialise all of the different stored metrics - done at the beginning of each validation epoch

        self.r1_c,  self.r5_c, self.r10_c, self.r20_c = [], [], [], []
        self.similarities_to_visual_cue = []

        self.diversity, self.diversity_patches = [], []

        self.L1_reconstruct, self.L1_real = [], []

        self.nll_paired = []

    def gather_all_metrics(self, outputs):

        retrieval_outputs = []
        diversity_outputs = []
        diversity_patch_outputs = []
        FID_outputs = []
        FID_patch_outputs = []
        L1_real = []
        L1_reconstruct = []
        NLL_outputs = []

        for part in outputs:
            retrieval_outputs.append(part[0])
            diversity_outputs.append(part[1])
            FID_outputs.append(part[2])
            diversity_patch_outputs.append(part[3])
            L1_real.append(part[4])
            L1_reconstruct.append(part[5])
            FID_patch_outputs.append(part[6])
            NLL_outputs.append(part[7])

        perceptual_diversity_score = self.gather_all_div(diversity_outputs)
        perceptual_diversity_score_patches = self.gather_all_div(diversity_patch_outputs)

        FID = self.gather_all_FID(FID_outputs)
        FID_patch = self.gather_all_FID(FID_patch_outputs, patches=True)
        Retrieval = self.gather_all_retrieval(retrieval_outputs)

        L1_real = self.gather_all_L1(L1_real)
        L1_reconstruct = self.gather_all_L1(L1_reconstruct)

        NLL_paired = self.gather_NLL(NLL_outputs)

        return {'diversity':perceptual_diversity_score, 'FID':FID, 'retrieval':Retrieval, 'diversity_patch':perceptual_diversity_score_patches,
                    'L1_real': L1_real, 'L1_reconstruct': L1_reconstruct, 'FID_patch':FID_patch, 'Paired_NLL':NLL_paired}

    def gather_NLL(self, inputs):

        paired_nll = []
        for part in inputs:
            paired_nll += part[0]

        return np.mean(paired_nll)

    def gather_all_retrieval(self,inputs):

        r1_c, r5_c,r10_c, r20_c  = [], [], [], []

        for part in inputs:
            r1_c += part[0]
            r5_c += part[1]
            r10_c += part[2]
            r20_c += part[3]
            
        print('retrieval length = '+str(len(r1_c)))
        return [np.mean(r1_c),np.mean(r5_c),np.mean(r10_c),np.mean(r20_c)]

    def gather_all_FID(self, inputs, patches=False):
        
        features = []
        for part in inputs:
            features += part

        print('FID length = '+str(len(features)))
        features = np.concatenate(features)
        # then need to array it
        if not patches:
            [mu_test, sigma_test, _, _] = self.inception_features
        else:
            [mu_test, sigma_test, _, _] = self.patch_inception_features

        mu_samples = np.mean(features, axis=0)
        sigma_samples = np.cov(features, rowvar=False)
        fid_value = calculate_frechet_distance(mu_test, sigma_test, mu_samples, sigma_samples)

        return fid_value

    def gather_all_div(self,inputs):

        diversity_outputs = []
        for part in inputs:
            diversity_outputs += part
        print('diversity length = '+str(len(diversity_outputs)))
        return np.mean(diversity_outputs)

    def gather_all_L1(self, inputs):

        L1_outputs = []
        for part in inputs:
            L1_outputs += part
        print('L1 length = '+str(len(L1_outputs)))

        return np.mean(L1_outputs)

    def log_nll_on_fly(self, paired_loss):

        self.nll_paired.append(paired_loss)

    def compute_L1_on_fly(self, sample, coordinates, image, reconstruct=True, segmentor=False, batch_info=False):

        # first, compute the L1 error between the two
        rec_loss = torch.abs(sample.contiguous() - image.contiguous()).cpu()
        
        # second, mask out the patch
        if not segmentor:
            rec_loss[:,:,coordinates[0].item():coordinates[1].item(),coordinates[2].item():coordinates[3].item()] = -100
            mask = rec_loss != -100
        else:
            mask = (batch_info['mask'] == 0).permute(0,3,1,2).cpu()
        L1_select = torch.masked_select(rec_loss, mask)

        # third, find the L1 mean
        L1_loss = torch.mean(L1_select).item()

        # add to the relevant lists
        if reconstruct:
            self.L1_reconstruct.append(L1_loss)
        else:
            self.L1_real.append(L1_loss)

    def compute_retrieval(self, query, retrieval_set, distance='euclid'):
        # given that the positive is the last item in the retrieval set, find its rank in terms of similarity to the query

        if distance == 'euclid':
            similarity = np.linalg.norm((retrieval_set-query), axis=1)

        else:
            normed_query = query / np.linalg.norm(query)
            normed_retrieval_set = retrieval_set / np.expand_dims(np.linalg.norm(retrieval_set,axis=1),axis=1).repeat(2048,axis=1)
            similarity = np.squeeze(np.dot(normed_query, np.transpose(normed_retrieval_set)))

        rank = np.where(np.argsort(similarity)== (len(retrieval_set)-1))[0].item()
        return rank, similarity[-1]

    def compute_retrieval_metrics_crop_on_fly(self, sample_ID, sample_feature, query, verbose=False,  distance='euclid',retrieval_set_size=100):
        # for this, the query is the crop image that was used as the conditioning
        # the sample feature is going to be the edit crop (this is a crop of the 'edit region' (the black part of the input image))
        # the retrieval set will be other random crops from other images (just load the patch inception features)
        held_out_test_set_indices_in_retrieval_set = self.retrieval_dict[sample_ID]

        retrieval_set_features = self.patch_inception_features[2][held_out_test_set_indices_in_retrieval_set][:retrieval_set_size]

        retrieval_set_features =np.concatenate((retrieval_set_features, sample_feature))

        rank_input, similarity = self.compute_retrieval(query, retrieval_set_features,distance=distance)

        self.similarities_to_visual_cue.append(similarity)


        self.r1_c.append(rank_input==0)
        self.r5_c.append(rank_input < 5)
        self.r10_c.append(rank_input < 10)
        self.r20_c.append(rank_input < 20)

    def compute_diversity_on_fly(self, LPIPS_model, diversity_features, patches=False):

        scores = []
        # compute all of the pairwise scores
        for ii in range(len(diversity_features)):

            feature_1 = diversity_features[ii]

            for jj in range(len(diversity_features)):
                if jj > ii:

                    feature_2 = diversity_features[jj]
                    scores.append(LPIPS_model.compute_score_between_feats(feature_1,feature_2).item())

        # the list 'scores' now contains all of the pairwise perceptual similarity scores, so simply
        # return the average of all of these (equal to the mean of the sample_ID-wise means, because
        # every sample_ID should have the same number of samples)

        assert(len(scores) == (( (len(diversity_features)**2)-len(diversity_features))/2))

        if not patches:
            self.diversity.append(np.mean(scores))
        else:
            self.diversity_patches.append(np.mean(scores))

