# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# Taming Transformers: https://github.com/CompVis/taming-transformers
# --------------------------------------------------------

import torch
import os
import PIL
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from utils.utils import auto_init_args

from E2EVE.modules.vqvae.quantize import GumbelQuantize
from E2EVE.main_functions import instantiate_from_config
from E2EVE.modules.diffusionmodules.model import Encoder, Decoder
from E2EVE.modules.vqvae.quantize import VectorQuantizer

from torchvision import transforms as T


class VQModel(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 image_key="source_image",
                 debug=False,
                 distributed_backend=None,
                 val_sample_dir=None,
                 non_paired_validation=None
                 ):
        super().__init__()
        
        auto_init_args(self)

        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.id = 'vqgan'

        self.loss = instantiate_from_config(lossconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        
        self.resolution = ddconfig['resolution']

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)

    def init_from_ckpt(self, path,):

        loaded_dict = torch.load(path, map_location="cpu")
        if "state_dict" in loaded_dict:
            sd = loaded_dict["state_dict"]
        else:
            sd = loaded_dict
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant, return_levels=False):
        quant = self.post_quant_conv(quant)

        if not return_levels:
            dec = self.decoder(quant)
            return dec
        else:

            dec, levels = self.decoder(quant,return_levels=return_levels)
            return dec, levels

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input,return_pred_indices=False):
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff

    def get_input(self, batch, k):

        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        return x.float()

    def training_step(self, batch, batch_idx, optimizer_idx):

        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        
        self.log("learning_rate", self.learning_rate, prog_bar=True,  on_step=True, on_epoch=True)

        global_batch_idx = self.global_step

        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")


            for value in log_dict_ae:
                self.log("train_"+value, log_dict_ae[value], prog_bar=True, on_step=True, on_epoch=True)


            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")

            for value in log_dict_disc:
                self.log("train_"+value, log_dict_disc[value], prog_bar=True, on_step=True, on_epoch=True)

            return discloss

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)

        xrec, qloss = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        for value in log_dict_ae:
            self.log(value, log_dict_ae[value], prog_bar=True, on_step=True, on_epoch=True)
        for value in log_dict_disc:
            self.log(value, log_dict_disc[value], prog_bar=True, on_step=True, on_epoch=True)

        return log_dict_ae


    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))

        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, **kwargs):

        if not isinstance(batch, dict):
            batch = {'image':batch}

        log = {}
        x = self.get_input(batch, self.image_key)[:4]
        x = x.to(self.device)

        xrec, _ = self(x)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec

        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x