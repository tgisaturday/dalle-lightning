import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import math
from einops import rearrange

from pl_dalle.modules.vqvae.vae import Encoder, Decoder
from pl_dalle.modules.vqvae.quantize import VectorQuantizer,GumbelQuantize
from pl_dalle.modules.losses.vqperceptual import VQLPIPSWithDiscriminator

class VQGAN(pl.LightningModule):
    def __init__(self,
                 args,batch_size, learning_rate,
                 ignore_keys=[]
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.args = args     
        self.image_size = args.resolution
        self.num_tokens = args.codebook_dim
        
        self.encoder = Encoder(hidden_dim=args.hidden_dim, in_channels=args.in_channels, ch_mult= args.ch_mult,
                                num_res_blocks=args.num_res_blocks, 
                                attn_resolutions=args.attn_resolutions,
                                dropout=args.dropout, 
                                resolution=args.resolution, z_channels=args.z_channels,
                                double_z=args.double_z)

        self.decoder = Decoder(hidden_dim=args.hidden_dim, out_channels=args.out_channels, ch_mult= args.ch_mult,
                                num_res_blocks=args.num_res_blocks, 
                                attn_resolutions=args.attn_resolutions,
                                dropout=args.dropout, in_channels=args.in_channels, 
                                resolution=args.resolution, z_channels=args.z_channels)
        
        self.loss = VQLPIPSWithDiscriminator(disc_start=args.disc_start, codebook_weight=args.codebook_weight,
                                            disc_in_channels=args.disc_in_channels,disc_weight=args.disc_weight)

        self.quantize = VectorQuantizer(args.codebook_dim, args.embed_dim, beta=0.25)
        self.quant_conv = torch.nn.Conv2d(args.z_channels, args.embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(args.embed_dim, args.z_channels, 1)

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h,self.device)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    @torch.no_grad()
    def get_codebook_indices(self, img):
        b = img.shape[0]
        img = (2 * img) - 1
        _, _, [_, _, indices] = self.model.encode(img)
        return rearrange(indices, '(b n) -> b n', b = b)

    def forward(self, input):
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff

    def training_step(self, batch, batch_idx, optimizer_idx):     
        x, _ = batch
        xrec, qloss = self(x)
        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("train/ae_loss", aeloss, prog_bar=True, logger=False)
            log_dict = log_dict_ae     
            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)
            log_dict["train/rec_loss"] = aeloss
            log_dict["train/embed_loss"] = qloss                     
            log_dict["train/inputs"] = x
            log_dict["train/reconstructions"] = xrec 
            self.log_dict(log_dict, prog_bar=False, logger=True)

            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("train/dis_closs", discloss, prog_bar=True,logger=False)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True)
            return discloss

            

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        xrec, qloss = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        self.log("val/ae_loss", aeloss, prog_bar=True, logger=False)

        log_dict = log_dict_ae.update(log_dict_disc)     
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log_dict["val/rec_loss"] = aeloss
        log_dict["val/disc_loss"] =discloss
        log_dict["val/embed_loss"] = qloss                     
        log_dict["val/inputs"] = x
        log_dict["val/reconstructions"] = xrec 
        self.log_dict(log_dict, prog_bar=False, logger=True)
        
        return self.log_dict


    def configure_optimizers(self):
        lr = self.hparams.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc],[]

    def get_last_layer(self):
        return self.decoder.conv_out.weight
        
    def log_images(self, batch, **kwargs):
        log = dict()
        x, _ = batch
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


class GumbelVQGAN(VQGAN):
    def __init__(self,
                 args, batch_size, learning_rate,
                 ignore_keys=[]
                 ):
        self.save_hyperparameters()
        self.args = args    
        super().__init__(args, batch_size, learning_rate,
                         ignore_keys=ignore_keys
                         )

        self.loss.n_classes = args.codebook_dim
        self.vocab_size = args.codebook_dim
        self.temperature = args.starting_temp
        self.anneal_rate = args.anneal_rate
        self.temp_min = args.temp_min
        self.quantize = GumbelQuantize(args.z_channels, 
                                       codebook_dim=args.codebook_dim,
                                       embedding_dim=args.embed_dim,
                                       kl_weight=args.kl_loss_weight, temp_init=args.starting_temp)


    def encode_to_prequant(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode_code(self, code_b):
        raise NotImplementedError

    @torch.no_grad()
    def get_codebook_indices(self, img):
        b = img.shape[0]
        img = (2 * img) - 1
        _, _, [_, _, indices] = self.model.encode(img)
        return rearrange(indices, 'b h w -> b (h w)', b=b)

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, _ = batch
        self.temperature = max(self.temperature * math.exp(-self.anneal_rate * self.global_step), self.temp_min)
        self.quantize.temperature = self.temperature
        xrec, qloss = self(x)

        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")

            self.log("train/ae_loss", aeloss, prog_bar=True, logger=False)
            log_dict = log_dict_ae     
            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)
            log_dict["train/rec_loss"] = aeloss
            log_dict["train/embed_loss"] = qloss                     
            log_dict["train/inputs"] = x
            log_dict["train/reconstructions"] = xrec 
            self.log_dict(log_dict, prog_bar=False, logger=True)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("train/dis_closs", discloss, prog_bar=True,logger=False)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True)
            return discloss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        xrec, qloss = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        self.log("val/ae_loss", aeloss, prog_bar=True, logger=False)
        log_dict = log_dict_ae.update(log_dict_disc)     
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log_dict["val/rec_loss"] = aeloss
        log_dict["val/disc_loss"] =discloss
        log_dict["val/embed_loss"] = qloss                     
        log_dict["val/inputs"] = x
        log_dict["val/reconstructions"] = xrec 
        self.log_dict(log_dict, prog_bar=False, logger=True)
        return self.log_dict

    def log_images(self, batch, **kwargs):
        log = dict()
        x, _ = batch
        x = x.to(self.device)
        # encode
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, _, _ = self.quantize(h)
        # decode
        x_rec = self.decode(quant)
        log["inputs"] = x
        log["reconstructions"] = x_rec
        return log        