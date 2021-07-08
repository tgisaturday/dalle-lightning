import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


from taming.modules.diffusionmodules.model import Encoder, Decoder, VUNet
from taming.modules.vqvae.quantize import VectorQuantizer
from taming.modules.vqvae.quantize import GumbelQuantize
from taming.modules.losses.vqperceptual import VQLPIPSWithDiscriminator

class VQModel(pl.LightningModule):
    def __init__(self,
                 args,batch_size, learning_rate,
                 ignore_keys=[],
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.args = args     
        
        self.encoder = Encoder(ch=args.ch, out_ch=args.out_ch, ch_mult= args.ch_mult,
                                num_res_blocks=args.num_res_blocks, 
                                attn_resolutions=args.attn_resolutions,
                                dropout=args.dropout, in_channels=args.in_channels, 
                                resolution=args.resolution, z_channels=args.z_channels,
                                double_z=args.double_z)

        self.decoder = Decoder(ch=args.ch, out_ch=args.out_ch, ch_mult= args.ch_mult,
                                num_res_blocks=args.num_res_blocks, 
                                attn_resolutions=args.attn_resolutions,
                                dropout=args.dropout, in_channels=args.in_channels, 
                                resolution=args.resolution, z_channels=args.z_channels)
        
        self.loss = VQLPIPSWithDiscriminator(disc_start=args.disc_start, codebook_weight=args.codebook_weight,
                                            disc_in_channels=args.disc_in_channels,disc_weight=args.disc_weight)

        self.quantize = VectorQuantizer(args.n_embed, args.embed_dim, beta=0.25,
                                        remap=remap, sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(args.z_channels, args.embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(args.embed_dim, args.z_channels, 1)


        if monitor is not None:
            self.monitor = monitor

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input):
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff

    def training_step(self, batch, batch_idx, optimizer_idx):
        #temporary fix for tpu pod training progress bar
        #if self.global_step % self.args.refresh_rate = 0:
        #    print('Step:', end='',flush=True)
        #print('Step: %s' %self.global_step)        
        
        x, _ = batch
        xrec, qloss = self(x)
        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("train/aeloss", aeloss, prog_bar=True, logger=False, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("train/discloss", discloss, prog_bar=True,logger=False, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        xrec, qloss = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        rec_loss = log_dict_ae["val/rec_loss"]
        self.log("val/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
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


class GumbelVQ(VQModel):
    def __init__(self,
                 args, batch_size, learning_rate,
                 ignore_keys=[],
                 monitor=None,
                 temperature = 0.9,
                 kl_weight=1e-8,
                 remap=None,
                 ):
        self.save_hyperparameters()
        self.args = args    
        super().__init__( args, batch_size, learning_rate,
                         ignore_keys=ignore_keys,
                         monitor=monitor,
                         )

        self.loss.n_classes = args.n_embed
        self.vocab_size = args.n_embed
        self.temperature = temperature
        self.quantize = GumbelQuantize(args.z_channels, args.embed_dim,
                                       n_embed=args.n_embed,
                                       kl_weight=kl_weight, temp_init=temperature,
                                       remap=remap)

    def encode_to_prequant(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode_code(self, code_b):
        raise NotImplementedError

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, _ = batch
        xrec, qloss = self(x)

        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")

            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        xrec, qloss = self(x, return_pred_indices=True)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        rec_loss = log_dict_ae["val/rec_loss"]
        self.log("val/rec_loss", rec_loss,
                 prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/aeloss", aeloss,
                 prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
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