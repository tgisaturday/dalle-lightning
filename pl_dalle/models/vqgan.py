import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import math

from pl_dalle.modules.vqvae.vae import Encoder, Decoder
from pl_dalle.modules.vqvae.quantize import VectorQuantizer, EMAVectorQuantizer, GumbelQuantizer
from pl_dalle.modules.losses.vqperceptual import VQLPIPSWithDiscriminator

class VQGAN(pl.LightningModule):
    def __init__(self,
                 args,batch_size, learning_rate, log_images=False,
                 ignore_keys=[]
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.args = args     
        self.log_images =log_images
        self.image_size = args.resolution
        self.num_tokens = args.codebook_dim
        
        f = self.image_size / self.args.attn_resolutions[0]
        self.num_layers = int(math.log(f)/math.log(2))
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
        self.quant_conv = torch.nn.Conv2d(args.z_channels, args.embed_dim, 1)
        self.quantize = VectorQuantizer(args.codebook_dim, args.embed_dim, beta=args.quant_beta)
        self.post_quant_conv = torch.nn.Conv2d(args.embed_dim, args.z_channels, 1)

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    @torch.no_grad()
    def get_codebook_indices(self, img):
        b = img.shape[0]
        img = (2 * img) - 1
        _, _, [_, _, indices] = self.model.encode(img)
        n = indices.shape[0] // b
        indices = indices.view(b,n)       
        return indices

    def forward(self, input):
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff

    def training_step(self, batch, batch_idx, optimizer_idx):     
        x, _ = batch
        xrec, qloss = self(x)
        
        if optimizer_idx == 0:
            # autoencode
            aeloss = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")

            self.log("train/rec_loss", aeloss, prog_bar=True, logger=True)
            self.log("train/embed_loss", qloss, prog_bar=True, logger=True)            
 
            if self.log_images:      
                log_dict = dict()           
                log_dict["train/inputs"] = x
                log_dict["train/reconstructions"] = xrec 
                self.log_dict(log_dict, prog_bar=False, logger=True)
            
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss= self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("train/disc_loss", discloss, prog_bar=True,logger=True)
            
            return discloss

            

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        xrec, qloss = self(x)
        aeloss = self.loss(qloss, x, xrec, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss = self.loss(qloss, x, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        self.log("val/rec_loss", aeloss, prog_bar=True, logger=True)
        self.log("val/disc_loss", discloss, prog_bar=True, logger=True)
        self.log("val/embed_loss", qloss, prog_bar=True, logger=True)        
 
        if self.log_images:   
            log_dict = dict()           
            log_dict["val/inputs"] = x
            log_dict["val/reconstructions"] = xrec 
            self.log_dict(log_dict, prog_bar=False, logger=True)
        return aeloss, discloss


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
        
class EMAVQGAN(VQGAN):
    def __init__(self,
                 args, batch_size, learning_rate, log_images=False,
                 ignore_keys=[]
                 ):
        super().__init__(args, batch_size, learning_rate, log_images,
                         ignore_keys=ignore_keys
                         )
     
        self.quantize = EMAVectorQuantizer(codebook_dim=args.codebook_dim,
                                       embedding_dim=args.embed_dim,
                                       beta=args.quant_beta, decay=args.quant_ema_decay, eps=args.quant_ema_eps)



class GumbelVQGAN(VQGAN):
    def __init__(self,
                 args, batch_size, learning_rate,log_images=False,
                 ignore_keys=[]
                 ): 
        super().__init__(args, batch_size, learning_rate, log_images,
                         ignore_keys=ignore_keys
                         )
        self.temperature = args.starting_temp
        self.anneal_rate = args.anneal_rate
        self.temp_min = args.temp_min
        #quant conv channel should be different for gumbel
        self.quant_conv = torch.nn.Conv2d(args.z_channels, args.codebook_dim, 1)       
        self.quantize = GumbelQuantizer(codebook_dim=args.codebook_dim,
                                       embedding_dim=args.embed_dim,
                                       kl_weight=args.kl_loss_weight, temp_init=args.starting_temp)


    def training_step(self, batch, batch_idx, optimizer_idx):
        x, _ = batch
        #temperature annealing
        self.temperature = max(self.temperature * math.exp(-self.anneal_rate * self.global_step), self.temp_min)
        self.quantize.temperature = self.temperature
        xrec, qloss = self(x)

        if optimizer_idx == 0:
            # autoencode
            aeloss = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")

            self.log("train/rec_loss", aeloss, prog_bar=True, logger=True)
            self.log("train/embed_loss", qloss, prog_bar=True, logger=True)            
 
            if self.log_images:      
                log_dict = dict()               
                log_dict["train/inputs"] = x
                log_dict["train/reconstructions"] = xrec 
                self.log_dict(log_dict, prog_bar=False, logger=True)
            
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss= self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("train/disc_loss", discloss, prog_bar=True,logger=True)
            
            return discloss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        self.quantize.temperature = 1.0        
        xrec, qloss = self(x)
        aeloss = self.loss(qloss, x, xrec, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss = self.loss(qloss, x, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        self.log("val/rec_loss", aeloss, prog_bar=True, logger=True)
        self.log("val/disc_loss", discloss, prog_bar=True, logger=True)
        self.log("val/embed_loss", qloss, prog_bar=True, logger=True)        
 
        if self.log_images:   
            log_dict = dict()           
            log_dict["val/inputs"] = x
            log_dict["val/reconstructions"] = xrec 
            self.log_dict(log_dict, prog_bar=False, logger=True)
        return aeloss, discloss
