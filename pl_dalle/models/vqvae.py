import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import math

from pl_dalle.modules.vqvae.vae import Encoder, Decoder
from pl_dalle.modules.vqvae.quantize import VectorQuantizer, EMAVectorQuantizer, GumbelQuantizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from einops import rearrange

class VQVAE(pl.LightningModule):
    def __init__(self,
                 args,batch_size, learning_rate,
                 ignore_keys=[]
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.args = args     
        self.image_size = args.resolution
        self.num_tokens = args.num_tokens
        
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
        

        self.smooth_l1_loss = args.smooth_l1_loss
        self.quant_conv = torch.nn.Conv2d(args.z_channels, args.codebook_dim, 1)        
        self.quantize = VectorQuantizer(args.num_tokens, args.codebook_dim, beta=0.25)
        self.post_quant_conv = torch.nn.Conv2d(args.codebook_dim, args.z_channels, 1)

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, input, feed_seq=False):
        if feed_seq:
            img_seq = input
            image_embeds = self.quantize.embedding(img_seq)
            b, n, d = image_embeds.shape
            h = w = int(math.sqrt(n))
            z = rearrange(image_embeds, 'b (h w) d -> b d h w', h = h, w = w)
        else:
            z = input

        quant = self.post_quant_conv(z)
        dec = self.decoder(quant)
        return dec


    @torch.no_grad()
    def get_codebook_indices(self, img):
        b = img.shape[0]
        img = (2 * img) - 1
        _, _, [_, _, indices] = self.encode(img)
        n = indices.shape[0] // b
        indices = indices.view(b,n)       
        return indices

    def forward(self, input):
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff

    def training_step(self, batch, batch_idx):     
        x, _ = batch
        xrec, qloss = self(x)
        if self.smooth_l1_loss:
            aeloss = F.smooth_l1_loss(x, xrec)
        else:
            aeloss = F.mse_loss(x, xrec)   
        loss = aeloss + qloss                     
        self.log("train/rec_loss", aeloss, prog_bar=True, logger=True)
        self.log("train/embed_loss", qloss, prog_bar=True, logger=True)
        self.log("train/total_loss", loss, prog_bar=True, logger=True)

        if self.args.log_images:
            return {'loss':loss, 'x':x.detach(), 'xrec':xrec.detach()}
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        xrec, qloss = self(x)
        if self.smooth_l1_loss:
            aeloss = F.smooth_l1_loss(x, xrec)
        else:
            aeloss = F.mse_loss(x, xrec)   
        loss = aeloss + qloss                     
        self.log("val/rec_loss", aeloss, prog_bar=True, logger=True)
        self.log("val/embed_loss", qloss, prog_bar=True, logger=True)
        self.log("val/total_loss", loss, prog_bar=True, logger=True)     

        if self.args.log_images:
            return {'loss':loss, 'x':x.detach(), 'xrec':xrec.detach()}
        return loss

    def configure_optimizers(self):
        lr = self.hparams.learning_rate
        opt = torch.optim.Adam(self.parameters(),lr=lr, betas=(0.5, 0.9))
        if self.args.lr_decay:
            scheduler = ReduceLROnPlateau(
            opt,
            mode="min",
            factor=0.5,
            patience=10,
            cooldown=10,
            min_lr=1e-6,
            verbose=True,
            )    
            sched = {'scheduler':scheduler, 'monitor':'val/total_loss'}            
            return [opt], [sched]
        else:
            return [opt], []   

    def get_last_layer(self):
        return self.decoder.conv_out.weight

class EMAVQVAE(VQVAE):
    def __init__(self,
                 args, batch_size, learning_rate, 
                 ignore_keys=[]
                 ):  
        super().__init__(args, batch_size, learning_rate,
                         ignore_keys=ignore_keys
                         )
        self.quantize = EMAVectorQuantizer(num_tokens=args.num_tokens,
                                       codebook_dim=args.codebook_dim,
                                       beta=args.quant_beta, decay=args.quant_ema_decay, eps=args.quant_ema_eps)        
    #exclude self.quantize from grad calculation since it is updated through EMA
    def configure_optimizers(self):
        lr = self.hparams.learning_rate
        opt = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                lr=lr, betas=(0.5, 0.9))
        if self.args.lr_decay:
            scheduler = ReduceLROnPlateau(
            opt,
            mode="min",
            factor=0.5,
            patience=10,
            cooldown=10,
            min_lr=1e-6,
            verbose=True,
            )    
            sched = {'scheduler':scheduler, 'monitor':'val/total_loss'}            
            return [opt], [sched]
        else:
            return [opt], [] 

class GumbelVQVAE(VQVAE):
    def __init__(self,
                 args, batch_size, learning_rate,
                 ignore_keys=[]
                 ):  
        super().__init__(args, batch_size, learning_rate, 
                         ignore_keys=ignore_keys
                         )
        self.temperature = args.starting_temp
        self.anneal_rate = args.anneal_rate
        self.temp_min = args.temp_min
        #quant conv channel should be different for gumbel
        self.quant_conv = torch.nn.Conv2d(args.z_channels, args.num_tokens, 1)           
        self.quantize = GumbelQuantizer(num_tokens=args.num_tokens,
                                       codebook_dim=args.codebook_dim,
                                       kl_weight=args.kl_loss_weight, temp_init=args.starting_temp)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        #temperature annealing
        self.temperature = max(self.temperature * math.exp(-self.anneal_rate * self.global_step), self.temp_min)
        self.quantize.temperature = self.temperature
        xrec, qloss = self(x)
        if self.smooth_l1_loss:
            aeloss = F.smooth_l1_loss(x, xrec)
        else:
            aeloss = F.mse_loss(x, xrec)   
        loss = aeloss + qloss                     
        self.log("train/rec_loss", aeloss, prog_bar=True, logger=True)
        self.log("train/embed_loss", qloss, prog_bar=True, logger=True)
        self.log("train/total_loss", loss, prog_bar=True, logger=True)                
        
        if self.args.log_images:
            return {'loss':loss, 'x':x.detach(), 'xrec':xrec.detach()}
        return loss


    def validation_step(self, batch, batch_idx):
        x, _ = batch
        self.quantize.temperature = 1.0
        xrec, qloss = self(x)
        if self.smooth_l1_loss:
            aeloss = F.smooth_l1_loss(x, xrec)
        else:
            aeloss = F.mse_loss(x, xrec)   
        loss = aeloss + qloss                     
        self.log("val/rec_loss", aeloss, prog_bar=True, logger=True)
        self.log("val/embed_loss", qloss, prog_bar=True, logger=True)
        self.log("val/total_loss", loss, prog_bar=True, logger=True)     

        if self.args.log_images:
            return {'loss':loss, 'x':x.detach(), 'xrec':xrec.detach()}
        return loss
