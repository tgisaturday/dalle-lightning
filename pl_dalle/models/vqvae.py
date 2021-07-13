import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import math
from einops import rearrange

from pl_dalle.modules.vqvae.vae import Encoder, Decoder
from pl_dalle.modules.vqvae.quantize import VectorQuantizer,GumbelQuantize



class VQVAE(pl.LightningModule):
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
        

        self.smooth_l1_loss = args.smooth_l1_loss
        self.quantize = VectorQuantizer(args.codebook_dim, args.embed_dim, beta=0.25)
        self.quant_conv = torch.nn.Conv2d(args.z_channels, args.embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(args.embed_dim, args.z_channels, 1)

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h, self.device)
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

    def training_step(self, batch, batch_idx):     
        x, _ = batch
        xrec, qloss = self(x)
        if self.smooth_l1_loss:
            aeloss = F.smooth_l1_loss(x, xrec)
        else:
            aeloss = F.mse_loss(x, xrec)   
        loss = aeloss + qloss                     
        self.log("train/ae_loss", aeloss, prog_bar=True, logger=False)

        log_dict = dict()      
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log_dict["train/rec_loss"] = aeloss
        log_dict["train/embed_loss"] = qloss
        log_dict["train/total_loss"] = loss                      
        log_dict["train/inputs"] = x
        log_dict["train/reconstructions"] = xrec 

        self.log_dict(log_dict, prog_bar=False, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        xrec, qloss = self(x)
        if self.smooth_l1_loss:
            aeloss = F.smooth_l1_loss(x, xrec)
        else:
            aeloss = F.mse_loss(x, xrec)   
        loss = aeloss + qloss                     
        self.log("val/ae_loss", aeloss, prog_bar=True, logger=False)

        log_dict = dict()      
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log_dict["val/rec_loss"] = aeloss
        log_dict["val/embed_loss"] = qloss
        log_dict["val/total_loss"] = loss                      
        log_dict["val/inputs"] = x
        log_dict["val/reconstructions"] = xrec 

        self.log_dict(log_dict, prog_bar=False, logger=True)
        return log_dict


    def configure_optimizers(self):
        lr = self.hparams.learning_rate
        opt = torch.optim.Adam(self.parameters(),lr=lr, betas=(0.5, 0.9))
        sched = torch.optim.lr_scheduler.ExponentialLR(optimizer = opt, gamma = self.args.lr_decay_rate)
        return [opt], [sched]

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


class GumbelVQVAE(VQVAE):
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

        if self.smooth_l1_loss:
            aeloss = F.smooth_l1_loss(x, xrec)
        else:
            aeloss = F.mse_loss(x, xrec) 
        loss = aeloss + qloss                      
        self.log("train/ae_loss", aeloss, prog_bar=True, logger=False)
        log_dict = dict()      
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log_dict["train/rec_loss"] = aeloss
        log_dict["train/embed_loss"] = qloss
        log_dict["train/total_loss"] = loss                      
        log_dict["train/inputs"] = x
        log_dict["train/reconstructions"] = xrec 

        self.log_dict(log_dict, prog_bar=False, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        xrec, qloss = self(x)
        self.quantize.temperature = 1.0
        if self.smooth_l1_loss:
            aeloss = F.smooth_l1_loss(x, xrec)
        else:
            aeloss = F.mse_loss(x, xrec)  
        loss = aeloss + qloss                      
        self.log("val/ae_loss", aeloss, prog_bar=True, logger=False)
  
        log_dict = dict()      
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log_dict["val/rec_loss"] = aeloss
        log_dict["val/embed_loss"] = qloss
        log_dict["val/total_loss"] = loss                      
        log_dict["val/inputs"] = x
        log_dict["val/reconstructions"] = xrec 

        self.log_dict(log_dict, prog_bar=False, logger=True)
        return log_dict

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