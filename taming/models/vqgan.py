import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from main import instantiate_from_config

from taming.modules.diffusionmodules.model import Encoder, Decoder, VUNet
from taming.modules.vqvae.quantize import VectorQuantizer
from taming.modules.losses.vqperceptual import VQLPIPSWithDiscriminator

class VQModel(pl.LightningModule):
    def __init__(self,
                 args,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None
                 ):
        super().__init__()
        self.image_key = image_key
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
        
        self.quantize = VectorQuantizer(args.n_embed, args.embed_dim, beta=0.25)
        self.quant_conv = torch.nn.Conv2d(args.z_channels, args.embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(args.embed_dim, args.z_channels, 1)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.image_key = image_key
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

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

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        return x.float()

    def training_step(self, batch, batch_idx, optimizer_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)

        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            return aeloss, log_dict_ae

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")

            return discloss, log_dict_disc

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        rec_loss = log_dict_ae["val/rec_loss"]

        return aeloss, discloss, rec_loss, log_dict_ae, log_dict_disc


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
        return opt_ae, opt_disc

    def get_last_layer(self):
        return self.decoder.conv_out.weight


