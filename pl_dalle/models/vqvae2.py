import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from torch import distributed as dist
from torch.optim.lr_scheduler import ReduceLROnPlateau
import math
from einops import rearrange

from pl_dalle.modules.losses.patch import PatchReconstructionDiscriminator

#from torch import distributed as dist
# import vqvae.distributed as dist_fn

# Copyright 2018 The Sonnet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================


# Borrowed from https://github.com/deepmind/sonnet and ported it to PyTorch

class VQVAE_N(pl.LightningModule):
    def __init__(self,
                 args, batch_size, learning_rate,
                 ignore_keys=[],
                 strides=[8, 2],
                 vocabs=[8192]
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        self.recon_loss = nn.MSELoss()
        self.latent_loss_weight = args.quant_beta
        self.image_size = args.resolution
        self.num_tokens = sum([v for v in vocabs])

        self.encoders = nn.ModuleList([])
        self.decoders = nn.ModuleList([])
        self.upsamples = nn.ModuleList([])
        self.quant_convs = nn.ModuleList([])

        for i, stride in enumerate(strides):
            if i == 0:
                in_ch = args.in_channels
            else:
                in_ch = args.hidden_dim
            enc = Encoder(in_ch, args.hidden_dim, args.num_res_blocks, args.num_res_ch, stride=stride)
            self.encoders.append(enc)

        for j, (i, stride) in enumerate(reversed(list(enumerate(strides)))):
            num_codes = j + 1
            prev_codes = args.codebook_dim * j
            in_ch = prev_codes + args.codebook_dim

            qconv = nn.Conv2d(prev_codes + args.hidden_dim, args.codebook_dim, 1)
            self.quant_convs.append(qconv)

            if i == 0:
                out_ch = args.in_channels
            else:
                out_ch = args.codebook_dim * num_codes

                ups = nn.ConvTranspose2d(in_ch, in_ch, 4, stride=stride, padding=1)
                self.upsamples.append(ups)

            dec = Decoder(in_ch, out_ch, args.hidden_dim, args.num_res_blocks, args.num_res_ch, stride=stride)
            self.decoders.append(dec)

        self.quantizers = nn.ModuleList([])
        for i, v in enumerate(vocabs):
            q = Quantize(args.codebook_dim, v, args.quant_ema_decay)
            if i + 1 < len(vocabs):
                self.quantizers.append(q)
            else:
                rest = len(strides) - i
                for _ in range(rest):
                    self.quantizers.append(q)
                break

        self.image_seq_len = 0
        res = self.image_size
        for stride in strides:
            res = res // stride
            self.image_seq_len += res ** 2

    def forward(self, input):
        quants, diff, _ = self.encode(input)
        dec = self.decode(quants)
        return dec, diff

    def encode(self, input):
        pq = input
        prequants = []
        for enc in self.encoders:
            pq = enc(pq)
            prequants.append(pq)
        prequants.reverse()
        quants = []
        ids = []
        diff = 0.0
        z = torch.tensor([], device=pq.device, dtype=pq.dtype)
        for i, (dec, qc, q) in enumerate(zip(self.decoders, self.quant_convs, self.quantizers)):
            pq = prequants[i]
            code = qc(torch.cat([z, pq], dim=1)).permute(0, 2, 3, 1)
            qf, qd, qi = q(code)
            qf = qf.permute(0, 3, 1, 2)
            diff = diff + qd.unsqueeze(0)
            quants.append(qf)
            ids.append(qi)
            # set up next iteration
            if i + 1 < len(self.decoders):
                z = dec(torch.cat([z, qf], dim=1))

        return quants, diff, ids

    def decode(self, quants):
        out = quants[0]
        for q, up in zip(quants[1:], self.upsamples):
            out = torch.cat([up(out), q], dim=1)
        out = self.decoders[-1](out)
        return out

    @torch.no_grad()
    def get_codebook_indices(self, img):
        b = img.shape[0]
        img = (2 * img) - 1
        _, _, ids = self.encode(img)
        indices = [ids.view(b, -1) for ids in ids]
        indices = torch.cat(indices, 1)
        return indices

    def training_step(self, batch, batch_idx):
        x = batch[0]
        xrec, qloss = self(x)

        recon_loss = self.recon_loss(xrec, x)
        latent_loss = qloss.mean()
        loss = recon_loss + self.latent_loss_weight * latent_loss

        self.log("train/rec_loss", recon_loss, prog_bar=True, logger=True)
        self.log("train/embed_loss", latent_loss, prog_bar=True, logger=True)
        self.log("train/total_loss", loss, prog_bar=True, logger=True)

        if self.args.log_images:
            return {'loss': loss, 'x': x.detach(), 'xrec': xrec.detach()}
        else:
            return loss

    def validation_step(self, batch, batch_idx):
        x = batch[0]
        xrec, qloss = self(x)

        recon_loss = self.recon_loss(xrec, x)
        latent_loss = qloss.mean()
        loss = recon_loss + self.latent_loss_weight * latent_loss
        
        self.log("val/rec_loss", recon_loss, prog_bar=True, logger=True)
        self.log("val/embed_loss", latent_loss, prog_bar=True, logger=True)
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


class VQVAE2(VQVAE_N):
    def __init__(self, args, batch_size, learning_rate, stride_1=8, stride_2=2):
        super().__init__(
            args,
            batch_size,
            learning_rate,
            strides=[stride_1, stride_2],
            vocabs=[args.num_tokens],
        )


class Quantize(nn.Module):
    def __init__(self, dim, num_tokens, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.num_tokens = num_tokens
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, num_tokens)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(num_tokens))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, input):
        flatten = input.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.num_tokens).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            #self.all_reduce(embed_onehot_sum)
            #self.all_reduce(embed_sum)

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.num_tokens * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()

        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))


class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out


class Encoder(nn.Module):
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()

        blocks = []
        strides = int(math.log2(stride))

        if strides == 0:
            blocks.append(nn.Conv2d(in_channel, channel // 2, 3, padding=1))
            blocks.append(nn.ReLU(inplace=True))

        for i in range(strides):
            # first stride
            if i == 0:
                blocks.append(nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1))
            # last stride
            elif i + 1 == strides:
                blocks.append(nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1))
            # middle stride
            else:
                blocks.append(nn.Conv2d(channel // 2, channel // 2, 4, stride=2, padding=1))
            blocks.append(nn.ReLU(inplace=True))

        if strides <= 1:
            blocks.append(nn.Conv2d(channel // 2, channel, 3, padding=1))
        else:
            blocks.append(nn.Conv2d(channel, channel, 3, padding=1))

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class Decoder(nn.Module):
    def __init__(
        self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride
    ):
        super().__init__()

        blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        strides = int(math.log2(stride))
        if strides == 1:
            blocks.append(nn.ConvTranspose2d(channel, out_channel, 4, stride=2, padding=1))
        else:
            for i in range(strides):
                if i == 0:
                    blocks.extend([nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1), nn.ReLU(inplace=True)])
                elif i + 1 < strides:
                    blocks.extend([nn.ConvTranspose2d(channel // 2, channel // 2, 4, stride=2, padding=1), nn.ReLU(inplace=True)])
                else:
                    blocks.append(nn.ConvTranspose2d(channel // 2, out_channel, 4, stride=2, padding=1))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)
