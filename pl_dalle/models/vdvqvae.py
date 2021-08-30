import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from torch import distributed as dist
from torch.optim.lr_scheduler import ReduceLROnPlateau
import math
from einops import rearrange
 
from typing import List, Set, Dict, Tuple, Optional

class VDVQVAE(pl.LightningModule):
    def __init__(
        self,
        strides: list[int] = [8,2],
        vocabs: list[int] = [8192],
        in_ch: int = 3,
        hidden_dim: int = 256,
        codebook_dim: int = 256,
        quant_beta: float = 0.99,
        quant_ema_decay: float = 0.99,
        num_res_blocks: int = 2,
        num_res_ch: int = 32,
        lr_decay: bool = False,
        base_lr: float = 4.5e-6,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.strides = strides
        self.vocabs = vocabs
        self.in_ch = in_ch
        self.hidden_dim = hidden_dim
        self.codebook_dim = codebook_dim
        self.quant_beta = quant_beta
        self.quant_ema_decay = quant_ema_decay
        self.num_res_blocks = num_res_blocks
        self.num_res_ch = num_res_ch
        self.num_tokens = sum(vocabs)        
        self.encoders = nn.ModuleList([])
        self.decoders = nn.ModuleList([])
        self.upsamples = nn.ModuleList([])
        self.quant_convs = nn.ModuleList([])
        self.base_lr = base_lr
        self.lr_decay = lr_decay

        for i, stride in enumerate(self.strides):
            if i == 0:
                in_ch = self.in_ch
            else:
                in_ch = self.hidden_dim
            enc = encoder(in_ch, self.hidden_dim, self.hidden_dim, self.num_res_blocks, self.num_res_ch, stride=stride)
            self.encoders.append(enc)
        
        self.max_stride = math.prod(self.strides)

        for j, (i, stride) in enumerate(reversed(list(enumerate(self.strides)))):
            num_codes = j + 1
            prev_codes = self.codebook_dim * j
            in_ch = prev_codes + self.codebook_dim

            qconv = nn.Conv2d(prev_codes + self.hidden_dim, self.codebook_dim, 1)
            self.quant_convs.append(qconv)

            if i == 0:
                out_ch = self.in_ch * 2
                num_res_blocks = self.num_res_blocks
            else:
                out_ch = self.codebook_dim * num_codes
                num_res_blocks = self.num_res_blocks
                self.upsamples.append(upsample_block(in_ch, in_ch, stride))
                
            dec = decoder(
                in_ch, out_ch, self.hidden_dim, 
                self.num_res_blocks, self.num_res_ch, 
                stride=stride
            )
            self.decoders.append(dec)
        
        self.quantizers = nn.ModuleList([])
        for i, v in enumerate(vocabs):
            q = Quantize(self.codebook_dim, v, self.quant_ema_decay)
            if i + 1 < len(vocabs):
                self.quantizers.append(q)
            else:
                rest = len(strides) - i
                for _ in range(rest):
                    self.quantizers.append(q)
                break

    def forward(self, input):
        quants, diff, _ = self.encode(input)
        xrec, logvar = self.decode(quants)
        return xrec, logvar, diff

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
            u = up(out)
            out = torch.cat([up(out), q], dim=1)
        out = self.decoders[-1](out)
        mean, logvar = out.chunk(2, dim=1)
        mean = clamp_with_grad(mean, -1.01, 1.01)
        logvar = clamp_with_grad(logvar, math.log(1e-5), 1.0)
        return mean, logvar
    
    @torch.no_grad()
    def get_codebook_indices(self, img, as_grids=False):
        b = img.shape[0]
        img = (2 * img) - 1
        _, _, ids = self.encode(img)
        if as_grids:
            return ids
        else:
            indices = [ids.view(b, -1) for ids in ids]
            indices = torch.cat(indices, 1)
            return indices

    def training_step(self, batch, batch_idx):
        x = batch[0]
        xrec, logvar, qloss = self(x)

        dist = torch.distributions.Normal(xrec, logvar.div(2).exp())
        recon_loss = -dist.log_prob(x).mean()
        latent_loss = qloss.mean()
        loss = recon_loss + self.quant_beta * latent_loss

        self.log("train/rec_loss", recon_loss, prog_bar=True, logger=True)
        self.log("train/embed_loss", latent_loss, prog_bar=True, logger=True)
        self.log("train/total_loss", loss, prog_bar=True, logger=True)
        self.log("train/mean_var", logvar.mean().exp(), logger=True)
        return {'loss': loss, 'x': x.detach(), 'xrec': xrec.detach()}

    def validation_step(self, batch, batch_idx):
        x = batch[0]
        xrec, logvar, qloss = self(x)

        dist = torch.distributions.Normal(xrec, logvar.div(2).exp())
        recon_loss = -dist.log_prob(x).mean()
        latent_loss = qloss.mean()
        loss = recon_loss + self.quant_beta * latent_loss
        
        self.log("val/rec_loss", recon_loss, prog_bar=True, logger=True)
        self.log("val/embed_loss", latent_loss, prog_bar=True, logger=True)
        self.log("val/total_loss", loss, prog_bar=True, logger=True)  
        self.log("val/mean_var", logvar.mean().exp(), logger=True)
           
        return {'loss':loss, 'x':x.detach(), 'xrec':xrec.detach()}

    def configure_optimizers(self):
        lr = self.base_lr
        opt = torch.optim.Adam(self.parameters(),lr=lr, betas=(0.5, 0.9))
        if self.lr_decay:
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
    
    def codebook_len(self, res):
        seq_len = 0
        for stride in self.strides:
            res = res // stride
            seq_len += res ** 2
        return seq_len

def stack_toks(small, big, v_low):
    bs, hs, ws = small.shape
    bb, hb, wb = big.shape
    assert bs == bb
    assert hs % hb == 0
    assert ws % wb == 0
    kh = hs // hb
    kw = ws // wb
    bases = v_low ** torch.arange(0, kh * kw + 1, device=small.device)
    base_high = bases[-1]
    base_low = bases[:-1].view(1, -1, 1, 1)
    smalls = rearrange(small, 'b (h kh) (w kw) -> b (kh kw) h w', kh=kh, kw=kw).mul(base_low).sum(dim=1)
    bigs = big * base_high
    return smalls + bigs    

def unstack_toks(toks, v_low, kh, kw):
    b, h, w = toks.shape
    assert h % kh == 0
    assert w % kw == 0
    bases = v_low ** torch.arange(0, kh * kw + 1, device=toks.device).view(1,-1,1,1)
    toks, bases = torch.broadcast_tensors(toks.unsqueeze(1), bases)
    rebased = torch.div(toks, bases, rounding_mode='floor')
    rebased_low = rearrange(rebased[:,:-1,:,:] % v_low, 'b (kh kw) h w -> b (h kh) (w kw)', kh=kh, kw=kw)
    rebased_high = rebased[:,-1,:,:]
    return rebased_low, rebased_high

def upsample_block(in_channel, out_channel, stride, hidden_dim=None):
    if hidden_dim is None:
        hidden_dim = out_channel
    blocks = [nn.Conv2d(in_channel, hidden_dim, 3, padding=1)]
    strides = int(math.log2(stride))
    for i in range(strides):
        blocks.extend([
            nn.ConvTranspose2d(hidden_dim, hidden_dim, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        ])
    blocks.append(nn.Conv2d(hidden_dim, out_channel, 1))
    return nn.Sequential(*blocks)
        
def decoder(in_channel, out_channel, channel, n_res_block, n_res_channel, stride):
    blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]
    blocks.append(nn.Sequential(*[ResBlock(channel, n_res_channel) for _ in range(n_res_block)]))
    blocks.append(upsample_block(channel, out_channel, stride))    
    return nn.Sequential(*blocks)

def downsample_block(in_channel, out_channel, stride, hidden_dim=None):
    if hidden_dim is None:
        hidden_dim = in_channel
    blocks = [nn.Conv2d(in_channel, hidden_dim, 3, padding=1)]    
    strides = int(math.log2(stride))
    for i in range(strides):
        blocks.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        ])
    blocks.append(nn.Conv2d(hidden_dim, out_channel, 1))
    return nn.Sequential(*blocks)

def encoder(in_channel, out_channel, channel, n_res_block, n_res_channel, stride):
    blocks = [downsample_block(in_channel, channel, stride, channel // 2)]
    blocks.append(nn.Sequential(*[ResBlock(channel, n_res_channel) for _ in range(n_res_block)]))
    blocks.append(nn.Conv2d(channel, out_channel, 1))
    return nn.Sequential(*blocks)

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
            nn.Conv2d(in_channel, channel, 3, padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out
    

class ClampWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, min, max):
        ctx.min = min
        ctx.max = max
        ctx.save_for_backward(input)
        return input.clamp(min, max)

    @staticmethod
    def backward(ctx, grad_in):
        input, = ctx.saved_tensors
        return grad_in * (grad_in * (input - input.clamp(ctx.min, ctx.max)) >= 0), None, None

clamp_with_grad = ClampWithGrad.apply