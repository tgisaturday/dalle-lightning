import torch
import torch.nn as nn
from torch import einsum
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """
    def __init__(self, codebook_dim, embedding_dim, beta, unknown_index="random"):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.codebook_dim = codebook_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.codebook_dim, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.codebook_dim, 1.0 / self.codebook_dim)

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp==1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits==False, "Only for interface compatible with Gumbel"
        assert return_logits==False, "Only for interface compatible with Gumbel"
        # reshape z -> (batch, height, width, channel) and flatten
        #z, 'b c h w -> b h w c'
        z = z.permute(0, 2, 3, 1)
        z_flattened = z.view(-1, self.embedding_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, self.embedding.weight.permute(1,0)) # 'n d -> d n'

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        perplexity = None
        min_encodings = None

        # compute loss for embedding

        loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
                   torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        #z_q, 'b h w c -> b c h w'
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)


class GumbelQuantize(nn.Module):
    """
    credit to @karpathy: https://github.com/karpathy/deep-vector-quantization/blob/main/model.py (thanks!)
    Gumbel Softmax trick quantizer
    Categorical Reparameterization with Gumbel-Softmax, Jang et al. 2016
    https://arxiv.org/abs/1611.01144
    """
    def __init__(self, num_hiddens,codebook_dim, embedding_dim, straight_through=True,
                 kl_weight=5e-4, temp_init=1.0, use_vqinterface=True, unknown_index="random"):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.codebook_dim = codebook_dim

        self.straight_through = straight_through
        self.temperature = temp_init
        self.kl_weight = kl_weight

        self.embed = nn.Embedding(codebook_dim, embedding_dim)

        self.use_vqinterface = use_vqinterface

    def forward(self, z, temp=None,return_logits=False):
        # force hard = True when we are in eval mode, as we must quantize. actually, always true seems to work
        hard = self.straight_through if self.training else True
        temp = self.temperature if temp is None else temp


        soft_one_hot = F.gumbel_softmax(z, tau=temp, dim=1, hard=hard)
        z_q = einsum('b n h w, n d -> b d h w', soft_one_hot, self.embed.weight)

        # + kl divergence to the prior loss
        qy = F.softmax(z, dim=1)
        diff = self.kl_weight * torch.sum(qy * torch.log(qy * self.codebook_dim + 1e-10), dim=1).mean()

        ind = soft_one_hot.argmax(dim=1)
        if self.use_vqinterface:
            if return_logits:
                return z_q, diff, (None, None, ind), z
            return z_q, diff, (None, None, ind)
        return z_q, diff, ind

