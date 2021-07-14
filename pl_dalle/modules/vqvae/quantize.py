import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    def __init__(self, codebook_dim, embedding_dim, beta):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.codebook_dim = codebook_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.codebook_dim, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.codebook_dim, 1.0 / self.codebook_dim)

    def forward(self, z):
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


class EMAVectorQuantizer(nn.Module):
    def __init__(self, codebook_dim, embedding_dim, beta, decay=0.99, eps=1e-5):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.codebook_dim = codebook_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.codebook_dim, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.codebook_dim, 1.0 / self.codebook_dim)

        self.embedding.weight.data.uniform_(-1.0 / self.codebook_dim, 1.0 / self.codebook_dim)
        self.register_buffer('cluster_size', torch.zeros(self.codebook_dim))
        self.ema_w = nn.Parameter(torch.Tensor(self.codebook_dim, self.embedding_dim))
        self.ema_w.data.uniform_(-1.0 / self.codebook_dim, 1.0 / self.codebook_dim)
        self.decay = decay
        self.eps = eps

    def forward(self, z):
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
        # Use EMA to update the embedding vectors
        if self.training:
            encoding_indices = min_encoding_indices.unsqueeze(1)
            encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=z.device)
            encodings.scatter_(1, encoding_indices, 1)
            self.cluster_size = self.cluster_size * self.decay + \
                                     (1 - self.decay) * torch.sum(encodings, 0)
            
            # Laplace smoothing of the cluster size
            n = torch.sum(self.cluster_size.data)
            self.cluster_size = (
                (self.cluster_size + self.eps)
                / (n + self.codebook_dim * self.eps) * n)
            
            dw = torch.matmul(encodings.t(), z_flattened)
            self.ema_w = nn.Parameter(self.ema_w * self.decay + (1 - self.decay) * dw)
            
            self.embedding.weight = nn.Parameter(self.ema_w / self.cluster_size.unsqueeze(1))
        # compute loss for embedding

        loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
                   torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        #z_q, 'b h w c -> b c h w'
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)

class GumbelQuantizer(nn.Module):
    def __init__(self, codebook_dim, embedding_dim, straight_through=True,
                 kl_weight=5e-4, temp_init=1.0, use_vqinterface=True):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.codebook_dim = codebook_dim

        self.straight_through = straight_through
        self.temperature = temp_init
        self.kl_weight = kl_weight

        self.embed = nn.Embedding(codebook_dim, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.codebook_dim, 1.0 / self.codebook_dim)

        self.use_vqinterface = use_vqinterface

    def forward(self, z, return_logits=False):
        # force hard = True when we are in eval mode, as we must quantize. actually, always true seems to work
        hard = self.straight_through if self.training else True
        temp = self.temperature 


        soft_one_hot = F.gumbel_softmax(z, tau=temp, dim=1, hard=hard)
        z_q = torch.einsum('b n h w, n d -> b d h w', soft_one_hot, self.embed.weight)

        # + kl divergence to the prior loss
        qy = F.softmax(z, dim=1)
        diff = self.kl_weight * torch.sum(qy * torch.log(qy * self.codebook_dim + 1e-10), dim=1).mean()

        ind = soft_one_hot.argmax(dim=1)
        if self.use_vqinterface:
            if return_logits:
                return z_q, diff, (None, None, ind), z
            return z_q, diff, (None, None, ind)
        return z_q, diff, ind

