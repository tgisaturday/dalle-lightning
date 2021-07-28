import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    def __init__(self, num_tokens, codebook_dim, beta):
        super().__init__()
        self.codebook_dim = codebook_dim
        self.num_tokens = num_tokens
        self.beta = beta
        embed_init = torch.randn(self.num_tokens, self.codebook_dim)
        self.embedding = nn.Embedding(self.num_tokens, self.codebook_dim)
        self.embedding.weight.data.copy_(embed_init.clone())

    def forward(self, z):
        # reshape z -> (batch, height, width, channel) and flatten
        #z, 'b c h w -> b h w c'
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.codebook_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, self.embedding.weight.permute(1,0)) # 'n d -> d n'

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        perplexity = None
        min_encodings = None

        # compute loss for embedding

        loss = self.beta * torch.mean((z_q.detach()-z)**2) + torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        #z_q, 'b h w c -> b c h w'
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)


class EMAVectorQuantizer(nn.Module):
    def __init__(self, num_tokens, codebook_dim, beta, decay=0.99, eps=1e-5):
        super().__init__()
        self.codebook_dim = codebook_dim
        self.num_tokens = num_tokens
        self.beta = beta
        embed_init = torch.randn(self.num_tokens, self.codebook_dim)
        self.embedding = nn.Embedding(self.num_tokens, self.codebook_dim)
        self.embedding.weight.data.copy_(embed_init.clone())
        self.cluster_size = nn.Parameter(torch.zeros(self.num_tokens))
        self.ema_w = nn.Parameter(embed_init.clone())
        self.decay = decay
        self.eps = eps

    def forward(self, z):
        # reshape z -> (batch, height, width, channel) and flatten
        #z, 'b c h w -> b h w c'
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.codebook_dim)
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
            encodings_onehot = F.one_hot(min_encoding_indices, self.num_tokens).type(z.dtype)
            #EMA cluster size
            self.cluster_size.data.mul_(self.decay).add_(torch.sum(encodings_onehot, 0), alpha=1 - self.decay)

            embedding_sum = torch.matmul(encodings_onehot.t(), z_flattened)
            #EMA embedding weight
            self.ema_w.data.mul_(self.decay).add_(embedding_sum, alpha=1 - self.decay)   

            # Laplace smoothing of the cluster size
            self.cluster_size.data.add_(self.eps).div_(torch.sum(self.cluster_size) + self.num_tokens * self.eps)

            embedding_normalized = self.ema_w / self.cluster_size.unsqueeze(1)
            #normalize embedding weight EMA and update current embedding weight
            self.embedding.weight.data.copy_(embedding_normalized)
        
        # compute loss for embedding
        loss = self.beta * torch.mean((z_q.detach()-z)**2)
        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        #z_q, 'b h w c -> b c h w'
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)

class GumbelQuantizer(nn.Module):
    def __init__(self, num_tokens, codebook_dim, straight_through=True,
                 kl_weight=5e-4, temp_init=1.0):
        super().__init__()

        self.codebook_dim = codebook_dim
        self.num_tokens = num_tokens

        self.straight_through = straight_through
        self.temperature = temp_init
        self.kl_weight = kl_weight

        self.embedding = nn.Embedding(num_tokens, codebook_dim)

    def forward(self, z):
        # force hard = True when we are in eval mode, as we must quantize. actually, always true seems to work
        hard = self.straight_through if self.training else True
        temp = self.temperature 

        soft_one_hot = F.gumbel_softmax(z, tau=temp, dim=1, hard=hard)
        z_q = torch.einsum('b n h w, n d -> b d h w', soft_one_hot, self.embedding.weight)

        # + kl divergence to the prior loss
        qy = F.softmax(z, dim=1)
        loss = self.kl_weight * torch.sum(qy * torch.log(qy * self.num_tokens + 1e-10), dim=1).mean()

        min_encoding_indices = soft_one_hot.argmax(dim=1)

        return z_q, loss, (None, None, min_encoding_indices)

