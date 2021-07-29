import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    def __init__(self, num_tokens, codebook_dim, beta):
        super().__init__()
        self.codebook_dim = codebook_dim
        self.num_tokens = num_tokens
        self.beta = beta

        self.embedding = nn.Embedding(self.num_tokens, self.codebook_dim)
        self.embedding.weight.data.normal_()

    def forward(self, z):
        # reshape z -> (batch, height, width, channel) and flatten
        #z, 'b c h w -> b h w c'
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.codebook_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened.pow(2), dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight.pow(2), dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, self.embedding.weight.permute(1,0)) # 'n d -> d n'

        encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(encoding_indices).view(z.shape)
        encodings = F.one_hot(encoding_indices, self.num_tokens).type(z.dtype)       
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # compute loss for embedding

        loss = self.beta * F.mse_loss(z_q.detach(), z) + F.mse_loss(z_q, z.detach())

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        #z_q, 'b h w c -> b c h w'
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q, loss, (perplexity, encodings, encoding_indices)


class EmbeddingEMA(nn.Module):
    def __init__(self, num_tokens, codebook_dim):
        super().__init__()
        weight = torch.randn(codebook_dim, num_tokens)
        self.register_buffer("weight", weight)
        self.register_buffer("cluster_size", torch.zeros(num_tokens))
        self.register_buffer("embed_avg", weight.clone())

    def forward(self, embed_id):
        return F.embedding(embed_id, self.weight.transpose(0, 1))

class EMAVectorQuantizer(nn.Module):
    def __init__(self, num_tokens, codebook_dim, beta, decay=0.99, eps=1e-5):
        super().__init__()
        self.codebook_dim = codebook_dim
        self.num_tokens = num_tokens
        self.decay = decay
        self.eps = eps
        self.beta = beta
        self.embedding = EmbeddingEMA(num_tokens,codebook_dim)

    def forward(self, z):
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.reshape(-1, self.codebook_dim)
        d = (
            z_flattened.pow(2).sum(1, keepdim=True)
            - 2 * z_flattened @ self.embedding.weight
            + self.embedding.weight.pow(2).sum(0, keepdim=True)
        )
        _, encoding_indices = (-d).max(1)
        encodings = F.one_hot(encoding_indices, self.num_tokens).type(z_flattened.dtype)
        encoding_indices = encoding_indices.view(*z.shape[:-1])
        z_q = self.embedding(encoding_indices)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        if self.training:
            encodings_sum = encodings.sum(0)
            embed_sum = z_flattened.transpose(0, 1) @ encodings
            #EMA cluster size
            self.embedding.cluster_size.data.mul_(self.decay).add_(encodings_sum, alpha=1 - self.decay)
            #EMA embedding average
            self.embedding.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

            #cluster size Laplace smoothing 
            n = self.embedding.cluster_size.sum()
            cluster_size = (
                (self.embedding.cluster_size + self.eps) / (n + self.num_tokens * self.eps) * n
            )
            #normalize embedding average with smoothed cluster size
            embed_normalized = self.embedding.embed_avg / cluster_size.unsqueeze(0)
            self.embedding.weight.data.copy_(embed_normalized)

        loss = self.beta * (z_q.detach() - z).pow(2).mean()
        z_q = z + (z_q - z).detach()
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q, loss, (perplexity, encodings, encoding_indices)



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

        encoding_indices = soft_one_hot.argmax(dim=1)
        encodings = F.one_hot(encoding_indices, self.num_tokens).type(z.dtype)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return z_q, loss, (perplexity, encodings, encoding_indices)

