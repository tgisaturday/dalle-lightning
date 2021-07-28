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
        perplexity = None
        encodings = None

        # compute loss for embedding

        loss = self.beta * F.mse_loss(z_q.detach()-z) + F.mse_loss(z_q - z.detach())

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        #z_q, 'b h w c -> b c h w'
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q, loss, (perplexity, encodings, encoding_indices)


class EMAVectorQuantizer(nn.Module):
    def __init__(self, num_tokens, codebook_dim, beta, decay=0.99, eps=1e-5):
        super().__init__()
        self.codebook_dim = codebook_dim
        self.num_tokens = num_tokens
        self.beta = beta

        self.embedding = nn.Embedding(self.num_tokens, self.codebook_dim)
        self.embedding.weight.data.normal_()
        self.register_buffer("cluster_size", torch.zeros(num_tokens))
        self.ema_w = nn.Parameter(torch.Tensor(self.num_tokens, self.codebook_dim))
        self.ema_w.data.normal_()
        self.decay = decay
        self.eps = eps


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
        perplexity = None
        
        encodings = F.one_hot(encoding_indices, self.num_tokens).type(z.dtype)

        # Use EMA to update the embedding vectors
        if self.training:
            #EMA cluster size
            new_cluster_size = torch.sum(encodings, 0)
            self.cluster_size.data.mul_(self.decay).add_(new_cluster_size, alpha=1 - self.decay)
            
            #EMA embedding weight
            new_ema_w = torch.matmul(encodings.t(), z_flattened)
            self.ema_w.data.mul_(self.decay).add_(new_ema_w, alpha=1 - self.decay)   

            # Laplace smoothing of the cluster size
            cluster_size_sum = torch.sum(self.cluster_size.data)
            self.cluster_size.data.add_(self.eps).div_(cluster_size_sum + self.num_tokens * self.eps)


            #normalize embedding weight EMA and update current embedding weight
            self.embedding.weight = nn.Parameter(self.ema_w / self.cluster_size.unsqueeze(1))
        
        # compute loss for embedding
        loss = self.beta * F.mse_loss(z_q.detach(), z)
        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        #z_q, 'b h w c -> b c h w'
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

        return z_q, loss, (None, None, encoding_indices)

