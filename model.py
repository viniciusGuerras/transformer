import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

#hyperparameters
head_size = 6
num_embeds = 512
dropout = 0.3
num_heads = 8
n_layers = 6
vocab_size = 500
context_window = 128

class HeadAttention(nn.Module):
    def __init__(self, n_embed, head_size):
        super().__init__()
        self.head_size = head_size
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.key = nn.Linear(n_embed, head_size, bias=False) 
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(context_window, context_window)))

    def forward(self, x):
        #the x will come from the model, it will be the input embedding + positional encoding
        #this will result in the matrice size (T, C) and we have various batches, so (B, T, C)
        #b batches, t (context length or time), c channels
        B, T, C = x.shape

        query = self.query(x) #(B, T, C)
        key = self.key(x) #(B, T, C)

        #transpose the dimensions of the key to become (B, C, T)
        #logo (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = query @ key.transpose(-2, -1) * C**-0.5

        #do the masking here and pass through the softmax
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)

        #lastly, we dot product the values after the pass
        value = self.value(x)
        ouput = wei @ value 
        return ouput

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, head_size):
        super().__init__()
        #create a bunch of attention heads
        self.heads = nn.ModuleList([HeadAttention(num_embeds, head_size) for _ in range(n_heads)])
        #make a linear projection for the values
        self.proj = nn.Linear(num_embeds, num_embeds)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

class FeedForward(nn.Module):
    def __init__(self, num_embeds):
        super().__init__()
        # basic ReLu with linear layers to increase dimensionality and pick important patterns
        self.net = nn.Sequential(
            nn.Linear(num_embeds, num_embeds * 4),
            nn.GELU(),
            nn.Linear(num_embeds * 4, num_embeds),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Expert(nn.Module):
    def __init__(self, num_embds):
        super().__init__()
        self.forward_layer = FeedForward(num_embds)
    def forward(self, x):
        return self.forward_layer(x)

class NoisytopkRouter(nn.Module):
    def __init__(self, n_embeds, n_experts, top_k):
        super().__init__()
        self.top_k = top_k
        self.linear = nn.Linear(n_embeds, n_experts)
        self.noise = nn.Linear(n_embeds, n_experts)

    def forward(self, x):
        logits = self.linear(x)

        noise_logits = self.noise(x)
        noise = torch.randn_like(logits)*F.softplus(noise_logits)

        noise_logits = logits + noise

        top_k_logits, indices = noise_logits.topk(self.top_k, dim=-1)
        zeroes =  torch.full_like(noise_logits, float('-inf'))
        sparse_logits = zeroes.scatter(-1, indices, top_k_logits)
        router_ouput = F.softmax(sparse_logits, dim=-1)
        return router_ouput, indices

class SparseMoE(nn.Module):
    def __init__(self, n_embed, num_experts, top_k):
        super().__init__()
        self.router = NoisytopkRouter(n_embed, num_experts, top_k)
        self.experts = nn.ModuleList([Expert(n_embed) for _ in range(num_experts)])
        self.top_k = top_k

    def forward(self, x):
        gating_output, indices = self.router(x)
        final_output = torch.zeros_like(x)

        # Reshape inputs for batch processing
        flat_x = x.view(-1, x.size(-1))
        flat_gating_output = gating_output.view(-1, gating_output.size(-1))

        # Process each expert in parallel
        for i, expert in enumerate(self.experts):
            # Create a mask for the inputs where the current expert is in top-k
            expert_mask = (indices == i).any(dim=-1)
            flat_mask = expert_mask.view(-1)

            if flat_mask.any():
                expert_input = flat_x[flat_mask]
                expert_output = expert(expert_input)

                # Extract and apply gating scores
                gating_scores = flat_gating_output[flat_mask, i].unsqueeze(1)
                weighted_output = expert_output * gating_scores

                # Update final output additively by indexing and adding
                final_output[expert_mask] += weighted_output.squeeze(1)

        return final_output
    
class Block(nn.Module):
    def __init__(self, n_embeds, n_heads, n_experts, top_k):
        super().__init__()
        head_size = n_embeds//n_heads
        # divide the attention in differents part of the embedding (to pick various types of connections)
        # communicate
        self.sa = MultiHeadAttention(n_heads, head_size)
        # computate
        self.moe = SparseMoE(n_embeds, n_experts, top_k)
        self.ln1 = nn.LayerNorm(n_embeds)
        self.ln2 = nn.LayerNorm(n_embeds)

    def forward(self, x):
        # residual connections
        x = x + self.ln1(self.sa(x))
        x = x + self.ln2(self.moe(x))
        return x

class Transformer(nn.Module):
    def __init__(self, num_embeds, context_window, n_experts, top_k):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, num_embeds)
        self.pos_embedding_table = nn.Embedding(context_window, num_embeds)
        self.blocks = nn.Sequential(*[Block(num_embeds, num_heads, n_experts, top_k) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(num_embeds)
        self.lm_head = nn.Linear(num_embeds, vocab_size)

    def forward(self, ids, targets=None):
        B, T = ids.shape
        toke_emb = self.token_embedding_table(ids)
        #i need to change to sinusoidal later
        pos_emb = self.pos_embedding_table(torch.arange(T))
        sum = toke_emb + pos_emb
        sum = self.blocks(sum)
        sum = self.ln_f(sum)
        logits = self.lm_head(sum)

        if targets == None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss


    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -context_window:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
