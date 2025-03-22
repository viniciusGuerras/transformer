import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from cnn_tokenizer import CnnTokenizer

dropout = 0.2

class HeadAttention(nn.Module):
    def __init__(self, n_embed, head_size, block_size):
        super().__init__()
        self.head_size = head_size
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.key = nn.Linear(n_embed, head_size, bias=False) 
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

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
    def __init__(self, n_embds, n_heads, head_size, block_size):
        super().__init__()
        #create a bunch of attention heads
        self.heads = nn.ModuleList([HeadAttention(n_embds, head_size, block_size) for _ in range(n_heads)])
        #make a linear projection for the values
        self.proj = nn.Linear(n_embds, n_embds)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

class FeedForward(nn.Module):
    def __init__(self, n_embds):
        super().__init__()
        # basic ReLu with linear layers to increase dimensionality and pick important patterns
        self.net = nn.Sequential(
            nn.Linear(n_embds, n_embds * 4),
            nn.GELU(),
            nn.Linear(n_embds * 4, n_embds),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

# a simple expert, whic is a feed forward layer
class Expert(nn.Module):
    def __init__(self, n_embds):
        super().__init__()
        self.forward_layer = FeedForward(n_embds)
    def forward(self, x):
        return self.forward_layer(x)

class NoisytopkRouter(nn.Module):
    def __init__(self, n_embds, n_experts, top_k):
        """
            top_k: number of experts selected
            linear: layer for the forward pass 
            noise: layer for the noise process
        """
        super().__init__()
        self.top_k = top_k
        self.linear = nn.Linear(n_embds, n_experts)
        self.noise = nn.Linear(n_embds, n_experts)

    def forward(self, x):
        # x = (B, T, C)  # Input shape
        # B = Batch size
        # T = Sequence length (number of time steps or tokens)
        # C = Number of channels (features, or embedding size)

        # Pass the input through the linear layers to produce logits
        logits = self.linear(x)  # (B, T, n_experts)
        # After applying self.linear(x), the shape is (B, T, n_experts), where:
        # B = Batch size
        # T = Sequence length
        # n_experts = Number of experts
        
        noise_logits = self.noise(x)  # (B, T, n_experts)
        # After applying self.noise(x), the shape is (B, T, n_experts), same as logits

        # Use the standard deviation, multiplied by the softplus of the noise logits to create the noise
        noise = torch.randn_like(logits) * F.softplus(noise_logits)  # (B, T, n_experts)
        # torch.randn_like(logits) produces a tensor with the same shape as logits filled with random values.
        # F.softplus(noise_logits) applies the softplus function element-wise to the noise logits.
        # This will result in the same shape as logits, which is (B, T, n_experts).
        
        # Add noise to logits
        noise_logits = logits + noise  # (B, T, n_experts)
        # The resulting shape is (B, T, n_experts), adding noise to logits element-wise.

        # Using the top-k function to get the top-k largest values and their indices
        top_k_logits, indices = noise_logits.topk(self.top_k, dim=-1)  # (B, T, top_k), (B, T, top_k)
        # top_k_logits will have the top-k values for each input in the batch, shape: (B, T, top_k)
        # indices will hold the indices of those top-k values, shape: (B, T, top_k)
        # top_k: number of experts to select.

        # Create a tensor of -inf values, with the same shape as noise_logits
        zeroes = torch.full_like(noise_logits, float('-inf'))  # (B, T, n_experts)
        # This will create a tensor with the same shape as noise_logits (B, T, n_experts), filled with -inf.

        # Add the top-k values to the -inf matrix at the corresponding positions
        sparse_logits = zeroes.scatter(-1, indices, top_k_logits)  # (B, T, n_experts)
        # zeroes.scatter(-1, indices, top_k_logits) will scatter the top_k_logits into the zeroes tensor
        # at the positions indicated by the indices. This will result in a sparse tensor where only the top-k
        # positions contain values and the rest are -inf.

        # Use the softmax function to zero out the -inf and retain the values that are non -inf
        router_output = F.softmax(sparse_logits, dim=-1)  # (B, T, n_experts)
        # The softmax function will compute probabilities across the last dimension (experts) by scaling the logits
        # such that their sum equals 1 for each (B, T) pair, and the -inf values will be turned into zeroes.

        # Return the final output and the indices of selected experts
        return router_output, indices  # (B, T, n_experts), (B, T, top_k)

class SparseMoE(nn.Module):
    def __init__(self, n_embed, num_experts, top_k):
        super().__init__()
        self.router = NoisytopkRouter(n_embed, num_experts, top_k)
        self.experts = nn.ModuleList([Expert(n_embed) for _ in range(num_experts)])
        self.top_k = top_k

    def forward(self, x):
        # Get the output of the router and the indices of the top-k experts for each input
        gating_output, indices = self.router(x)  # gating_output: (B, T, n_experts), indices: (B, T, top_k)
        
        # Initialize the final output tensor with zeros, having the same shape as the input
        final_output = torch.zeros_like(x)  # final_output: (B, T, C)
        
        # Reshape inputs and gating outputs for efficient batch processing (leaves only the last dimension)
        flat_x = x.view(-1, x.size(-1))  # flat_x: (B*T, C)
        flat_gating_output = gating_output.view(-1, gating_output.size(-1))  # flat_gating_output: (B*T, n_experts)

        # Loop over each expert to process in parallel
        for i, expert in enumerate(self.experts):
            # Create a mask for inputs where the current expert is in the top-k (based on the indices)
            expert_mask = (indices == i).any(dim=-1)  # expert_mask: (B, T), indicates which inputs are assigned to expert i
            flat_mask = expert_mask.view(-1)  # flat_mask: (B*T), flattened mask for batch processing

            # If there are any inputs assigned to the current expert (i.e., expert_mask has any True values)
            if flat_mask.any():
                # Select the input data that corresponds to the current expert using the mask
                expert_input = flat_x[flat_mask]  # expert_input: (N, C) where N is the number of inputs assigned to expert i
                # Pass the expert input through the expert (e.g., a neural network layer)
                expert_output = expert(expert_input)  # expert_output: (N, output_size) where output_size is the expert output size

                # Extract the gating scores for the current expert and apply them to the expert's output
                gating_scores = flat_gating_output[flat_mask, i].unsqueeze(1)  # gating_scores: (N, 1)
                weighted_output = expert_output * gating_scores  # weighted_output: (N, output_size)

                # Add the weighted output to the final output tensor at the positions indicated by the expert mask
                final_output[expert_mask] += weighted_output.squeeze(1)  # final_output: (B, T, C)

        # Return the final output after processing all experts and applying gating
        return final_output  # final_output: (B, T, C)
    
class Block(nn.Module):
    def __init__(self, n_embds, n_heads, n_experts, top_k, block_size):
        super().__init__()
        head_size = n_embds//n_heads
        # divide the attention in differents part of the embedding (to pick various types of connections)
        # communicate
        self.sa = MultiHeadAttention(n_embds, n_heads, head_size, block_size)
        # computate
        self.moe = SparseMoE(n_embds, n_experts, top_k)
        self.ln1 = nn.LayerNorm(n_embds)
        self.ln2 = nn.LayerNorm(n_embds)

    def forward(self, x):
        # residual connections
        x = x + self.ln1(self.sa(x))
        x = x + self.ln2(self.moe(x))
        return x

# i still need to use this in the attention
class KVCache:
    def __init__(self, max_cache_size):
        self.max_cache_size = max_cache_size
        self.k_cache = []
        self.v_cache = []

    def update(self, key, value):
        if len(self.k_cache) >= self.max_cache_size:
            self.k_cache.pop(0)
            self.v_cache.pop(0)

        self.k_cache.append(key)
        self.v_cache.append(value)

    def get_cache(self):
        return torch.stack(self.k_cache), torch.stack(self.v_cache)

class Transformer(nn.Module):
    def __init__(self, vocab_size, n_layers, n_heads, n_embds, block_size, n_experts, top_k):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embds)
        self.pos_embedding_table = nn.Embedding(block_size, n_embds)
        self.blocks = nn.Sequential(*[Block(n_embds, n_heads, n_experts, top_k, block_size) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embds)
        self.lm_head = nn.Linear(n_embds, vocab_size)

    def forward(self, ids, targets=None):
        B, T = ids.shape
        toke_emb = self.token_embedding_table(ids)
        #i need to change to rotatory later
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
            idx_cond = idx[:, -self.block_size:]
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
