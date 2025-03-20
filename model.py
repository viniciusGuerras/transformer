import pandas as pd
import numpy as np
import torch
import regex as re
from torch import nn
from torch.nn import functional as F
#hyperparameters
context_window = 8
head_size = 6
num_heads = 6
num_embeds = 512
dropout = 0.1
vocab_size = 280
n_layers = 6
epochs = 5000
learning_rate = 1e-3
#-----

# data loading and pre-processing
data = pd.read_csv("datasets/Game_of_Thrones_Script.csv")
data_array = np.array(data.iloc[:, 5])

n = int(0.9 * float(data_array.shape[0]))

data_array = ["<START> " + str(phrase) + " <END>" for phrase in data_array]

"""
pattern = re.compile("'s|'t|'re|'ve|'m|'ll|'d|(?:\s{2,}(?=\S))| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+|\$+")
res = re.compile(pattern)
"""

print(data_array)

train_data = data_array[:n]
val_data = data_array[n:]

def get_batch(ids):
    pass
 
class HeadAttention(nn.Module):
    def __init__(self, n_embed, head_size):
        super().__init__()
        self.head_size = head_size
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.key = nn.Linear(n_embed, head_size, bias=False) 
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(context_window, context_window)))

    def forward(self, x):
        #the x will come from the model, it will be the input embedding + positiona encoding
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
        # basic ReLu with linear layers to increase dimensionality and pick important patterns
        self.net = nn.Sequential(
            nn.Linear(num_embeds, num_embeds * 4),
            nn.ReLU(),
            nn.Linear(num_embeds * 4, num_embeds),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embeds, n_heads):
        super().__init__()
        head_size = n_embeds//n_heads
        # divide the attention in differents part of the embedding (to pick various types of connections)
        # communicate
        self.sa = MultiHeadAttention(n_heads, head_size)
        # computate
        self.ffwd = FeedForward(n_embeds)
        self.ln1 = nn.LayerNorm(n_embeds)
        self.ln2 = nn.LayerNorm(n_embeds)

    def forward(self, x):
        # residual connections
        x = x + self.ln1(self.sa(x))
        x = x + self.ln2(self.ffwd(x))
        return x

class Transformer(nn.Module):
    def __init__(self):
        super().__init__(self, num_embeds, context_window)
        self.token_embedding_table = nn.Embedding(vocab_size, num_embeds)
        self.pos_embedding_table = nn.Embedding(context_window, num_embeds)
        self.blocks = nn.Sequential(*[Block(num_embeds, num_heads) for _ in range(n_layers)])
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



"""
model = Transformer()

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(epochs):

    # sample a batch of data
    xb, yb = get_batch('train')

    if epochs%100 == 0:
        print(loss)

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

"""