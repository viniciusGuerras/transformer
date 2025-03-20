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
 
class HeadAttention(nn.Module):
    def __init__(self, n_embed, head_size):
        self.head_size = head_size
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.key = nn.Linear(n_embed, head_size, bias=False) 
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_module("tril", torch.tril(torch.zeros(context_window, context_window)))

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
        #do the masking here
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        value = self,value(x)
        ouput = wei @ value 

class MutiHeadAttention(nn.Module):
    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([HeadAttention(head_size) for _ in range(n_heads)])
        self.proj = nn.Linear(num_embeds, num_embeds)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)