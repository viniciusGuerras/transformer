from model import Transformer
from tokenizer import Tokenizer
import pandas as pd
import numpy as np
import torch

#to-do 
"""
fix - special tokens and tokenizer saving and loading
add - kv cache and RoPe <- really important
add - 16 bit/8 bit processing
study - modern initialization methods
optimize - all of tat
Rotary Positional Embeddings (RoPE)
Relative Positional Encodings
Dynamic Attention Span
Load Balancing for MoE
Memory Augmentation / External Memory Modules
Efficient Transformer Variants
Multi-Task Learning or Auxiliary Objectives
Adaptive Computation Time (ACT)
Incorporating Pretrained Knowledge
"""

#hyperparameters
epochs = 2000
learning_rate = 1e-4
batch_size = 64
num_embeds = 512
special_tokens_size = 2
vocab_size = 300
n_layers = 6

"""
this can impact the multiplications because of division
sometimes the head size (head_size = n_embeds//n_heads) 
inteferes with the matrice size (because of float division)
"""

n_heads = 4
load_model = True    

context_window = 32
current_context_window = 8
context_window_step = 8

n_experts = 8
n_common_experts = 2
top_k = 2
#-----

toke = Tokenizer()

data = pd.read_parquet("datasets/train-00000-of-00001-090b52ccb189d47a.parquet")
data_array = np.array(data["text"])

data_array = toke.encode(data_array[:1000],vocab_size)

n = int(0.9 * float(len(data_array)))
train_data = torch.tensor(data_array[:n])
val_data = torch.tensor(data_array[n:])

vocab_size = vocab_size + special_tokens_size

def save_checkpoint(state, filename="checkpoints/model/checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

def get_batch(split, size):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - context_window, (batch_size,))
    x = torch.stack([data[i:i+size] for i in ix])
    y = torch.stack([data[i+1:i+size+1] for i in ix])
    return x, y

model = Transformer(vocab_size, n_layers, n_heads, num_embeds, context_window, n_experts, n_common_experts, top_k)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

if load_model:
    checkpoint = torch.load("checkpoints/model/checkpoint.pth.tar")
    load_checkpoint(checkpoint, model, optimizer)

# TRAINING LOOP 
for iter in range(epochs):

    xb, yb = get_batch('train', context_window)
    if current_context_window < context_window:
        current_context_window += context_window_step

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if iter % 100 == 0:
        print(iter)
    if iter%1000==0:
        checkpoint = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)
        print(f"epoch: {iter} and loss:{loss}")

context = torch.zeros((1, 1), dtype=torch.long)
print(toke.decode(model.generate(context, max_new_tokens=5000)[0].tolist(), load=True))
