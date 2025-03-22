from model import Transformer
from tokenizer import Tokenizer
import pandas as pd
import numpy as np
import torch

#to-do 
"""
fix - special tokens and tokenizer saving and loading
add - kv cache and RoPe
optimize - all of tat
"""


#hyperparameters
epochs = 2000
learning_rate = 1e-4
batch_size = 64
num_embeds = 512
vocab_size = 255
n_layers = 12

"""
this can impact the multiplications because of division
sometimes the head size (head_size = n_embeds//n_heads) 
inteferes with the matrice size (because of float division)
"""

n_heads = 8
load_model = False    

context_window = 
current_context_window = 8
context_window_step = 8

n_experts = 8
top_k = 2
#-----

toke = Tokenizer()

# data loading ad pre-processing
data = pd.read_csv("datasets/obras_machado_de_assis.csv")
data_array = np.array(data["texto"])

toke = Tokenizer()

data_array = toke.encode(data_array,vocab_size)

n = int(0.9 * float(len(data_array)))
train_data = torch.tensor(data_array[:n])
val_data = torch.tensor(data_array[n:])


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


model = Transformer(vocab_size, n_layers, n_heads, num_embeds, context_window, n_experts, top_k)

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