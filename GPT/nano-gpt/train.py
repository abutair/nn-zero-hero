import torch
import torch.nn as nn
from torch.nn import functional 


## Declear hyperparameters

batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
# ------------


with open('C:/Users/abutair/workspace/nn-zero-to-hero/GPT/nano-gpt/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}

encode = lambda s :[stoi[c] for c in s]
decode = lambda l:''.join([itos[i] for i in l])


data = torch.tensor(encode(text),dtype=torch.long)

n= int(0.9 * len(data))
train_data = data[:n]
val_data= data[n:]


def get_batch(split):
    if split=="train":
        data= train_data
    else:
        data = val_data

    ix= torch.randint(len(data)-block_size,(batch_size))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y= torch.stack([data[i+1:i+block_size+1] for i in ix ])
    x,y = x.to(device),y.to(device)
    return  x,y 


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

