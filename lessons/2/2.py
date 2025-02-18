import torch
import torch.nn as nn
from torch.nn import functional as F


block_size = 10
batch_size = 4
n_embd = 10
device = "mps"

with open("1984.txt", "r", encoding="utf-8") as file:
    text = file.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

def encode(word):
    return [chars.index(char) for char in word]


def decode(vector):
    return "".join(chars[i] for i in vector)


data = torch.tensor(encode(text))

split_index = int(0.8 * len(data))
training_data = data[:split_index]
validation_data = data[split_index:]


def get_batch(action):
    data = training_data if action == "training" else validation_data
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


class GPTLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, n_embd)
        self.positions_embeddings = nn.Embedding(block_size, n_embd)
        # self.blocks = 
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, index):
        B, T = index.shape

        tok_embed = self.token_embeddings(index)
        pos_embed = self.token_embeddings(torch.arange(T, device=device))
        x = tok_embed + pos_embed
        # x = self.blocks
        x = self.ln_f(x)
        logits = self.lm_head(x)

        # here will be the loss calculation part

        return logits


    def generate(self, index, max_new_tokens):
        for _ in range(max_new_tokens):
            index_cond = index[:, -block_size:]
            logits = self.forward(index_cond)
            logits = logits[:, -1, :]

            probs = F.softmax(logits, dim=1)
            index_next = torch.multinomial(probs, num_samples=1)

            index = torch.cat((index, index_next), dim=1)
        
        return index


model = GPTLanguageModel(vocab_size)
m = model.to(device)
print(encode("Hello! How are you?"))
prompt = "Hello! How are you?"
context = torch.tensor(encode(prompt), dtype=torch.long, device=device)
print(decode(m.generate(context.unsqueeze(0), max_new_tokens=100)[0].tolist()))