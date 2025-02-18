import torch


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


