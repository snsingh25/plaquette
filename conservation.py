import torch
import torch.nn as nn

# Parameters
p = 13
d_model = 64
d_head  = 32
epochs  = 6000

# Dataset
pairs  = torch.cartesian_prod(torch.arange(p), torch.arange(p))
labels = (pairs[:, 0] + pairs[:, 1]) % p
perm   = torch.randperm(len(pairs))
pairs, labels = pairs[perm], labels[perm]
split  = int(len(pairs) * 0.7)
train_pairs, train_labels = pairs[:split], labels[:split]


class AttentionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(p, d_model)
        self.W_Q = nn.Linear(d_model, d_head, bias=False)
        self.W_K = nn.Linear(d_model, d_head, bias=False)
        self.W_V = nn.Linear(d_model, d_head, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(d_head, d_head * 4),
            nn.ReLU(),
            nn.Linear(d_head * 4, d_head),
        )
        self.out = nn.Linear(d_head, p)

    def forward(self, x):
        e = self.embed(x)
        Q = self.W_Q(e)
        K = self.W_K(e)
        V = self.W_V(e)
        scores    = Q @ K.transpose(-2, -1) / d_head ** 0.5
        attention = torch.softmax(scores, dim=-1)
        mixed     = attention @ V
        mixed     = mixed + self.mlp(mixed)
        return self.out(mixed[:, 0, :])


def conserved_charge(model):
    # W_Q.weight is (d_head, d_model), so W @ W.T gives (d_head, d_head)
    WQ = model.W_Q.weight
    WK = model.W_K.weight
    charge = WQ @ WQ.T - WK @ WK.T
    return charge.norm().item()


def train(optimizer_name):
    torch.manual_seed(42)
    model   = AttentionHead()
    loss_fn = nn.CrossEntropyLoss()

    if optimizer_name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    elif optimizer_name == "SGD + weight decay":
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.01)
    elif optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

    charges = []
    for epoch in range(epochs):
        logits = model(train_pairs)
        loss   = loss_fn(logits, train_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            charges.append((epoch, conserved_charge(model)))

    return charges


# Run all three
results = {}
for name in ["SGD", "SGD + weight decay", "Adam"]:
    print(f"Training with {name}...")
    results[name] = train(name)

# Print comparison
print(f"\n{'epoch':>6}  {'SGD':>10}  {'SGD+wd':>10}  {'Adam':>10}")
print("-" * 42)
for i in range(len(results["SGD"])):
    epoch = results["SGD"][i][0]
    sgd   = results["SGD"][i][1]
    sgdwd = results["SGD + weight decay"][i][1]
    adam  = results["Adam"][i][1]
    if i % 4 == 0:  # print every 200 epochs
        print(f"{epoch:6d}  {sgd:10.4f}  {sgdwd:10.4f}  {adam:10.4f}")
