import torch
import torch.nn as nn

# Parameters
p = 13
d_model = 64
d_head  = 32

# Dataset
pairs  = torch.cartesian_prod(torch.arange(p), torch.arange(p))
labels = (pairs[:, 0] + pairs[:, 1]) % p
perm   = torch.randperm(len(pairs))
pairs, labels = pairs[perm], labels[perm]
split  = int(len(pairs) * 0.7)
train_pairs, train_labels = pairs[:split], labels[:split]
test_pairs,  test_labels  = pairs[split:], labels[split:]


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

        mixed = mixed + self.mlp(mixed)

        return self.out(mixed[:, 0, :])


# Training
model = AttentionHead()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.3)
loss_fn   = nn.CrossEntropyLoss()

for epoch in range(6000):
    logits = model(train_pairs)
    loss   = loss_fn(logits, train_labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        with torch.no_grad():
            test_logits = model(test_pairs)
            test_pred   = test_logits.argmax(dim=-1)
            accuracy    = (test_pred == test_labels).float().mean()
        print(f"epoch {epoch:4d}  loss {loss.item():.4f}  test acc {accuracy.item():.2%}")
