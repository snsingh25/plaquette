import torch
import torch.nn as nn
import matplotlib.pyplot as plt

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
        mixed     = mixed + self.mlp(mixed)
        return self.out(mixed[:, 0, :])


def conserved_charge(model):
    WQ = model.W_Q.weight
    WK = model.W_K.weight
    return (WQ @ WQ.T - WK @ WK.T).norm().item()


def train(optimizer_name, record_grokking=False):
    torch.manual_seed(42)
    model   = AttentionHead()
    loss_fn = nn.CrossEntropyLoss()

    if optimizer_name == "SGD":
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
    elif optimizer_name == "SGD + weight decay":
        opt = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.01)
    elif optimizer_name == "Adam":
        opt = torch.optim.Adam(model.parameters(), lr=0.003)
    elif optimizer_name == "AdamW (grokking)":
        opt = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.3)

    charges, losses, accuracies = [], [], []
    for epoch in range(epochs):
        logits = model(train_pairs)
        loss   = loss_fn(logits, train_labels)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if epoch % 20 == 0:
            charges.append((epoch, conserved_charge(model)))
            if record_grokking:
                losses.append((epoch, loss.item()))
                with torch.no_grad():
                    pred = model(test_pairs).argmax(dim=-1)
                    acc  = (pred == test_labels).float().mean().item()
                accuracies.append((epoch, acc))

    if record_grokking:
        return charges, losses, accuracies
    return charges


# ── Plot 1: Grokking ────────────────────────────────────────
print("Training grokking model...")
_, losses, accuracies = train("AdamW (grokking)", record_grokking=True)

fig, ax1 = plt.subplots(figsize=(10, 5))
ax2 = ax1.twinx()

ep_l = [x[0] for x in losses]
ep_a = [x[0] for x in accuracies]

ax1.plot(ep_l, [x[1] for x in losses], color="steelblue", linewidth=1.5, label="Train loss")
ax2.plot(ep_a, [x[1] for x in accuracies], color="tomato", linewidth=1.5, label="Test accuracy")

ax1.set_xlabel("Epoch", fontsize=12)
ax1.set_ylabel("Train loss", color="steelblue", fontsize=12)
ax2.set_ylabel("Test accuracy", color="tomato", fontsize=12)
ax1.set_title("Grokking: memorisation then generalisation", fontsize=14)
ax1.axvline(x=200, color="gray", linestyle="--", alpha=0.5, label="Memorised (~200)")
ax1.axvline(x=5100, color="gray", linestyle=":", alpha=0.5, label="Generalised (~5100)")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=10)

plt.tight_layout()
plt.savefig("docs/plot_grokking.png", dpi=150)
plt.close()
print("  Saved docs/plot_grokking.png")


# ── Plot 2: Conserved charge ────────────────────────────────
print("Training conservation models...")
results = {}
for name in ["SGD", "SGD + weight decay", "Adam"]:
    print(f"  {name}...")
    results[name] = train(name)

fig, ax = plt.subplots(figsize=(10, 5))
colors = {"SGD": "steelblue", "SGD + weight decay": "seagreen", "Adam": "tomato"}
for name, charges in results.items():
    ep = [c[0] for c in charges]
    ch = [c[1] for c in charges]
    ax.plot(ep, ch, color=colors[name], linewidth=1.5, label=name)

ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("||W_Q^T W_Q  -  W_K^T W_K||", fontsize=12)
ax.set_title("Conserved charge under different optimisers", fontsize=14)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig("docs/plot_conservation.png", dpi=150)
plt.close()
print("  Saved docs/plot_conservation.png")


# ── Plot 3: Flat directions ────────────────────────────────
print("Computing flat directions...")
d_model_small, d_head_small = 8, 4
torch.manual_seed(42)

embed = nn.Embedding(p, d_model_small)
WQ = nn.Linear(d_model_small, d_head_small, bias=False)
WK = nn.Linear(d_model_small, d_head_small, bias=False)

all_pairs = torch.cartesian_prod(torch.arange(p), torch.arange(p))
rows = []
for i in range(len(all_pairs)):
    x = all_pairs[i].unsqueeze(0)
    e = embed(x)
    Q, K = WQ(e), WK(e)
    scores = Q @ K.transpose(-2, -1) / d_head_small ** 0.5
    attn = torch.softmax(scores, dim=-1).flatten()
    for j in range(len(attn)):
        WQ.weight.grad = None
        WK.weight.grad = None
        attn[j].backward(retain_graph=True)
        gq = WQ.weight.grad.flatten().clone()
        gk = WK.weight.grad.flatten().clone()
        rows.append(torch.cat([gq, gk]))

J = torch.stack(rows)
S = torch.linalg.svdvals(J)
threshold = S.max() * 1e-5

fig, ax = plt.subplots(figsize=(10, 5))
colors_sv = ["tomato" if s < threshold else "steelblue" for s in S]
ax.bar(range(len(S)), S.detach().numpy(), color=colors_sv, width=0.8)
ax.axhline(y=threshold, color="gray", linestyle="--", alpha=0.6, label=f"Threshold ({threshold:.2e})")
ax.set_xlabel("Singular value index", fontsize=12)
ax.set_ylabel("Singular value", fontsize=12)
ax.set_title(f"Jacobian singular values — {(S < threshold).sum().item()} flat directions (predicted: {d_head_small**2})", fontsize=14)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig("docs/plot_flat_directions.png", dpi=150)
plt.close()
print("  Saved docs/plot_flat_directions.png")

print("\nAll plots saved to docs/")
