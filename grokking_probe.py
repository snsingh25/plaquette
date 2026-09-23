import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

p = 13
d_model = 64
d_head  = 32
epochs  = 6000
snapshot_every = 20

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


# Gauge-invariant probes
def eigenvalues_of_QK(model):
    WQ = model.W_Q.weight  # (d_head, d_model)
    WK = model.W_K.weight
    QK = WQ @ WK.T  # (d_head, d_head) — gauge invariant
    eigs = torch.linalg.eigvalsh(QK @ QK.T)  # real eigenvalues, sorted
    return eigs.detach().numpy()

def attention_entropy(model):
    with torch.no_grad():
        e = model.embed(train_pairs)
        Q = model.W_Q(e)
        K = model.W_K(e)
        scores = Q @ K.transpose(-2, -1) / d_head ** 0.5
        attn = torch.softmax(scores, dim=-1)
        # Shannon entropy per token, averaged
        entropy = -(attn * (attn + 1e-10).log()).sum(dim=-1).mean().item()
    return entropy

def embedding_fourier(model):
    with torch.no_grad():
        E = model.embed.weight  # (p, d_model)
        F = torch.fft.fft(E, dim=0)  # DFT along the token dimension
        power = (F.abs() ** 2).mean(dim=1)  # average power per frequency
    return power.numpy()[:p // 2 + 1]  # only positive frequencies


# Training with snapshots
torch.manual_seed(42)
model = AttentionHead()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.3)
loss_fn = nn.CrossEntropyLoss()

history = {
    "epoch": [], "loss": [], "accuracy": [],
    "eigenvalues": [], "entropy": [], "fourier": [],
}

print("Training and collecting snapshots...")
for epoch in range(epochs):
    logits = model(train_pairs)
    loss   = loss_fn(logits, train_labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % snapshot_every == 0:
        with torch.no_grad():
            pred = model(test_pairs).argmax(dim=-1)
            acc  = (pred == test_labels).float().mean().item()

        history["epoch"].append(epoch)
        history["loss"].append(loss.item())
        history["accuracy"].append(acc)
        history["eigenvalues"].append(eigenvalues_of_QK(model))
        history["entropy"].append(attention_entropy(model))
        history["fourier"].append(embedding_fourier(model))

        if epoch % 500 == 0:
            print(f"  epoch {epoch:4d}  loss {loss.item():.4f}  acc {acc:.2%}")

print(f"  epoch {epoch:4d}  loss {loss.item():.4f}  acc {acc:.2%}")
print("Done training. Building animation...")

# Build animation
n_frames = len(history["epoch"])
ep = history["epoch"]
n_freqs = len(history["fourier"][0])

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Gauge-invariant probing of grokking", fontsize=16, fontweight="bold")

# Panel 1: Loss and accuracy
ax1 = axes[0, 0]
ax1b = ax1.twinx()
loss_line, = ax1.plot([], [], color="steelblue", linewidth=1.5, label="Train loss")
acc_line, = ax1b.plot([], [], color="tomato", linewidth=1.5, label="Test accuracy")
ax1.set_xlim(0, epochs)
ax1.set_ylim(0, 3)
ax1b.set_ylim(0, 1.05)
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss", color="steelblue")
ax1b.set_ylabel("Accuracy", color="tomato")
ax1.set_title("Training progress")
lines_1 = [loss_line, acc_line]
ax1.legend(lines_1, [l.get_label() for l in lines_1], loc="center right")
vline1 = ax1.axvline(x=0, color="gray", linestyle="--", alpha=0.4)

# Panel 2: Eigenvalues of W_Q @ W_K.T
ax2 = axes[0, 1]
eig_bars = ax2.bar(range(d_head), np.zeros(d_head), color="steelblue", width=0.7)
ax2.set_xlim(-0.5, d_head - 0.5)
all_eigs = np.array(history["eigenvalues"])
ax2.set_ylim(0, all_eigs.max() * 1.1)
ax2.set_xlabel("Index")
ax2.set_ylabel("Eigenvalue")
ax2.set_title("Eigenvalues of (W_Q W_K^T)(W_Q W_K^T)^T")

# Panel 3: Attention entropy
ax3 = axes[1, 0]
entropy_line, = ax3.plot([], [], color="seagreen", linewidth=1.5)
ax3.set_xlim(0, epochs)
all_ent = history["entropy"]
ax3.set_ylim(0, max(all_ent) * 1.2)
ax3.set_xlabel("Epoch")
ax3.set_ylabel("Entropy (nats)")
ax3.set_title("Attention entropy")
vline3 = ax3.axvline(x=0, color="gray", linestyle="--", alpha=0.4)

# Panel 4: Fourier spectrum of embeddings
ax4 = axes[1, 1]
freq_bars = ax4.bar(range(n_freqs), np.zeros(n_freqs), color="darkorange", width=0.7)
ax4.set_xlim(-0.5, n_freqs - 0.5)
all_fourier = np.array(history["fourier"])
ax4.set_ylim(0, all_fourier.max() * 1.1)
ax4.set_xlabel("Frequency")
ax4.set_ylabel("Power")
ax4.set_title("Embedding Fourier spectrum")

epoch_text = fig.text(0.5, 0.92, "", ha="center", fontsize=13)

plt.tight_layout(rect=[0, 0, 1, 0.9])


def animate(i):
    # Loss/accuracy
    loss_line.set_data(ep[:i+1], history["loss"][:i+1])
    acc_line.set_data(ep[:i+1], history["accuracy"][:i+1])
    vline1.set_xdata([ep[i]])

    # Eigenvalues
    eigs = history["eigenvalues"][i]
    for bar, h in zip(eig_bars, eigs):
        bar.set_height(h)

    # Entropy
    entropy_line.set_data(ep[:i+1], history["entropy"][:i+1])
    vline3.set_xdata([ep[i]])

    # Fourier
    fourier = history["fourier"][i]
    for bar, h in zip(freq_bars, fourier):
        bar.set_height(h)

    epoch_text.set_text(f"Epoch {ep[i]}")
    return loss_line, acc_line, entropy_line, epoch_text, vline1, vline3


# Save animation — sample every 3rd frame for reasonable file size
frame_indices = list(range(0, n_frames, 3))
ani = animation.FuncAnimation(fig, animate, frames=frame_indices, blit=False, interval=80)
ani.save("docs/grokking_probe.gif", writer="pillow", dpi=100)
plt.close()

print("Saved docs/grokking_probe.gif")
