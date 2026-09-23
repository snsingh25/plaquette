import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

p = 13
d_model = 64
n_heads = 4
d_head  = d_model // n_heads  # 16 per head
epochs  = 8000
snapshot_every = 20

# Dataset
pairs  = torch.cartesian_prod(torch.arange(p), torch.arange(p))
labels = (pairs[:, 0] + pairs[:, 1]) % p
perm   = torch.randperm(len(pairs))
pairs, labels = pairs[perm], labels[perm]
split  = int(len(pairs) * 0.7)
train_pairs, train_labels = pairs[:split], labels[:split]
test_pairs,  test_labels  = pairs[split:], labels[split:]


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(p, d_model)

        # Each head gets its own Q, K, V
        self.W_Q = nn.ModuleList([nn.Linear(d_model, d_head, bias=False) for _ in range(n_heads)])
        self.W_K = nn.ModuleList([nn.Linear(d_model, d_head, bias=False) for _ in range(n_heads)])
        self.W_V = nn.ModuleList([nn.Linear(d_model, d_head, bias=False) for _ in range(n_heads)])

        # Combine heads back to d_model
        self.combine = nn.Linear(n_heads * d_head, d_model, bias=False)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model),
        )

        self.out = nn.Linear(d_model, p)

    def forward(self, x):
        e = self.embed(x)

        head_outputs = []
        for h in range(n_heads):
            Q = self.W_Q[h](e)
            K = self.W_K[h](e)
            V = self.W_V[h](e)
            scores = Q @ K.transpose(-2, -1) / d_head ** 0.5
            attn   = torch.softmax(scores, dim=-1)
            head_outputs.append(attn @ V)

        # Concatenate all heads
        mixed = torch.cat(head_outputs, dim=-1)
        mixed = self.combine(mixed)
        mixed = mixed + self.mlp(mixed)

        return self.out(mixed[:, 0, :])


# Gauge-invariant probes per head
def eigenvalues_per_head(model):
    eigs = []
    for h in range(n_heads):
        WQ = model.W_Q[h].weight
        WK = model.W_K[h].weight
        QK = WQ @ WK.T
        e = torch.linalg.eigvalsh(QK @ QK.T).detach().numpy()
        eigs.append(e)
    return eigs

def entropy_per_head(model):
    with torch.no_grad():
        e = model.embed(train_pairs)
        entropies = []
        for h in range(n_heads):
            Q = model.W_Q[h](e)
            K = model.W_K[h](e)
            scores = Q @ K.transpose(-2, -1) / d_head ** 0.5
            attn = torch.softmax(scores, dim=-1)
            ent = -(attn * (attn + 1e-10).log()).sum(dim=-1).mean().item()
            entropies.append(ent)
    return entropies

def embedding_fourier(model):
    with torch.no_grad():
        E = model.embed.weight
        F = torch.fft.fft(E, dim=0)
        power = (F.abs() ** 2).mean(dim=1)
    return power.numpy()[:p // 2 + 1]


# Training
torch.manual_seed(42)
model = MultiHeadAttention()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1.0)
loss_fn = nn.CrossEntropyLoss()

history = {
    "epoch": [], "loss": [], "accuracy": [],
    "eigenvalues": [], "entropy": [], "fourier": [],
}

print(f"Training {n_heads}-head model...")
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
        history["eigenvalues"].append(eigenvalues_per_head(model))
        history["entropy"].append(entropy_per_head(model))
        history["fourier"].append(embedding_fourier(model))

        if epoch % 500 == 0:
            print(f"  epoch {epoch:4d}  loss {loss.item():.4f}  acc {acc:.2%}")

print(f"  epoch {epoch:4d}  loss {loss.item():.4f}  acc {acc:.2%}")


# Flat directions per head
print("\nCounting flat directions per head...")
d_model_small, d_head_small, n_heads_small = 8, 4, 4
torch.manual_seed(42)
embed_s = nn.Embedding(p, d_model_small)
WQ_s = nn.ModuleList([nn.Linear(d_model_small, d_head_small, bias=False) for _ in range(n_heads_small)])
WK_s = nn.ModuleList([nn.Linear(d_model_small, d_head_small, bias=False) for _ in range(n_heads_small)])

all_pairs = torch.cartesian_prod(torch.arange(p), torch.arange(p))

for h in range(n_heads_small):
    rows = []
    for i in range(len(all_pairs)):
        x = all_pairs[i].unsqueeze(0)
        e = embed_s(x)
        Q = WQ_s[h](e)
        K = WK_s[h](e)
        scores = Q @ K.transpose(-2, -1) / d_head_small ** 0.5
        attn = torch.softmax(scores, dim=-1).flatten()
        for j in range(len(attn)):
            WQ_s[h].weight.grad = None
            WK_s[h].weight.grad = None
            attn[j].backward(retain_graph=True)
            gq = WQ_s[h].weight.grad.flatten().clone()
            gk = WK_s[h].weight.grad.flatten().clone()
            rows.append(torch.cat([gq, gk]))

    J = torch.stack(rows)
    S = torch.linalg.svdvals(J)
    threshold = S.max() * 1e-5
    n_flat = (S < threshold).sum().item()
    print(f"  Head {h}: {n_flat} flat directions (predicted: {d_head_small**2} = {d_head_small}^2)")

total_flat = n_heads_small * d_head_small ** 2
print(f"  Total: {n_heads_small} heads x {d_head_small}^2 = {total_flat} flat directions")


# Build animation
print("\nBuilding animation...")
n_frames = len(history["epoch"])
ep = history["epoch"]
n_freqs = len(history["fourier"][0])
head_colors = ["steelblue", "tomato", "seagreen", "darkorange"]

fig, axes = plt.subplots(3, 2, figsize=(14, 12))
fig.suptitle(f"Gauge-invariant probing of grokking — {n_heads} heads", fontsize=16, fontweight="bold")

# Row 1 left: Loss/accuracy
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
ax1.legend([loss_line, acc_line], ["Train loss", "Test accuracy"], loc="center right", fontsize=8)

# Row 1 right: Fourier spectrum
ax_f = axes[0, 1]
freq_bars = ax_f.bar(range(n_freqs), np.zeros(n_freqs), color="darkorange", width=0.7)
ax_f.set_xlim(-0.5, n_freqs - 0.5)
all_fourier = np.array(history["fourier"])
ax_f.set_ylim(0, all_fourier.max() * 1.1)
ax_f.set_xlabel("Frequency")
ax_f.set_ylabel("Power")
ax_f.set_title("Embedding Fourier spectrum")

# Row 2: Eigenvalues per head (2 panels, 2 heads each)
all_eigs = np.array(history["eigenvalues"])  # (n_frames, n_heads, d_head)
eig_max = all_eigs.max() * 1.1

eig_bar_sets = []
for panel in range(2):
    ax = axes[1, panel]
    bars_in_panel = []
    x_offset = 0
    tick_positions = []
    tick_labels_list = []
    for h_idx in range(2):
        h = panel * 2 + h_idx
        x_positions = np.arange(d_head) + x_offset
        bars = ax.bar(x_positions, np.zeros(d_head), color=head_colors[h], width=0.7, label=f"Head {h}")
        bars_in_panel.append((h, bars))
        tick_positions.append(x_offset + d_head / 2 - 0.5)
        tick_labels_list.append(f"Head {h}")
        x_offset += d_head + 2
    ax.set_xlim(-1, x_offset - 1)
    ax.set_ylim(0, eig_max)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels_list)
    ax.set_ylabel("Eigenvalue")
    ax.set_title(f"Eigenvalues of (W_Q W_K^T)(W_Q W_K^T)^T")
    ax.legend(fontsize=8)
    eig_bar_sets.append(bars_in_panel)

# Row 3: Entropy per head
ax_e = axes[2, 0]
entropy_lines = []
for h in range(n_heads):
    line, = ax_e.plot([], [], color=head_colors[h], linewidth=1.5, label=f"Head {h}")
    entropy_lines.append(line)
ax_e.set_xlim(0, epochs)
all_ent = np.array(history["entropy"])
ax_e.set_ylim(0, all_ent.max() * 1.2)
ax_e.set_xlabel("Epoch")
ax_e.set_ylabel("Entropy (nats)")
ax_e.set_title("Attention entropy per head")
ax_e.legend(fontsize=8)

# Row 3 right: flat directions summary (static)
ax_flat = axes[2, 1]
flat_labels = [f"Head {h}" for h in range(n_heads_small)] + ["Total"]
flat_predicted = [d_head_small**2] * n_heads_small + [n_heads_small * d_head_small**2]
flat_colors = head_colors[:n_heads_small] + ["gray"]
ax_flat.bar(flat_labels, flat_predicted, color=flat_colors, width=0.6)
ax_flat.set_ylabel("Flat directions")
ax_flat.set_title(f"Flat directions: {d_head_small}^2 = {d_head_small**2} per head")
for i, v in enumerate(flat_predicted):
    ax_flat.text(i, v + 0.5, str(v), ha="center", fontsize=10, fontweight="bold")

epoch_text = fig.text(0.5, 0.93, "", ha="center", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.91])


def animate(i):
    loss_line.set_data(ep[:i+1], history["loss"][:i+1])
    acc_line.set_data(ep[:i+1], history["accuracy"][:i+1])

    for bar, h_val in zip(freq_bars, history["fourier"][i]):
        bar.set_height(h_val)

    for panel_bars in eig_bar_sets:
        for h, bars in panel_bars:
            eigs = history["eigenvalues"][i][h]
            for bar, val in zip(bars, eigs):
                bar.set_height(val)

    for h in range(n_heads):
        ent_so_far = [history["entropy"][j][h] for j in range(i+1)]
        entropy_lines[h].set_data(ep[:i+1], ent_so_far)

    epoch_text.set_text(f"Epoch {ep[i]}")
    return ()


frame_indices = list(range(0, n_frames, 3))
ani = animation.FuncAnimation(fig, animate, frames=frame_indices, blit=False, interval=80)
ani.save("docs/multihead_grokking.gif", writer="pillow", dpi=100)
plt.close()

print("Saved docs/multihead_grokking.gif")
