import torch
import torch.nn as nn

p = 13
d_model = 8
d_head  = 4

torch.manual_seed(42)

# Random model (flat directions are architectural, no training needed)
embed = nn.Embedding(p, d_model)
W_Q = nn.Linear(d_model, d_head, bias=False)
W_K = nn.Linear(d_model, d_head, bias=False)
W_V = nn.Linear(d_model, d_head, bias=False)

pairs = torch.cartesian_prod(torch.arange(p), torch.arange(p))

# Build Jacobian of attention pattern w.r.t. (W_Q, W_K) over all 169 inputs
rows = []
for i in range(len(pairs)):
    x = pairs[i].unsqueeze(0)
    e = embed(x)
    Q, K, V = W_Q(e), W_K(e), W_V(e)
    scores = Q @ K.transpose(-2, -1) / d_head ** 0.5
    attn = torch.softmax(scores, dim=-1).flatten()

    for j in range(len(attn)):
        W_Q.weight.grad = None
        W_K.weight.grad = None
        attn[j].backward(retain_graph=True)
        gq = W_Q.weight.grad.flatten().clone()
        gk = W_K.weight.grad.flatten().clone()
        rows.append(torch.cat([gq, gk]))

J = torch.stack(rows)

# SVD
S = torch.linalg.svdvals(J)
threshold = S.max() * 1e-5
n_flat = (S < threshold).sum().item()
n_params = W_Q.weight.numel() + W_K.weight.numel()

print(f"W_Q + W_K parameters:      {n_params}")
print(f"Predicted flat (d_head^2):  {d_head ** 2}")
print(f"Measured flat:              {n_flat}")
print(f"\nAll {len(S)} singular values:")
for i, s in enumerate(S):
    label = "  <-- flat" if s < threshold else ""
    print(f"  {i:2d}: {s.item():.6f}{label}")
