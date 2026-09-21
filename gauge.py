import torch

# Dimensions
d_model = 6
d_head  = 4
seq_len = 3

# Random input and weights
X   = torch.randn(seq_len, d_model)
W_Q = torch.randn(d_model, d_head)
W_K = torch.randn(d_model, d_head)
W_V = torch.randn(d_model, d_head)

# Original attention pattern
Q = X @ W_Q
K = X @ W_K

scores    = Q @ K.T / d_head ** 0.5
attention = torch.softmax(scores, dim=-1)

# Gauge transformation: W_Q -> W_Q @ M, W_K -> W_K @ inv(M).T
M = torch.randn(d_head, d_head)

W_Q_new = W_Q @ M
W_K_new = W_K @ torch.inverse(M).T

# Transformed attention pattern
Q_new = X @ W_Q_new
K_new = X @ W_K_new

scores_new    = Q_new @ K_new.T / d_head ** 0.5
attention_new = torch.softmax(scores_new, dim=-1)

# Check
print("Original:\n", attention)
print("\nTransformed:\n", attention_new)
print("\nIdentical?", torch.allclose(attention, attention_new))
