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

# Forward pass
Q = X @ W_Q
K = X @ W_K
V = X @ W_V

scores    = Q @ K.T / d_head ** 0.5
attention = torch.softmax(scores, dim=-1)
output    = attention @ V

# Results
print("Attention pattern:\n", attention)
print("\nOutput:\n", output)
