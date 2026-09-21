import torch

# Parameters
p = 13  # modulus
train_frac = 0.7

# Generate all (a, b) pairs and labels
pairs  = torch.cartesian_prod(torch.arange(p), torch.arange(p))
labels = (pairs[:, 0] + pairs[:, 1]) % p

# Shuffle
perm   = torch.randperm(len(pairs))
pairs  = pairs[perm]
labels = labels[perm]

# Train/test split
split  = int(len(pairs) * train_frac)
train_pairs, test_pairs   = pairs[:split], pairs[split:]
train_labels, test_labels = labels[:split], labels[split:]

# Verify
print(f"Total: {len(pairs)}  Train: {len(train_pairs)}  Test: {len(test_pairs)}")
print(f"\nFirst 5 training examples:")
for i in range(5):
    a, b = train_pairs[i]
    print(f"  {a.item()} + {b.item()} mod {p} = {train_labels[i].item()}")
