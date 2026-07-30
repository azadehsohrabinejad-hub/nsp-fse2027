import torch
import torch.optim as optim
import json
import os
from model import NSPModel
from filter import NSPFilter

# 1. Load all synthetic sequences
synth_dir = r"D:\raz\razieh\data\nsp_sequences"
print("Loading synthetic data...")
sequences = []
for file in os.listdir(synth_dir):
    if file.endswith(".json"):
        with open(os.path.join(synth_dir, file), 'r') as f:
            data = json.load(f)
        sequences.append(torch.tensor(data["Y"], dtype=torch.float32))

print(f"Loaded {len(sequences)} sequences.")

# 2. Initialize Model and Filter
model = NSPModel(state_dim=2, obs_dim=21)
nsp_filter = NSPFilter(model)

# 3. Setup Optimizer (This acts as the M-step of EM)
optimizer = optim.Adam(model.parameters(), lr=0.05)
epochs = 50

print("\nStarting Training (NSP-FIT-EM)...")
for epoch in range(epochs):
    total_loss = 0
    optimizer.zero_grad()
    
    # We want to MAXIMIZE log-likelihood, which means MINIMIZE negative log-likelihood
    for Y in sequences:
        _, log_lik = nsp_filter(Y)
        total_loss -= log_lik  # Accumulate negative log-likelihood
        
    # Backpropagation
    total_loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs} - Loss (NLL): {total_loss.item():.2f}")

print("\nTraining finished!")

# 4. Inspect learned parameters
print("\n--- Learned Transition Matrix (A) ---")
print(model.A.detach().numpy())

print("\n--- True Transition Matrix (A_true) ---")
print([[0.9, 0.1], [-0.1, 0.9]])