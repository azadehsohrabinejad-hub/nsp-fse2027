import torch
import os
import json

# 1. Define true parameters (The ground truth we want the model to learn)
state_dim = 2
obs_dim = 21
seq_length = 10  # T = 10 (multi-step sequences)
num_sequences = 100

# True Transition and Observation Matrices
A_true = torch.tensor([[0.9, 0.1], [-0.1, 0.9]])
C_true = torch.randn(obs_dim, state_dim) * 0.5

# Noise covariances
Q_true = torch.tensor([0.01, 0.01])  # Process noise
R_true = torch.ones(obs_dim) * 0.1   # Observation noise

# 2. Output directory
out_dir = r"D:\raz\razieh\data\synthetic_sequences"
os.makedirs(out_dir, exist_ok=True)

print(f"Generating {num_sequences} synthetic sequences (T={seq_length})...")

# 3. Generate sequences
for i in range(num_sequences):
    # Initialize hidden states and observations
    z = torch.zeros(seq_length, state_dim)
    y = torch.zeros(seq_length, obs_dim)
    
    # Random initial state
    z[0] = torch.randn(state_dim)
    
    for t in range(seq_length):
        if t > 0:
            # z_t = A * z_{t-1} + noise
            noise_z = torch.randn(state_dim) * torch.sqrt(Q_true)
            z[t] = torch.matmul(A_true, z[t-1]) + noise_z
        
        # y_t = C * z_t + noise
        noise_y = torch.randn(obs_dim) * torch.sqrt(R_true)
        y[t] = torch.matmul(C_true, z[t]) + noise_y
        
    # Save in the same format as our real NSP sequences
    data = {
        "run_id": f"synthetic_run_{i}",
        "sequence_length": seq_length,
        "feature_vector_length": obs_dim,
        "Y": y.tolist(),  # The 21-dimensional observation sequence
        "semantic_label": None
    }
    
    file_path = os.path.join(out_dir, f"synthetic_run_{i}.json")
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

print(f"Done! Synthetic data saved to: {out_dir}")