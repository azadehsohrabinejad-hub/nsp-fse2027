import torch
import os
import json

# True parameters (Ground Truth)
A_true = torch.tensor([[0.9, 0.1], [-0.1, 0.9]])
C_true = torch.randn(21, 2) * 0.5
Q_true = torch.tensor([0.01, 0.01])
R_true = torch.ones(21) * 0.1

out_dir = r"D:\raz\razieh\data\labeled_synthetic_sequences"
os.makedirs(out_dir, exist_ok=True)

print("Generating 100 labeled synthetic trajectories...")
for i in range(100):
    T = 10
    z = torch.zeros(T, 2)
    y = torch.zeros(T, 21)
    
    # Random initial state
    z[0] = torch.randn(2)
    
    for t in range(T):
        if t > 0:
            noise_z = torch.randn(2) * torch.sqrt(Q_true)
            z[t] = torch.matmul(A_true, z[t-1]) + noise_z
        
        noise_y = torch.randn(21) * torch.sqrt(R_true)
        y[t] = torch.matmul(C_true, z[t]) + noise_y
        
    # Calculate Ground Truth Drift
    # Convert to numpy to avoid PyTorch max() type issues
    p_true = z[:, 0].numpy()
    s_true = z[:, 1].numpy()
    
    drifts = []
    for t in range(1, T):
        delta_p = p_true[t] - p_true[t-1]
        delta_s = s_true[t] - s_true[t-1]
        d = max(0.0, delta_p) * max(0.0, -delta_s)
        drifts.append(float(d))
        
    total_drift = sum(drifts)
    
    # Assign Semantic Label (Completeness): High drift -> Low completeness
    semantic_label = max(0.0, 1.0 - total_drift)
    
    data = {
        "run_id": f"labeled_synth_{i}",
        "sequence_length": T,
        "feature_vector_length": 21,
        "Y": y.tolist(),
        "semantic_label": semantic_label
    }
    
    with open(os.path.join(out_dir, f"labeled_synth_{i}.json"), 'w') as f:
        json.dump(data, f, indent=2)

print(f"Done! 100 labeled trajectories saved to {out_dir}")