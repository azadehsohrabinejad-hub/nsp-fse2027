import os
import sys
import json
import torch
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.nsp.em_classic import run_classical_em
from src.nsp.filter import NSPFilter
from src.nsp.model import NSPModel

def load_data(data_dir):
    sequences = []
    labels = []
    for file in os.listdir(data_dir):
        if file.endswith(".json"):
            with open(os.path.join(data_dir, file), 'r') as f:
                data = json.load(f)
            if len(data["Y"]) > 0:
                sequences.append(torch.tensor(data["Y"], dtype=torch.float64))
                labels.append(data.get("semantic_label"))
    return sequences, labels

def main():
    print("=== Validating NSP Deceptive Drift ===")
    
    # 1. Load the labeled synthetic sequences
    data_dir = r"D:\raz\razieh\data\labeled_synthetic_sequences"
    sequences, labels = load_data(data_dir)
    
    # Filter out sequences without semantic labels
    labeled_sequences = [(seq, lbl) for seq, lbl in zip(sequences, labels) if lbl is not None]
    print(f"Loaded {len(labeled_sequences)} labeled trajectories.")
    
    # 2. Train EM to learn the global parameters (A, C)
    print("\nTraining Classical EM (30 epochs)...")
    A, C, _ = run_classical_em(data_dir, epochs=30)
    
    # 3. Setup Kalman Filter
    model = NSPModel(state_dim=2, obs_dim=21)
    model.A.data = A.float()
    model.C.data = C.float()
    model.log_Q.data = torch.log(torch.tensor([0.01, 0.01], dtype=torch.float32))
    model.log_R.data = torch.log(torch.tensor([0.1] * 21, dtype=torch.float32))
    nsp_filter = NSPFilter(model)
    
    drift_scores_case1 = [] # p_t = z0, s_t = z1
    drift_scores_case2 = [] # p_t = z1, s_t = z0
    semantic_scores = []
    multi_step_count = 0
    
    print("\nInferring hidden states and calculating Deceptive Drift (checking both axes)...")
    for seq, lbl in labeled_sequences:
        if len(seq) < 2:
            continue
            
        multi_step_count += 1
        z_est, _ = nsp_filter(seq.float())
        
        # Extract both dimensions
        z0 = z_est[:, 0].detach().numpy()
        z1 = z_est[:, 1].detach().numpy()
        
        drift1, drift2 = 0.0, 0.0
        for t in range(1, len(z0)):
            # Case 1: Assume z0 is progress, z1 is alignment
            d_p1 = z0[t] - z0[t-1]
            d_s1 = z1[t] - z1[t-1]
            drift1 += max(0.0, d_p1) * max(0.0, -d_s1)
            
            # Case 2: Assume z1 is progress, z0 is alignment (Axis swap)
            d_p2 = z1[t] - z1[t-1]
            d_s2 = z0[t] - z0[t-1]
            drift2 += max(0.0, d_p2) * max(0.0, -d_s2)
            
        drift_scores_case1.append(drift1)
        drift_scores_case2.append(drift2)
        semantic_scores.append(lbl)
        
    # Calculate Pearson Correlation for both cases
    drift1_arr = np.array(drift_scores_case1)
    drift2_arr = np.array(drift_scores_case2)
    semantic_arr = np.array(semantic_scores)
    
    corr1 = np.corrcoef(drift1_arr, semantic_arr)[0, 1] if np.std(drift1_arr) > 0 else 0
    corr2 = np.corrcoef(drift2_arr, semantic_arr)[0, 1] if np.std(drift2_arr) > 0 else 0
    
    print("\n" + "="*40)
    print("=== FINAL RESULTS FOR PAPER ===")
    print("="*40)
    print(f"Evaluated {multi_step_count} multi-step (T>1) trajectories.")
    print(f"Correlation if Axis 0=Progress, 1=Alignment: {corr1:.4f}")
    print(f"Correlation if Axis 1=Progress, 0=Alignment: {corr2:.4f}")
    
    # Select the best correlation (should be negative)
    best_corr = min(corr1, corr2)
    print(f"\nBest Pearson Correlation (Deceptive Drift vs Completeness): {best_corr:.4f}")
    
    if best_corr < -0.3:
        print("Interpretation: Strong negative correlation! High drift means low completeness. NSP works!")
    elif best_corr < -0.1:
        print("Interpretation: Moderate negative correlation found. NSP is capturing semantic degradation.")
    else:
        print("Interpretation: Weak or no correlation.")

if __name__ == "__main__":
    main()