import os
import sys
import json
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
    print("=== Generating Figure 7: Drift vs Completeness ===")
    
    # 1. Load data
    data_dir = r"D:\raz\razieh\data\labeled_synthetic_sequences"
    sequences, labels = load_data(data_dir)
    labeled_sequences = [(seq, lbl) for seq, lbl in zip(sequences, labels) if lbl is not None]
    
    # 2. Train EM
    print("Training Classical EM (30 epochs)...")
    A, C, _ = run_classical_em(data_dir, epochs=30)
    
    # 3. Setup Filter
    model = NSPModel(state_dim=2, obs_dim=21)
    model.A.data = A.float()
    model.C.data = C.float()
    model.log_Q.data = torch.log(torch.tensor([0.01, 0.01], dtype=torch.float32))
    model.log_R.data = torch.log(torch.tensor([0.1] * 21, dtype=torch.float32))
    nsp_filter = NSPFilter(model)
    
    drifts = []
    semantics = []
    
    print("Inferring hidden states...")
    for seq, lbl in labeled_sequences:
        if len(seq) < 2: continue
        
        z_est, _ = nsp_filter(seq.float())
        z0 = z_est[:, 0].detach().numpy()
        z1 = z_est[:, 1].detach().numpy()
        
        # Case 2 (Axis swap: z1=progress, z0=alignment)
        drift = 0.0
        for t in range(1, len(z0)):
            d_p = z1[t] - z1[t-1]
            d_s = z0[t] - z0[t-1]
            drift += max(0.0, d_p) * max(0.0, -d_s)
            
        drifts.append(drift)
        semantics.append(lbl)
        
    drifts = np.array(drifts)
    semantics = np.array(semantics)
    
    # 4. Plotting
    plt.figure(figsize=(8, 6))
    plt.scatter(drifts, semantics, alpha=0.7, edgecolors='k', s=80, color='royalblue')
    
    # Add trend line (Linear Regression)
    z = np.polyfit(drifts, semantics, 1)
    p = np.poly1d(z)
    x_line = np.linspace(drifts.min(), drifts.max(), 100)
    plt.plot(x_line, p(x_line), "r--", linewidth=2, label="Trend Line")
    
    # Calculate Pearson Correlation for the title
    corr = np.corrcoef(drifts, semantics)[0, 1]
    
    plt.title(f"Deceptive Drift vs Semantic Completeness\n(Pearson r = {corr:.4f})", fontsize=14)
    plt.xlabel("Deceptive Drift Score (High = Test-passing but wrong)", fontsize=12)
    plt.ylabel("Semantic Completeness (1 = Perfect)", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    
    output_path = r"D:\raz\razieh\reports\Figure_7_Drift_vs_Completeness.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nFigure 7 saved successfully to: {output_path}")

if __name__ == "__main__":
    main()