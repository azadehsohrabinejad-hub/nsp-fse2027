import os
import sys
import json
import torch
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root to path to import nsp modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.nsp.model import NSPModel
from src.nsp.filter import NSPFilter

# True parameters for synthetic data (must match generate_synthetic.py)
A_true = torch.tensor([[0.9, 0.1], [-0.1, 0.9]])
C_true = torch.randn(21, 2) * 0.5

def load_data(data_dir):
    """Loads sequences from a directory."""
    sequences = []
    for file in os.listdir(data_dir):
        if file.endswith(".json"):
            with open(os.path.join(data_dir, file), 'r') as f:
                data = json.load(f)
            if len(data["Y"]) > 0:
                sequences.append(torch.tensor(data["Y"], dtype=torch.float32))
    return sequences

def run_robust_validation(data_type="synthetic", epochs=30, num_seeds=3):
    print(f"\n=== Starting Robust Validation ({data_type}) ===")
    
    # 1. Setup paths
    if data_type == "synthetic":
        data_dir = r"D:\raz\razieh\data\synthetic_sequences"
    else:
        data_dir = r"D:\raz\razieh\data\nsp_sequences"
        
    report_dir = r"D:\raz\razieh\reports\likelihood_validation"
    os.makedirs(report_dir, exist_ok=True)
    
    # 2. Load Data and Split (80% Train, 20% Val)
    sequences = load_data(data_dir)
    np.random.seed(42) # Fixed split for fair comparison across seeds
    indices = np.random.permutation(len(sequences))
    split = int(0.8 * len(sequences))
    train_seqs = [sequences[i] for i in indices[:split]]
    val_seqs = [sequences[i] for i in indices[split:]]
    
    # 3. Normalize based on Train set ONLY
    all_train_data = torch.cat(train_seqs, dim=0)
    mean = torch.mean(all_train_data, dim=0)
    std = torch.std(all_train_data, dim=0)
    std[std == 0] = 1.0  # Prevent division by zero
    
    train_seqs_norm = [(seq - mean) / std for seq in train_seqs]
    val_seqs_norm = [(seq - mean) / std for seq in val_seqs]
    
    seed_results = []
    
    # 4. Loop over multiple Seeds
    for seed in range(1, num_seeds + 1):
        print(f"\n--- Seed {seed}/{num_seeds} ---")
        torch.manual_seed(seed)
        
        model = NSPModel(state_dim=2, obs_dim=21)
        nsp_filter = NSPFilter(model)
        optimizer = optim.Adam(model.parameters(), lr=0.05)
        
        best_val_ll = -float('inf')
        history = []
        
        # 5. Training Loop
        for epoch in range(epochs):
            # --- Train Phase ---
            model.train()
            optimizer.zero_grad()
            train_ll = 0
            for Y in train_seqs_norm:
                _, log_lik = nsp_filter(Y)
                train_ll += log_lik
            loss = -train_ll
            loss.backward()
            
            # Gradient Clipping for numerical stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # --- Validation Phase ---
            model.eval()
            with torch.no_grad():
                val_ll = 0
                for Y in val_seqs_norm:
                    _, log_lik = nsp_filter(Y)
                    val_ll += log_lik
            
            avg_train_ll = train_ll.item() / len(train_seqs_norm)
            avg_val_ll = val_ll.item() / len(val_seqs_norm)
            
            history.append({"epoch": epoch+1, "train_ll": avg_train_ll, "val_ll": avg_val_ll})
            
            # Save Best Model based on Validation LL (Fixes the checkpoint timing issue)
            if avg_val_ll > best_val_ll:
                best_val_ll = avg_val_ll
                torch.save(model.state_dict(), os.path.join(report_dir, f"best_model_{data_type}_seed{seed}.pt"))
        
        # 6. Parameter Recovery (Only for Synthetic Data)
        param_recovery = {}
        if data_type == "synthetic":
            with torch.no_grad():
                A_learned = model.A.detach()
                C_learned = model.C.detach()
                # Calculate Frobenius norm distance
                param_recovery["A_error"] = torch.norm(A_learned - A_true).item()
                param_recovery["C_error"] = torch.norm(C_learned - C_true).item()
        
        final_train_ll = history[-1]["train_ll"]
        initial_train_ll = history[0]["train_ll"]
        
        seed_results.append({
            "seed": seed,
            "initial_train_ll": initial_train_ll,
            "final_train_ll": final_train_ll,
            "best_val_ll": best_val_ll,
            "param_recovery": param_recovery
        })
        
        # Plot learning curve for this seed
        plt.figure(figsize=(8, 4))
        plt.plot([h["epoch"] for h in history], [h["train_ll"] for h in history], label='Train LL', color='blue')
        plt.plot([h["epoch"] for h in history], [h["val_ll"] for h in history], label='Val LL', color='red', linestyle='--')
        plt.title(f"Robust Validation - {data_type} (Seed {seed})")
        plt.xlabel("Epoch")
        plt.ylabel("Log-Likelihood")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(report_dir, f"{data_type}_seed{seed}_learning_curve.png"))
        plt.close()
        
    # 7. Save Summary JSON
    summary_path = os.path.join(report_dir, "robust_validation_summary.json")
    existing_summary = {}
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            existing_summary = json.load(f)
    existing_summary[data_type] = seed_results
    with open(summary_path, 'w') as f:
        json.dump(existing_summary, f, indent=4)
        
    print(f"\nRobust Validation ({data_type}) completed. Results saved to JSON.")

if __name__ == "__main__":
    # Run for Synthetic (Checks parameter recovery)
    run_robust_validation(data_type="synthetic", epochs=30, num_seeds=3)
    
    # Run for Project (Checks generalization)
    run_robust_validation(data_type="project", epochs=30, num_seeds=3)