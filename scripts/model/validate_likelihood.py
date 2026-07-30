import os
import sys
import json
import csv
import torch
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg') # Use non-GUI backend for server/script compatibility
import matplotlib.pyplot as plt

# Add project root to path to import nsp modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.nsp.model import NSPModel
from src.nsp.filter import NSPFilter

def load_and_normalize_data(data_dir):
    """Loads sequences and standardizes features (Mean=0, Std=1)."""
    sequences = []
    for file in os.listdir(data_dir):
        if file.endswith(".json"):
            with open(os.path.join(data_dir, file), 'r') as f:
                data = json.load(f)
            if len(data["Y"]) > 0:
                sequences.append(torch.tensor(data["Y"], dtype=torch.float32))
    
    if not sequences:
        raise ValueError(f"No sequences found in {data_dir}")

    # Calculate mean and std across all data points
    all_data = torch.cat(sequences, dim=0)
    mean = torch.mean(all_data, dim=0)
    std = torch.std(all_data, dim=0)
    std[std == 0] = 1.0  # Prevent division by zero for constant features

    # Normalize sequences
    normalized_sequences = [(seq - mean) / std for seq in sequences]
    return normalized_sequences

def run_validation(data_type="synthetic", epochs=50):
    print(f"\n=== Starting Likelihood Validation ({data_type}) ===")
    
    # 1. Setup paths
    if data_type == "synthetic":
        data_dir = r"D:\raz\razieh\data\synthetic_sequences"
    else:
        data_dir = r"D:\raz\razieh\data\nsp_sequences"
        
    report_dir = r"D:\raz\razieh\reports\likelihood_validation"
    os.makedirs(report_dir, exist_ok=True)
    
    # 2. Load and Normalize Data
    print("Loading and normalizing data...")
    sequences = load_and_normalize_data(data_dir)
    print(f"Loaded {len(sequences)} sequences.")

    # 3. Initialize Model and Optimizer
    model = NSPModel(state_dim=2, obs_dim=21)
    nsp_filter = NSPFilter(model)
    optimizer = optim.Adam(model.parameters(), lr=0.05)

    history = []
    best_ll = -float('inf')
    best_epoch = 0
    nan_inf_count = 0
    positive_changes = 0
    prev_avg_ll = -float('inf')

    # 4. Training Loop
    for epoch in range(epochs):
        total_ll = 0
        optimizer.zero_grad()
        
        for Y in sequences:
            _, log_lik = nsp_filter(Y)
            total_ll += log_lik
            
        # Loss is Negative Log-Likelihood
        loss = -total_ll
        
        # Check for numerical stability
        has_nan = torch.isnan(loss).item() or torch.isinf(loss).item()
        if has_nan:
            nan_inf_count += 1
            print(f"Epoch {epoch+1}: WARNING! NaN or Inf detected in loss.")
            continue
            
        loss.backward()
        
        # Calculate Gradient Norm
        grad_norm = sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
        
        optimizer.step()
        
        avg_ll = total_ll.item() / len(sequences)
        
        # Track metrics
        history.append({
            "epoch": epoch + 1,
            "total_log_likelihood": total_ll.item(),
            "average_log_likelihood": avg_ll,
            "negative_log_likelihood": loss.item(),
            "gradient_norm": grad_norm,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "finite_parameters": not has_nan
        })
        
        # Track best model
        if avg_ll > best_ll:
            best_ll = avg_ll
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(report_dir, f"best_model_{data_type}.pt"))
            
        # Track positive changes
        if epoch > 0 and avg_ll > prev_avg_ll:
            positive_changes += 1
            
        prev_avg_ll = avg_ll
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Avg LL: {avg_ll:.4f} - Grad Norm: {grad_norm:.4f}")

    # 5. Evaluation and Acceptance Criteria
    initial_ll = history[0]["average_log_likelihood"]
    final_ll = history[-1]["average_log_likelihood"]
    positive_ratio = positive_changes / (epochs - 1) if epochs > 1 else 0
    
    validation_summary = {
        "data_type": data_type,
        "total_epochs": epochs,
        "initial_log_likelihood": initial_ll,
        "final_log_likelihood": final_ll,
        "best_log_likelihood": best_ll,
        "best_epoch": best_epoch,
        "nan_inf_count": nan_inf_count,
        "positive_epoch_change_ratio": positive_ratio,
        "criteria_passed": {
            "LL_final > LL_initial": final_ll > initial_ll,
            "Best_LL > Initial_LL": best_ll > initial_ll,
            "NaN_Inf_count_zero": nan_inf_count == 0,
            "Positive_ratio >= 0.60": positive_ratio >= 0.60
        }
    }

    # 6. Save Reports
    # Save CSV history
    csv_path = os.path.join(report_dir, f"{data_type}_training_history.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    # Save Plot
    ll_values = [h["average_log_likelihood"] for h in history]
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs+1), ll_values, label='Avg Log-Likelihood', color='blue')
    # Add moving average (window=5)
    if len(ll_values) >= 5:
        moving_avg = np.convolve(ll_values, np.ones(5)/5, mode='valid')
        plt.plot(range(5, epochs+1), moving_avg, label='5-Epoch Moving Avg', color='red', linestyle='--')
    plt.title(f"Training Likelihood Validation ({data_type})")
    plt.xlabel("Epoch")
    plt.ylabel("Log-Likelihood")
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(report_dir, f"{data_type}_log_likelihood.png")
    plt.savefig(plot_path)
    plt.close()

    # Save JSON Summary
    summary_path = os.path.join(report_dir, "validation_summary.json")
    # Merge with existing summary if exists
    existing_summary = {}
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            existing_summary = json.load(f)
    existing_summary[data_type] = validation_summary
    with open(summary_path, 'w') as f:
        json.dump(existing_summary, f, indent=4)

    print("\n--- Validation Summary ---")
    print(json.dumps(validation_summary, indent=4))
    print(f"Reports saved to: {report_dir}")

if __name__ == "__main__":
    # Run for Synthetic Data (Most important)
    run_validation(data_type="synthetic", epochs=50)
    
    # Run for Project Data (Pilot)
    run_validation(data_type="project", epochs=50)