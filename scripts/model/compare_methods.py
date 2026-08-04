import os
import sys
import json
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.nsp.model import NSPModel
from src.nsp.filter import NSPFilter
from src.nsp.particle_filter import NSPParticleFilter

def load_data(data_dir):
    sequences = []
    for file in os.listdir(data_dir):
        if file.endswith(".json"):
            with open(os.path.join(data_dir, file), 'r') as f:
                data = json.load(f)
            if len(data["Y"]) > 0:
                sequences.append(torch.tensor(data["Y"], dtype=torch.float32))
    return sequences

def run_comparison():
    print("=== Starting Method Comparison (Step 3.10) ===")
    
    data_dir = r"D:\raz\razieh\data\synthetic_sequences"
    sequences = load_data(data_dir)
    
    # Initialize Model with random parameters
    torch.manual_seed(42)
    model = NSPModel(state_dim=2, obs_dim=21)
    
    # 1. Kalman Filter
    print("\nRunning Kalman Filter...")
    kf = NSPFilter(model)
    kf_total_ll = 0
    for Y in sequences:
        _, ll = kf(Y)
        kf_total_ll += ll.item()
    kf_avg_ll = kf_total_ll / len(sequences)
    
    # 2. Particle Filter
    print("Running Particle Filter (1000 particles)...")
    pf = NSPParticleFilter(model, num_particles=1000)
    pf_total_ll = 0
    for Y in sequences:
        _, ll = pf(Y)
        pf_total_ll += ll.item()
    pf_avg_ll = pf_total_ll / len(sequences)
    
    # Summary
    print("\n--- Comparison Summary ---")
    print(f"Total Sequences: {len(sequences)}")
    print(f"Kalman Filter Avg LL:   {kf_avg_ll:.4f}")
    print(f"Particle Filter Avg LL: {pf_avg_ll:.4f}")
    
    difference = abs(kf_avg_ll - pf_avg_ll)
    print(f"\nAbsolute Difference: {difference:.4f}")
    print("Note: Lower (more negative) is better for Log-Likelihood.")
    
    # Save summary
    report_dir = r"D:\raz\razieh\reports\likelihood_validation"
    os.makedirs(report_dir, exist_ok=True)
    summary_path = os.path.join(report_dir, "method_comparison_summary.json")
    
    summary = {
        "kalman_filter_avg_ll": kf_avg_ll,
        "particle_filter_avg_ll": pf_avg_ll,
        "absolute_difference": difference
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
        
    print(f"\nResults saved to: {summary_path}")

if __name__ == "__main__":
    run_comparison()