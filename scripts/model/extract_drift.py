import os
import sys
import json
import torch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.nsp.em_classic import run_classical_em
from src.nsp.filter import NSPFilter
from src.nsp.model import NSPModel

def load_data(data_dir):
    sequences = []
    for file in os.listdir(data_dir):
        if file.endswith(".json"):
            with open(os.path.join(data_dir, file), 'r') as f:
                data = json.load(f)
            if len(data["Y"]) > 0:
                seq = torch.tensor(data["Y"], dtype=torch.float64)
                # FILTER: Only keep sequences that have some activity (non-zero)
                if torch.sum(torch.abs(seq)) > 0:
                    sequences.append(seq)
    return sequences
    
def main():
    print("=== Step 4: Drift Extraction on Real Pilot Data ===")
    
        # Change path to the REAL pilot sequences
    data_dir = r"D:\raz\razieh\data\nsp_pilot_sequences"
    sequences = load_data(data_dir)
    
    if not sequences:
        print("No sequences found! Run extract_pilot_sequences.py first.")
        return
        
    print(f"Loaded {len(sequences)} real trajectories (T={sequences[0].shape[0]}).")
    
    # 1. Train EM to learn the global parameters (A, C)
    # 1. Train EM to learn the global parameters (A, C)
    print("\nTraining Classical EM (10 epochs)...")
    A, C, _ = run_classical_em(data_dir, epochs=10)
    
    # 2. Use Kalman Filter to infer hidden states (z_t) for each trajectory
    print("\nInferring hidden states (z_t) for each trajectory...")
    
    # Create a model instance with learned parameters
    model = NSPModel(state_dim=2, obs_dim=21)
    # IMPORTANT: Cast EM outputs (float64) to float32 to match the Kalman Filter
    model.A.data = A.float()
    model.C.data = C.float()
    
    # Set noise to small values learned from data (also as float32)
    model.log_Q.data = torch.log(torch.tensor([0.01, 0.01], dtype=torch.float32))
    model.log_R.data = torch.log(torch.tensor([0.1] * 21, dtype=torch.float32))
    
    nsp_filter = NSPFilter(model)
    
    drift_results = []
    
    for i, Y in enumerate(sequences):
        # Y shape: (T, 21)
        z_est, _ = nsp_filter(Y.float()) # Forward pass
        
        # Extract Semantic Alignment (s_t) and Repair Progress (p_t)
        # Assuming z_t[0] is progress and z_t[1] is alignment
        p_t = z_est[:, 0].tolist()
        s_t = z_est[:, 1].tolist()
        
        # Calculate Drift
        drifts = []
        for t in range(1, len(p_t)):
            delta_p = p_t[t] - p_t[t-1]
            delta_s = s_t[t] - s_t[t-1]
            
            # Deceptive Drift formula from drift_definition.md
            # D = max(0, delta_p) * max(0, -delta_s)
            d_deceptive = max(0, delta_p) * max(0, -delta_s)
            drifts.append(d_deceptive)
            
        total_drift = sum(drifts)
        
        result = {
            "run_id": f"trajectory_{i+1}",
            "length_T": len(p_t),
            "repair_progress": p_t,
            "semantic_alignment": s_t,
            "drift_per_round": drifts,
            "total_deceptive_drift": total_drift
        }
        drift_results.append(result)
        
        print(f"\n--- Trajectory {i+1} ---")
        print(f"Length (T): {len(p_t)}")
        print(f"Repair Progress (p_t): {[round(p, 3) for p in p_t]}")
        print(f"Semantic Alignment (s_t): {[round(s, 3) for s in s_t]}")
        print(f"Total Deceptive Drift: {total_drift:.4f}")
        
    # Save results
    report_path = r"D:\raz\razieh\reports\pilot_drift_analysis.json"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(drift_results, f, indent=4)
        
    print(f"\nDrift analysis saved to: {report_path}")

if __name__ == "__main__":
    main()