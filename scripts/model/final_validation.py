import os
import sys
import json
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.nsp.em_classic import run_classical_em
from src.nsp.filter import NSPFilter
from src.nsp.model import NSPModel

def load_data(seq_dir, judge_path):
    sequences = []
    labels = []
    run_ids = []
    
    # Load Judge scores
    with open(judge_path, 'r') as f:
        judge_data = json.load(f)
        
    # Create a dictionary for fast lookup: run_id -> score
    judge_scores = {item["run_id"]: item["judge_score"] for item in judge_data}
    
    # Load sequences
    for file in os.listdir(seq_dir):
        if file.endswith(".json"):
            with open(os.path.join(seq_dir, file), 'r') as f:
                data = json.load(f)
            if len(data["Y"]) > 0:
                seq = torch.tensor(data["Y"], dtype=torch.float64)
                run_id = data.get("run_id", file.replace(".json", ""))
                
                # Only keep sequences that have a judge score and are multi-step (T>1)
                if run_id in judge_scores and len(seq) > 1:
                    # Filter out all-zero sequences
                    if torch.sum(torch.abs(seq)) > 0:
                        sequences.append(seq)
                        labels.append(judge_scores[run_id])
                        run_ids.append(run_id)
                        
    return sequences, labels, run_ids

def main():
    print("=== Final Real-World Validation: NSP Drift vs LLM-Judge ===")
    
    seq_dir = r"D:\raz\razieh\data\nsp_pilot_sequences"
    judge_path = r"D:\raz\razieh\reports\llm_judge_results.json"
    
    sequences, labels, run_ids = load_data(seq_dir, judge_path)
    print(f"Loaded {len(sequences)} valid multi-step trajectories with Judge scores.")
    
    if len(sequences) == 0:
        print("No valid data found! Make sure Runner generated non-zero traces.")
        return
        
    # 1. Train EM
    print("\nTraining Classical EM (20 epochs)...")
    A, C, _ = run_classical_em(seq_dir, state_dim=2, epochs=20)
    
    # 2. Setup Kalman Filter
    model = NSPModel(state_dim=2, obs_dim=21)
    model.A.data = A.float()
    model.C.data = C.float()
    model.log_Q.data = torch.log(torch.tensor([0.01, 0.01], dtype=torch.float32))
    model.log_R.data = torch.log(torch.tensor([0.1] * 21, dtype=torch.float32))
    nsp_filter = NSPFilter(model)
    
    drifts = []
    semantics = []
    
    print("\nInferring hidden states and calculating Drift...")
    for seq, lbl, rid in zip(sequences, labels, run_ids):
        z_est, _ = nsp_filter(seq.float())
        z0 = z_est[:, 0].detach().numpy()
        z1 = z_est[:, 1].detach().numpy()
        
        # Case 2 (Axis swap: z1=progress, z0=alignment) based on previous findings
        drift = 0.0
        for t in range(1, len(z0)):
            d_p = z1[t] - z1[t-1]
            d_s = z0[t] - z0[t-1]
            drift += max(0.0, d_p) * max(0.0, -d_s)
            
        drifts.append(drift)
        semantics.append(lbl)
        
    drifts = np.array(drifts)
    semantics = np.array(semantics)
    
    # 3. Calculate Pearson Correlation
    if np.std(drifts) == 0:
        print("Drift variance is 0.")
        return
        
    corr = np.corrcoef(drifts, semantics)[0, 1]
    
    print("\n" + "="*40)
    print("=== FINAL REAL-WORLD RESULTS FOR PAPER ===")
    print("="*40)
    print(f"Evaluated {len(drifts)} real trajectories (scored by GPT-4o).")
    print(f"Pearson Correlation (Deceptive Drift vs LLM-Judge Score): {corr:.4f}")
    
    if corr < -0.3:
        print("Interpretation: Strong negative correlation! High drift means low semantic correctness.")
    elif corr < -0.1:
        print("Interpretation: Moderate negative correlation found.")
    else:
        print("Interpretation: Weak or no correlation. Need more multi-round data.")

if __name__ == "__main__":
    main()