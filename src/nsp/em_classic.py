import torch
import os
import json
import numpy as np

def run_classical_em(data_dir, state_dim=2, obs_dim=21, epochs=30):
    print(f"\n=== Starting Classical EM on {data_dir} ===")
    
    # 1. Load Data
    sequences = []
    for file in os.listdir(data_dir):
        if file.endswith(".json"):
            with open(os.path.join(data_dir, file), 'r') as f:
                data = json.load(f)
            if len(data["Y"]) > 0:
                sequences.append(torch.tensor(data["Y"], dtype=torch.float64)) # Use float64 for EM precision
    
    N = len(sequences)
    if N == 0:
        print("No sequences found!")
        return

    # 2. Initialize Parameters randomly
    torch.manual_seed(42)
    A = torch.eye(state_dim, dtype=torch.float64) + 0.01 * torch.randn(state_dim, state_dim, dtype=torch.float64)
    C = torch.randn(obs_dim, state_dim, dtype=torch.float64)
    Q = torch.eye(state_dim, dtype=torch.float64) * 0.1
    R = torch.eye(obs_dim, dtype=torch.float64) * 0.1
    
    def forward_backward(Y, A, C, Q, R):
        """E-step: Kalman Filter + RTS Smoother"""
        T = Y.shape[0]
        state_dim = A.shape[0]
        
        # Forward (Filter)
        z_pred = torch.zeros(T, state_dim, dtype=torch.float64)
        P_pred = torch.zeros(T, state_dim, state_dim, dtype=torch.float64)
        z_filt = torch.zeros(T, state_dim, dtype=torch.float64)
        P_filt = torch.zeros(T, state_dim, state_dim, dtype=torch.float64)
        
        z = torch.zeros(state_dim, dtype=torch.float64)
        P = torch.eye(state_dim, dtype=torch.float64)
        
        log_likelihood = 0.0
        
        for t in range(T):
            # Predict
            z_p = torch.matmul(A, z)
            P_p = torch.matmul(A, torch.matmul(P, A.t())) + Q
            P_p = 0.5 * (P_p + P_p.t())
            
            z_pred[t] = z_p
            P_pred[t] = P_p
            
            # Update
            y_pred = torch.matmul(C, z_p)
            residual = Y[t] - y_pred
            S = torch.matmul(C, torch.matmul(P_p, C.t())) + R
            S = 0.5 * (S + S.t()) + 1e-8 * torch.eye(C.shape[0], dtype=torch.float64)
            
            K = torch.linalg.solve(S, torch.matmul(C, P_p)).t()
            z = z_p + torch.matmul(K, residual)
            P = P_p - torch.matmul(K, torch.matmul(C, P_p))
            
            z_filt[t] = z
            P_filt[t] = 0.5 * (P + P.t())
            
            # Log-Likelihood
            sign, logdet = torch.slogdet(S)
            mahalanobis = torch.matmul(residual, torch.linalg.solve(S, residual))
            log_likelihood += -0.5 * (logdet + mahalanobis + len(residual) * torch.log(torch.tensor(2 * torch.pi, dtype=torch.float64)))
            
        # Backward (Smoother)
        z_smooth = torch.zeros(T, state_dim, dtype=torch.float64)
        P_smooth = torch.zeros(T, state_dim, state_dim, dtype=torch.float64)
        P_cross = torch.zeros(T-1, state_dim, state_dim, dtype=torch.float64) # E[z_t z_{t-1}^T]
        
        z_smooth[-1] = z_filt[-1]
        P_smooth[-1] = P_filt[-1]
        
        for t in range(T - 2, -1, -1):
            J = torch.matmul(P_filt[t], torch.matmul(A, torch.linalg.inv(P_pred[t+1])))
            z_smooth[t] = z_filt[t] + torch.matmul(J, z_smooth[t+1] - z_pred[t+1])
            P_smooth[t] = P_filt[t] + torch.matmul(J, torch.matmul(P_smooth[t+1] - P_pred[t+1], J.t()))
            P_cross[t] = torch.matmul(P_smooth[t+1], J.t())
            
        return z_smooth, P_smooth, P_cross, log_likelihood

    # 3. EM Loop
    history = []
    for epoch in range(epochs):
        # Initialize accumulators for M-step
        sum_E_zz = torch.zeros(state_dim, state_dim, dtype=torch.float64)
        sum_E_zz_prev = torch.zeros(state_dim, state_dim, dtype=torch.float64)
        sum_E_z_z_prev = torch.zeros(state_dim, state_dim, dtype=torch.float64)
        sum_yz = torch.zeros(obs_dim, state_dim, dtype=torch.float64)
        sum_yy = torch.zeros(obs_dim, obs_dim, dtype=torch.float64)
        sum_transitions = 0
        total_ll = 0.0
        
        # E-step
        for Y in sequences:
            T = Y.shape[0]
            z_s, P_s, P_c, ll = forward_backward(Y, A, C, Q, R)
            total_ll += ll
            
            for t in range(T):
                E_zz = P_s[t] + torch.outer(z_s[t], z_s[t])
                sum_E_zz += E_zz
                sum_yz += torch.outer(Y[t], z_s[t])
                sum_yy += torch.outer(Y[t], Y[t])
                
            for t in range(T-1):
                E_z_z_prev = P_c[t] + torch.outer(z_s[t+1], z_s[t])
                sum_E_z_z_prev += E_z_z_prev
                sum_E_zz_prev += P_s[t] + torch.outer(z_s[t], z_s[t])
                
            sum_transitions += (T - 1)
            
        # M-step (Closed-form updates)
        A = torch.matmul(sum_E_z_z_prev, torch.linalg.inv(sum_E_zz_prev))
        
        # Q update
        Q = (sum_E_zz - torch.matmul(A, sum_E_z_z_prev.t()) - torch.matmul(sum_E_z_z_prev, A.t()) + torch.matmul(A, torch.matmul(sum_E_zz_prev, A.t())))
        Q = Q / sum_transitions
        Q = 0.5 * (Q + Q.t()) + 1e-8 * torch.eye(state_dim, dtype=torch.float64)
        
        # C update
        C = torch.matmul(sum_yz, torch.linalg.inv(sum_E_zz))
        
        # R update
        R = (sum_yy - torch.matmul(C, sum_yz.t()) - torch.matmul(sum_yz, C.t()) + torch.matmul(C, torch.matmul(sum_E_zz, C.t())))
        R = R / (N * sequences[0].shape[0]) # Simplified: assumes similar lengths
        R = 0.5 * (R + R.t()) + 1e-8 * torch.eye(obs_dim, dtype=torch.float64)
        
        avg_ll = total_ll / N
        history.append({"epoch": epoch+1, "log_likelihood": avg_ll.item()})
        print(f"Epoch {epoch+1}/{epochs} - Log-Likelihood: {avg_ll:.4f}")
        
    return A, C, history

if __name__ == "__main__":
    # Test on Synthetic Data to verify parameter recovery
    synth_dir = r"D:\raz\razieh\data\synthetic_sequences"
    A_true = [[0.9, 0.1], [-0.1, 0.9]]
    
    A_learned, C_learned, hist = run_classical_em(synth_dir, epochs=30)
    
    print("\n--- EM Results ---")
    print("Learned Transition Matrix (A):")
    print(A_learned.numpy())
    print("\nTrue Transition Matrix (A_true):")
    print(A_true)
    
    ll_increase = hist[-1]["log_likelihood"] - hist[0]["log_likelihood"]
    print(f"\nTotal LL Increase: {ll_increase:.4f}")