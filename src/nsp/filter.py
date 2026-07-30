import torch
import torch.nn as nn
from src.nsp.model import NSPModel

class NSPFilter(nn.Module):
    """
    Algorithm 2: NSP-FILTER (Kalman Filter implementation)
    Estimates hidden states z_t given observations y_t.
    """
    def __init__(self, model: NSPModel):
        super(NSPFilter, self).__init__()
        self.model = model

    def forward(self, Y):
        """
        Runs the forward pass (filtering) for a given sequence.
        Y shape: (T, obs_dim)
        """
        T = Y.shape[0]
        state_dim = self.model.state_dim
        
        # Initialize hidden state (z_0) and covariance (P_0)
        z = torch.zeros(state_dim, dtype=Y.dtype)
        P = torch.eye(state_dim, dtype=Y.dtype) * 1.0  # Initial uncertainty
        
        # Extract parameters from model and ensure positivity for variances
        Q = torch.diag(torch.exp(self.model.log_Q))  # Process noise covariance
        R = torch.diag(torch.exp(self.model.log_R))  # Observation noise covariance
        A = self.model.A
        C = self.model.C
        
        estimated_states = []
        log_likelihoods = []
        
        for t in range(T):
            # --- Predict Step ---
            z_pred = torch.matmul(A, z)
            P_pred = torch.matmul(torch.matmul(A, P), A.t()) + Q
            
            # --- Update Step ---
            y_pred = torch.matmul(C, z_pred)
            y_obs = Y[t]
            
            # Innovation (residual)
            residual = y_obs - y_pred
            
            # Innovation covariance
            S = torch.matmul(torch.matmul(C, P_pred), C.t()) + R
            
            # Kalman Gain
            # We want to solve S * K^T = C * P_pred to get K^T, then transpose it
            K_t = torch.linalg.solve(S, torch.matmul(C, P_pred))
            K = K_t.t()
            
            # Updated state and covariance
            z = z_pred + torch.matmul(K, residual)
            P = P_pred - torch.matmul(torch.matmul(K, C), P_pred)
            
            estimated_states.append(z)
            
            # Calculate log-likelihood for EM algorithm later
            log_det_S = torch.logdet(S)
            mahalanobis = torch.matmul(residual, torch.linalg.solve(S, residual))
            log_likelihood = -0.5 * (log_det_S + mahalanobis + self.model.obs_dim * torch.log(torch.tensor(2 * torch.pi)))
            log_likelihoods.append(log_likelihood)
            
        # Stack states to shape (T, state_dim)
        estimated_states = torch.stack(estimated_states)
        total_log_likelihood = torch.stack(log_likelihoods).sum()
        
        return estimated_states, total_log_likelihood

# --- Test the Filter on Synthetic Data ---
if __name__ == "__main__":
    print("Testing NSP-FILTER on synthetic data...")
    
    # 1. Load synthetic data
    import json
    import os
    synth_file = r"D:\raz\razieh\data\synthetic_sequences\synthetic_run_0.json"
    with open(synth_file, 'r') as f:
        data = json.load(f)
    Y = torch.tensor(data["Y"], dtype=torch.float32)
    
    # 2. Initialize Model and Filter
    # Note: The model parameters (A, C) are random right now, not trained.
    model = NSPModel(state_dim=2, obs_dim=21)
    nsp_filter = NSPFilter(model)
    
    # 3. Run Filter
    states, log_lik = nsp_filter(Y)
    
    print(f"Input sequence length (T): {Y.shape[0]}")
    print(f"Estimated hidden states shape: {states.shape}")
    print(f"Total Log-Likelihood: {log_lik.item():.4f}")
    print("Filter executed successfully without errors!")