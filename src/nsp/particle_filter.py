import torch
import torch.nn as nn
import math
import os
import sys

# Add project root to path to find the 'src' module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
class NSPParticleFilter(nn.Module):
    """
    Algorithm 4: Particle Filter (Bootstrap Filter / SIR)
    A non-linear, non-Gaussian alternative to the Kalman Filter.
    """
    def __init__(self, model, num_particles=500):
        super(NSPParticleFilter, self).__init__()
        self.model = model
        self.num_particles = num_particles

    def forward(self, Y):
        """
        Runs the particle filter for a given sequence.
        Y shape: (T, obs_dim)
        """
        T = Y.shape[0]
        state_dim = self.model.state_dim
        obs_dim = self.model.obs_dim
        dtype = Y.dtype

        # Extract parameters
        A = self.model.A
        C = self.model.C
        Q_std = torch.exp(0.5 * self.model.log_Q)  # Standard deviation for process noise
        R_std = torch.exp(0.5 * self.model.log_R)  # Standard deviation for observation noise

        # 1. Initialize Particles (Sample from prior)
        # Shape: (num_particles, state_dim)
        particles = torch.randn(self.num_particles, state_dim, dtype=dtype)
        
        log_likelihoods = []
        estimated_states = []

        for t in range(T):
            # 2. Prediction Step: Move particles according to transition model
            # z_t = A * z_{t-1} + noise
            noise_pred = torch.randn_like(particles) * Q_std
            particles = torch.matmul(particles, A.t()) + noise_pred

            # 3. Weighting Step: Evaluate observation likelihood
            # y_t = C * z_t + noise
            y_pred = torch.matmul(particles, C.t())  # Shape: (num_particles, obs_dim)
            residual = Y[t].unsqueeze(0) - y_pred    # Shape: (num_particles, obs_dim)
            
            # Calculate log-weights (Gaussian likelihood)
            # w_i = exp(-0.5 * ||residual||^2 / R^2)
            log_weights = -0.5 * torch.sum((residual / R_std) ** 2, dim=1)
            
            # Normalize weights (Softmax trick for numerical stability)
            max_log_w = torch.max(log_weights)
            weights = torch.exp(log_weights - max_log_w)
            sum_weights = torch.sum(weights)
            
            if sum_weights == 0 or torch.isnan(sum_weights):
                # Fallback if all weights collapse to zero
                weights = torch.ones_like(weights) / self.num_particles
            else:
                weights = weights / sum_weights

            # 4. Log-Likelihood estimation
            # p(y_t | y_{1:t-1}) ≈ mean of unnormalized weights
            ll_t = max_log_w + torch.log(sum_weights / self.num_particles)
            log_likelihoods.append(ll_t)
            
            # 5. State Estimation (Weighted mean)
            z_est = torch.sum(particles * weights.unsqueeze(1), dim=0)
            estimated_states.append(z_est)

            # 6. Resampling Step (Systematic Resampling to avoid particle degeneracy)
            if t < T - 1:
                positions = (torch.arange(self.num_particles, dtype=dtype) + torch.rand(1, dtype=dtype)) / self.num_particles
                cumsum = torch.cumsum(weights, dim=0)
                cumsum[-1] = 1.0  # Ensure exact 1.0 to avoid index out of bounds
                
                # Find indices for resampling
                indices = torch.searchsorted(cumsum, positions)
                particles = particles[indices]

        estimated_states = torch.stack(estimated_states)
        total_log_likelihood = torch.stack(log_likelihoods).sum()
        
        return estimated_states, total_log_likelihood

# --- Test the Particle Filter on Synthetic Data ---
if __name__ == "__main__":
    print("Testing NSP Particle Filter...")
    
    # Load synthetic data
    import json
    import os
    from src.nsp.model import NSPModel
    
    synth_file = r"D:\raz\razieh\data\synthetic_sequences\synthetic_run_0.json"
    if not os.path.exists(synth_file):
        print("Synthetic data not found. Please run generate_synthetic.py first.")
        exit()
        
    with open(synth_file, 'r') as f:
        data = json.load(f)
    Y = torch.tensor(data["Y"], dtype=torch.float32)
    
    # Initialize Model and Particle Filter
    model = NSPModel(state_dim=2, obs_dim=21)
    pf = NSPParticleFilter(model, num_particles=1000)
    
    # Run Filter
    states, log_lik = pf(Y)
    
    print(f"Input sequence length (T): {Y.shape[0]}")
    print(f"Estimated hidden states shape: {states.shape}")
    print(f"Total Log-Likelihood: {log_lik.item():.4f}")
    print("Particle Filter executed successfully without errors!")