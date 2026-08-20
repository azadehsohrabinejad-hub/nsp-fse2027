import torch
import torch.nn as nn

class NSPModel(nn.Module):
    """
    NSP State-Space Model (Linear Gaussian Baseline).
    z_t = A * z_{t-1} + w_t  (Hidden state transition)
    y_t = C * z_t + v_t      (Observation generation)
    """
    def __init__(self, state_dim=4, obs_dim=21):
        super(NSPModel, self).__init__()
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        
        # 1. Transition Matrix (A) - How hidden state evolves
        # Initialized as identity matrix (assuming state persists)
        self.A = nn.Parameter(torch.eye(state_dim))
        
        # 2. Observation Matrix (C) - Maps hidden state to 21 features
        # Initialized randomly
        self.C = nn.Parameter(torch.randn(obs_dim, state_dim) * 0.1)
        
        # 3. Process Noise Covariance (Q) - w_t ~ N(0, Q)
        # Using log-parameterization to ensure positive variance
        self.log_Q = nn.Parameter(torch.zeros(state_dim))
        
        # 4. Observation Noise Covariance (R) - v_t ~ N(0, R)
        self.log_R = nn.Parameter(torch.zeros(obs_dim))

    def forward(self, z_prev):
        """
        Predicts the next hidden state.
        z_t = A * z_{t-1}
        """
        # Apply transition matrix
        z_pred = torch.matmul(z_prev, self.A.t())
        return z_pred

# Test the model definition
if __name__ == "__main__":
    print("Initializing NSP Model...")
    model = NSPModel(state_dim=4, obs_dim=21)
    
    # Create a dummy hidden state (batch_size=1, state_dim=4)
    z_prev = torch.tensor([[0.5, -0.5]], dtype=torch.float32)
    
    # Predict next state
    z_next = model(z_prev)
    
    print(f"Previous State (z_1): {z_prev}")
    print(f"Predicted Next State (z_2): {z_next}")
    print(f"Observation Matrix (C) shape: {model.C.shape}")
    print("Model initialized successfully!")