import torch

class LyapunovRisk(torch.nn.Module):
    def __init__(self, lyapunov_factor=1., lie_factor=1., equilibrium_factor=1., 
                 lie_offset=0.5):
        super(LyapunovRisk, self).__init__()
        self.relu = torch.nn.ReLU()
        self.lyapunov_factor = lyapunov_factor
        self.lie_factor = lie_factor
        self.equilibrium_factor = equilibrium_factor
        self.lie_offset = lie_offset

    def forward(self, V_candidate, L_V, V_X0):
        '''
        V_candidate: Candidate lyapunov function from model output
        L_V:         Lie derivative of lyapunov function (L_V = ∑∂V/∂xᵢ*fᵢ)
        V_X0:        Candidate lyapunov function evaluated at equilibrium conditions for state
        '''

        # lyapunov function must always be positive. Penalize negative outputs
        V_loss = self.relu(-V_candidate)
        # lie derivative must be negative 
        lie_loss = self.relu(L_V + self.lie_offset)
        # Lyapuvonv function evaluated at equilibrium points should be 0
        eq_loss = V_X0**2
        
        # weight loss factors individually
        total_risk = (self.lyapunov_factor*V_loss +  self.lie_factor*lie_loss).mean() + self.equilibrium_factor*eq_loss
        return total_risk