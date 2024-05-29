
import torch
import numpy as np
from torch.utils.data import Dataset

class LyapunovDataset(Dataset):
    def __init__(self, filename, state_dim=39, action_dim=9):
        '''
        filename:
            trajectories.npz file generated from lyapunov_dataset.py
        '''
        super().__init__()
        self.data = self.load_torch(filename)
        self.state_dim = state_dim
        self.action_dim = action_dim

    def load_torch(self, filename):
        '''
        Loads trajectories.npz as a flattened torch tensor of (s, a, s')
        '''
        loaded = np.load(filename, allow_pickle=True)
        loaded_data = [loaded[key] for key in loaded]
        states, actions, next_states = loaded_data
        dataset = np.hstack([states, actions, next_states])
        data_tensor = torch.tensor(dataset)
        return data_tensor.to(torch.float32)     
    
    def __getitem__(self, idx):
        '''
        Each row contains state, action, next_state pairs
        '''
        x = self.data[idx, :self.state_dim]
        a = self.data[idx, self.state_dim:(self.state_dim + self.action_dim)]
        x_prime = self.data[idx, -self.state_dim:]
        
        return x, a, x_prime
        
    def __len__(self):
        return self.data.shape[0]