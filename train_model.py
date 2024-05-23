
import torch
from model import NeuralLyapunovController
from loss import LyapunovRisk
from matplotlib import pyplot as plt
from lyapunov_dataset import LyapunovDataset
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

class Trainer():
    def __init__(self, model, lr, optimizer, loss_fn, loss_mode,
                 seed=42):
        self.model = model
        self.lr = lr
        self.optimizer = optimizer
        self.lyapunov_loss = loss_fn
        self.loss_mode = loss_mode
        self.seed = seed

    def get_lie_derivative(self, X, V_candidate, f):
        '''
        Calculates L_V = ∑∂V/∂xᵢ*fᵢ
        '''
        w1 = self.model.layer1.weight
        b1 = self.model.layer1.bias
        w2 = self.model.layer2.weight
        b2 = self.model.layer2.bias
        # running through model again 
        z1 = X @ w1.t() + b1
        a1 = torch.tanh(z1)
        z2 = a1 @ w2.t() + b2
        d_z2 = 1. - V_candidate**2
        partial_z2_a1 = w2
        partial_a1_z1 = 1 - torch.tanh(z1)**2
        partial_z1_x = w1

        d_a1 = (d_z2 @ partial_z2_a1)
        d_z1 = d_a1 * partial_a1_z1

        # gets final ∂V/∂x
        d_x = d_z1 @ partial_z1_x

        lie_derivative = torch.diagonal((d_x @ f.t()), 0)
        return lie_derivative
    
    def get_approx_lie_derivative(self, V_candidate, V_candidate_next, dt=0.1):
        '''
        Calculates L_V = ∑∂V/∂xᵢ*fᵢ by forward finite difference
                    L_V = (V' - V) / dt
        '''
        return (V_candidate_next - V_candidate) / dt

    def get_dataloader(self, dataset, batch_size):
        train_loader = DataLoader(dataset, batch_size=batch_size,
                                shuffle=True, generator=torch.Generator().manual_seed(self.seed))
        return train_loader
    
    def _train(self, train_loader, x_0):
        self.model.train()
        total_loss = 0.
        count = 0.
        t_loader = tqdm(enumerate(train_loader), desc='train', total=len(train_loader), leave=False)
        for batch_idx, data in t_loader:
            x, pi, x_prime = data

            # zero gradients
            self.optimizer.zero_grad()

            # get lyapunov function and input from model
            V_candidate, u = self.model(x)
            # get lyapunov function evaluated at equilibrium point
            V_X0, u_X0 = self.model(x_0)

            # get loss
            if self.loss_mode == 'approx_dynamics':
                # compute approximate f_dot and compare to true f
                f_approx = approx_f_value(x, x_prime, dt=0.1)
                L_V_approx = self.get_lie_derivative(x, V_candidate, f_approx)
                loss = self.lyapunov_loss(V_candidate, L_V_approx, V_X0)

            elif self.loss_mode == 'approx_lie':
                # compute approximate f_dot and compare to true f
                V_candidate_prime, u_prime = self.model(x_prime)
                L_V_approx = self.get_approx_lie_derivative(V_candidate, V_candidate_prime, dt=0.1)
                loss = self.lyapunov_loss(V_candidate, L_V_approx, V_X0)

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            count += 1
            t_loader.set_postfix(loss=loss.item())
        # return avg loss over all runs
        return total_loss / count
    
    def train(self, dataset, x_0, epochs=2000, batch_size=64, verbose=False, every_n_epochs=10):
        valid = False
        loss_list = []
        train_loader = self.get_dataloader(dataset, batch_size=batch_size)
        pbar = trange(epochs, desc='progress')
        for epoch in pbar:
            if valid == True:
                # TODO Update valid once falsifier is used
                if verbose:
                    print('Found valid solution.')
                break
            loss = self._train(train_loader, x_0)

            loss_list.append(loss)
            if verbose and (epoch+1 % every_n_epochs == 0):
                print('Epoch:\t{}\tLyapunov Risk: {:.4f}'.format(epoch+1, loss))

            # TODO Add in falsifier here
            # add counterexamples
            metrics = {'loss' : loss}
            pbar.set_postfix(metrics)
        return loss_list
    
def load_model(d_in=39, n_hidden=16, d_out=9):
    controller = NeuralLyapunovController(d_in, n_hidden, d_out)
    return controller

def approx_f_value(X, X_prime, dt=0.1):
    # Approximate f value with S, a, S'
    y = (X_prime - X) / dt
    return y

def plot_losses(approx_dynamics_loss, approx_lie_loss):
    fig = plt.figure(figsize=(8, 6))
    x = range(len(approx_dynamics_loss))
    plt.plot(x, approx_dynamics_loss, label='Approximate Dynamics Loss')
    plt.plot(x, approx_lie_loss, label='Approximate Lie Derivative Loss')

    plt.ylabel('Lyapunov Risk', size=16)
    plt.xlabel('Epochs', size=16)
    plt.grid()
    plt.legend()
    plt.savefig('loss_comparison.png')
    
if __name__ == '__main__':
    lr = 0.01
    batch_size=500
    epochs=50
    loss_fn = LyapunovRisk(lyapunov_factor=1., lie_factor=1., equilibrium_factor=1.)

    state_dim = 39
    action_dim = 9
    
    ### Load training data ###
    print('Loading dataset...')
    lyapunov_dataset = LyapunovDataset(filename='trajectories.npz', 
                                       state_dim=state_dim, action_dim=action_dim)

    # TODO: Find equilibrium for KSTAR env
    X_0 = torch.zeros(state_dim)

    print("Training with approx dynamics loss...")
    d_in, n_hidden, d_out = 39, 16, 9

    model_1 = load_model(d_in, n_hidden, d_out)
    optimizer_1 = torch.optim.Adam(model_1.parameters(), lr=lr)
    trainer_1 = Trainer(model_1, lr, optimizer_1, loss_fn, loss_mode='approx_dynamics')
    # calculate lie derivative when system dynamics are unknown (this model compares the approximate f to the ground truth)
    approx_dynamics_loss = trainer_1.train(lyapunov_dataset, X_0, epochs=epochs,
                                           batch_size=batch_size, verbose=True)

    print("Training with approx lie derivative loss...")
    model_2 = load_model()
    optimizer_2 = torch.optim.Adam(model_2.parameters(), lr=lr)
    trainer_2 = Trainer(model_2, lr, optimizer_2, loss_fn, loss_mode='approx_lie')
    # calculate lie derivative when system dynamics are unknown (this model compares the approximate f to the ground truth)
    approx_lie_loss = trainer_2.train(lyapunov_dataset, X_0, epochs=epochs,
                                      batch_size=batch_size, verbose=True)
    
    plot_losses(approx_dynamics_loss, approx_lie_loss)