
import torch
from model import NeuralLyapunovController
from loss import LyapunovRisk
from matplotlib import pyplot as plt
from lyapunov_dataset import KSTARLyapunovDataset
class Trainer():
    def __init__(self, model, lr, optimizer, loss_fn):
        self.model = model
        self.lr = lr
        self.optimizer = optimizer
        self.lyapunov_loss = loss_fn
    
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

    def train(self, X, x_0, epochs=2000, verbose=False, every_n_epochs=10, check_approx=False):
        model.train()
        valid = False
        loss_list = []
        approx_loss_list = []
        if check_approx == True:
            env = gym.make('CartPole-v1')

        for epoch in range(1, epochs+1):
            if valid == True:
                if verbose:
                    print('Found valid solution.')
                break

            # zero gradients
            optimizer.zero_grad()

            # get lyapunov function and input from model
            V_candidate, u = self.model(X)
            # get lyapunov function evaluated at equilibrium point
            V_X0, u_X0 = self.model(x_0)
            # Compute lie derivative of V : L_V = ∑∂V/∂xᵢ*fᵢ
            f = f_value(X, u)
            L_V = self.get_lie_derivative(X, V_candidate, f)
            # get loss
            loss = self.lyapunov_loss(V_candidate, L_V, V_X0)
            
            # compute approximate f_dot and compare to true f
            if check_approx == True:
                X_prime = step(X, u, env)
                f_approx = approx_f_value(X, X_prime, dt=0.02)

                # check dx/dt estimates are close
                # epsilon for x_dot. cart velocity and angular velocity are easier to approximate than accelerations.
                # TODO is there a better way to approximate without running throught the simulator multiple times?
                epsilon = torch.tensor([1e-4, 10., 1e-4, 10.])

                assert(torch.all(abs(f - f_approx) < epsilon))

                # could replace loss function 
                L_V_approx = self.get_lie_derivative(X, V_candidate, f_approx)
                approx_loss = self.lyapunov_loss(V_candidate, L_V_approx, V_X0)
                approx_loss_list.append(approx_loss.item())
            

            loss_list.append(loss.item())
            loss.backward()
            self.optimizer.step() 
            if verbose and (epoch % every_n_epochs == 0):
                print('Epoch:\t{}\tLyapunov Risk: {:.4f}'.format(epoch, loss.item()))

            # TODO Add in falsifier here
            # add counterexamples

        if check_approx == True:
            return loss_list, approx_loss_list
        else:
            return loss_list
    
def load_model():
    d_in, n_hidden, d_out = 39, 6, 1
    controller = NeuralLyapunovController(d_in, n_hidden, d_out)
    return controller

def approx_f_value(X, X_prime, dt=0.1):
    # Approximate f value with S, a, S'
    y = (X_prime - X) / dt
    return y

def plot_losses(true_loss, approx_loss):
    fig = plt.figure(figsize=(8, 6))
    x = range(len(true_loss))
    plt.plot(x, true_loss, label='True Loss')
    plt.plot(x, approx_loss, label='Approximate Loss')
    plt.ylabel('Lyapunov Risk', size=16)
    plt.xlabel('Epochs', size=16)
    plt.grid()
    plt.legend()
    plt.savefig('results/loss_comparison.png')
    
if __name__ == '__main__':
    ### load model and training pipeline with initialized LQR weights ###
    model = load_model()
    lr = 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = LyapunovRisk(lyapunov_factor=1., lie_factor=1., equilibrium_factor=1.)
    trainer = Trainer(model, lr, optimizer, loss_fn)

    ### Generate random training data ###
    print('Loading dataset...')
    data = KSTARLyapunovDataset.load('trajectories.npz')
    print(len(data), len(data[0]))
    # TODO: Find equilibrium for KSTAR env
    X_0 = torch.zeros(39)

    # ### Start training process ##
    # approx = True # calculate lie derivative when system dynamics are unknown (this model compares the approximate f to the ground truth)
    # true_loss, approx_loss = trainer.train(X, X_0, epochs=200, verbose=True, check_approx=approx)
    # plot_losses(true_loss, approx_loss)