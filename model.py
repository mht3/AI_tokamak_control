import torch

class NeuralLyapunovController(torch.nn.Module):
    
    def __init__(self, n_input, n_hidden, n_output):
        super(NeuralLyapunovController, self).__init__()

        self.layer1 = torch.nn.Linear(n_input, n_hidden)
        self.layer2 = torch.nn.Linear(n_hidden, 1)
        self.activation = torch.nn.Tanh()
        # control layer should have same dimensions as input
        self.control = torch.nn.Linear(n_input, n_output, bias=False)

    def forward(self,x):
        h_1 = self.activation(self.layer1(x))
        # V is potential lyapunov function
        V = self.activation(self.layer2(h_1))
        u = self.control(x)
        return V, u

if __name__ == '__main__':
    # test out function
    # number of samples
    N = 1
    # inputs 
    D_in = 4            # input dimension
    H1 = 6              # hidden dimension
    # dimension of U and lyapunov output
    D_out = 1           # output dimension

    x = torch.Tensor(N, D_in).uniform_(-6, 6)       
    
    controller = NeuralLyapunovController(D_in, H1, D_out)
    V, u = controller(x)

    print(V)
    print(u)