import torch

def get_action(x_i):
    '''
    Take in single state x_i -> plicy -> get action
    '''
    pass

def get_next_state():
    pass

def get_trajectory(x_i, trajectory_length):
    pi_s = get_action(x_i)
    # get action
    # get next state
    for j in range(trajectory_length - 1):
        # perturb state
        # get action
        # get next state
        pass

    # return trajectory

def load_state(N):
    target_mins = [0.8, 4.0, 0.80]
    target_maxs = [2.1, 7.0, 1.05]
    # bounds for position, velocity, angle, and angular velocity

    # X: Nx3 tensor of initial states
    X = torch.empty(N, 0)
    for i in range(len(target_mins)):
        t_min = target_mins[i]
        t_max = target_maxs[i]
        x = torch.Tensor(N, 1).uniform_(t_min, t_max)
        X = torch.cat([X, x], dim=1)

    return X

if __name__ == '__main__':

    N=500
    X = load_state(N)
    print(X.shape)
    print(X[0:4, :])

    # get 3 (s, pi(s), s') pairs
    trajectory_length = 2

    for i in range(N):
        # length 3 vector for state
        x_i = X[i, :]
        trajectory = get_trajectory(x_i, trajectory_length)
