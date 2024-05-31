import torch
from common.model_structure import *
from kstar_env import KSTAREnv

class Falsifier():
    def __init__(self, lower_bound, upper_bound, epsilon=0., scale=0.1, frequency=100, num_samples=10, env=None):
        self.epsilon = epsilon
        self.counterexamples_added = 0
        self.lower_bound = torch.Tensor(lower_bound)
        self.upper_bound = torch.Tensor(upper_bound)
        self.scale = scale
        self.frequency = frequency
        self.num_samples = num_samples
        # AI Gym environment. Used to get min and max state values and optionally get next_state
        self.env = KSTAREnv()
        self.load_rl_model()


    ## Copied from "generate_lyapunov_dataset.py"
    def load_rl_model(self):
        # Load RL model
        base_path = os.path.abspath(os.path.dirname(sys.argv[0]))
        # RL model (stable baselines TD3)
        rl_model_path = os.path.join(base_path, 'weights', 'rl', 'rt_control',
                                    '3frame_v220505', 'best_model.zip')
        self.rl_model = SB2_model(
            model_path = rl_model_path, 
            low_state = self.env.low_state, 
            high_state = self.env.high_state, 
            low_action = self.env.low_action, 
            high_action = self.env.high_action, 
            activation='relu', 
            last_actv='tanh', 
            norm=True, 
            bavg=0.0
        )
        # initialize previous action to low action (required input to this RL model)
        self.action = np.array(self.env.low_action)
    
    ## Copied from "generate_lyapunov_dataset.py"
    def get_action(self, x):
        '''
        Take in single observation x to get action 
        '''
        # this RL model needs the previous action as input, so we store a class variable for action
        self.action = self.rl_model.predict(x, yold=self.action)
        return self.action
    
    ## Copied from "generate_lyapunov_dataset.py"
    def get_next_state(self, x, pi):
        '''
        Generates next state given current state and current action
        x: current state
        pi: current action based on policy
        '''
        # take step in environment (0.1 seconds)
        # environment already knows about current state x so we don't need to use it here
        observation, reward, terminated, truncated, info = self.env.step(pi)
        return observation


    def get_frequency(self):
        return self.frequency

    @torch.no_grad
    def check_lyapunov(self, X, V_candidate, L_V):    
        '''
        Checks if the lyapunov conditions are violated for any sample. 
        Data points that are unsatisfiable will be sampled and added 
        '''
        N = X.shape[0]

        # Ensure lyapunov function and lie derivative are 1D tensors
        if V_candidate.dim() != 1:
            V_candidate = V_candidate.reshape(N)
        if L_V.dim() != 1:
            L_V = L_V.reshape(N)
        
        lyapunov_mask = (V_candidate < 0.)
        lie_mask = (L_V > self.epsilon)

        # bitwise or for falsification conditions
        union = lyapunov_mask.logical_or(lie_mask)

        # get batch indices that violate the lyapunov conditions
        indices = torch.nonzero(union).squeeze()
        # check num elements > 0
        if indices.numel() > 0:
            return  X[indices].reshape(indices.numel(), self.lower_bound.shape[0])
        else:
            return None
    
    @torch.no_grad
    def add_counterexamples(self, X, counterexamples):
        '''
        Take counter examples and sample points around them.
        X: current training data
        counterexamples: all new examples from training data that don't satisfy the lyapunov conditions
        '''
        self.counterexamples_added += self.num_samples
        for i in range(counterexamples.shape[0]):
            samples = torch.empty(self.num_samples, 0)
            counterexample = counterexamples[i]
            for j in range(self.upper_bound.shape[0]):
                lb = self.lower_bound[j]
                ub = self.upper_bound[j]
                value = counterexample[j]
                # TODO: Add noise to only 0D parameters close to current counterexample -- (not sure what this means?)
                # Determine the min and max values for each feature in the chosen counterexamples
                min_value = torch.max(lb, value - self.scale * (value - lb))
                max_value = torch.min(ub, value + self.scale * (ub - value))
                sample = torch.Tensor(self.num_samples, 1).uniform_(min_value, max_value)
                samples = torch.cat([samples, sample], dim=1)
            ### Moved these TODOs out here, since it seems like the above loop is still building the samples. Here, they are done being built for a given counterexample
            # TODO: Run through RL model to get actions
                # For each sample (which should be a state), call 'get_action(sample)'. Store the actions in an 'actions' tensor corresponding to the 'samples' tensor
            # TODO: run through simulator to get next state
                # For each sample state: 
                #          - Reset self.env to that state (this may require changing the env.reset() function, so that it can reset to an entire state instead of just targets/nb1?). 
                #            This way, each sample will temporarily use our single self.env
                #          - Call get_next_state(), and build out a 'next_states' tensor corresponding to the 'actions' and 'samples tensors
            # TODO: Add x, pi, x' to samples
                # Concatenate the 'samples', 'actions', and 'next_states' tensors into tuples, which can be added back into X
            X = torch.cat((X, samples), dim=0)
            # TODO: project x and x' back to 3D space
        return X
    
if __name__ == '__main__':
    # small test for falsifier
    falsifier = Falsifier(lower_bound=[-1., -0.5], upper_bound=[1., 0.5])
    x = torch.Tensor(5, 1).uniform_(-1., 1.)
    x = torch.cat((x, torch.Tensor(5, 1).uniform_(-0.5, 0.5)), dim=1)

    # Fake lyapunov funcions
    V = torch.ones(size=(5,1))
    V[3, 0] = -0.5
    V[2, 0] = -0.1

    L_V = -torch.ones(size=(5,1))
    L_V[1, 0] = 2.5

    ##
    print(f"x: {x}")
    print(f"V: {V}")
    print(f"L_V: {L_V}")

    counterexamples = falsifier.check_lyapunov(x, V, L_V)
    print(f"counterexamples: {counterexamples}")
    assert(counterexamples.size(0) == 3)
    num_samples=2
    x_new = falsifier.add_counterexamples(x, counterexamples)
    #assert(x_new.size(0) == num_samples*counterexamples.size(0) + x.size(0))
    #print(x)
    #print(x_new)
    print(f"x_new: {x_new}")