'''
Policy Robustness check on trained TD3 RL policy (uses same model from rt_control_v2).
States: Control sliders in the GUI (Beta_p,                q_95,               l_i)
                                    poloidal plasma beta   safety factor     internal inductance
Actions: Graphs in the GUI (Ip,                K,              delta_u,                delta_l)
                    Total plasma current    Enlongation       upper triangularity     lower triangularity
'''
import os, sys, time
from matplotlib import pyplot as plt
from common.model_structure import *
from common.wall import *
from common.setting import *

class ModelTester:
    def __init__(self, rl_model_path, nn_model_path, target_params, target_init, 
                 input_params, input_init, output_params0, output_params2, verbose=True):
        self.model = self.load_agent(rl_model_path, verbose=verbose)
        # kstar_lstm
        self.kstar_lstm = self.load_sim(nn_model_path, verbose=verbose)
        self.targets = {}

        # originally targetSliderDict in rt_control
        for i, target_param in enumerate(target_params):
            self.targets[target_param] = [target_init[i], target_init[i]]

        # originally inputSliderDict in rt_control
        self.inputs = {}
        for j, input_param in enumerate(input_params):
            self.inputs[input_param] = input_init[j]

        self.new_action = np.array(self.low_action)
        seq_len = 10
        self.x = np.zeros([seq_len, 18])
        self.histories = [list(self.low_action) + list(target_init)] * self.lookback
        self.output_params0 = output_params0
        self.output_params2 = output_params2

        self.outputs = {}
        for p in output_params2:
            self.outputs[p] = [0.]
        self.input_params = input_params
        self.input_init = input_init
        # plot length in ms
        self.plot_length = 40

    def load_sim(self, nn_model_path, verbose=False):
        if verbose:
            print('Loading simulator...')
        self.kstar_nn = kstar_nn(model_path=nn_model_path, n_models=1)
        if verbose:
            print('Done.')

    def load_agent(self, rl_model_path, verbose=False):
        # how far to lookback in history
        self.lookback = 3
        # wide parameters from rt_control_v2
        self.low_target  = [0.8,4.0,0.80]
        self.high_target = [2.1,7.0,1.05]
        self.low_action  = [0.3, 0.0, 0.0,0.0, 1.6,0.15,0.5 ,1.265,2.14]
        self.high_action = [0.8,1.75,1.75,1.5,1.95, 0.5,0.85, 1.36, 2.3]
        self.low_state  = (self.low_action + self.low_target) * self.lookback + self.low_target
        self.high_state = (self.high_action + self.high_target) * self.lookback + self.high_target

        # Load agents
        if verbose:
            print('Loading model...')
        rl_model = SB2_model(
            model_path = rl_model_path, 
            low_state = self.low_state, 
            high_state = self.high_state, 
            low_action = self.low_action, 
            high_action = self.high_action, 
            activation='relu', 
            last_actv='tanh', 
            norm=True, 
            bavg=0.0
        )
        if verbose:
            print('Done.')
        return rl_model

    def step(self):
        '''
        Use data driven simulator to get next state and reward from a given action
        '''
        x = np.zeros(17)
        idx_convert = [0,1,3,4,5,6,7,8,9,10,11,12,13,14,10,2]
        for i in range(len(x)-1):
            x[i] = self.inputs[input_params[idx_convert[i]]]
        x[9], x[10] = 0.5*(x[9]+x[10]), 0.5*(x[10]-x[9])
        x[14] = 1 if x[14]>1.265+1.e-4 else 0

        # fixed year_in setting
        year_in = 2021
        x[-1] = year_in

        # predict state from simulator 
        y = self.kstar_nn.predict(x)

        # update output parameters
        for i in range(len(self.output_params0)):
            output_param = self.output_params0[i]
            if len(self.outputs[output_param]) >= self.plot_length:
                del self.outputs[output_param][0]
            elif len(self.outputs[output_param]) == 1:
                self.outputs[output_param][0] = y[i]
            self.outputs[output_param].append(y[i])
        self.x[:,:len(self.output_params0)] = y
        idx_convert = [0, 1, 2, 12, 13 ,14 ,10, 11, 3, 4, 5, 6, 10]
        for i in range(len(self.x[0]) - 1 - 4):
            self.x[:,i+4] = self.inputs[input_params[idx_convert[i]]]
        self.x[:, 11 + 4] += self.inputs[input_params[7]]
        self.x[:, 12 + 4] = 1 if self.x[-1, 12 + 4] > 1.265 + 1.e-4 else 0
        self.x[:, -1] = year_in

    def get_action(self, state, noise=True, sigma=1.):
        # Predict based upon given state
        observation = np.zeros_like(self.low_state)
        for i in range(self.lookback):
            observation[i * len(self.histories[0]) : (i + 1) * len(self.histories[0])] = self.histories[i]
        
        # get current state values (list of 3 elements in order of  Bp, q95, l_i) along with previous actions as input
        observation[self.lookback * len(self.histories[0]) :] = state
        self.new_action = self.model.predict(observation, yold=self.new_action)

        # TODO PERTURB ACTION with noise
        idx_convert = [0, 3, 4, 5, 12, 13, 14, 10, 11]
        if noise == True:
            noise = np.random.normal(0, sigma, len(idx_convert))
        else:
            noise = np.zeros(shape=len(idx_convert))

        for i, idx in enumerate(idx_convert):
            # add to simulator inputs 
            self.inputs[self.input_params[idx]] = self.new_action[i] + noise[i]

if __name__ == '__main__':
    base_path = os.path.abspath(os.path.dirname(sys.argv[0]))
    rl_model_path = os.path.join(base_path, 'weights', 'rl', 'rt_control',
                                 '3frame_v220505', 'best_model.zip')
    nn_model_path   = os.path.join(base_path, 'weights', 'nn')

    input_params = ['Ip [MA]','Bt [T]','GW.frac. [-]',\
                    'Pnb1a [MW]','Pnb1b [MW]','Pnb1c [MW]',\
                    'Pec2 [MW]','Pec3 [MW]','Zec2 [cm]','Zec3 [cm]',\
                    'In.Mid. [m]','Out.Mid. [m]','Elon. [-]','Up.Tri. [-]','Lo.Tri. [-]']
    input_init = [0.5,1.8,0.33, 1.5, 1.5, 0.5, 0.0,0.0,0.0,0.0, 1.32, 2.22,1.7,0.3,0.75]

    target_params = ['βp','q95','li']
    target_init   = [1.45, 5.5, 0.925]

    output_params0 = ['βn','q95','q0','li']
    output_params2 = ['βn','βp','h89','h98','q95','q0','li','wmhd']

    model_tester = ModelTester(rl_model_path, nn_model_path, target_params=target_params, target_init=target_init, 
                               input_params=input_params, input_init=input_init, output_params0=output_params0,
                               output_params2=output_params2, verbose=True)
    
    # bounds of state
    target_mins   = [0.8, 4.0, 0.80]
    target_maxs   = [2.1, 7.0, 1.05]

    # bounds of action
    input_mins = [0.3,1.5,0.2,  0.0, 0.0, 0.0, 0.0,0.0,-10,-10, 1.265,2.18,1.6,0.1,0.5 ]
    input_maxs = [0.8,2.7,0.6,  1.75,1.75,1.5, 0.8,0.8, 10, 10, 1.36, 2.29,2.0,0.5,0.9 ]

    model_tester.get_action(target_init, noise=True)

    # take a step to get next state (current action is updated as class variable)
    model_tester.step()

    # print(model_tester.inputs)
    # print(model_tester.targets)
