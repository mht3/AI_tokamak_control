import numpy as np
import gymnasium as gym
import sys, os
from common.model_structure import *
from common.wall import *
from common.setting import *

class KSTAREnv(gym.Env):
    '''
    AI Gym Environment for Seo et al's Date Driven Simulator framework.
    Goal is to estimate feedforward control.
    '''
    def __init__(self, verbose=False):
        '''
        params
            verbose: bool

        '''
        super(KSTAREnv).__init__()
        self.verbose = verbose
        # how far to lookback in history
        self.lookback = 3

        # wide parameters from rt_control_v2
        self.low_target  = [0.8,4.0,0.80]
        self.high_target = [2.1,7.0,1.05]
        self.low_action  = [0.3, 0.0, 0.0,0.0, 1.6,0.15,0.5 ,1.265,2.14]
        self.high_action = [0.8,1.75,1.75,1.5,1.95, 0.5,0.85, 1.36, 2.3]
        # state takes into account previous actions
        self.low_state  = (self.low_action + self.low_target) * self.lookback + self.low_target
        self.high_state = (self.high_action + self.high_target) * self.lookback + self.high_target

        # bounds of main part of state
        self.observation_space = gym.spaces.Box(np.array(self.low_state), np.array(self.high_state))

        # bounds of action
        self.action_space = gym.spaces.Box(np.array(self.low_action), np.array(self.high_action))

        # remnant of GUI (still used for Bp)
        self.plot_length = 40

        self.setup_simulator()

        self.state = self.reset()

    def set_state(self, state):
        self.state = state

    def load_sim(self, nn_model_path, lstm_model_path, bpw_model_path, num_models=1):
        if self.verbose:
            print('Loading simulator...')
        self.kstar_nn = kstar_nn(model_path=nn_model_path, n_models=1)
        self.kstar_lstm = kstar_v220505(model_path=lstm_model_path, n_models=num_models)
        self.bpw_nn = bpw_nn(model_path=bpw_model_path, n_models=num_models)
        if self.verbose:
            print('Done.')

    def setup_simulator(self):
        # setting up simulator
        base_path = os.path.abspath(os.path.dirname(sys.argv[0]))
        # nn/lstm model are simulator which output output_params0 (βn, q95, q0, li)
        nn_model_path   = os.path.join(base_path, 'weights', 'nn')
        lstm_model_path = os.path.join(base_path, 'weights', 'lstm', 'v220505')
        # final network that outputs output_params1 (βp, wmhd)
        bpw_model_path  = os.path.join(base_path, 'weights', 'bpw')
        # initialize input and target parameters
        self.input_params = ['Ip [MA]','Bt [T]','GW.frac. [-]',\
                        'Pnb1a [MW]','Pnb1b [MW]','Pnb1c [MW]',\
                        'Pec2 [MW]','Pec3 [MW]','Zec2 [cm]','Zec3 [cm]',\
                        'In.Mid. [m]','Out.Mid. [m]','Elon. [-]','Up.Tri. [-]','Lo.Tri. [-]']
        
        self.input_init = [0.5,1.8,0.33, 1.5, 1.5, 0.5, 0.0,0.0,0.0,0.0, 1.32, 2.22,1.7,0.3,0.75]

        self.target_params = ['βp','q95','li']
        self.target_init   = [1.45,5.5,0.925]    # output parameters used (output params 0 is output of simulator)
        self.output_params0 = ['βn','q95','q0','li']
        self.output_params1 = ['βp','wmhd']
        self.output_params2 = ['βn','βp','h89','h98','q95','q0','li','wmhd']
        self.dummy_params = ['Ip [MA]', 'Elon. [-]', 'Up.Tri. [-]', 'Lo.Tri. [-]', 'In.Mid. [m]', 'Out.Mid. [m]', 'Pnb1a [MW]','Pnb1b [MW]','Pnb1c [MW]']
        # kstar_lstm
        self.load_sim(nn_model_path, lstm_model_path, bpw_model_path)
        # If steady=false, an average is taken for data driven simulator outputs
        self.steady = False
        

    def run_simulator(self, action):
        '''
        Use data driven simulator to get next state and reward from a given action.
        Each run through the simulator is 0.1 seconds.
        '''
        # add predicted action from RL model to simulator inputs
        # index 3, 4, 5 are nbi powers
        action_idx_convert = [0, 3, 4, 5, 12, 13, 14, 10, 11]
        for i, idx in enumerate(action_idx_convert):
            self.inputs[self.input_params[idx]] = action[i]

        if self.steady or self.first:
            x = np.zeros(17)
            idx_convert = [0,1,3,4,5,6,7,8,9,10,11,12,13,14,10,2]
            for i in range(len(x)-1):
                x[i] = self.inputs[self.input_params[idx_convert[i]]]
            x[9], x[10] = 0.5*(x[9]+x[10]), 0.5*(x[10]-x[9])
            x[14] = 1 if x[14]>1.265+1.e-4 else 0

            # fixed year_in setting
            year_in = 2021
            x[-1] = year_in

            # predict state from simulator 
            # Predicts output_params0 (βn, q95, q0, li)
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
                self.x[:,i+4] = self.inputs[self.input_params[idx_convert[i]]]
            self.x[:, 11 + 4] += self.inputs[self.input_params[7]]
            self.x[:, 12 + 4] = 1 if self.x[-1, 12 + 4] > 1.265 + 1.e-4 else 0
            self.x[:, -1] = year_in
        else:
            self.x[:-1, len(self.output_params0):] = self.x[1:, len(self.output_params0):]
            idx_convert = [0, 1, 2, 12, 13 ,14 ,10, 11, 3, 4, 5, 6, 10]
            for i in range(len(self.x[0])-1-4):
                self.x[-1,i+4] = self.inputs[self.input_params[idx_convert[i]]]
            self.x[-1, 11 + 4] += self.inputs[self.input_params[7]]
            self.x[-1, 12 + 4] = 1 if self.x[-1, 12 + 4] > 1.265 + 1.e-4 else 0

            # run through sim with averaged results instead of only 1 output
            y = self.kstar_lstm.predict(self.x)
            
            self.x[:-1,:len(self.output_params0)] = self.x[1:,:len(self.output_params0)]
            self.x[-1,:len(self.output_params0)] = y
            for i in range(len(self.output_params0)):
                output_param = self.output_params0[i]
                if len(self.outputs[output_param]) >= self.plot_length:
                    del self.outputs[output_param][0]
                elif len(self.outputs[output_param]) == 1:
                    self.outputs[output_param][0] = y[i]
                self.outputs[output_param].append(y[i])

        
        # Predict output_params1 (βp, wmhd)
        x = np.zeros(8)
        idx_convert = [0,0,1,10,11,12,13,14]
        x[0] = self.outputs['βn'][-1]
        for i in range(1,len(x)):
            x[i] = self.inputs[self.input_params[idx_convert[i]]]
        x[3],x[4] = 0.5*(x[3]+x[4]),0.5*(x[4]-x[3])
        y = self.bpw_nn.predict(x)
        for i in range(len(self.output_params1)):
            output_param = self.output_params1[i]
            if len(self.outputs[output_param]) >= self.plot_length:
                del self.outputs[output_param][0]
            elif len(self.outputs[output_param]) == 1:
                self.outputs[output_param][0] = y[i]
            self.outputs[output_param].append(y[i])

        
        # Store dummy parameters
        for p in self.dummy_params:
            if len(self.dummy[p]) >= self.plot_length:
                del self.dummy[p][0]
            elif len(self.dummy[p]) == 1:
                self.dummy[p][0] = self.inputs[p]
            self.dummy[p].append(self.inputs[p])
        

        # update targets list
        if not self.first:
            for i,target_param in enumerate(self.target_params):
                if len(self.targets[target_param]) >= self.plot_length:
                    del self.targets[target_param][0]
                elif len(self.targets[target_param]) == 1:
                    self.targets[target_param][0] = self.targets[target_param][-1]

                self.targets[target_param].append(self.targets[target_param][-1])


        # Update histories based on action and predicted outputs
        self.histories[:-1] = self.histories[1:]
        self.histories[-1] = list(action) + list([self.outputs['βp'][-1], self.outputs['q95'][-1], self.outputs['li'][-1]])

        # Use robust simulator after first step
        if self.first:
            self.first = False
        
        # get state and return
        self.state = self.get_state()
        return self.state

    def calculate_reward(self):
        '''
        Get reward based on current state
        '''
        if np.isclose(self.fusion_duration, 0):
            # negative rmse as reward
            eps_bp, eps_q95, eps_li = 0.2, 0.5, 0.05
            
            # get rmse over entire state (includes lookback)
            y = np.empty(shape=(3, self.lookback))
            for i in range(self.lookback):
                y[i, :] = self.state[i * len(self.histories[0]) : (i + 1) * len(self.histories[0])][-3:]

            error_bp = y[:, 0] - self.target_init[0]
            rms_bp = np.sqrt(np.mean((error_bp / eps_bp)**2))

            error_q95 = y[:, 1] - self.target_init[1]
            rms_q95 = np.sqrt(np.mean((error_q95 / eps_q95)**2))

            error_li = y[:, 2] - self.target_init[2]
            rms_li = np.sqrt(np.mean((error_li / eps_li)**2))
            return - rms_bp - rms_q95 - rms_li
        
        else:
            # 0 reward if not at end of episode
            return 0.
        
    def step(self, action):
        # apply action to state via simulator
        self.state = self.run_simulator(action)

        # each step in simulator takes 0.1 seconds
        self.fusion_duration -= 0.1
        reward = self.calculate_reward()

        if self.fusion_duration <= 0.:
            terminated = True
        else:
            terminated = False
        
        truncated = False
        info = {}
        return self.state, reward, terminated, truncated, info

    def get_state(self):
        # get current state values (list of 3 elements in order of  Bp, q95, l_i) along with previous actions as input
        observation = np.zeros_like(self.low_state)
        for i in range(self.lookback):
            observation[i * len(self.histories[0]) : (i + 1) * len(self.histories[0])] = self.histories[i]
        
        # add current target to state
        observation[self.lookback * len(self.histories[0]) :] = [self.targets[self.target_params[i]][-1] for i in [0, 1, 2]]
        return observation

    def reset(self, seed=None):
        np.random.seed(seed)
        # randomly initialize target state (table 2 in Seo et al paper)
        target_init_low = [1.1, 3.8, 0.84]
        target_init_high = [2.1, 6.2, 1.06]
        self.target_init = np.random.uniform(low=target_init_low, high=target_init_high, size=len(target_init_high))
        # randomly initialize NBI powers (Idx 3, 4, and 5 of input_params)
        # NB1 powers min and max are shown below (from table 2 in Seo et al paper)
        nb1_low = [1.15, 1.15, 0.45]
        nb1_high = [1.75, 1.75, 0.6]
        self.input_init[3:6] = np.random.uniform(low=nb1_low, high=nb1_high, size=len(nb1_low))

        self.targets = {}
        # originally targetSliderDict in rt_control
        for i, target_param in enumerate(self.target_params):
            self.targets[target_param] = [self.target_init[i], self.target_init[i]]

        # originally inputSliderDict in rt_control
        self.inputs = {}
        for j, input_param in enumerate(self.input_params):
            self.inputs[input_param] = self.input_init[j]

        self.outputs = {}
        for p in self.output_params2:
            self.outputs[p] = [0.]

        self.dummy = {}
        for p in self.dummy_params:
            self.dummy[p] = [0.]

        seq_len = 10
        self.first = True
        # initialize action to start sim

        self.x = np.zeros([seq_len, 18])
        # initialize histories for LSTM model
        self.histories = [list(self.low_action) + list(self.target_init)] * self.lookback
        # reset initial state with noise
        self.state = self.get_state()

        # reset episode duration to 4 seconds
        self.fusion_duration = 4.
        
        info = {}
        return self.state, info


def test_environment(env):
    '''
    Sample random actions and pass through simulator
    '''
    env.action_space.seed(42)
    observation, info = env.reset(seed=42)
    print("Running loop...")
    # run for 8 seconds
    timesteps=80
    rewards = []
    for i in range(timesteps):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            rewards.append(reward)
            observation, info = env.reset()

    env.close()
    print('Done.')
    print(rewards)

def test_environment_with_rl_model(env):
    '''
    Load TD3 algorithm trained on simulator as test to make sure reward is close to 0
    '''
    # Load RL model and simulator
    base_path = os.path.abspath(os.path.dirname(sys.argv[0]))
    # RL model (stable baselines TD3)
    rl_model_path = os.path.join(base_path, 'weights', 'rl', 'rt_control',
                                 '3frame_v220505', 'best_model.zip')
    print('Loading model...')
    rl_model = SB2_model(
        model_path = rl_model_path, 
        low_state = env.low_state, 
        high_state = env.high_state, 
        low_action = env.low_action, 
        high_action = env.high_action, 
        activation='relu', 
        last_actv='tanh', 
        norm=True, 
        bavg=0.0
    )



    env.action_space.seed(42)
    observation, info = env.reset(seed=42)
    print("Running loop...")
    # run for 12 seconds (in simulator time)
    timesteps=120
    rewards = []

    # initialize old action (needed for RL model)
    action = np.array(env.low_action)
    for i in range(timesteps):
        action = rl_model.predict(observation, yold=action)

        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            rewards.append(reward)
            observation, info = env.reset()

    env.close()
    print('Done.')
    print(rewards)

if __name__ == '__main__':
    # Example training loop for KStar Environment
    env = KSTAREnv(verbose=True)
    # TODO Find way to suppress printing from simulator
    test_environment_with_rl_model(env)