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


### Needed for graphing:

# (under "RL setting" in original script)
low_target  = [1.0, 4.0]
high_target = [2.0, 7.0]
target_mins, target_maxs = low_target, high_target

# (under "Setting" in original script)
dpi = 1
decimals = np.log10(1000)


def i2f(i,decimals=decimals):
    return float(i/10**decimals)


class ModelTester:
    def __init__(self, rl_model_path, nn_model_path, lstm_model_path, bpw_model_path,
                 target_params, target_init, input_params, input_init,
                 output_params0, output_params1, output_params2,
                 dummy_params, verbose=True):
        self.model = self.load_agent(rl_model_path, verbose=verbose)
        # kstar_lstm
        self.load_sim(nn_model_path, lstm_model_path, bpw_model_path, verbose=verbose)
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
        self.output_params1 = output_params1
        self.output_params2 = output_params2
        self.dummy_params = dummy_params


        self.outputs = {}
        for p in output_params2:
            self.outputs[p] = [0.]
        self.input_params = input_params
        self.input_init = input_init
        # plot length (4 seconds in total)
        self.plot_length = 40
        self.time = np.linspace(-0.1 * (self.plot_length - 1), 0, self.plot_length) # negative time enables real-time-graph in GUI, we will use same convention here

        # If false, an average is taken for data driven simulator outputs
        self.steady = False
        self.first = True

        self.dummy = {}
        for p in dummy_params:
            self.dummy[p] = [0.]


    def load_sim(self, nn_model_path, lstm_model_path, bpw_model_path,
                 max_models=10, verbose=False):
        if verbose:
            print('Loading simulator...')
        self.kstar_nn = kstar_nn(model_path=nn_model_path, n_models=1)
        self.kstar_lstm = kstar_v220505(model_path=lstm_model_path, n_models=max_models)
        self.bpw_nn = bpw_nn(model_path=bpw_model_path, n_models=max_models)
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
        if self.steady or self.first:
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
                self.x[:,i+4] = self.inputs[input_params[idx_convert[i]]]
            self.x[:, 11 + 4] += self.inputs[input_params[7]]
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
            for i in range(len(output_params0)):
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
            x[i] = self.inputs[input_params[idx_convert[i]]]
        x[3],x[4] = 0.5*(x[3]+x[4]),0.5*(x[4]-x[3])
        y = self.bpw_nn.predict(x)
        for i in range(len(self.output_params1)):
            output_param = self.output_params1[i]
            if len(self.outputs[output_param]) >= self.plot_length:
                del self.outputs[output_param][0]
            elif len(self.outputs[output_param]) == 1:
                self.outputs[output_param][0] = y[i]
            self.outputs[output_param].append(y[i])
        """
        # Store dummy parameters
        for p in dummy_params:
            if len(self.dummy[p]) >= self.plot_length:
                del self.dummy[p][0]
            elif len(self.dummy[p]) == 1:
                self.dummy[p][0] = i2f(self.inputSliderDict[p].value())
            self.dummy[p].append(i2f(self.inputSliderDict[p].value()))
        """
        if not self.first:
            for i,target_param in enumerate(target_params):
                if len(self.targets[target_param]) >= self.plot_length:
                    del self.targets[target_param][0]
                elif len(self.targets[target_param]) == 1:
                    self.targets[target_param][0] = self.outputs[target_param][-1]

                print(target_param)
                self.targets[target_param].append(self.outputs[target_param][-1])


        # Update histories based on action and predicted outputs
        self.histories[:-1] = self.histories[1:]
        self.histories[-1] = list(self.new_action) + list([self.outputs['βp'][-1], self.outputs['q95'][-1], self.outputs['li'][-1]])

        # Use robust simulator after first step
        if self.first:
            self.first = False

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


def policy_noise_injection(env):
    # bounds of state
    target_mins   = [0.8, 4.0, 0.80]
    target_maxs   = [2.1, 7.0, 1.05]

    # bounds of action
    input_mins = [0.3,1.5,0.2,  0.0, 0.0, 0.0, 0.0,0.0,-10,-10, 1.265,2.18,1.6,0.1,0.5 ]
    input_maxs = [0.8,2.7,0.6,  1.75,1.75,1.5, 0.8,0.8, 10, 10, 1.36, 2.29,2.0,0.5,0.9 ]

    print("Running RL policy and simulator...")
    # get action using RL policy
    env.get_action(target_init, noise=True)
    # take a step to get next state (current action is updated as class variable)
    env.step()


    print("Running RL policy and simulator...")
    # Take step again with noise
    env.get_action(target_init, noise=True)
    env.step()

    # print actions (code calls these inputs)
    print("env.inputs: ")
    print(env.inputs)

    # print states (includes history)
    # can use env.outputs and env.targets for plotting code (hopefully)
    print("env.outputs: ")
    print(env.outputs)


def make_graphs(env):
    # making basic graphs for now (removed limits, legends, ticks from original code)
    # ignoring "dpi" (original code used this, but it blows up the scale of the graphs to make lines way too big)

    ts = env.time[-len(env.outputs['βn']):]

    #print("env.outputs['βn']: ", env.outputs['βn'])
    #print("env.time: ", env.time)
    print("ts: ", ts)
    print("env.outputs['βp']: ", env.outputs['βp'])
    print("env.dummy: ", env.dummy)

    ##############
    # Plotting operation trajectory
    ##############
    """
    plt.subplot(3,3,2)
    pnb = np.sum([env.dummy['Pnb1a [MW]'], env.dummy['Pnb1b [MW]'], env.dummy['Pnb1c [MW]']], axis=0)
    plt.title('AI operation trajectory')
    plt.plot(ts,env.dummy['Ip [MA]'],'k',linewidth=2*(100/dpi),label='Ip [MA]')
    plt.step(ts,0.1*pnb,'grey',linewidth=2*(100/dpi),label='0.1*Pnb [MW]',where='mid')
    plt.grid(linewidth=0.5*(100/dpi))
    plt.legend(loc='upper left',fontsize=7.5*(100/dpi),frameon=False)
    plt.xlim([-0.1 * plot_length - 0.2, 0.2])
    plt.ylim([0.1, 0.75])
    plt.xticks(color='w')
    """
    
    ##############
    # Plotting targets/outputs of 0D parameters
    ##############
    alpha = 0.5
    gaps = 0.5 * np.subtract(target_maxs, target_mins)
    
    # Plot βp:
    plt.subplot(3,3,3)
    plt.title('Response and target')
    plt.plot(ts,env.outputs['βp'],'k',label='βp')
    plt.plot(ts,env.targets['βp'],'b',alpha=alpha,linestyle='-',label='Target')
    plt.grid()
    plt.legend(loc='upper left',frameon=False)

    # Plot q95:
    plt.subplot(3,3,6)
    plt.plot(ts,env.outputs['q95'],'k',label='q95')
    plt.plot(ts,env.targets['q95'],'b',alpha=alpha,linestyle='-',label='Target')
    plt.grid()
    plt.legend(loc='upper left',frameon=False)

    # Plot li: 
    # (rt_control_v3 doesn't plot this, but v2 does. Including here for completeness)
    plt.subplot(3,3,9)
    plt.plot(ts,env.outputs['li'],'k',label='li')
    plt.plot(ts,env.targets['li'],'b',alpha=alpha,linestyle='-',label='Target')
    plt.grid()
    plt.legend(loc='upper left',frameon=False)

    plt.show()

    return



if __name__ == '__main__':

    # Load RL model and simulator
    base_path = os.path.abspath(os.path.dirname(sys.argv[0]))
    # RL model (stable baselines TD3)
    rl_model_path = os.path.join(base_path, 'weights', 'rl', 'rt_control',
                                 '3frame_v220505', 'best_model.zip')
    
    # nn/lstm model are simulator which output output_params0 (βn, q95, q0, li)
    nn_model_path   = os.path.join(base_path, 'weights', 'nn')
    lstm_model_path = os.path.join(base_path, 'weights', 'lstm', 'v220505')
    # final network that outputs output_params1 (βp, wmhd)
    bpw_model_path  = os.path.join(base_path, 'weights', 'bpw')


    # initialize input and target parameters
    input_params = ['Ip [MA]','Bt [T]','GW.frac. [-]',\
                    'Pnb1a [MW]','Pnb1b [MW]','Pnb1c [MW]',\
                    'Pec2 [MW]','Pec3 [MW]','Zec2 [cm]','Zec3 [cm]',\
                    'In.Mid. [m]','Out.Mid. [m]','Elon. [-]','Up.Tri. [-]','Lo.Tri. [-]']
    input_init = [0.5,1.8,0.33, 1.5, 1.5, 0.5, 0.0,0.0,0.0,0.0, 1.32, 2.22,1.7,0.3,0.75]

    target_params = ['βp','q95','li']
    target_init   = [1.45, 5.5, 0.925]

    # output parameters used (output params 0 is output of simulator)
    output_params0 = ['βn','q95','q0','li']
    output_params1 = ['βp','wmhd']
    output_params2 = ['βn','βp','h89','h98','q95','q0','li','wmhd']
    dummy_params = ['Ip [MA]', 'Elon. [-]', 'Up.Tri. [-]', 'Lo.Tri. [-]', 'In.Mid. [m]', 'Out.Mid. [m]', 'Pnb1a [MW]','Pnb1b [MW]','Pnb1c [MW]']


    env = ModelTester(rl_model_path, nn_model_path, lstm_model_path, bpw_model_path,
                               target_params=target_params, target_init=target_init, 
                               input_params=input_params, input_init=input_init, 
                               output_params0=output_params0, output_params1=output_params1,
                               output_params2=output_params2, 
                               dummy_params=dummy_params,
                               verbose=True)
    
    policy_noise_injection(env)
    make_graphs(env)