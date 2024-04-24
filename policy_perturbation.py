'''
Policy Robustness check on trained TD3 RL policy (uses same model from rt_control_v2).
'''
import os, sys, time
from matplotlib import pyplot as plt
from common.model_structure import *
from common.wall import *
from common.setting import *

def load_agent(rl_model_path, verbose=False):
    lookback = 3
    # wide parameters from rt_control_v2
    low_target  = [0.8,4.0,0.80]
    high_target = [2.1,7.0,1.05]
    low_action  = [0.3, 0.0, 0.0,0.0, 1.6,0.15,0.5 ,1.265,2.14]
    high_action = [0.8,1.75,1.75,1.5,1.95, 0.5,0.85, 1.36, 2.3]
    low_state  = (low_action + low_target) * lookback + low_target
    high_state = (high_action + high_target) * lookback + high_target

    # Load agents
    if verbose:
        print('Loading model...')
    rl_model = SB2_model(
        model_path = rl_model_path, 
        low_state = low_state, 
        high_state = high_state, 
        low_action = low_action, 
        high_action = high_action, 
        activation='relu', 
        last_actv='tanh', 
        norm=True, 
        bavg=0.0
    )
    if verbose:
        print('Done.')
    return rl_model

class ModelTester:
    def __init__(self, model):
        self.model = model

    def predict(self):
        # Predict output_params0 (βn, q95, q0, li)
        pass

if __name__ == '__main__':
    base_path = os.path.abspath(os.path.dirname(sys.argv[0]))
    rl_model_path = os.path.join(base_path, 'weights', 'rl', 'rt_control',
                                 '3frame_v220505', 'best_model.zip')
    td3_model = load_agent(rl_model_path, verbose=True)
    
    model_tester = ModelTester(model=td3_model)

    input_params = ['Ip [MA]','Bt [T]','GW.frac. [-]',\
                    'Pnb1a [MW]','Pnb1b [MW]','Pnb1c [MW]',\
                    'Pec2 [MW]','Pec3 [MW]','Zec2 [cm]','Zec3 [cm]',\
                    'In.Mid. [m]','Out.Mid. [m]','Elon. [-]','Up.Tri. [-]','Lo.Tri. [-]']
    input_mins = [0.3,1.5,0.2,  0.0, 0.0, 0.0, 0.0,0.0,-10,-10, 1.265,2.18,1.6,0.1,0.5 ]
    input_maxs = [0.8,2.7,0.6,  1.75,1.75,1.5, 0.8,0.8, 10, 10, 1.36, 2.29,2.0,0.5,0.9 ]
    input_init = [0.5,1.8,0.33, 1.5, 1.5, 0.5, 0.0,0.0,0.0,0.0, 1.32, 2.22,1.7,0.3,0.75]

    output_params0 = ['βn','q95','q0','li']
    output_params1 = ['βp','wmhd']
    output_params2 = ['βn','βp','h89','h98','q95','q0','li','wmhd']
    dummy_params = ['Ip [MA]', 'Elon. [-]', 'Up.Tri. [-]', 'Lo.Tri. [-]', 'In.Mid. [m]', 'Out.Mid. [m]', 'Pnb1a [MW]','Pnb1b [MW]','Pnb1c [MW]']

    target_params = ['βp','q95','li']
    target_mins   = [0.8, 4.0, 0.80]
    target_maxs   = [2.1, 7.0, 1.05]
    target_init   = [1.45, 5.5, 0.925]