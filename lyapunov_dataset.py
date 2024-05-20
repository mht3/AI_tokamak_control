import numpy as np
from kstar_env import KSTAREnv
import sys, os
from common.model_structure import *

class KSTARLyapunovDataset():

    def __init__(self, trajectory_length=40, N=500):
        # AI Gym environment with data-driven KSTAR simulator
        self.env = KSTAREnv()

        # get low and high state directly from AI gym env
        # current state consists of (9 actions, 3 0D parameters) repeated 3 times and 
        # finally the 3 targets. Total size is 39
        self.state_min = np.array(self.env.low_state)
        self.state_max = np.array(self.env.high_state)

        # scale standard deviation based on min/max values (used for adding noise)
        self.scaled_sigma = np.zeros_like(self.state_max)
        # only change current 0D parameters (indices 33-35 in state)
        self.scaled_sigma[-6:-3] = ((self.state_max - self.state_min) / 6)[-6:-3]
    
        # how long of a trajectory to unroll (defaults to 40 0.1 second intervals which is a single episode)
        self.trajectory_length = trajectory_length
        if self.trajectory_length > 40:
            raise ValueError('Trajectory length must be less than 40 timesteps (maximum episode length)')
        # numer of samples in dataset (total # of different trajectories)
        self.N = N

        # initialize trained TD3 model to generate actions for dataset
        self.load_rl_model()

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

    def get_action(self, x):
        '''
        Take in single observation x to get action 
        '''
        # this RL model needs the previous action as input, so we store a class variable for action
        self.action = self.rl_model.predict(x, yold=self.action)
        return self.action
    
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

    def build(self):

        initial_targets = self.load_initial_targets()

        # get 3 (s, pi(s), s') pairs
        # trajectory_size = self.trajectory_length * len(self.sta)
        trajectories = []
        for i in range(self.N):
            # numpy array for initial targets
            target = initial_targets[i, :]
            # reset environment for new trajectory
            self.env.reset(target_init=target)

            # get current state based on initialization
            x = self.env.get_state()
            # list of tuples of (s, pi(s), s') pairs
            trajectory = self.get_trajectory(x)
            trajectories.append(trajectory)

        return trajectories


    def get_trajectory(self, x):
        trajectory = []
        # get action for current state x
        pi = self.get_action(x)
        # get next state
        x_prime = self.get_next_state(x, pi)
        trajectory.append((x, pi, x_prime))


        for j in range(self.trajectory_length - 1):
            # perturb state and use as next trajectory
            x = x_prime + np.random.normal(0, self.scaled_sigma, len(x_prime))
            # get action
            pi = self.get_action(x)
            # get next state
            x_prime = self.get_next_state(x, pi)

            # add to trajectories
            trajectory.append((x, pi, x_prime))
            

        return trajectory

        # return trajectory

    def load_initial_targets(self):

        # X: Nx3 numpy array of initial states
        X = np.empty(shape=(self.N, 0))
        for i in range(len(self.env.low_target)):
            t_min = self.env.low_target[i]
            t_max = self.env.high_target[i]
            x = np.random.uniform(t_min, t_max, size=(self.N, 1))
            X = np.concatenate([X, x], axis=1)
        return X

    def save(self, data, filename='trajectories.npz'):
        flattened_data = self.convert_data(data)
        np.savez_compressed(filename, *flattened_data)

    def convert_data(self, data):
        flattened = []
        for trajectory in data:
            for state,action,next_state in trajectory:
                flattened.append((state,action,next_state))
        all_states, all_actions, all_next_states = zip(*flattened)
        return np.array(all_states), np.array(all_actions), np.array(all_next_states)

def load(filename, trajectory_length=40):
    '''
    Loads trajectories.npz
    '''
    loaded = np.load(filename, allow_pickle=True)
    loaded_data = [loaded[key] for key in loaded]
    states, actions, next_states = loaded_data
    trajectories = []
    for i in range(0, len(states), trajectory_length):
        trajectory = [(states[j], actions[j], next_states[j]) for j in range(i, i + trajectory_length)]
        trajectories.append(trajectory)
    return trajectories


if __name__ == '__main__':
    trajectory_length = 40
    dataset = KSTARLyapunovDataset(trajectory_length=trajectory_length, N=500)
    trajectories = dataset.build()
    print('##### Trajectories ######')
    print(len(trajectories[0]))
    dataset.save(trajectories)
    print('##### Loaded Data ######')
    loaded_data = load('trajectories.npz', trajectory_length)
    print(len(loaded_data[0]))
 



