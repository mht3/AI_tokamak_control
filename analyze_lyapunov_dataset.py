"""
A small analysis script to analyze a dataset('trajectories.npz'), generated using 'generate_lyapunov_dataset.py'

Not using lookback from the state variable (just current state)
"""


import numpy as np
import matplotlib.pyplot as plt



## copied from "generate_lyapunov_dataset.py"
def load(filename, trajectory_length=40):
    '''
    Loads trajectories.npz -- AS TRAJECTORIES (not flattened)
    '''
    loaded = np.load(filename, allow_pickle=True)
    loaded_data = [loaded[key] for key in loaded]
    states, actions, next_states = loaded_data
    trajectories = []
    for i in range(0, len(states), trajectory_length):
        trajectory = [(states[j], actions[j], next_states[j]) for j in range(i, i + trajectory_length)]
        trajectories.append(trajectory)
    return trajectories



def plot_trajectory_bounds(trajectory_index, trajectory_target, target_min, target_max, points_all):
    time = np.arange(len(points_all))
    # Create a figure with three subplots
    fig, axs = plt.subplots(3, 1, figsize=(5,6))

    print("trajectory_target: ", trajectory_target)
    # Plot each variable versus time in a separate subplot
    variables = ['0D[1]', '0D[2]', '0D[3]']
    for i, ax in enumerate(axs):
        ax.plot(time, points_all[:, i], label=f'{variables[i]}')
        ax.plot(time, np.full_like(time, trajectory_target[i]), '--', label=f'Target for this trajectory ', color='black')
        ax.plot(time, np.full_like(time, target_min[i]), '--', label=f'Min {variables[i]}', color='r')
        ax.plot(time, np.full_like(time, target_max[i]), '--', label=f'Max {variables[i]}', color='g')
        ax.set_xlabel('Time')
        ax.set_ylabel(f'{variables[i]} Value')
        ax.set_title(f'{variables[i]} across trajectory {trajectory_index}')
        #ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()
    return




if __name__ == '__main__':
    state_dim = 39
    action_dim = 9

    ### Define target bounds for state (copied from kstar_env)
    target_min = np.array([0.8, 4.0, 0.80])
    target_max = np.array([2.1, 7.0, 1.05])
    

    ### Load the dataset from file
    trajectories_dataset = load('trajectories.npz')
    #print(f"Total number of trajectories {len(trajectories_dataset)}")
    #print(f"Length of first trajectory: {len(trajectories_dataset[0])}")
    
    
    for i,trajectory in enumerate(trajectories_dataset):
        print("----------------------")
        print("Trajectory index: ", i)

        # For a single trajectory:
        trajectory_target = trajectory[0][0][36:39]
        print("trajectory_target: ", trajectory_target)
        points_all     = []
        points_inside  = []
        points_outside = []

        for index,tuple in enumerate(trajectory):
            # Tuple is defined as (s,a,s')
            s = np.array(tuple[0])
            a = np.array(tuple[1])
            s_prime = np.array(tuple[2])

            current_0D = s[33:36]
            target = s[36:39]
            #print("----------------------")
            #print(f"current_0D: {current_0D}")
            #print(f"target: {target}")

            #print("Tuple index: ", index)
            #print(f"Current state: {s}")
            #print(f"Length of current state: {len(s)}")
            #print(f"Lookback pos0: {s[:12]}")
            #print(f"Lookback pos1: {s[12:24]}")
            #print(f"Lookback pos2 (current t): {s[24:36]}")
            #print(f"Target: {s[36:39]}")
            
            within_bounds = np.all((current_0D >= target_min) & (current_0D <= target_max))
            #print(f"Is s for this (s,a,s') within bounds of [target_min, target_max]?: {within_bounds}")

            points_all.append(current_0D)
            if within_bounds:
                points_inside.append(current_0D)
            else:
                points_outside.append(current_0D)

            
            #if(index>2): break
        
        
        print(f"num_of_tuples_within_bounds: {len(points_inside)}")
        print(f"num_of_tuples_outside_bounds: {len(points_outside)}")
        
        points_all = np.array(points_all)
        plot_trajectory_bounds(i+1, trajectory_target, target_min, target_max, points_all)

        #if(i>5):break
        break