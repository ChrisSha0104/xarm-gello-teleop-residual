import numpy as np
import matplotlib.pyplot as plt
import torch

def load_from_txt(filename, return_type='numpy'):
    """
    Load a .txt file into a numpy array or torch tensor.

    Each line in the file should contain space- or comma-separated floats,
    representing an action of dimension A.

    Args:
        filename (str): Path to the .txt file.
        return_type (str): 'numpy' or 'torch' to specify output format.

    Returns:
        np.ndarray or torch.Tensor: Loaded data of shape (N, A)
    """
    data = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Support both comma-separated and space-separated values
            delimiter = ',' if ',' in line else None
            elements = line.split(delimiter)
            data.append([float(x) for x in elements])

    array = np.array(data, dtype=np.float32)

    if return_type == 'numpy':
        return array
    elif return_type == 'torch':
        return torch.from_numpy(array)
    else:
        raise ValueError("return_type must be either 'numpy' or 'torch'")
    
# base traj
base_traj = load_from_txt('sim2real_vis_traj2/base_traj/teleop_comm_b.txt', 'numpy') # (400, 10) # not used in plot

# sim data
sim_robot_obs = load_from_txt('sim2real_vis_traj2/sim_traj2_data/robot_obs/sim_robot_ee_b.txt', 'numpy') # (400, 10)
sim_ee_goal = load_from_txt('sim2real_vis_traj2/sim_traj2_data/ee_with_residual/sim_ee_goal_with_res.txt', 'numpy') # (400, 10)

# real data
real_robot_obs = load_from_txt('sim2real_vis_traj2/real_traj2_data/robot_obs/real_robot_ee_b.txt', 'numpy') # (400, 10)
real_ee_goal = load_from_txt('sim2real_vis_traj2/real_traj2_data/ee_with_residual/real_ee_goal_with_res.txt', 'numpy') # (400, 10)

# Time axis and labels
timesteps = np.arange(1, 401)
dim_labels = [
    "X Pos", "Y Pos", "Z Pos",
    "Orn 1", "Orn 2", "Orn 3", "Orn 4", "Orn 5", "Orn 6",
    "Gripper"
]

# ---------- FIGURE 1: sim vs real robot obs ----------
fig1, axs1 = plt.subplots(2, 5, figsize=(20, 8))
axs1 = axs1.flatten()

for i in range(10):
    ax = axs1[i]
    ax.plot(timesteps, sim_robot_obs[:, i], label='Sim Robot Obs', linestyle='-')
    ax.plot(timesteps, real_robot_obs[:, i], label='Real Robot Obs', linestyle='-.')
    ax.set_title(f'Robot Obs - {dim_labels[i]}')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Value')
    ax.grid(True)

axs1[0].legend(loc='upper right')
fig1.suptitle('Sim vs Real Robot Observations', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# ---------- FIGURE 2: sim vs real ee goal ----------
fig2, axs2 = plt.subplots(2, 5, figsize=(20, 8))
axs2 = axs2.flatten()

for i in range(10):
    ax = axs2[i]
    ax.plot(timesteps, sim_ee_goal[:, i], label='Sim EE Goal', linestyle='--')
    ax.plot(timesteps, real_ee_goal[:, i], label='Real EE Goal', linestyle=':')
    ax.set_title(f'EE Goal - {dim_labels[i]}')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Value')
    ax.grid(True)

axs2[0].legend(loc='upper right')
fig2.suptitle('Sim vs Real EE Goals', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()