import torch
import matplotlib.pyplot as plt
import os

def plot_sim_vs_real(sim_path: str, real_path: str, title: str = "Sim vs Real", dim_labels=None):
    """
    Plots the sim and real tensors along each dimension as separate lines.

    Args:
        sim_path (str): Path to the saved torch tensor from simulation (shape: [time, dim]).
        real_path (str): Path to the saved torch tensor from real world (shape: [time, dim]).
        title (str): Title of the entire figure.
        dim_labels (list of str, optional): Custom labels for each dimension.
    """
    import torch
    import matplotlib.pyplot as plt
    import os

    assert os.path.exists(sim_path), f"Sim path not found: {sim_path}"
    assert os.path.exists(real_path), f"Real path not found: {real_path}"

    sim = torch.load(sim_path).detach().cpu()
    real = torch.load(real_path).detach().cpu()
    real = real[:-1,:]
    import pdb; pdb.set_trace()
    assert sim.shape == real.shape, "Sim and real tensors must have the same shape"
    time, dim = sim.shape

    if dim_labels is not None:
        assert len(dim_labels) == dim, f"dim_labels must have length {dim}"

    fig, axes = plt.subplots(dim, 1, figsize=(10, 3 * dim), sharex=True)
    fig.suptitle(title, fontsize=16)

    for i in range(dim):
        ax = axes[i] if dim > 1 else axes
        label = dim_labels[i] if dim_labels else f'Dim {i}'
        ax.plot(range(time), sim[:, i], label='Sim', color='tab:blue')
        ax.plot(range(time), real[:, i], label='Real', color='tab:orange', linestyle='--')
        ax.set_ylabel(label)
        ax.legend()
        ax.grid(True)

    ax.set_xlabel('Time step')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

dim_labels = ['x', 'y', 'z', 'orn1', 'orn2', 'orn3', 'orn4', 'orn5', 'orn6', 'binary gripper']
# plot_sim_vs_real("experiments/real_world/modules_teleop/RRL/tasks/cube/sim2real/traj2/sim_state_obs.pt", "experiments/real_world/modules_teleop/RRL/tasks/cube/sim2real/traj2/real_state_obs.pt", title="Robot State difference when deploying the policy using same teleop comm", dim_labels=dim_labels)
plot_sim_vs_real("experiments/real_world/modules_teleop/RRL/tasks/cube/sim2real/traj2/sim_residual_output.pt", "experiments/real_world/modules_teleop/RRL/tasks/cube/sim2real/traj2/real_residual.pt", title="residual output diff in normalized space", dim_labels=dim_labels)