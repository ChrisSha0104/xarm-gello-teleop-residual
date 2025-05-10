import torch
import os
import matplotlib.pyplot as plt
import numpy as np

def plot_xyz_distribution_with_noise(
    traj_dir: str,
    title: str = "XYZ Workspace Coverage with Dense Noise",
    noise_std_frac: float = 0.05,
    noise_multiplier: int = 10,
    max_files: int = None,
):
    """
    Plot clean xyz positions overlaid with multiple noisy samples (dense bubbles).

    Args:
        traj_dir (str): Directory with torch trajectories of shape (T, 10).
        title (str): Plot title.
        noise_std_frac (float): Fractional noise level (e.g., 0.03 = 3%).
        noise_multiplier (int): Number of noisy samples per clean point.
        max_files (int, optional): Max number of files to load.
    """
    all_pos = []

    for i, fname in enumerate(sorted(os.listdir(traj_dir))):
        if not fname.endswith(".pt") and not fname.endswith(".pth"):
            continue
        path = os.path.join(traj_dir, fname)
        data = torch.load(path)  # shape (T, 10)
        pos = data[:, :3]  # (T, 3)
        all_pos.append(pos)
        if max_files and i + 1 >= max_files:
            break

    pos_tensor = torch.cat(all_pos, dim=0).cpu()  # (N, 3)

    # Noise parameters
    std = pos_tensor.std(dim=0)
    noise_std = noise_std_frac * std

    # Create dense noise: repeat and add noise
    clean = pos_tensor.numpy()
    repeated = pos_tensor.repeat(noise_multiplier, 1)  # (N * k, 3)
    noise = torch.randn_like(repeated) * noise_std
    noisy = (repeated + noise).numpy()

    # Plot
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(title, fontsize=16)

    def scatter(ax, x1, y1, x2, y2, xlabel, ylabel):
        ax.scatter(x1, y1, s=1, alpha=0.3, color="tab:red", label="Clean")
        ax.scatter(x2, y2, s=1, alpha=0.1, color="tab:blue", label="Noisy sampled {} times".format(noise_multiplier))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True)
        ax.legend()

    scatter(axs[0], clean[:, 1], clean[:, 0], noisy[:, 1], noisy[:, 0], "y", "x")
    axs[0].set_title("Top-down view (y vs x)")

    scatter(axs[1], clean[:, 0], clean[:, 2], noisy[:, 0], noisy[:, 2], "x", "z")
    axs[1].set_title("Side-view (x vs z)")

    scatter(axs[2], clean[:, 1], clean[:, 2], noisy[:, 1], noisy[:, 2], "y", "z")
    axs[2].set_title("Front-view (y vs z)")

    plt.tight_layout()
    plt.show()

plot_xyz_distribution_with_noise(
    traj_dir = "experiments/real_world/modules_teleop/RRL/tasks/insertion/training_set3"
)

