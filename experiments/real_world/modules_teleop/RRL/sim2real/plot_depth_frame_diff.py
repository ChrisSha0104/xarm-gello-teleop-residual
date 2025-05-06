import numpy as np
import matplotlib.pyplot as plt

def plot_pixelwise_mse_heatmap(depth_path_1: str, depth_path_2: str, title="Average Pixel-wise MSE Over Time"):
    """
    Computes and plots a 2D heatmap of pixel-wise MSE averaged across all frames.

    Args:
        depth_path_1 (str): Path to first .npy file (shape: [num_frames, 120, 120]).
        depth_path_2 (str): Path to second .npy file (same shape).
        title (str): Title for the heatmap.
    """
    # Load and verify data
    depth1 = np.load(depth_path_1)[:398]
    depth2 = np.load(depth_path_2)[:398]
    assert depth1.shape == depth2.shape, "Shape mismatch between depth arrays"

    # Compute pixel-wise MSE averaged over frames
    mse_map = np.mean((depth1 - depth2) ** 2, axis=0)  # shape: (120, 120)

    # Plot heatmap
    plt.figure(figsize=(6, 5))
    im = plt.imshow(mse_map, cmap='inferno', interpolation='nearest')
    plt.colorbar(im, label='MSE')
    plt.title(title)
    plt.xlabel("Pixel X")
    plt.ylabel("Pixel Y")
    plt.tight_layout()
    plt.show()


plot_pixelwise_mse_heatmap('experiments/real_world/modules_teleop/RRL/tasks/cube/sim2real/traj3/sim_depth_obs.npy', 'experiments/real_world/modules_teleop/RRL/tasks/cube/sim2real/traj3/real_depth_obs.npy')