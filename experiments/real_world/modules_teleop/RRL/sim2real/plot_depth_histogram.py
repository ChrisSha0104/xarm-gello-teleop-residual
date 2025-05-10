import numpy as np
import matplotlib.pyplot as plt

def plot_two_depth_histograms(filepath1, filepath2, bins=15):
    """
    Plot histograms of two depth datasets on the same figure.

    Args:
        filepath1 (str): Path to the first .npy depth file (e.g., sim).
        filepath2 (str): Path to the second .npy depth file (e.g., real).
        bins (int): Number of bins to use for histograms.
    """
    # Load depth data
    depth_data1 = np.load(filepath1)
    depth_data2 = np.load(filepath2)
    print(f"Loaded depth data shapes: {depth_data1.shape}, {depth_data2.shape}")

    plt.imshow(depth_data1[0], cmap='gray')
    plt.show()
    plt.imshow(depth_data2[0], cmap='gray')
    plt.show()

    # Flatten and filter depths between 0.1 and 0.5
    all_depths1 = depth_data1.flatten()
    all_depths2 = depth_data2.flatten()

    selected_depths1 = all_depths1[(all_depths1 >= 0.1) & (all_depths1 <= 0.45)]
    selected_depths2 = all_depths2[(all_depths2 >= 0.1) & (all_depths2 <= 0.45)]

    print(f"Selected points: {selected_depths1.shape[0]} (dataset 1), {selected_depths2.shape[0]} (dataset 2)")

    # Plot
    plt.figure(figsize=(8, 6))
    plt.hist(selected_depths1, bins=bins, range=(0.1, 0.45), alpha=0.5, label='Sim', edgecolor='black')
    plt.hist(selected_depths2, bins=bins, range=(0.1, 0.45), alpha=0.5, label='Real', edgecolor='black')
    plt.title('Depth Value Histogram Comparison')
    plt.xlabel('Depth (meters)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Example usage: adjust filenames accordingly
    plot_two_depth_histograms('experiments/real_world/modules_teleop/RRL/tasks/insertion/sim2real/no_res_traj1/sim_depth_obs.npy', 'experiments/real_world/modules_teleop/RRL/tasks/insertion/sim2real/no_res_traj1/real_depth_obs.npy', bins=100)
