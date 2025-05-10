import torch
import torch.nn.functional as F

import numpy as np
import cv2

# def normalize_depth_01(depth_input, min_depth=0.07, max_depth=2.0):
#     """Normalize depth to range [0, 1] for CNN input."""
#     depth_input = torch.nan_to_num(depth_input, nan=0.0)
#     depth_input = depth_input.reshape(depth_input.shape[0], -1)
#     depth_input = torch.clamp(depth_input, min_depth, max_depth)  # Ensure valid depth range
#     depth_input = (depth_input - min_depth) / (max_depth - min_depth)
#     return depth_input

# import torch

def normalize_depth_01(depth_input, min_depth=0.10, max_depth=0.5):
    """Normalize depth to range [0, 1] while preserving spatial structure."""
    depth_input = torch.nan_to_num(depth_input, nan=0.0)  # Replace NaNs with 0

    # Ensure depth values are within the min-max range
    depth_input = torch.clamp(depth_input, min=min_depth, max=max_depth)

    # Normalize to range [0, 1]
    normalized_depth = (depth_input - min_depth) / (max_depth - min_depth)

    return normalized_depth.reshape(-1,120*120)

def clamp_depth_01(depth_input, min_depth=0.07, max_depth=0.5):
    """Normalize depth to range [0, 1] while preserving spatial structure."""
    depth_input = torch.nan_to_num(depth_input, nan=0.0)  # Replace NaNs with 0

    # Ensure depth values are within the min-max range
    depth_input = torch.clamp(depth_input, min=min_depth, max=max_depth)

    return depth_input.reshape(-1,120*120)


def add_depth_dependent_noise_torch(depth, base_std=0.005, scale=0.02):
    """Adds Gaussian noise where noise increases with depth distance."""
    noise_std = base_std + scale * depth  # Standard deviation grows with depth
    noise = torch.randn_like(depth) * noise_std  # Generate Gaussian noise
    return torch.clamp(depth + noise, min=0)  # Ensure depth is non-negative

def add_salt_and_pepper_noise_torch(depth, prob=0.02):
    """Adds salt-and-pepper noise by setting some pixels to 0 (missing depth) or max depth."""
    noisy_depth = torch.clone(depth)
    mask = torch.randint(0, 100, depth.shape, device=depth.device) / 100.0  # Random values between 0 and 1

    noisy_depth[mask < (prob / 2)] = 0  # Set some pixels to 0 (black, missing depth)
    noisy_depth[(mask >= (prob / 2)) & (mask < prob)] = depth.max()  # Set some pixels to max depth (white noise)
    
    return noisy_depth

def add_perlin_noise(depth_images, scale=0.1, octaves=4, device="cuda"):
    """
    Adds Perlin noise to a batch of depth images.
    
    Args:
        depth_images (torch.Tensor): Tensor of shape (num_envs, H, W).
        scale (float): Scaling factor for noise intensity.
        device (str): Device to run computation on.

    Returns:
        torch.Tensor: Noisy depth images of shape (num_envs, H, W).
    """
    depth_images = depth_images.to(device)  # Move to GPU
    perlin_map = generate_perlin_noise(depth_images.shape, scale, octaves, device=device) #type: ignore
    return depth_images + perlin_map

def generate_perlin_noise(shape, scale=20, threshold=0.5):
    """
    Generate Perlin-like structured noise for masking depth data.

    Args:
        shape (tuple): (num_envs, 1, H, W) tensor shape.
        scale (int): Controls smoothness of the noise (higher = larger structures).
        threshold (float): Controls how much area is zeroed out.
    
    Returns:
        torch.Tensor: A (num_envs, 1, H, W) binary mask where 1s are kept and 0s are occlusions.
    """
    num_envs, _, H, W = shape
    noise = torch.rand(num_envs, 1, H // scale, W // scale, device='cuda')  # Low-res noise
    noise = F.interpolate(noise, size=(H, W), mode='bilinear', align_corners=False)  # Upscale
    return (noise > threshold).float()  # Convert to binary mask (1s = keep, 0s = occlusions)

def add_realistic_sim2real_noise(depth_map: torch.Tensor, zero_density=0.3):
    """
    Adds Perlin-noise-based structured dropout to simulate realistic sensor occlusions.

    Args:
        depth_map (torch.Tensor): A (num_envs, 1, 120, 120) depth map tensor.
        zero_density (float): Controls the proportion of occluded regions.
        
    Returns:
        torch.Tensor: Depth map with realistic structured noise.
    """
    num_envs, _, H, W = depth_map.shape
    assert (H, W) == (120, 120), "Depth map must have shape (num_envs, 1, 120, 120)"

    noisy_depth = depth_map.clone()

    # Generate Perlin-like structured noise masks
    gripper_noise_mask = generate_perlin_noise((num_envs, 1, H, W), scale=20, threshold=1 - zero_density)
    finger_noise_mask = generate_perlin_noise((num_envs, 1, H, W), scale=5, threshold=1 - zero_density)

    # Region 1: Gripper base (y < 30 and depth < 0.1)
    y_indices = torch.arange(H, device=depth_map.device).view(1, 1, H, 1)
    gripper_mask = (y_indices > 90) & (depth_map < 0.1)
    noisy_depth[gripper_mask & (gripper_noise_mask == 0)] = 0.0

    # Region 2: Fingers (x in (0,40) U (80,120) and y in (30,70))
    x_indices = torch.arange(W, device=depth_map.device).view(1, 1, 1, W)
    finger_mask = ((x_indices < 40) | (x_indices >= 80)) & ((y_indices >= 50) & (y_indices < 90))
    noisy_depth[finger_mask & (finger_noise_mask == 0)] = 0.0

    return noisy_depth

def save_tensor_as_txt(tensor, filename_prefix="depth_map"):
    """
    Save a (num_envs, 120*120) PyTorch tensor as separate .txt files.

    Args:
        tensor (torch.Tensor): Tensor of shape (num_envs, 120*120).
        filename_prefix (str): Prefix for the saved file names.
    """
    assert tensor.ndim == 2, "Tensor must have shape (num_envs, 120*120)"
    
    num_envs = tensor.shape[0]
    
    # Move tensor to CPU and convert to NumPy
    tensor_np = tensor.reshape(-1, 120, 120).cpu().numpy()  # Shape (num_envs, 120, 120)

    for i in range(num_envs):
        filename = f"{filename_prefix}_env{i}.txt"
        np.savetxt(filename, np.round(tensor_np[i], 2), fmt="%.2f")
        print(f"Saved: {filename}")

def simulate_depth_noise(depth):
    """Simulates realistic depth noise including Gaussian noise and salt-and-pepper noise."""
    depth = add_depth_dependent_noise_torch(depth, base_std=0.005, scale=0.01)
    depth = add_realistic_sim2real_noise(depth, zero_density=0.7)
    # depth = add_salt_and_pepper_noise_torch(depth, prob=0.01)
    # depth = add_perlin_noise(depth, scale=0.05, octaves=4)
    return depth


def transform_point_eef_to_cam(p_eef: torch.Tensor, T_cam2eef: torch.Tensor) -> torch.Tensor:
    """
    Transforms a point from the EEF frame to the camera frame.
    
    Args:
        p_eef (torch.Tensor): A point in the EEF frame (shape: (3,) or (N,3)).
        T_eef2cam (torch.Tensor): 4x4 homogeneous transform from EEF to camera frame.
    
    Returns:
        torch.Tensor: The point in the camera frame (shape: (3,) or (N,3)).
    """
    T_eef2cam = T_cam2eef.inverse()  # Inverse transform to go from EEF to camera frame

    if p_eef.dim() == 1:
        p_eef_h = torch.cat([p_eef, torch.tensor([1.0], device=p_eef.device)])
        p_cam_h = T_eef2cam @ p_eef_h
        return p_cam_h[:3] / p_cam_h[3]
    else:
        ones = torch.ones((p_eef.shape[0], 1), device=p_eef.device, dtype=p_eef.dtype)
        p_eef_h = torch.cat([p_eef, ones], dim=1)  # shape (N,4)
        p_cam_h = (T_eef2cam.unsqueeze(0) @ p_eef_h.unsqueeze(-1)).squeeze(-1)  # (N,4)
        return p_cam_h[:, :3] / p_cam_h[:, 3:4]
    

def project_points(points_3d: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """
    Projects 3D points in the camera frame onto the 2D image plane using the pinhole camera model.

    Args:
        points_3d (torch.Tensor): Tensor of shape (N, 3) representing 3D points in the camera coordinate frame.
        K (torch.Tensor): Camera intrinsics matrix of shape (3, 3) in the form:
                          [[fx, 0,  cx],
                           [0,  fy, cy],
                           [0,  0,  1]].
    
    Returns:
        torch.Tensor: Tensor of shape (N, 2) containing the projected 2D pixel coordinates.
    """
    # Compute the homogeneous projection: shape (N, 3)
    # Multiply each 3D point by K:
    points_hom = (K @ points_3d.T).T  # (N, 3)
    
    # Avoid division by zero: if any Z is zero, we add a small epsilon.
    eps = 1e-6
    Z = points_hom[:, 2:3] + eps  # shape (N, 1)
    
    # Normalize to get pixel coordinates.
    u = points_hom[:, 0:1] / Z
    v = points_hom[:, 1:2] / Z
    
    # Concatenate to shape (N, 2)
    points_2d = torch.cat([u, v], dim=1)
    return points_2d

def visualize_points_on_image(points_2d: torch.Tensor, image: np.ndarray, color=(0, 255, 0), radius=0.3):
    """
    Overlays 2D points onto an image.
    
    Args:
        points_2d (torch.Tensor): Tensor of shape (N, 2) with pixel coordinates.
        image (np.ndarray): Image in BGR format (as from cv2).
        color (tuple): Color for the drawn points (B, G, R).
        radius (int): Radius of the circle to draw.
    
    Returns:
        np.ndarray: Image with overlaid points.
    """
    # Convert the points to CPU numpy int32 array.
    points_np = points_2d.cpu().numpy().astype(np.int32)
    for point in points_np:
        cv2.circle(image, (point[0], point[1]), radius, color, -1) # type: ignore
    return image

def filter_depth_for_visualization(depth: np.ndarray, max_depth: float = 0.5, min_depth: float = 0.1, unit: str = "mm", crop_depth = False) -> np.ndarray:
    """
    filter depth to be visualized using cv2.
    args:
        unit: "m" in real, "mm" in simulation
        crop_depth: whether to crop the depth image, true in real and false in simulation
    """

    if unit == "mm":
        depth = depth / 1000.0
    elif unit == "cm":
        depth = depth / 100.0

    # remove NaN and Inf values
    depth_filtered = np.nan_to_num(
        depth,
        nan=0.0,
        posinf=max_depth,
        neginf=min_depth,
    )

    if crop_depth:
        # crop depth to be square
        h, w = depth_filtered.shape
        crop_w = h

        x1 = (w - crop_w) // 2
        x2 = x1 + crop_w 

        depth_filtered = depth_filtered[:, x1:x2]

    # clip depth
    depth = np.clip(depth_filtered, min_depth, max_depth)

    # Normalize to [0, 255] and convert to uint8
    depth_uint8 = (depth * 255.0).astype(np.uint8)
    
    depth_vis = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)
    depth_vis = cv2.resize(depth_vis, (480, 480))
    
    return depth_vis

def filter_depth_real(real_depth, min_depth=0.1, max_depth=0.45):
    h, w = real_depth.shape
    crop_w = h
    x1 = (w - crop_w) // 2
    x2 = x1 + crop_w 

    depth_cropped = real_depth[:, x1:x2]
    depth_resized = cv2.resize(depth_cropped, (120, 120))
    depth_scaled = depth_resized / 1000.0  # Convert mm to m
    depth_filtered = np.nan_to_num(
            depth_scaled,
            nan=0.0,
            posinf=max_depth,
            neginf=min_depth,
        )
    depth_filtered = np.clip(depth_filtered, min_depth, max_depth)
    return depth_filtered

import numpy as np

def add_noise_in_depth_band_np(depth: np.ndarray,
                               min_d: float = 0.10,
                               max_d: float = 0.25,
                               mean: float = 0.0,
                               std: float = 0.01,
                               clip_min: float = 0.1,
                               clip_max: float = 0.45) -> np.ndarray:
    """
    Add zero-mean Gaussian noise only to pixels whose depth lies in [min_d, max_d].

    Args:
        depth   : np.ndarray of shape (120, 120), depth in meters.
        min_d   : lower bound of depth band to perturb.
        max_d   : upper bound of depth band to perturb.
        mean    : noise mean (meters).
        std     : noise standard deviation (meters).
        clip_min: minimum valid depth after noise.
        clip_max: maximum valid depth after noise (None to skip upper clamp).

    Returns:
        np.ndarray of shape (120, 120): depth + noise in the specified band.
    """
    # 1) create mask of pixels to perturb
    mask = (depth >= min_d) & (depth <= max_d)

    # 2) sample full noise map, then zero out outside the band
    noise = np.random.randn(*depth.shape) * std + mean
    noise *= mask  # only keep noise inside the band

    # 3) add noise and clamp
    depth_noisy = depth + noise
    depth_noisy = np.maximum(depth_noisy, clip_min)
    if clip_max is not None:
        depth_noisy = np.minimum(depth_noisy, clip_max)

    return depth_noisy
