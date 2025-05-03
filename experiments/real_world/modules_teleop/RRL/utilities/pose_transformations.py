import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from typing import Callable
import math

from .math_utils import sample_uniform, euler_xyz_from_quat, quat_from_euler_xyz, quat_from_matrix, subtract_frame_transforms, combine_frame_transforms, quat_mul, matrix_from_quat
"""
Conventions:
    quat: [w, x, y, z]

"""

def normalize(q: torch.Tensor) -> torch.Tensor:
    """
    Normalizes a batch of quaternions.
    
    Args:
        q (torch.Tensor): Tensor of shape (batch_size, 4).
    
    Returns:
        torch.Tensor: Normalized quaternions of shape (batch_size, 4).
    """
    norm = torch.norm(q, dim=1, keepdim=True)
    return q / norm

def slerp(q1: torch.Tensor, q2: torch.Tensor, t: float) -> torch.Tensor:
    """
    Performs spherical linear interpolation between two batches of quaternions.

    Args:
        q1 (torch.Tensor): Tensor of shape (batch_size, 4) representing the first set of quaternions.
        q2 (torch.Tensor): Tensor of shape (batch_size, 4) representing the second set of quaternions.
        t (float): Interpolation factor in [0, 1]. At t=0 returns q1, at t=1 returns q2.

    Returns:
        torch.Tensor: Tensor of shape (batch_size, 4) containing the interpolated unit quaternions.
    """
    # Normalize input quaternions to ensure they are unit quaternions
    q1 = normalize(q1)
    q2 = normalize(q2)
    
    # Compute the dot product along the quaternion dimension
    dot = torch.sum(q1 * q2, dim=1, keepdim=True)
    
    # If the dot product is negative, flip q2 to take the shortest path.
    mask = dot < 0
    if mask.any():
        q2 = torch.where(mask, -q2, q2)
        dot = torch.where(mask, -dot, dot)
    
    # Clamp dot values to avoid numerical issues with arccos
    dot = torch.clamp(dot, -1.0, 1.0)

    # Calculate the angle between the quaternions
    theta = torch.acos(dot)
    
    # For small angles, use qlerp (nlerp) to avoid division by zero.
    # Here we set a threshold (e.g., dot > 0.9995)
    if (dot > 0.9995).all():
        # Fall back to linear interpolation followed by normalization.
        result = (1 - t) * q1 + t * q2
        return normalize(result)
    
    sin_theta = torch.sin(theta)
    
    # Compute interpolation factors for q1 and q2
    factor1 = torch.sin((1 - t) * theta) / sin_theta
    factor2 = torch.sin(t * theta) / sin_theta
    
    # Interpolate and normalize
    q_interp = factor1 * q1 + factor2 * q2
    return normalize(q_interp)

def slerp_batch(q1: torch.Tensor, q2: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
    """
    Vectorized spherical linear interpolation (slerp) between two batches of quaternions
    for multiple interpolation timesteps.

    Args:
        q1 (torch.Tensor): Tensor of shape (batch_size, 4) for the first set of quaternions.
        q2 (torch.Tensor): Tensor of shape (batch_size, 4) for the second set of quaternions.
        ts (torch.Tensor): 1D tensor of interpolation factors in [0, 1] with shape (num_steps,).
                           Each entry defines a timestep; t=0 returns q1 and t=1 returns q2.

    Returns:
        torch.Tensor: Tensor of interpolated unit quaternions of shape (batch_size, num_steps, 4).
    """
    # Normalize q1 and q2 to ensure they're unit quaternions.
    q1 = normalize(q1)  # shape (B, 4)
    q2 = normalize(q2)  # shape (B, 4)
    
    # Compute dot product (B, 1)
    dot = torch.sum(q1 * q2, dim=1, keepdim=True)
    
    # If dot is negative, flip q2 to take the shorter path.
    mask = dot < 0
    if mask.any():
        q2 = torch.where(mask, -q2, q2)
        dot = torch.where(mask, -dot, dot)
    
    # Clamp to avoid numerical issues.
    dot = torch.clamp(dot, -1.0, 1.0)
    theta = torch.acos(dot)  # shape (B, 1)
    
    # Expand ts to shape (1, num_steps) so that it broadcasts over batch.
    ts_exp = ts.view(1, -1)  # shape (1, num_steps)
    
    # If all dot values are very close to 1, fall back to linear interpolation (nlerp).
    if (dot > 0.9995).all():
        # Expand q1 and q2 for broadcasting: (B, 1, 4)
        q1_exp = q1.unsqueeze(1)
        q2_exp = q2.unsqueeze(1)
        result = (1 - ts_exp) * q1_exp + ts_exp * q2_exp
        # Normalize result: here we can use a batched normalization.
        norm = torch.norm(result, dim=-1, keepdim=True)
        return result / norm

    # Otherwise, proceed with slerp.
    # Compute sin(theta) and ensure proper broadcasting.
    sin_theta = torch.sin(theta)  # shape (B, 1) — broadcasts to (B, num_steps)
    
    # Compute interpolation factors:
    # factor1 = sin((1-t)*theta)/sin(theta)
    # factor2 = sin(t*theta)/sin(theta)
    factor1 = torch.sin((1 - ts_exp) * theta) / sin_theta  # shape (B, num_steps)
    factor2 = torch.sin(ts_exp * theta) / sin_theta         # shape (B, num_steps)
    
    # Expand q1 and q2 to shape (B, 1, 4) to allow broadcasting with factors.
    q1_exp = q1.unsqueeze(1)  # shape (B, 1, 4)
    q2_exp = q2.unsqueeze(1)  # shape (B, 1, 4)
    
    # Apply the factors (unsqueeze factors to shape (B, num_steps, 1))
    q_interp = factor1.unsqueeze(2) * q1_exp + factor2.unsqueeze(2) * q2_exp  # shape (B, num_steps, 4)
    
    # Normalize the results along the quaternion dimension.
    norm = torch.norm(q_interp, dim=-1, keepdim=True)
    q_interp = q_interp / norm

    return q_interp

def qlerp(q1: torch.Tensor, q2: torch.Tensor, t: float) -> torch.Tensor:
    """
    Performs normalized linear interpolation (nlerp) between two batches of quaternions.

    Args:
        q1 (torch.Tensor): Tensor of shape (batch_size, 4) for the first set of quaternions.
        q2 (torch.Tensor): Tensor of shape (batch_size, 4) for the second set of quaternions.
        t (float): Interpolation factor in [0, 1] (0 gives q1, 1 gives q2).

    Returns:
        torch.Tensor: Tensor of shape (batch_size, 4) containing the normalized interpolated quaternions.
    """
    # Normalize inputs to be safe
    q1 = normalize(q1)
    q2 = normalize(q2)
    
    # Compute the weighted combination and normalize the result
    q_interp = (1 - t) * q1 + t * q2
    return normalize(q_interp)

def rotation_matrix_to_6d(R: torch.Tensor) -> torch.Tensor:
    """
    Convert a rotation matrix to a 6D orientation representation.
    
    The 6D representation is obtained by concatenating the first two columns of the rotation matrix 
    in column-major order: [r11, r21, r31, r12, r22, r32].
    
    Args:
        R (torch.Tensor): Rotation matrix of shape (3, 3) or (batch_size, 3, 3).
    
    Returns:
        torch.Tensor: 6D representation as a tensor of shape (6,) or (batch_size, 6).
    """
    squeeze_output = False
    if R.dim() == 2:  # Single rotation matrix
        R = R.unsqueeze(0)
        squeeze_output = True

    # Extract the first two columns of R: shape (batch_size, 3, 2)
    R_6d = R[..., :2]
    # Permute to get column-major ordering: shape (batch_size, 2, 3)
    R_6d = R_6d.permute(0, 2, 1)
    # Flatten the last two dimensions to get shape (batch_size, 6)
    R_6d = R_6d.reshape(R.shape[0], 6)

    return R_6d.squeeze(0) if squeeze_output else R_6d

def validate_rotation_matrix(R: torch.Tensor, tol=1e-3) -> bool:
    """
    Check whether each rotation matrix in R is orthonormal.
    
    Args:
        R (torch.Tensor): Rotation matrix of shape (..., 3, 3).
        tol (float): Tolerance for the difference from the identity.
        
    Returns:
        bool: True if all matrices are within tolerance of being orthonormal.
    """
    # Create an identity matrix of shape (3, 3) on the same device and dtype.
    I = torch.eye(3, device=R.device, dtype=R.dtype)
    # Compute R R^T and compare to I.
    # If R is batched, this computes the norm over the last two dims.
    err = torch.norm(torch.matmul(R, R.transpose(-2, -1)) - I, dim=(-2, -1))
    return (err < tol).all() # type: ignore

def project_rotation_matrix(R: torch.Tensor) -> torch.Tensor:
    """
    Project each rotation matrix in R onto SO(3) using SVD.
    
    Args:
        R (torch.Tensor): Rotation matrix of shape (..., 3, 3).
    
    Returns:
        torch.Tensor: Projected rotation matrix of the same shape.
    """
    squeeze_output = False
    if R.dim() == 2:
        R = R.unsqueeze(0)
        squeeze_output = True
    U, S, Vh = torch.linalg.svd(R)
    R_proj = U @ Vh
    # Ensure determinant is 1: if det(R_proj) is negative, flip the sign of the last column of U.
    det = torch.det(R_proj)
    mask = (det < 0)
    if mask.any():
        U[mask, :, -1] = -U[mask, :, -1]
        R_proj = U @ Vh
    if squeeze_output:
        R_proj = R_proj.squeeze(0)
    return R_proj

def sixd_to_rotation_matrix(sixd: torch.Tensor, eps: float = 1e-6, collinearity_thresh: float = 1e-3, tol: float = 1e-3) -> torch.Tensor:
    """
    Convert a 6D orientation representation to a rotation matrix 
    This function uses a Gram–Schmidt process to form a rotation matrix.
    After computing the matrix, it checks whether the matrix is orthonormal and, if not, projects it onto SO(3).
    
    Args:
        sixd (torch.Tensor): 6D orientation tensor of shape (6,) or (batch_size, 6).
        eps (float): Small constant to avoid division by zero.
        collinearity_thresh (float): Dot-product threshold to detect near-collinearity.
        tol (float): Tolerance for the validation of the rotation matrix.
    
    Returns:
        torch.Tensor: Rotation matrix of shape (3, 3) or (batch_size, 3, 3).
    """
    single_input = sixd.dim() == 1
    if single_input:
        sixd = sixd.unsqueeze(0)  # (1, 6)

    x_raw = sixd[:, 0:3]
    y_raw = sixd[:, 3:6]

    # Normalize x
    x = torch.nn.functional.normalize(x_raw, dim=1, eps=eps)

    # Make y orthogonal to x
    dot = (x * y_raw).sum(dim=1, keepdim=True)
    y = torch.nn.functional.normalize(y_raw - dot * x, dim=1, eps=eps)

    # Compute z as cross product
    z = torch.cross(x, y, dim=1)

    # Stack as rotation matrix
    rot = torch.stack((x, y, z), dim=-1)  # shape (B, 3, 3)

    # Check orthonormality and project to SO(3) if needed
    def project_to_so3(matrix):
        u, _, v = torch.svd(matrix)
        return torch.matmul(u, v.transpose(-2, -1))

    # Validate each matrix
    identity = torch.eye(3, device=sixd.device).unsqueeze(0)
    rot_T = rot.transpose(-2, -1)
    prod = torch.matmul(rot_T, rot)
    diff = torch.norm(prod - identity, dim=(1, 2))

    needs_projection = diff > tol
    if needs_projection.any():
        rot[needs_projection] = project_to_so3(rot[needs_projection])

    return rot.squeeze(0) if single_input else rot

def quat_to_6d(quat: torch.Tensor) -> torch.Tensor:
    R = matrix_from_quat(quat)
    return rotation_matrix_to_6d(R)

def quat_from_6d(sixd: torch.Tensor) -> torch.Tensor:
    R = sixd_to_rotation_matrix(sixd)
    return quat_from_matrix(R)

def gripper_real2sim(real_gripper_status: float) -> float:
    sim_gripper_joint = (800 - real_gripper_status) / 800
    if sim_gripper_joint < 0:
        sim_gripper_joint = 0

    return sim_gripper_joint

# def slerp_batch(q1: torch.Tensor, q2: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
#     """
#     Vectorized SLERP between two batches of quaternions.
    
#     Args:
#         q1 (torch.Tensor): Batch of quaternions, shape (B, 4)
#         q2 (torch.Tensor): Batch of quaternions, shape (B, 4)
#         ts (torch.Tensor): 1D tensor of interpolation factors, shape (num_steps,)
    
#     Returns:
#         torch.Tensor: Interpolated quaternions, shape (B, num_steps, 4)
#     """
#     # Normalize input quaternions.
#     q1 = q1 / q1.norm(dim=-1, keepdim=True)
#     q2 = q2 / q2.norm(dim=-1, keepdim=True)
    
#     # Dot product (cosine of the angle between quaternions).
#     cosTheta = torch.sum(q1 * q2, dim=-1, keepdim=True)  # (B, 1)
    
#     # Ensure the shortest path is taken.
#     mask = (cosTheta < 0)
#     q2 = torch.where(mask, -q2, q2)
#     cosTheta = torch.where(mask, -cosTheta, cosTheta)
    
#     # Expand q1 and q2 for broadcasting: (B, 1, 4)
#     q1_exp = q1.unsqueeze(1)
#     q2_exp = q2.unsqueeze(1)
    
#     # Reshape ts to (1, num_steps, 1) for broadcasting.
#     ts_exp = ts.view(1, -1, 1)
    
#     # Compute the angle between quaternions.
#     theta = torch.acos(torch.clamp(cosTheta, -1.0, 1.0))  # shape: (B, 1)
#     theta = theta.unsqueeze(1)  # Now shape: (B, 1, 1)
#     sinTheta = torch.sin(theta) + 1e-6  # (B, 1, 1)
    
#     # Compute interpolation factors.
#     factor1 = torch.sin((1 - ts_exp) * theta) / sinTheta  # (B, num_steps, 1)
#     factor2 = torch.sin(ts_exp * theta) / sinTheta        # (B, num_steps, 1)
    
#     q_interp = factor1 * q1_exp + factor2 * q2_exp  # (B, num_steps, 4)
#     q_interp = q_interp / q_interp.norm(dim=-1, keepdim=True)
    
#     return q_interp

def subtract_frame_transforms_10D(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
    """
    args:
        p1: coordinate frame of p1 in f0, shape (num_envs, 10)
        p2: coordiante frame of p2 in f0, shape (num_envs, 10)
    return:
        p2_in_p1: coordinate frame of p2 in p1 frame, shape (num_envs, 10)
    """

    pos1 = p1[:,:3]
    pos2 = p2[:,:3]

    quat1 = quat_from_6d(p1[:, 3:9])
    quat2 = quat_from_6d(p2[:, 3:9])

    pos_2in1, quat_2in1 = subtract_frame_transforms(pos1, quat1, pos2, quat2)
    orient_6D_2in1 = quat_to_6d(quat_2in1)

    p2_in_p1 = torch.cat([pos_2in1, orient_6D_2in1, p2[:,-1:]], dim=-1)
    return p2_in_p1

def combine_frame_transforms_10D(p10: torch.Tensor, p21: torch.Tensor) -> torch.Tensor:
    """
    args:
        p10: coordinate frame of f1 in f0, shape (num_envs, 10)
        p21: coordiante frame of f2 in f1, shape (num_envs, 10)
    return:
        p20: coordinate frame of f2 in f0 frame, shape (num_envs, 10)
    """

    pos1 = p10[:,:3]
    pos2 = p21[:,:3]

    quat1 = quat_from_6d(p10[:, 3:9])
    quat2 = quat_from_6d(p21[:, 3:9])

    pos20, quat20 = combine_frame_transforms(pos1, quat1, pos2, quat2)
    orient_6D_20 = quat_to_6d(quat20)

    p20 = torch.cat([pos20, orient_6D_20, p21[:,-1:]], dim=-1)

    return p20


def ee_7D_to_9D(ee_7D: torch.Tensor) -> torch.Tensor:
    """
    args:
        ee 7D
    return:
        ee 10D
    """
    if ee_7D.shape[1] != 7:
        raise ValueError("Input poses must be 7D.")

    position = ee_7D[:, :3]
    quat = ee_7D[:, 3:7]

    orient_6D = quat_to_6d(quat)
    return torch.cat([position, orient_6D], dim=-1)

def ee_8D_to_10D(ee_8D: torch.Tensor) -> torch.Tensor:
    """
    args:
        ee 8D
    return:
        ee 10D
    """
    if ee_8D.shape[1] != 8:
        raise ValueError("Input poses must be 8D.")

    position = ee_8D[:, :3]
    quat = ee_8D[:, 3:7]
    gripper = ee_8D[:,-1:]

    orient_6D = quat_to_6d(quat)
    return torch.cat([position, orient_6D, gripper], dim=-1)

def ee_10D_to_8D(ee_10D: torch.Tensor) -> torch.Tensor:
    """
    args:
        ee 10D
    return:
        ee 8D
    """
    if ee_10D.shape[1] != 10:
        raise ValueError("Input poses must be 10D.")

    position = ee_10D[:, :3]
    orient_6D = ee_10D[:, 3:9]
    gripper = ee_10D[:,-1:]

    quat = quat_from_6d(orient_6D)
    return torch.cat([position, quat, gripper], dim=-1)

def compute_relative_state(prev_state: torch.Tensor, curr_state: torch.Tensor) -> torch.Tensor:
    """
    Compute the relative robot state (current state expressed in the previous state's frame).

    The function supports two state types:
    
    1. 10-dimensional state: [position (3), orientation (6D) (columns 3-8), gripper (1)].
    2. 16-dimensional state: [position (3), orientation (6D) (columns 3-8),
                               linear velocity (3) (columns 9-11),
                               angular velocity (3) (columns 12-14), gripper (1) (column 15)].
    
    For both cases, the relative state is computed as follows:
      - Relative position:  p_rel = R_prev^T (p_curr - p_prev)
      - Relative orientation: R_rel = R_prev^T * R_curr, converted back to a 6D representation.
      - For the 16-dimensional state, the linear and angular velocities are also rotated:
          v_lin_rel = R_prev^T (v_lin_curr - v_lin_prev)
          v_ang_rel = R_prev^T (v_ang_curr - v_ang_prev)
      - The gripper value is taken from the current state.
      
    The output will have the same dimensionality as the input state.

    Args:
        prev_state (torch.Tensor): Previous state, shape (num_envs, 10) or (num_envs, 16).
        curr_state (torch.Tensor): Current state, shape (num_envs, 10) or (num_envs, 16).

    Returns:
        torch.Tensor: Relative state, of the same shape as the input states.
    """
    # Check that both states have the same dimension.
    if prev_state.shape[1] != curr_state.shape[1]:
        raise ValueError("prev_state and curr_state must have the same dimensionality.")
    
    state_dim = prev_state.shape[1]
    if state_dim not in (10, 16):
        raise ValueError("State dimensionality must be either 10 or 16.")

    # Extract common components: position, orientation, gripper.
    prev_pos = prev_state[:, :3]  # (num_envs, 3)
    curr_pos = curr_state[:, :3]  # (num_envs, 3)
    
    # Orientation is stored as 6D (columns 3 to 8).
    prev_R = sixd_to_rotation_matrix(prev_state[:, 3:9])  # (num_envs, 3, 3)
    curr_R = sixd_to_rotation_matrix(curr_state[:, 3:9])  # (num_envs, 3, 3)
    
    # Gripper is the last element.
    prev_gripper = prev_state[:, -1]  # (num_envs,)
    curr_gripper = curr_state[:, -1]  # (num_envs,)

    # Compute the inverse of the previous rotation (transpose, since R is orthogonal).
    prev_R_T = prev_R.transpose(1, 2)  # (num_envs, 3, 3)

    # Compute relative position: rotate the difference into the previous frame.
    delta_pos = curr_pos - prev_pos
    rel_pos = torch.bmm(prev_R_T, delta_pos.unsqueeze(-1)).squeeze(-1)  # (num_envs, 3)

    # Compute relative orientation: R_rel = R_prev^T * R_curr, then convert to 6D.
    rel_R = torch.bmm(prev_R_T, curr_R)  # (num_envs, 3, 3)
    rel_orient = rotation_matrix_to_6d(rel_R)  # (num_envs, 6)

    if state_dim == 16:
        # Extract velocities.
        prev_lin_vel = prev_state[:, 9:12]    # (num_envs, 3)
        prev_ang_vel = prev_state[:, 12:15]    # (num_envs, 3)
        curr_lin_vel = curr_state[:, 9:12]     # (num_envs, 3)
        curr_ang_vel = curr_state[:, 12:15]     # (num_envs, 3)
        
        # Compute relative velocities by rotating the differences.
        delta_lin_vel = curr_lin_vel - prev_lin_vel
        rel_lin_vel = torch.bmm(prev_R_T, delta_lin_vel.unsqueeze(-1)).squeeze(-1)  # (num_envs, 3)
        
        delta_ang_vel = curr_ang_vel - prev_ang_vel
        rel_ang_vel = torch.bmm(prev_R_T, delta_ang_vel.unsqueeze(-1)).squeeze(-1)  # (num_envs, 3)
        
        # Assemble relative state for 16-dimensional input:
        # [rel_pos (3), rel_orient (6), rel_lin_vel (3), rel_ang_vel (3), gripper (1)]
        rel_state = torch.cat([
            rel_pos,
            rel_orient,
            rel_lin_vel,
            rel_ang_vel,
            curr_gripper.unsqueeze(1)
        ], dim=1)  # (num_envs, 16)
    else:
        # Assemble relative state for 10-dimensional input:
        # [rel_pos (3), rel_orient (6), gripper (1)]
        rel_state = torch.cat([
            rel_pos,
            rel_orient,
            curr_gripper.unsqueeze(1)
        ], dim=1)  # (num_envs, 10)

    return rel_state

def convex_combination_10D_poses(p1: torch.Tensor, p2: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """
    Compute a convex combination of two 10D poses by p1 * (1 - alpha) + p2 * alpha, i.e., alpha = 0 gives p1 and alpha = 1 gives p2.
    Args:
        p1 (torch.Tensor): First 10D pose [position, 6D orientation, gripper status], shape (num_envs, 10).
        p2 (torch.Tensor): Second 10D pose [position, 6D orientation, gripper status], shape (num_envs, 10).
        alpha (float): Interpolation factor in the range [0, 1].
    Returns:
        convex combination: position added linearly, 6D orientation interpolated using SLERP, gripper status added linearly.
    """
    if p1.shape != p2.shape:
        raise ValueError("p1 and p2 must have the same shape.")
    if p1.shape[1] != 10:
        raise ValueError("Input poses must be 10D.")
    # Interpolate position
    pos1 = p1[:, :3]
    pos2 = p2[:, :3]
    pos_combined = pos1 * (1-alpha) + pos2 * alpha

    # Interpolate orientation using SLERP
    q1 = quat_from_6d(p1[:, 3:9])
    q2 = quat_from_6d(p2[:, 3:9])

    q_interp = slerp(q1, q2, alpha)
    orient_combined = quat_to_6d(q_interp)
    
    # Interpolate gripper status
    gripper1 = p1[:, 9:10]
    gripper2 = p2[:, 9:10]
    gripper_combined = gripper1 * alpha + gripper2 * (1 - alpha)
    
    # 4. Concatenate the interpolated parts into a new 10D pose:
    #    [position (3), orientation (6), gripper status (1)]
    return torch.cat([pos_combined, orient_combined, gripper_combined], dim=1)

def linear_combination_10D_poses(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
    if p1.shape != p2.shape:
        raise ValueError("p1 and p2 must have the same shape.")
    if p1.shape[1] != 10:
        raise ValueError("Input poses must be 10D.")
    # Interpolate position
    pos1 = p1[:, :3]
    pos2 = p2[:, :3]
    pos_combined = pos1 + pos2 

    # Interpolate orientation using SLERP
    q1 = quat_from_6d(p1[:, 3:9])
    q2 = quat_from_6d(p2[:, 3:9])

    q_interp = slerp(q1, q2, 0.5)
    orient_combined = quat_to_6d(q_interp)
    
    # Interpolate gripper status
    gripper1 = p1[:, 9:10]
    gripper2 = p2[:, 9:10]
    gripper_combined = gripper1 + gripper2
    
    # 4. Concatenate the interpolated parts into a new 10D pose:
    #    [position (3), orientation (6), gripper status (1)]
    return torch.cat([pos_combined, orient_combined, gripper_combined], dim=-1)


def plot_interpolation(quaternions, method_name, ax):
    """
    Plots the rotated x-axis (first column of rotation matrices) for a series of quaternions.
    
    Args:
        quaternions (torch.Tensor): Tensor of shape (N, 4) containing a series of unit quaternions.
        method_name (str): Label for the interpolation method (e.g., 'SLERP' or 'QLERP').
        ax (Axes3D): A matplotlib 3D axis object.
    """
    # Convert quaternions to rotation matrices and extract the rotated x-axis
    rot_mats = matrix_from_quat(quaternions)
    # The rotated x-axis is the first column of the rotation matrix
    x_axes = rot_mats[:, :, 0].cpu().numpy()
    xs, ys, zs = x_axes[:, 0], x_axes[:, 1], x_axes[:, 2]
    
    # Plot the path on a sphere
    ax.plot(xs, ys, zs, marker='o', label=method_name)
    ax.scatter(xs[0], ys[0], zs[0], color='green', s=80, label='Start')
    ax.scatter(xs[-1], ys[-1], zs[-1], color='red', s=80, label='End')
    ax.set_title(method_name)
    ax.set_xlim([-1.0, 1.0])
    ax.set_ylim([-1.0, 1.0])
    ax.set_zlim([-1.0, 1.0])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

def visualize_interpolations():
    """
    Visualizes a series of interpolated quaternions (using slerp and qlerp) by plotting
    the path of a rotated x-axis vector on the unit sphere.
    """
    # Define two example quaternions (batch size 1 for our endpoints, but later expanded to a series)
    # Each quaternion is in [w, x, y, z] format.
    q1 = torch.tensor([[0.92388, 0.38268, 0.0, 0.0]])
    q2 = torch.tensor([[0.70711, 0.0, 0.70711, 0.0]])
    
    N = 50  # Number of interpolation steps
    ts = np.linspace(0, 1, N)
    
    slerp_list = []
    qlerp_list = []
    for t in ts:
        slerp_q = slerp(q1, q2, float(t))
        qlerp_q = qlerp(q1, q2, float(t))
        slerp_list.append(slerp_q)
        qlerp_list.append(qlerp_q)
    
    # Concatenate along batch dimension: shape (N, 4)
    slerp_quats = torch.cat(slerp_list, dim=0)
    qlerp_quats = torch.cat(qlerp_list, dim=0)
    
    # Create a figure with two subplots: one for slerp and one for qlerp
    fig = plt.figure(figsize=(12, 6))
    
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    
    plot_interpolation(slerp_quats, "SLERP", ax1)
    plot_interpolation(qlerp_quats, "QLERP", ax2)
    
    plt.suptitle("Visualization of Quaternion Interpolation\n(Rotated X-axis Path, i.e., [1,0,0], on Unit Sphere)")
    plt.show()

def random_quaternions(num_samples: int) -> torch.Tensor:
    """
    Generate random unit quaternions.
    
    Args:
        num_samples (int): Number of quaternions to generate.
        
    Returns:
        torch.Tensor: Tensor of shape (num_samples, 4) with random unit quaternions.
    """
    q = torch.randn(num_samples, 4)
    q = torch.nn.functional.normalize(q, dim=1)
    return q

def visualize_orientation_consistency(num_samples: int = 100):
    """
    Visualize the consistency of converting between rotation matrices and their 6D representations.
    
    The process is as follows:
      1. Generate a batch of random rotation matrices (via random quaternions).
      2. Convert these matrices to a 6D orientation representation using rotation_matrix_to_6d.
      3. Convert the 6D representations back to rotation matrices via sixd_to_rotation_matrix.
      4. Compute the reconstruction error (Frobenius norm difference).
      5. Also, for a fixed vector (e.g. the x-axis), show how both the original and 
         the reconstructed rotations transform it.
         
    Args:
        num_samples (int): Number of random rotation matrices (poses) to generate.
    """
    # Step 1: Generate random quaternions and convert them to rotation matrices.
    q_orig = random_quaternions(num_samples)  # shape: (num_samples, 4)
    R_orig = matrix_from_quat(q_orig)         # shape: (num_samples, 3, 3)
    
    # Step 2: Convert the rotation matrices to 6D representations.
    sixd = rotation_matrix_to_6d(R_orig)      # shape: (num_samples, 6)
    
    # Step 3: Convert back from 6D to rotation matrices.
    R_recon = sixd_to_rotation_matrix(sixd)    # shape: (num_samples, 3, 3)
    
    # Step 4: Compute reconstruction errors.
    diff = R_orig - R_recon
    # Compute Frobenius norm error per sample.
    errors = diff.view(num_samples, -1).norm(dim=1)
    mean_error = errors.mean().item()
    
    # Plot a histogram of the reconstruction errors.
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(errors.detach().cpu().numpy(), bins=20, color='skyblue', edgecolor='black')
    plt.title(f'Rotation Matrix Reconstruction Error\nMean Frobenius Norm Error: {mean_error:.4f}')
    plt.xlabel('Frobenius Norm Error')
    plt.ylabel('Frequency')
    
    # Step 5: Visualize how a fixed vector is transformed.
    # Use the x-axis (unit vector along x).
    x_axis = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32).unsqueeze(0)  # shape: (1, 3)
    x_axis = x_axis.unsqueeze(-1)  # shape: (1, 3, 1) for batch matrix multiplication
    
    # Apply original and reconstructed rotation matrices to the x-axis.
    x_orig = torch.matmul(R_orig, x_axis).squeeze(-1)    # shape: (num_samples, 3)
    x_recon = torch.matmul(R_recon, x_axis).squeeze(-1)    # shape: (num_samples, 3)
    
    # Create a 3D scatter plot of the transformed x-axis vectors.
    ax = plt.subplot(1, 2, 2, projection='3d')
    ax.scatter(x_orig[:, 0].cpu().numpy(), x_orig[:, 1].cpu().numpy(), x_orig[:, 2].cpu().numpy(),
               label='Original Rotation', color='blue', alpha=0.6)
    ax.scatter(x_recon[:, 0].cpu().numpy(), x_recon[:, 1].cpu().numpy(), x_recon[:, 2].cpu().numpy(),
               label='Reconstructed Rotation', color='red', alpha=0.6)
    ax.set_title("Transformed X-Axis Vectors")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z") # type: ignore
    ax.legend()
    
    print("Original rotation matrix [0]:\n", R_orig[0])
    print("Reconstructed rotation matrix [0]:\n", R_recon[0])

    print("Original rotation matrix [11]:\n", R_orig[11])
    print("Reconstructed rotation matrix [11]:\n", R_recon[11])

    plt.tight_layout()
    plt.show()
    # plt.savefig("output.png")

def visualize_and_validate_slerp_batch():
    """
    Visualize and validate the vectorized slerp_batch method.
    
    This function:
      1. Defines two quaternions: one for no rotation and one for a 90° rotation about the z-axis.
      2. Interpolates between these quaternions over a number of timesteps.
      3. Validates that the interpolated quaternions at t=0 and t=1 match the input quaternions.
      4. Converts each interpolated quaternion to a rotation matrix.
      5. Applies each rotation to the x-axis ([1, 0, 0]) and plots the path in 3D.
    """
    batch_size = 1  # For simplicity, we use a single pair
    num_steps = 20

    # Define two known quaternions (in [w, x, y, z] order).
    # Identity quaternion: no rotation.
    q1 = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    # 90° rotation around the z-axis.
    q2 = torch.tensor([[0.7071, 0.0, 0.0, 0.7071]])

    # Generate interpolation timesteps as a tensor.
    ts = torch.linspace(0, 1, num_steps)  # shape: (num_steps,)

    # Call your vectorized slerp_batch method.
    q_interp = slerp_batch(q1, q2, ts)  # Expected shape: (batch_size, num_steps, 4)

    # Validate endpoints:
    # At t=0, the result should match q1; at t=1, it should match q2.
    err_start = torch.norm(q_interp[:, 0, :] - q1, dim=1)
    err_end = torch.norm(q_interp[:, -1, :] - q2, dim=1)
    print("Endpoint errors:")
    print("t=0 error:", err_start.item())
    print("t=1 error:", err_end.item())

    # Now visualize the interpolation by plotting the trajectory of a transformed reference vector.
    # For each interpolated quaternion (for our single batch element), convert to a rotation matrix.
    # Then, apply the rotation matrix to the x-axis (i.e. [1, 0, 0]) to see how it moves.
    x_axis = torch.tensor([1.0, 0.0, 0.0])
    trajectory = []

    for qi in q_interp[0]:  # q_interp[0] has shape (num_steps, 4)
        R = matrix_from_quat(qi)  # shape: (3,3)
        x_transformed = R @ x_axis  # rotated x-axis, shape: (3,)
        trajectory.append(x_transformed.unsqueeze(0))
    
    trajectory = torch.cat(trajectory, dim=0).cpu().numpy()  # shape: (num_steps, 3)

    # Plot the transformed x-axis trajectory.
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
            marker='o', linestyle='-', color='blue', label='Transformed X-Axis')
    
    # Mark the endpoints clearly.
    ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2],
               color='green', s=100, label='t = 0 (q1)')
    ax.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2],
               color='red', s=100, label='t = 1 (q2)')
    
    ax.set_title("SLERP Batch Interpolation: X-Axis Trajectory")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.show()

# torch methods


if __name__ == "__main__":
    visualize_and_validate_slerp_batch()
    # visualize_orientation_consistency()
    # visualize_interpolations()