import os
import copy
import time
import numpy as np
import torch
import transforms3d
from pathlib import Path
import open3d as o3d

import warnings
warnings.filterwarnings("always", category=RuntimeWarning)

import sapien.core as sapien

from typing import Union
# from utils import get_root
def get_root(path: Union[str, Path], name: str = '.root') -> Path:
    root = Path(path).resolve()
    while not (root / name).is_file():
        root = root.parent
    return root
root: Path = get_root(__file__)

def np2o3d(pcd, color=None):
    # pcd: (n, 3)
    # color: (n, 3)
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
    if color is not None:
        assert pcd.shape[0] == color.shape[0]
        assert color.max() <= 1
        assert color.min() >= 0
        pcd_o3d.colors = o3d.utility.Vector3dVector(color)
    return pcd_o3d

class KinHelper():
    def __init__(self, robot_name, headless=True):
        # load robot
        if "xarm7" in robot_name:
            urdf_path = str(root / "assets/robots/xarm7/xarm7.urdf")
            self.eef_name = 'link7'
        else:
            raise RuntimeError('robot name not supported')
        self.robot_name = robot_name

        # load sapien robot
        self.engine = sapien.Engine()
        self.scene = self.engine.create_scene()
        loader = self.scene.create_urdf_loader()
        self.sapien_robot = loader.load(urdf_path)
        self.robot_model = self.sapien_robot.create_pinocchio_model()
        self.sapien_eef_idx = -1
        for link_idx, link in enumerate(self.sapien_robot.get_links()):
            if link.name == self.eef_name:
                self.sapien_eef_idx = link_idx
                break

        # load meshes and offsets from urdf_robot
        self.meshes = {}
        self.scales = {}
        self.offsets = {}
        self.pcd_dict = {}
        self.tool_meshes = {}

    def compute_fk_sapien_links(self, qpos, link_idx):
        fk = self.robot_model.compute_forward_kinematics(qpos)
        link_pose_ls = []
        for i in link_idx:
            link_pose_ls.append(self.robot_model.get_link_pose(i).to_transformation_matrix())
        return link_pose_ls

    def compute_ik_sapien(self, initial_qpos, cartesian, verbose=False):
        """
        Compute IK using sapien
        initial_qpos: (N, ) numpy array
        cartesian: (6, ) numpy array, x,y,z in meters, r,p,y in radians
        """
        tf_mat = np.eye(4)
        tf_mat[:3, :3] = transforms3d.euler.euler2mat(ai=cartesian[3], aj=cartesian[4], ak=cartesian[5], axes='sxyz')
        tf_mat[:3, 3] = cartesian[0:3]
        pose = sapien.Pose.from_transformation_matrix(tf_mat)

        if 'xarm7' in self.robot_name:
            active_qmask = np.array([True, True, True, True, True, True, True])
        qpos = self.robot_model.compute_inverse_kinematics(
            link_index=self.sapien_eef_idx, 
            pose=pose,
            initial_qpos=initial_qpos, 
            active_qmask=active_qmask, 
            )
        if verbose:
            print('ik qpos:', qpos)

        # verify ik
        fk_pose = self.compute_fk_sapien_links(qpos[0], [self.sapien_eef_idx])[0]
        
        if verbose:
            print('target pose for IK:', tf_mat)
            print('fk pose for IK:', fk_pose)
        
        pose_diff = np.linalg.norm(fk_pose[:3, 3] - tf_mat[:3, 3])
        rot_diff = np.linalg.norm(fk_pose[:3, :3] - tf_mat[:3, :3])
        
        if pose_diff > 0.01 or rot_diff > 0.01:
            print('ik pose diff:', pose_diff)
            print('ik rot diff:', rot_diff)
            warnings.warn('ik pose diff or rot diff too large. Return initial qpos.', RuntimeWarning, stacklevel=2, )
            return initial_qpos
        return qpos[0]

    def compute_cartesian_vel(self, qpos, qvel):
        """
        Compute Cartesian linear and angular velocity from joint positions and velocities.

        Args:
            qpos: (N,) numpy array (Joint positions)
            qvel: (N,) numpy array (Joint velocities)

        Returns:
            linear_vel: (3,) numpy array [vx, vy, vz]
            angular_vel: (3,) numpy array [wx, wy, wz]
        """
        # Compute Jacobian at qpos
        J = self.robot_model.compute_single_link_local_jacobian(qpos, self.sapien_eef_idx)  # Shape (6, N)

        # Compute end-effector velocity
        ee_vel = J @ qvel  # Shape (6,)

        # Extract linear and angular velocities
        linear_vel = ee_vel[:3]  # [vx, vy, vz]
        angular_vel = ee_vel[3:]  # [wx, wy, wz]

        return linear_vel, angular_vel


def test_fk():
    robot_name = 'xarm7'
    init_qpos = np.array([0, 0, 0, 0, 0, 0, 0])
    end_qpos = np.array([0, 0, 0, 0, 0, 0, 0])
    
    kin_helper = KinHelper(robot_name=robot_name, headless=False)
    START_ARM_POSE = [0, 0, 0, 0, 0, 0, 0]

    for i in range(100):
        curr_qpos = init_qpos + (end_qpos - init_qpos) * i / 100
        fk = kin_helper.compute_fk_sapien_links(curr_qpos, [kin_helper.sapien_eef_idx])[0]
        fk_euler = transforms3d.euler.mat2euler(fk[:3, :3], axes='sxyz')

        if i == 0:
            init_ik_qpos = np.array(START_ARM_POSE)
        ik_qpos = kin_helper.compute_ik_sapien(init_ik_qpos, np.array(list(fk[:3, 3]) + list(fk_euler)).astype(np.float32))
        re_fk_pos_mat = kin_helper.compute_fk_sapien_links(ik_qpos, [kin_helper.sapien_eef_idx])[0]
        re_fk_euler = transforms3d.euler.mat2euler(re_fk_pos_mat[:3, :3], axes='sxyz')
        re_fk_pos = re_fk_pos_mat[:3, 3]
        print('re_fk_pos diff:', np.linalg.norm(re_fk_pos - fk[:3, 3]))
        print('re_fk_euler diff:', np.linalg.norm(np.array(re_fk_euler) - np.array(fk_euler)))
        

        init_ik_qpos = ik_qpos.copy()
        qpos_diff = np.linalg.norm(ik_qpos[:6] - curr_qpos[:6])
        if qpos_diff > 0.01:
            warnings.warn('qpos diff too large', RuntimeWarning, stacklevel=2, )

        time.sleep(0.1)

def test_cartesian_vel():
    qpos = np.array([0.1, 0.2, 0.3, 0.1, 0.2, 0.1, 0.0])  # Joint positions
    qvel = np.array([0.5, -0.3, 0.2, 0.1, -0.2, 0.3, 0.0])  # Joint velocities (rad/s)
    
    robot_name = 'xarm7'
    kin_helper = KinHelper(robot_name=robot_name, headless=False)
    linear_vel, angular_vel = kin_helper.compute_cartesian_vel(qpos, qvel)
    print("Linear Velocity (m/s):", linear_vel)
    print("Angular Velocity (rad/s):", angular_vel)

import numpy as np

def validate_rotation_matrix_np(R, tol=1e-3):
    """
    Check whether each rotation matrix in R is orthonormal.
    
    Args:
        R (np.ndarray): Rotation matrix of shape (..., 3, 3).
        tol (float): Tolerance for the difference from the identity.
        
    Returns:
        bool: True if all matrices are within tolerance of being orthonormal.
    """
    I = np.eye(3, dtype=R.dtype)
    # Compute R R^T - I (works for both batched and single matrices)
    diff = np.matmul(R, np.swapaxes(R, -2, -1)) - I
    err = np.linalg.norm(diff, axis=(-2, -1))
    return np.all(err < tol)


def project_rotation_matrix_np(R):
    """
    Project each rotation matrix in R onto SO(3) using SVD.
    
    Args:
        R (np.ndarray): Rotation matrix of shape (..., 3, 3).
    
    Returns:
        np.ndarray: Projected rotation matrix of the same shape.
    """
    squeeze_output = False
    if R.ndim == 2:
        R = R[np.newaxis, ...]
        squeeze_output = True

    # For batched SVD, loop over the first dimension if needed.
    def batched_svd(mats):
        U_list, S_list, Vh_list = [], [], []
        for i in range(mats.shape[0]):
            U, S, Vh = np.linalg.svd(mats[i])
            U_list.append(U)
            S_list.append(S)
            Vh_list.append(Vh)
        return (np.stack(U_list, axis=0),
                np.stack(S_list, axis=0),
                np.stack(Vh_list, axis=0))
    
    if R.ndim == 3:
        U, S, Vh = batched_svd(R)
    else:
        # If R has more than one batch dimension, flatten them to a single batch.
        batch_shape = R.shape[:-2]
        R_flat = R.reshape(-1, 3, 3)
        U, S, Vh = batched_svd(R_flat)
        U = U.reshape(*batch_shape, 3, 3)
        S = S.reshape(*batch_shape, 3)
        Vh = Vh.reshape(*batch_shape, 3, 3)
    
    R_proj = np.matmul(U, Vh)
    det = np.linalg.det(R_proj)
    # If any projected matrix has a negative determinant, flip the sign of the last column of U.
    if np.any(det < 0):
        # We iterate over the batch indices. (For large batches, vectorize as needed.)
        it = np.nditer(det, flags=['multi_index'])
        while not it.finished:
            if it[0] < 0:
                idx = it.multi_index
                U[idx + (slice(None),)][..., -1] = -U[idx + (slice(None),)][..., -1]
            it.iternext()
        R_proj = np.matmul(U, Vh)
    
    if squeeze_output:
        R_proj = np.squeeze(R_proj, axis=0)
    return R_proj


def sixd_to_rotation_matrix_np(sixd, eps=1e-6, collinearity_thresh=0.99, tol=1e-3):
    """
    Convert a 6D orientation representation to a rotation matrix with checks to avoid degenerate cases.
    
    The 6D representation is assumed to consist of two 3D vectors (typically the first two columns
    of the rotation matrix). A Gram–Schmidt process is applied to obtain an orthonormal rotation matrix.
    
    Args:
        sixd (np.ndarray): 6D orientation tensor of shape (6,) or (batch_size, 6).
        eps (float): Small constant to avoid division by zero.
        collinearity_thresh (float): Dot-product threshold to detect near-collinearity.
        tol (float): Tolerance for the validation of the rotation matrix.
    
    Returns:
        np.ndarray: Rotation matrix of shape (3, 3) or (batch_size, 3, 3).
    """
    squeeze_output = False
    if sixd.ndim == 1:
        sixd = sixd[np.newaxis, :]
        squeeze_output = True

    # Split into two 3D vectors.
    a = sixd[:, :3].copy()  # candidate for first column
    b = sixd[:, 3:6].copy()  # candidate for second column

    # Normalize first vector 'a'
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    if np.isnan(a_norm).any() or np.isinf(a_norm).any():
        raise ValueError("Invalid values in a_norm")
    # Replace near-zero vectors with default direction [1, 0, 0]
    mask = (a_norm < eps).flatten()
    if np.any(mask):
        a[mask] = np.array([1.0, 0.0, 0.0])
        a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    r1 = a / (a_norm + eps)

    # Normalize second vector 'b'
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    mask = (b_norm < eps).flatten()
    if np.any(mask):
        b[mask] = np.array([0.0, 1.0, 0.0])
        b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    b = b / (b_norm + eps)

    # Check for near-collinearity.
    dot = np.sum(r1 * b, axis=1, keepdims=True)
    nearly_collinear = np.abs(dot) > collinearity_thresh
    if np.any(nearly_collinear):
        # Use a candidate that is guaranteed not to be collinear.
        default = np.tile(np.array([0.0, 0.0, 1.0]), (r1.shape[0], 1))
        alternative = np.tile(np.array([0.0, 1.0, 0.0]), (r1.shape[0], 1))
        # For each row, choose alternative if r1 is nearly equal to default.
        condition = (np.linalg.norm(r1 - default, axis=1, keepdims=True) < eps)
        candidate = np.where(condition, alternative, default)
        b = np.cross(r1, candidate)
        b = b / (np.linalg.norm(b, axis=1, keepdims=True) + eps)
    else:
        # Standard Gram–Schmidt orthogonalization.
        b_proj = dot * r1
        b_ortho = b - b_proj
        b = b_ortho / (np.linalg.norm(b_ortho, axis=1, keepdims=True) + eps)

    # Compute third column as cross product.
    r3 = np.cross(r1, b)
    # Stack the three basis vectors as columns (shape: (batch, 3, 3)).
    R = np.stack((r1, b, r3), axis=2)
    
    # Validate the rotation matrix and project if necessary.
    if not validate_rotation_matrix_np(R, tol=tol):
        R = project_rotation_matrix_np(R)
        
    if squeeze_output:
        R = np.squeeze(R, axis=0)
    return R


# -- Quaternion utilities --
# These functions assume a quaternion is represented as a 4-element array in the order [w, x, y, z].
# (Adjust the ordering as needed for your application.)

def matrix_from_quat_np(quat):
    """
    Convert a quaternion to a rotation matrix.
    
    Args:
        quat (np.ndarray): Quaternion of shape (..., 4) in [w, x, y, z] order.
    
    Returns:
        np.ndarray: Rotation matrix of shape (..., 3, 3).
    """
    # Normalize quaternion
    quat = quat / np.linalg.norm(quat, axis=-1, keepdims=True)
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    
    # Allocate output array with the same batch shape as quat.
    shape = quat.shape[:-1]
    R = np.empty(shape + (3, 3), dtype=quat.dtype)
    
    R[..., 0, 0] = 1 - 2*(y**2 + z**2)
    R[..., 0, 1] = 2*(x*y - z*w)
    R[..., 0, 2] = 2*(x*z + y*w)
    R[..., 1, 0] = 2*(x*y + z*w)
    R[..., 1, 1] = 1 - 2*(x**2 + z**2)
    R[..., 1, 2] = 2*(y*z - x*w)
    R[..., 2, 0] = 2*(x*z - y*w)
    R[..., 2, 1] = 2*(y*z + x*w)
    R[..., 2, 2] = 1 - 2*(x**2 + y**2)
    return R

def sixd_to_rotmat(x):
    """
    Converts a 6D orientation (two 3D vectors) into a 3x3 rotation matrix.
    
    Parameters:
        x (np.ndarray): 6D vector (shape (6,)), where the first three elements
                        represent the first 3D vector and the last three the second.
                        
    Returns:
        R (np.ndarray): 3x3 rotation matrix.
    """
    # Split the 6D vector into two 3D vectors
    a1 = x[:3]
    a2 = x[3:]
    
    # Normalize the first vector
    b1 = a1 / np.linalg.norm(a1)
    
    # Make the second vector orthogonal to b1
    # Remove the projection of a2 onto b1
    a2_proj = np.dot(b1, a2) * b1
    b2 = a2 - a2_proj
    b2 = b2 / np.linalg.norm(b2)
    
    # Compute the third basis vector as the cross product of b1 and b2
    b3 = np.cross(b1, b2)
    
    # Stack the vectors as columns to form the rotation matrix
    R = np.column_stack((b1, b2, b3))
    return R

def rotation_matrix_to_6d_np(R):
    """
    Convert a rotation matrix to its 6D representation by taking its first two columns.
    
    Args:
        R (np.ndarray): Rotation matrix of shape (..., 3, 3).
        
    Returns:
        np.ndarray: 6D representation of shape (..., 6).
    """
    # Extract first two columns and flatten the last two dimensions.
    r6d = R[..., :, :2].flatten(order='F')
    return r6d


def quat_from_matrix_np(R):
    """
    Convert a rotation matrix to a quaternion.
    
    Args:
        R (np.ndarray): Rotation matrix of shape (..., 3, 3).
        
    Returns:
        np.ndarray: Quaternion of shape (..., 4) in [w, x, y, z] order.
    """
    # Following a standard algorithm (see e.g., EuclideanSpace.com)
    m00 = R[..., 0, 0]
    m11 = R[..., 1, 1]
    m22 = R[..., 2, 2]
    trace = m00 + m11 + m22

    # Allocate arrays for quaternion components.
    shape = R.shape[:-2]
    quat = np.empty(shape + (4,), dtype=R.dtype)
    
    # Case 1: trace > 0
    cond1 = trace > 0
    s1 = np.where(cond1, 0.5 / np.sqrt(trace + 1.0 + 1e-8), 0.0)
    w1 = np.where(cond1, 0.25 / s1, 0.0)
    x1 = np.where(cond1, (R[..., 2, 1] - R[..., 1, 2]) * s1, 0.0)
    y1 = np.where(cond1, (R[..., 0, 2] - R[..., 2, 0]) * s1, 0.0)
    z1 = np.where(cond1, (R[..., 1, 0] - R[..., 0, 1]) * s1, 0.0)
    
    # Case 2: m00 is the largest diagonal term.
    cond2 = (~cond1) & (R[..., 0, 0] > R[..., 1, 1]) & (R[..., 0, 0] > R[..., 2, 2])
    s2 = np.where(cond2, 2.0 * np.sqrt(1.0 + R[..., 0, 0] - R[..., 1, 1] - R[..., 2, 2] + 1e-8), 0.0)
    w2 = np.where(cond2, (R[..., 2, 1] - R[..., 1, 2]) / s2, 0.0)
    x2 = np.where(cond2, 0.25 * s2, 0.0)
    y2 = np.where(cond2, (R[..., 0, 1] + R[..., 1, 0]) / s2, 0.0)
    z2 = np.where(cond2, (R[..., 0, 2] + R[..., 2, 0]) / s2, 0.0)
    
    # Case 3: m11 is the largest diagonal term.
    cond3 = (~cond1) & (~cond2) & (R[..., 1, 1] > R[..., 2, 2])
    s3 = np.where(cond3, 2.0 * np.sqrt(1.0 + R[..., 1, 1] - R[..., 0, 0] - R[..., 2, 2] + 1e-8), 0.0)
    w3 = np.where(cond3, (R[..., 0, 2] - R[..., 2, 0]) / s3, 0.0)
    x3 = np.where(cond3, (R[..., 0, 1] + R[..., 1, 0]) / s3, 0.0)
    y3 = np.where(cond3, 0.25 * s3, 0.0)
    z3 = np.where(cond3, (R[..., 1, 2] + R[..., 2, 1]) / s3, 0.0)
    
    # Case 4: m22 is the largest diagonal term.
    cond4 = ~(cond1 | cond2 | cond3)
    s4 = np.where(cond4, 2.0 * np.sqrt(1.0 + R[..., 2, 2] - R[..., 0, 0] - R[..., 1, 1] + 1e-8), 0.0)
    w4 = np.where(cond4, (R[..., 1, 0] - R[..., 0, 1]) / s4, 0.0)
    x4 = np.where(cond4, (R[..., 0, 2] + R[..., 2, 0]) / s4, 0.0)
    y4 = np.where(cond4, (R[..., 1, 2] + R[..., 2, 1]) / s4, 0.0)
    z4 = np.where(cond4, 0.25 * s4, 0.0)
    
    w = np.where(cond1, w1, np.where(cond2, w2, np.where(cond3, w3, w4)))
    x = np.where(cond1, x1, np.where(cond2, x2, np.where(cond3, x3, x4)))
    y = np.where(cond1, y1, np.where(cond2, y2, np.where(cond3, y3, y4)))
    z = np.where(cond1, z1, np.where(cond2, z2, np.where(cond3, z3, z4)))
    
    quat[..., 0] = w
    quat[..., 1] = x
    quat[..., 2] = y
    quat[..., 3] = z
    # Normalize the quaternion
    quat = quat / (np.linalg.norm(quat, axis=-1, keepdims=True) + 1e-8)
    return quat


def quat_to_6d_np(quat):
    """
    Convert a quaternion to its 6D rotation representation.
    
    Args:
        quat (np.ndarray): Quaternion of shape (..., 4) in [w, x, y, z] order.
    
    Returns:
        np.ndarray: 6D rotation representation of shape (..., 6).
    """
    R = matrix_from_quat_np(quat)
    return rotation_matrix_to_6d_np(R)


def quat_from_6d_np(sixd):
    """
    Convert a 6D rotation representation to a quaternion.
    
    Args:
        sixd (np.ndarray): 6D rotation representation of shape (6,) or (batch_size, 6).
    
    Returns:
        np.ndarray: Quaternion of shape (4,) or (batch_size, 4) in [w, x, y, z] order.
    """
    R = sixd_to_rotmat(sixd)
    return quat_from_matrix_np(R)

def quat_from_matrix_np(R):
    """
    Convert a 3x3 rotation matrix to a quaternion in (w, x, y, z) format.
    
    Parameters:
        R (np.ndarray): A 3x3 rotation matrix.
        
    Returns:
        np.ndarray: A 4-element array containing the quaternion (w, x, y, z).
    """
    # Ensure R is a 3x3 matrix
    assert R.shape == (3, 3), "Input must be a 3x3 matrix."
    
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2  # S = 4 * w
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # S = 4 * x
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # S = 4 * y
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # S = 4 * z
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S
        
    return np.array([w, x, y, z])

def matrix_from_quat_np(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions (np.ndarray): The quaternion orientation in (w, x, y, z).
                                  Shape is (..., 4).

    Returns:
        np.ndarray: Rotation matrices of shape (..., 3, 3).

    Reference:
        Adapted from the PyTorch3D implementation:
        https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py
    """
    # Unpack the quaternion components.
    r = quaternions[..., 0]
    i = quaternions[..., 1]
    j = quaternions[..., 2]
    k = quaternions[..., 3]
    
    # Compute the scaling factor.
    two_s = 2.0 / np.sum(quaternions * quaternions, axis=-1)
    
    # Compute the entries of the rotation matrix.
    o0 = 1 - two_s * (j * j + k * k)
    o1 = two_s * (i * j - k * r)
    o2 = two_s * (i * k + j * r)
    o3 = two_s * (i * j + k * r)
    o4 = 1 - two_s * (i * i + k * k)
    o5 = two_s * (j * k - i * r)
    o6 = two_s * (i * k - j * r)
    o7 = two_s * (j * k + i * r)
    o8 = 1 - two_s * (i * i + j * j)
    
    # Stack into the last dimension and reshape to (..., 3, 3).
    o = np.stack([o0, o1, o2, o3, o4, o5, o6, o7, o8], axis=-1)
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def compute_relative_state_np(prev_state, curr_state):
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
        prev_state (np.ndarray): Previous state, shape (num_envs, 10) or (num_envs, 16).
        curr_state (np.ndarray): Current state, shape (num_envs, 10) or (num_envs, 16).

    Returns:
        np.ndarray: Relative state, of the same shape as the input states.
    """
    if prev_state.shape[0] != curr_state.shape[0 ]:
        raise ValueError("prev_state and curr_state must have the same dimensionality.")
    
    state_dim = prev_state.shape[0]
    if state_dim not in (10, 16):
        raise ValueError("State dimensionality must be either 10 or 16.")
        
    # Extract common components: position, orientation, gripper.
    prev_pos = prev_state[:3]  # shape: (3)
    curr_pos = curr_state[:3]  # shape: (3)
    
    # Orientation is stored as 6D (columns 3 to 9).
    prev_R = sixd_to_rotation_matrix_np(prev_state[3:9])  # shape: (3, 3)
    curr_R = sixd_to_rotation_matrix_np(curr_state[3:9])   # shape: (3, 3)
    
    # Gripper is the last element.
    curr_gripper = curr_state[-1].reshape(1)  # shape: (1)
    
    # Compute the inverse of the previous rotation (transpose, since R is orthogonal).
    prev_R_T = prev_R.T  # shape: (3, 3)
    
    # Compute relative position: rotate the difference into the previous frame.
    delta_pos = curr_pos - prev_pos  # shape: (3)
    rel_pos = (prev_R_T @ delta_pos).reshape(-1)  # shape: (3)
    
    # Compute relative orientation: R_rel = R_prev^T * R_curr, then convert to 6D.
    rel_R = np.matmul(prev_R_T, curr_R)  # shape: (3, 3)
    rel_orient = rotation_matrix_to_6d_np(rel_R)  # shape: (6)
    
    if state_dim == 16:
        # Extract velocities.
        prev_lin_vel = prev_state[9:12]    # shape: (3)
        prev_ang_vel = prev_state[12:15]     # shape: (3)
        curr_lin_vel = curr_state[9:12]      # shape: (3)
        curr_ang_vel = curr_state[12:15]      # shape: (3)
        
        # Compute relative velocities by rotating the differences.
        delta_lin_vel = curr_lin_vel - prev_lin_vel  # shape: (3)
        rel_lin_vel = (prev_R_T @ delta_lin_vel).reshape(-1) # shape: (3)
        
        delta_ang_vel = curr_ang_vel - prev_ang_vel  # shape: (3)
        rel_ang_vel = (prev_R_T @ delta_ang_vel).reshape(-1) # shape: (3)
        
        # Assemble relative state for 16-dimensional input:
        # [rel_pos (3), rel_orient (6), rel_lin_vel (3), rel_ang_vel (3), gripper (1)]
        rel_state = np.concatenate([
            rel_pos,
            rel_orient,
            rel_lin_vel,
            rel_ang_vel,
            curr_gripper,
        ], axis=0)  # shape: (16)
    else:
        # Assemble relative state for 10-dimensional input:
        # [rel_pos (3), rel_orient (6), gripper (1)]
        rel_state = np.concatenate([
            rel_pos,
            rel_orient,
            curr_gripper,
        ], axis=0)  # shape: (10)
    
    return rel_state

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
    return (err < tol).all()

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

def sixd_to_rotation_matrix(sixd: torch.Tensor, eps=1e-6, collinearity_thresh=0.99, tol=1e-3) -> torch.Tensor:
    """
    Convert a 6D orientation representation to a rotation matrix with checks to avoid degenerate cases.
    
    This function uses a Gram–Schmidt process (with safety checks) to form a rotation matrix.
    After computing the matrix, it checks whether the matrix is orthonormal and, if not, projects it onto SO(3).
    
    Args:
        sixd (torch.Tensor): 6D orientation tensor of shape (6,) or (batch_size, 6).
        eps (float): Small constant to avoid division by zero.
        collinearity_thresh (float): Dot-product threshold to detect near-collinearity.
        tol (float): Tolerance for the validation of the rotation matrix.
    
    Returns:
        torch.Tensor: Rotation matrix of shape (3, 3) or (batch_size, 3, 3).
    """
    squeeze_output = False
    if sixd.dim() == 1:
        sixd = sixd.unsqueeze(0)
        squeeze_output = True

    # Split into two 3D vectors.
    a = sixd[:, :3]  # candidate for first column
    b = sixd[:, 3:6]  # candidate for second column

    # Check norm of first vector.
    a_norm = a.norm(dim=1, keepdim=True)
    if torch.isnan(a_norm).any() or torch.isinf(a_norm).any():
        import pdb; pdb.set_trace()
        raise ValueError("Invalid values in a_norm")
    if (a_norm < eps).any():
        # Replace near-zero vectors with a default direction.
        a = torch.where(a_norm < eps, torch.tensor([1.0, 0.0, 0.0], device=sixd.device).expand_as(a), a)
        a_norm = a.norm(dim=1, keepdim=True)
    r1 = a / (a_norm + eps)

    # Normalize second vector candidate.
    b_norm = b.norm(dim=1, keepdim=True)
    if (b_norm < eps).any():
        b = torch.tensor([0.0, 1.0, 0.0], device=sixd.device).expand_as(b)
        b_norm = b.norm(dim=1, keepdim=True)
    b = b / (b_norm + eps)

    # Check for near-collinearity.
    dot = (r1 * b).sum(dim=1, keepdim=True)
    nearly_collinear = (dot.abs() > collinearity_thresh)
    if nearly_collinear.any():
        # For problematic cases, choose a candidate vector that is guaranteed not to be collinear.
        default = torch.tensor([0.0, 0.0, 1.0], device=sixd.device).expand_as(r1)
        alternative = torch.tensor([0.0, 1.0, 0.0], device=sixd.device).expand_as(r1)
        condition = (r1 - default).norm(dim=1, keepdim=True) < eps
        candidate = torch.where(condition, alternative, default)
        b = torch.cross(r1, candidate, dim=1)
        b = torch.nn.functional.normalize(b, dim=1)
    else:
        # Standard Gram–Schmidt orthogonalization.
        b_proj = dot * r1
        b_ortho = b - b_proj
        b = torch.nn.functional.normalize(b_ortho, dim=1)

    # Compute third column as cross product.
    r3 = torch.cross(r1, b, dim=1)
    R = torch.stack((r1, b, r3), dim=2)

    # Validate the resulting rotation matrix.
    if not validate_rotation_matrix(R, tol=tol):
        R = project_rotation_matrix(R)
    return R.squeeze(0) if squeeze_output else R

def quat_to_6d(quat: torch.Tensor) -> torch.Tensor:
    R = matrix_from_quat(quat)
    return rotation_matrix_to_6d(R)

def quat_from_6d(sixd: torch.Tensor) -> torch.Tensor:
    R = sixd_to_rotation_matrix(sixd)
    return quat_from_matrix(R)

def matrix_from_quat(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: The quaternion orientation in (w, x, y, z). Shape is (..., 4).

    Returns:
        Rotation matrices. The shape is (..., 3, 3).

    Reference:
        https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py#L41-L70
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def quat_from_matrix(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: The rotation matrices. Shape is (..., 3, 3).

    Returns:
        The quaternion in (w, x, y, z). Shape is (..., 4).

    Reference:
        https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py#L102-L161
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(matrix.reshape(batch_dim + (9,)), dim=-1)

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    return quat_candidates[torch.nn.functional.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :].reshape(
        batch_dim + (4,)
    )


if __name__ == "__main__":
    test_cartesian_vel()
    # test_fk()
