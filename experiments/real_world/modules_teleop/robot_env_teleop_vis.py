from typing import Callable, List
from enum import Enum
import numpy as np
import multiprocess as mp
import time
import threading
import cv2
import pygame
import os
import pickle
import transforms3d
import subprocess
import torch
import math
import logging

from .RRL.RL_models.residual_actor_critic import ResidualActorCritic
from .RRL.RL_models.residual_teacher_student import ResidualStudentTeacher
from .RRL.utilities.math_utils import *
from .RRL.utilities.action_normalizer import ActionNormalizer
from .RRL.utilities.state_hist_buffer import HistoryBuffer

from pynput import keyboard
from pathlib import Path
from copy import deepcopy
import matplotlib.pyplot as plt

from utils import get_root, mkdir
root: Path = get_root(__file__)

from modules_teleop.perception import Perception
from modules_teleop.perception_zed import PerceptionZED
from modules_teleop.xarm_controller import XarmController
from modules_teleop.teleop_keyboard import KeyboardTeleop
from modules_teleop.kinematics_utils import *
from experiments.real_world.modules_teleop.RRL.utilities.pose_transformations import *
from experiments.real_world.modules_teleop.RRL.utilities.traj_generator import *
from experiments.real_world.modules_teleop.RRL.utilities.vision_helper import *
from gello.teleop_gello import GelloTeleop
from gello.teleop_gello_residual import GelloTeleopResidual
from camera.multi_realsense import MultiRealsense
from camera.single_realsense import SingleRealsense

from rsl_rl.modules import EmpiricalNormalization
import pyzed.sl as sl

class EnvEnum(Enum):
    NONE = 0
    INFO = 1
    DEBUG = 2
    VERBOSE = 3

class RobotTeleopEnvVision(mp.Process):
    kin_helper = KinHelper(robot_name='xarm7')

    def __init__(
        self, 
        mode: str = '2D', # 2D or 3D
        exp_name: str = "recording",

        realsense: MultiRealsense | SingleRealsense | None = None,
        shm_manager: mp.managers.SharedMemoryManager | None = None,
        serial_numbers: list[str] | None= None,
        resolution: tuple[int, int] = (1280, 720),
        capture_fps: int = 30,
        enable_depth: bool = True,
        enable_color: bool = True,

        perception: Perception | None = None,
        record_fps: int | None = 0,
        record_time: float | None = 60 * 10,  # 10 minutes
        perception_process_func: Callable | None = lambda x: x,  # default is identity function

        use_robot: bool = False,
        use_gello: bool = False,
        use_residual_policy: bool = False,
        load_model: str = '',
        # xarm_controller: XarmController | None = None,
        robot_ip: str | None = '192.168.1.196',
        gripper_enable: bool = False,
        calibrate_result_dir: Path = root / "log" / "latest_calibration",
        data_dir: Path = "data",
        debug: bool | int | None = False,
        
        bimanual: bool = False,
        bimanual_robot_ip: List[str] | None = ['192.168.1.196', '192.168.1.224'],

    ) -> None:

        # Debug level
        self.debug = 0 if debug is None else (2 if debug is True else debug)

        self.mode = mode
        self.exp_name = exp_name
        self.data_dir = data_dir

        self.bimanual = bimanual

        self.capture_fps = capture_fps
        self.record_fps = record_fps

        self.state = mp.Manager().dict()  # should be main explict exposed variable to the child class / process

        # Realsense camera setup
        # camera is always required for real env
        if realsense is not None:
            assert isinstance(realsense, MultiRealsense) or isinstance(realsense, SingleRealsense)
            self.realsense = realsense
            self.serial_numbers = list(self.realsense.cameras.keys())
        else:
            self.realsense = MultiRealsense(
                shm_manager=shm_manager,
                serial_numbers=serial_numbers,
                resolution=resolution,
                capture_fps=capture_fps,
                enable_depth=enable_depth,
                enable_color=enable_color,
                verbose=self.debug > EnvEnum.VERBOSE.value
            )
            self.serial_numbers = list(self.realsense.cameras.keys())

        # Perception setup
        if perception is not None:
            assert isinstance(perception, Perception)
            self.perception = perception
        else:
            self.perception = PerceptionZED(
                capture_fps=self.realsense.capture_fps,
                record_fps=record_fps,
                record_time=record_time,
                process_func=perception_process_func,
                exp_name=exp_name,
                data_dir=data_dir,
                verbose=self.debug > EnvEnum.VERBOSE.value,
            )

        # Robot setup
        self.use_robot = use_robot
        self.use_gello = use_gello

        # use teleop without residual policy
        if use_robot and not use_residual_policy:
            if bimanual:
                self.left_xarm_controller = XarmController(
                    start_time=time.time(),
                    ip=bimanual_robot_ip[0],
                    gripper_enable=gripper_enable,
                    mode=mode,
                    command_mode='joints' if use_gello else 'cartesian',
                    robot_id=0,
                    verbose=False,
                )
                self.right_xarm_controller = XarmController(
                    start_time=time.time(),
                    ip=bimanual_robot_ip[1],
                    gripper_enable=gripper_enable,
                    mode=mode,
                    command_mode='joints' if use_gello else 'cartesian',
                    robot_id=1,
                    verbose=False,
                )
                self.xarm_controller = None
            else:
                self.xarm_controller = XarmController(
                    start_time=time.time(),
                    ip=robot_ip,
                    gripper_enable=gripper_enable,
                    mode=mode,
                    command_mode='joints' if use_gello else 'cartesian',
                    robot_id=-1,
                    verbose=False,
                )
                self.left_xarm_controller = None
                self.right_xarm_controller = None
        # use teleop with residual policy
        elif use_robot and use_residual_policy:
            if bimanual:
                self.left_xarm_controller = XarmController(
                    start_time=time.time(),
                    ip=bimanual_robot_ip[0],
                    gripper_enable=gripper_enable,
                    mode=mode,
                    command_mode='joints' if use_gello else 'cartesian',
                    robot_id=0,
                    verbose=False,
                )
                self.right_xarm_controller = XarmController(
                    start_time=time.time(),
                    ip=bimanual_robot_ip[1],
                    gripper_enable=gripper_enable,
                    mode=mode,
                    command_mode='joints' if use_gello else 'cartesian',
                    robot_id=1,
                    verbose=False,
                )
                self.xarm_controller = None
            else:
                self.xarm_controller = XarmController(
                    start_time=time.time(),
                    ip=robot_ip,
                    gripper_enable=gripper_enable,
                    mode=mode,
                    command_mode='joints' if use_gello else 'cartesian',
                    robot_id=-1,
                    verbose=False,
                )
                self.left_xarm_controller = None
                self.right_xarm_controller = None
        # disable teleop
        else:
            self.left_xarm_controller = None
            self.right_xarm_controller = None
            self.xarm_controller = None

        # subprocess can only start a process object created by current process
        self._real_alive = mp.Value('b', False)

        self.start_time = 0
        mp.Process.__init__(self)
        self._alive = mp.Value('b', False)

        # pygame
        # Initialize a separate Pygame window for image display
        img_w, img_h = 480, 480
        col_num = 1 # TODO: add rgb image column
        self.screen_width, self.screen_height = img_w * col_num, img_h
        self.image_window = None

        # Shared memory for image data
        self.image_data = mp.Array('B', np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8).flatten())

        # robot eef
        self.eef_point = np.array([[0.0, 0.0, 0.175]])  # the eef point in the gripper frame

        # residual policy inputs
        self.use_residual_policy = use_residual_policy
        self.model_path = load_model
        self.prev_time = 0
        self.update_flag = False


        if self.use_gello:
            self.teleop = GelloTeleopResidual(bimanual=self.bimanual)
        else:
            self.teleop = KeyboardTeleop()

        # ------------------------------------------------- Residual Policy -------------------------------------------------        
        # init buffers
        self.visual_obs = np.zeros((120*120), dtype=np.float32)
        self.teleop_states_obs = np.zeros((10), dtype=np.float32) # 10D state
        self.robot_states_obs = np.zeros((10), dtype=np.float32) # 10D state
        self.robot_qpos = np.zeros((7), dtype=np.float32) # 8D qpos
        self.teleop_command = mp.Array('d', [0.0]*8) # 8D qpos

        # policy params
        self.alpha = 0.1
        num_samples = 3
        sample_interval = 5

        # init teleop history
        self.teleop_hist = HistoryBuffer(num_envs=1, 
                            hist_length=num_samples * sample_interval + 1,
                            state_dim=10,
                            num_samples=num_samples,
                            sample_spacing=sample_interval,
                            device='cpu' # NOTE: use cpu for real env
                            )

        # normalizaiton params
        fingertip_lower_limit = [0.20, -0.4, 0.01, 
                                -1.05, -1.05, -1.05, -1.05, -1.05, -1.05, 
                                -1.0]
        fingertip_upper_limit = [0.60, 0.4, 0.5, 
                                1.05, 1.05, 1.05, 1.05, 1.05, 1.05, 
                                1.0]

        self.fingertip_low = torch.tensor([fingertip_lower_limit])
        self.fingertip_high = torch.tensor([fingertip_upper_limit])
        self.fingertip_normalizer = ActionNormalizer(self.fingertip_low, self.fingertip_high) # NOTE:

        state_obs_lower_limit = fingertip_lower_limit * (num_samples+1) 
        state_obs_upper_limit = fingertip_upper_limit * (num_samples+1) 
        state_obs_low = torch.tensor([state_obs_lower_limit])
        state_obs_high = torch.tensor([state_obs_upper_limit])
        self.state_obs_normalizer = ActionNormalizer(state_obs_low, state_obs_high)

        # load policy
        # self.cube_net = ResidualStudentTeacher(num_student_obs=60+120*120, num_teacher_obs=69, num_actions=10, student_hidden_size=512, teacher_hidden_size=256)
        self.insertion_mlp = ResidualStudentTeacher(num_student_obs=40+120*120, num_teacher_obs=53, num_actions=10, student_hidden_size=512, teacher_hidden_size=256)
        self.policy = self.get_policy(self.insertion_mlp, self.model_path)
        assert(use_gello, "Residual policy requires gello")
        print("Residual policy loaded")

        # fingertip2ee real
        self.fingertip2ee_offset_real = torch.tensor([[0.0, 0.0, 0.155]]) # check values

        # collect training teleop data
        self.collect_teleop_traj = False
        self.teleop_traj_real = []
        self.task = "insertion"
        self.training_set_num = 4
        self.demo_num = 1

        # visualization options:
        self.print_all_intermediate_value = True
        self.show_camera = True

        # Options for storing and play traj for sim2real
        self.enable_residual = False
        self.use_offline_teleop_traj = True
        self.store_traj_data = True
        self.storing_path = f"experiments/real_world/modules_teleop/RRL/tasks/{self.task}/sim2real/no_res_traj1/"

        if self.use_offline_teleop_traj:
            self.traj_length, _ = torch.load(f"{self.storing_path}/sim_teleop_obs.pt", weights_only=True).shape
            self.offline_teleop_fingertip_traj = torch.load(f"{self.storing_path}/sim_teleop_obs.pt", weights_only=True).to('cpu')
        if self.store_traj_data:
            os.makedirs(os.path.dirname(self.storing_path), exist_ok=True)
            self.state_list = []
            self.depth_list = []
            self.nres_list = []

    def get_policy(self, network, model_path) -> Callable:
        loaded_dict = torch.load(model_path, weights_only=False)
        network.load_state_dict(loaded_dict['model_state_dict'])
        network.eval()

        policy = lambda x: network.act_inference(x)
        return policy
    
    def get_observations(self, i=None):
        if self.use_offline_teleop_traj:
            teleop_hist = self.offline_teleop_fingertip_traj[i].clone().reshape(1, -1) # playing sim recorded base traj
            teleop_fingertip_base = teleop_hist[:, :10].clone() # shape (1, 10)
        else:
            if i == 0:
                self.teleop_hist.initialize(self.fingertip_10D.clone())
            teleop_fingertip_base = self.teleop_fingertip_10D.clone().reshape(1, -1) # shape (1, 10)
            self.teleop_hist.append(teleop_fingertip_base.clone())
            teleop_hist = self.teleop_hist.get_history() 

        state_obs = torch.cat((
            self.fingertip_10D,
            teleop_hist,
        ), dim=-1)

        normalized_state_obs = self.state_obs_normalizer.normalize(state_obs)
        normalized_actor_obs = torch.cat((
            normalized_state_obs,
            self.normalized_visual_obs
        ), dim=-1)

        if self.store_traj_data:
            self.state_list.append(self.fingertip_10D.clone())
            self.depth_list.append(self.depth_noised.copy())

        if self.print_all_intermediate_value:
            print("teleop fingertip base", teleop_fingertip_base)
            print("robot fingertip", self.fingertip_10D)

        return teleop_fingertip_base, normalized_actor_obs        

    def real_start(self, start_time) -> None:
        self._real_alive.value = True
        print("starting real env")
        
        # Realsense camera setup
        self.realsense.start()
        self.realsense.restart_put(start_time + 1)
        # time.sleep(2)

        # Perception setup
        if self.perception is not None:
            self.perception.start()
    
        # Robot setup
        if self.use_robot:
            if self.bimanual:
                self.left_xarm_controller.start()
                self.right_xarm_controller.start()
            else:
                self.xarm_controller.start()

        if self.use_gello:
            self.teleop.start()
        
        while not self.real_alive:
            self._real_alive.value = True
            print(".", end="")
            time.sleep(0.5)
        
        print("real env started")

        self.update_real_state_t = threading.Thread(name="update_real_state", target=self.update_real_state)
        self.update_real_state_t.start()

    def real_stop(self, wait=False) -> None:
        self._real_alive.value = False
        if self.use_robot:
            if self.bimanual and self.left_xarm_controller.is_controller_alive:
                self.left_xarm_controller.stop()
            if self.bimanual and self.right_xarm_controller.is_controller_alive:
                self.right_xarm_controller.stop()
            if not self.bimanual and self.xarm_controller.is_controller_alive:
                self.xarm_controller.stop()
        if self.perception is not None and self.perception.alive.value:
            self.perception.stop()
        self.realsense.stop(wait=False)

        if self.use_gello:
            self.teleop.stop()

        # self.image_display_thread.join()
        self.update_real_state_t.join()
        print("real env stopped")

    @property
    def real_alive(self) -> bool:
        alive = self._real_alive.value
        if self.perception is not None:
            alive = alive and self.perception.alive.value
        if self.use_robot:
            controller_alive = \
                (self.bimanual and self.left_xarm_controller.is_controller_alive and self.right_xarm_controller.is_controller_alive) \
                or (not self.bimanual and self.xarm_controller.is_controller_alive)
            alive = alive and controller_alive
        self._real_alive.value = alive
        return self._real_alive.value

    def _update_perception(self) -> None:
        if self.perception.alive.value:
            if not self.perception.perception_q.empty():
                self.state["perception_out"] = {
                    "time": time.time(),
                    "value": self.perception.perception_q.get()
                }
        return

    def _update_teleop_command(self) -> None:
        if self.xarm_controller.is_controller_alive:
            if all(value == 0.0 for value in self.teleop_command):
                teleop_comm_ee_10D = np.zeros(10)
            else:
                teleop_qpos_np = np.frombuffer(self.teleop_command.get_obj(), dtype=np.float64) # shape (8,)
                teleop_ee_7D = self.fk(teleop_qpos_np) # np array (8,)
                teleop_binary_gripper = np.array([-1.0]) if teleop_qpos_np[-1] < 0.5 else np.array([1.0]) # -1.0 = open, 1.0 = close

                teleop_comm_ee_10D = np.concatenate([teleop_ee_7D[:3], 
                                                     quat_to_6d_np(teleop_ee_7D[3:7]), 
                                                     teleop_binary_gripper], axis=0) # np array (10,)
            self.state["teleop_ee_10D"] = { # 10D state
                "time": time.time(),
                "value": teleop_comm_ee_10D # NOTE [pos, 6D orientation, gripper_qpos]
            }
        return

    def _update_robot(self) -> None:
        if self.bimanual:
            if self.left_xarm_controller.is_controller_alive and self.right_xarm_controller.is_controller_alive:
                if not self.left_xarm_controller.cur_trans_q.empty() and not self.right_xarm_controller.cur_trans_q.empty():
                    self.state["robot_out"] = {
                        "time": time.time(),
                        "left_value": self.left_xarm_controller.cur_trans_q.get(),
                        "right_value": self.right_xarm_controller.cur_trans_q.get()
                    }
                if not self.left_xarm_controller.cur_gripper_q.empty() and not self.right_xarm_controller.cur_trans_q.empty():
                    self.state["gripper_out"] = {
                        "time": time.time(),
                        "left_value": self.left_xarm_controller.cur_gripper_q.get(),
                        "right_value": self.right_xarm_controller.cur_gripper_q.get()
                    }
        else:
            if self.xarm_controller.is_controller_alive:
                if not self.xarm_controller.cur_trans_q.empty():
                    self.state["robot_out"] = {
                        "time": time.time(),
                        "value": self.xarm_controller.cur_trans_q.get()
                    }
                if not self.xarm_controller.cur_gripper_q.empty():
                    self.state["gripper_out"] = {
                        "time": time.time(),
                        "value": self.xarm_controller.cur_gripper_q.get()
                    }
                if not self.xarm_controller.cur_qvel_q.empty():
                    self.state["robot_qvel_out"] = {
                        "time": time.time(),
                        "value": self.xarm_controller.cur_qvel_q.get() # shape(7,)
                    }
                if not self.xarm_controller.cur_qpos_q.empty():
                    self.state["robot_qpos_out"] = {
                        "time": time.time(),
                        "value": self.xarm_controller.cur_qpos_q.get() # shape(7,)
                    }
        return

    def _update_robot_states_obs(self) -> None:
        # update 10D ee state np
        fk = self.state['robot_out']['value']
        self.robot_states_obs[:3] = fk[:3,3]
        self.robot_states_obs[3:9] = rotation_matrix_to_6d_np(fk[:3,:3]) 
        gripper_qpos = self.gripper_real2sim(self.state['gripper_out']['value']) 
        self.robot_states_obs[-1] = -1.0 if gripper_qpos < 0.5 else 1.0 # -1.0 = open, 1.0 = close

        # update qpos
        self.robot_qpos = self.state["robot_qpos_out"]["value"] # np array (7,)

        # update 10D fingertip state torch
        curr_robot_ee_b = torch.from_numpy(self.robot_states_obs.copy()).float().reshape(1, -1)        
        fingertip_pos = curr_robot_ee_b[:, :3].clone() + tf_vector(quat_from_6d(curr_robot_ee_b[:,3:9]), self.fingertip2ee_offset_real)
        self.fingertip_10D = torch.cat((fingertip_pos, curr_robot_ee_b[:, 3:]), dim=-1) # shape (1, 10)

    def _update_teleop_states_obs(self) -> None: 
        # update 10D ee state np
        self.teleop_states_obs = self.state['teleop_ee_10D']['value'] # (pos, orn, gripper binary)

        # update 10D fingertip state torch
        curr_teleop_comm_b = torch.from_numpy(self.teleop_states_obs.copy()).float().reshape(1, -1)
        teleop_fingertip_pos = curr_teleop_comm_b[:, :3].clone() + tf_vector(quat_from_6d(curr_teleop_comm_b[:,3:9]), self.fingertip2ee_offset_real)
        self.teleop_fingertip_10D = torch.cat((teleop_fingertip_pos, curr_teleop_comm_b[:, 3:]), dim=-1) # shape (1, 10)

    def _update_vision_obs(self) -> None:
        raw_depth = self.state["perception_out"]["value"].copy()
        depth_filtered = filter_depth_real(raw_depth) # shape (120, 120), should be same reading as sim
        self.depth_noised = add_noise_in_depth_band_np(depth_filtered)
        normalized_depth = self.normalize_depth_01(self.depth_noised.flatten(), 
                                                  min_depth=0.1, 
                                                  max_depth=0.5,
                                                  unit="m")
        self.normalized_visual_obs = torch.from_numpy(normalized_depth.copy()).float().reshape(1, -1)

    def update_real_state(self) -> None:
        while self.real_alive:
            try: # NOTE: by design, updates in rotation, i.e., don't need to synchronize time steps
                if self.use_robot:
                    start = time.time()
                    self._update_robot()
                    # print("robot state obs update freq: ", 1/(time.time()-start))
                if self.perception is not None:
                    start = time.time()
                    self._update_perception()
                    # print("perception obs update freq: ", 1/(time.time()-start))
                if self.use_gello and self.use_residual_policy:
                    start = time.time()
                    self._update_teleop_command()
                    # print("teleop command obs update freq: ", 1/(time.time()-start))
            except:
                print(f"Error in update_real_state")
                break
        print("update_real_state stopped")

    def display_image(self):
        pygame.init()
        self.image_window = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption('Image Display Window')
        while self._alive.value:
            # Extract image data from the shared array
            image = np.frombuffer(self.image_data.get_obj(), dtype=np.uint8).reshape((self.screen_height, self.screen_width, 3))
            pygame_image = pygame.surfarray.make_surface(image.swapaxes(0, 1))

            # Blit the image to the window
            self.image_window.blit(pygame_image, (0, 0))
            pygame.display.update()

            # Handle events (e.g., close window)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.stop()
                    pygame.quit()
                    return

            time.sleep(1 / self.realsense.capture_fps)  # 30 FPS
        print("Image display stopped")

    def start_image_display(self):
        # Start a thread for the image display loop
        self.image_display_thread = threading.Thread(name="display_image", target=self.display_image)
        self.image_display_thread.start()

    def get_dones(self, i) -> None:
        if self.use_offline_teleop_traj and i == self.traj_length:
            print("end of traj")
            
            # save all tensors
            if self.store_traj_data:
                state_tensor = torch.stack(self.state_list, dim=0).reshape(-1, 10)  # shape: (T, 10)
                torch.save(state_tensor, os.path.join(self.storing_path, "real_state_obs.pt")) 
                residual_tensor = torch.stack(self.nres_list, dim=0).reshape(-1, 10)  # shape: (T, 10)
                torch.save(residual_tensor, os.path.join(self.storing_path, "real_residual.pt"))
                depth_array = np.stack(self.depth_list, axis=0)  # shape: (T, 120, 120) where T is number of timesteps
                np.save(os.path.join(self.storing_path, "real_depth_obs.npy"), depth_array)  # Save to disk

                print(f"init pos teleop: {self.offline_teleop_fingertip_traj[0,:3]}")
                print(f"init pos real robot: {state_tensor[0,:3]}")
                print(f"real data stored at {self.storing_path}")
                
                print(f"Saved {depth_array.shape} depth frames to real_depth_obs.npy")
            exit()

    def pre_physics_step(self, teleop_base, nres_10D) -> np.ndarray: 
        if self.enable_residual:
            nbase = self.fingertip_normalizer.normalize(teleop_base.clone())
            n_fingertip_goal_10D = nbase + self.alpha * nres_10D.clone()
            fingertip_goal = self.fingertip_normalizer.denormalize(n_fingertip_goal_10D.clone())
        else:
            fingertip_goal = teleop_base.clone()
        fingertip_goal[:,:3] = torch.clamp(fingertip_goal[:,:3], self.fingertip_low[:,:3], self.fingertip_high[:,:3])
        fingertip_goal[:,-1] = torch.where(fingertip_goal[:, -1] > 0.0, 
                                        torch.tensor(1.0, device=fingertip_goal.device),  # closed
                                        torch.tensor(-1.0, device=fingertip_goal.device)) # open

        if torch.norm(fingertip_goal[:,:3] - teleop_base[:,:3]) > 1.0:
            print("fingertip goal is too far from teleop fingertip")
            exit()

        # compute ee goal
        ee_pos = fingertip_goal[:, :3].clone() + tf_vector(quat_from_6d(fingertip_goal[:, 3:9]), -1*self.fingertip2ee_offset_real)
        ee_goal = torch.cat((ee_pos, fingertip_goal[:, 3:]), dim=-1)
        
        # compute ik
        quat = quat_from_6d(ee_goal[:,3:9])
        r, p, y = euler_xyz_from_quat(quat) # type: ignore
        finger_qpos = 0.56 if fingertip_goal[0,-1] > 0.0 else 0.0

        ee_goal_6D = torch.cat([ee_goal[0,:3], r, p, y], dim=0) # shape (8,)
        ee_goal_6D_np = ee_goal_6D.detach().cpu().numpy() # np array (8,)
        qpos_arm_goal = self.ik(self.robot_qpos, ee_goal_6D_np) # np array (7,)
        qpos_goal = np.append(qpos_arm_goal, finger_qpos) # np array (8,)

        # apply action
        self.teleop.comm_with_residual[:] = qpos_goal # np array ( 8,)
                
        if self.print_all_intermediate_value:
            print("fingertip goal: ", fingertip_goal)

        return qpos_goal

    def run(self) -> None:
        fps = 30
        print(f"running policy at {fps} Hz")

        i = 0
        while self.alive:
            try:
                tic = time.time()
                self.teleop_command[:] = list(self.teleop.command)

                # compute intermediate values
                self._update_vision_obs()
                self._update_robot_states_obs()
                self._update_teleop_states_obs()

                # store teleop traj
                if self.collect_teleop_traj and self.teleop.start_recording.value:
                    # store teleop traj
                    if self.collect_teleop_traj:
                        print("current teleop comm: ", teleop_base)
                        self.teleop_traj_real.append(teleop_base.clone())

                # controlling using residual policy or teleop
                if self.teleop.start_residual_policy.value or self.use_offline_teleop_traj:
                    # check if reached end of offline teleop traj
                    self.get_dones(i) 

                    # get obs
                    teleop_base, nobs = self.get_observations(i)

                    # run policy
                    nres_10D = self.policy(nobs)
                    if self.store_traj_data:
                        self.nres_list.append(nres_10D.clone())

                    # get teleop command
                    qpos_goal = self.pre_physics_step(teleop_base, nres_10D)
                    self.teleop.comm_with_residual[:] = qpos_goal

                    # update step timer and match control freq
                    i += 1
                else:
                    self.teleop.comm_with_residual[:] = self.teleop_command[:].copy()

                if self.show_camera:
                    raw_depth = self.state["perception_out"]["value"].copy()
                    depth_vis = filter_depth_for_visualization(raw_depth, crop_depth=True)
                    cv2.imshow("depth", depth_vis)
                    cv2.waitKey(1)

                time.sleep(max(0, 1 / fps - (time.time() - tic)))
                # print("freq: ", 1 / (time.time() - tic))
            
            except BaseException as e:
                print(f"Error in robot teleop env: {e.with_traceback()}")
                break
        
        self.stop()
        print("RealEnv process stopped")

    def euler_from_quat_np(self, quat: np.ndarray) -> np.ndarray:
        """
        Convert a quaternion (w, x, y, z) to Euler angles (roll, pitch, yaw) in ZYX order.

        Args:
            quat (np.ndarray): Quaternion in (w, x, y, z) format.

        Returns:
            np.ndarray: Euler angles in (roll, pitch, yaw) format (radians).
        """
        if quat.shape != (4,):
            raise ValueError(f"Invalid quaternion shape {quat.shape}. Expected (4,).")

        w, x, y, z = quat

        # Compute roll (φ)
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))

        # Compute pitch (θ)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.sign(sinp) * (np.pi / 2)  # Use 90 degrees if out of range
        else:
            pitch = np.arcsin(sinp)

        # Compute yaw (ψ)
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))

        return np.array([roll, pitch, yaw])  # (φ, θ, ψ)

    def quat_from_matrix_np(self, matrix: np.ndarray) -> np.ndarray:
        """
        Convert a 3×3 rotation matrix to a quaternion (w, x, y, z) using NumPy.

        Args:
            matrix (np.ndarray): A 3×3 rotation matrix.

        Returns:
            np.ndarray: Quaternion in (w, x, y, z) format.
        """
        if matrix.shape != (3, 3):
            raise ValueError(f"Invalid rotation matrix shape {matrix.shape}. Expected (3, 3).")

        m00, m01, m02 = matrix[0, 0], matrix[0, 1], matrix[0, 2]
        m10, m11, m12 = matrix[1, 0], matrix[1, 1], matrix[1, 2]
        m20, m21, m22 = matrix[2, 0], matrix[2, 1], matrix[2, 2]

        trace = m00 + m11 + m22

        if trace > 0.0:
            S = np.sqrt(trace + 1.0) * 2  # S = 4 * qw
            qw = 0.25 * S
            qx = (m21 - m12) / S
            qy = (m02 - m20) / S
            qz = (m10 - m01) / S
        elif m00 > m11 and m00 > m22:
            S = np.sqrt(1.0 + m00 - m11 - m22) * 2  # S = 4 * qx
            qw = (m21 - m12) / S
            qx = 0.25 * S
            qy = (m01 + m10) / S
            qz = (m02 + m20) / S
        elif m11 > m22:
            S = np.sqrt(1.0 + m11 - m00 - m22) * 2  # S = 4 * qy
            qw = (m02 - m20) / S
            qx = (m01 + m10) / S
            qy = 0.25 * S
            qz = (m12 + m21) / S
        else:
            S = np.sqrt(1.0 + m22 - m00 - m11) * 2  # S = 4 * qz
            qw = (m10 - m01) / S
            qx = (m02 + m20) / S
            qy = (m12 + m21) / S
            qz = 0.25 * S

        return np.array([qw, qx, qy, qz])

    def gripper_real2sim(self, real_gripper_status: float) -> float:
        """
        Maps the real robot's gripper status (800 open, 0 closed) to the simulator's (0.0 open, 0.84 closed)
        using a quadratic function.

        Args:
            real_gripper_status (float): Gripper status from the real robot (range: 840 to 0).

        Returns:
            float: Mapped gripper status for the simulator (range: 0.0 to 0.84).
        """
        qpos = (840 - real_gripper_status) / 840
        qpos = np.clip(qpos, 0.0, 0.84)  # Ensure qpos is within [0.0, 1.0]
        return qpos

    def mp2np(self, mp_array: mp.Array) -> np.array:
        return np.frombuffer(mp_array.get_obj())  

    def ik(self, curr_qpos: np.array, cartesian_goal: np.array) -> np.array:
        '''
        ee_goal: shape (6, )
        '''
        qpos = self.kin_helper.compute_ik_sapien(curr_qpos, cartesian_goal)
        return qpos.reshape(-1,)

    def fk(self, joints: np.array) -> np.array:
        fk = self.kin_helper.compute_fk_sapien_links(joints[:7], [self.kin_helper.sapien_eef_idx])[0]
        pos = fk[:3, 3]
        quat = self.quat_from_matrix_np(fk[:3, :3])
        return np.concatenate([pos, quat])

    def normalize_depth_01(self, depth_input, min_depth=0.1, max_depth=0.5, unit="m"):
        """
        Normalize depth values to the range [0, 1].
        args:
            depth_input: np.array, shape (120, 120)
        """
        if unit == "mm":
            depth_input = depth_input / 1000.0
        elif unit == "cm":
            depth_input = depth_input / 100.0

        # # remove NaN and Inf values
        # depth_filtered = np.nan_to_num(
        #     depth_input,
        #     nan=0.0,
        #     posinf=max_depth,
        #     neginf=min_depth,
        # )
        
        # # clip depth
        # depth_out = np.clip(depth_filtered, min_depth, max_depth)
        depth_out = (depth_input - min_depth) / (max_depth - min_depth)
        return depth_out.flatten().astype(np.float32)  # Ensure final dtype is float32

    def get_intrinsics(self):
        return self.realsense.get_intrinsics()

    def get_extrinsics(self):
        return self.state["extr"]

    @property
    def alive(self) -> bool:
        alive = self._alive.value and self.real_alive
        self._alive.value = alive
        return alive

    def start(self) -> None:
        self.start_time = time.time()
        self._alive.value = True
        self.real_start(time.time())
        # self.start_image_display()
        super().start()

    def stop(self) -> None:
        self._alive.value = False
        if self.collect_teleop_traj:
            path = os.path.join("experiments/real_world/modules_teleop/RRL/tasks", f"{self.task}/training_set{self.training_set_num}/demo_traj{self.demo_num}.pt") #TODO change path
            os.makedirs(os.path.dirname(path), exist_ok=True)
            teleop_fingertip_traj = torch.stack(self.real_traj, dim=0).reshape(-1, 10)  # shape: (T, 10)
            torch.save(teleop_fingertip_traj, path) 
            print(f"teleop data for {self.task} task. Training set {self.training_set_num}, demo traj {self.demo_num} of {teleop_fingertip_traj.shape[0]} timesteps saved at {path}")
            print("----------------------------------------------------------------------------")
        self.real_stop()


