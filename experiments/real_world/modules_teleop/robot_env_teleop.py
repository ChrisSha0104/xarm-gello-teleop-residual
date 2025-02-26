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

from .encoded_actor_critic import ActorCriticVisual
from pynput import keyboard
from pathlib import Path
from copy import deepcopy

from utils import get_root, mkdir
root: Path = get_root(__file__)

from modules_teleop.perception import Perception
from modules_teleop.xarm_controller import XarmController
from modules_teleop.teleop_keyboard import KeyboardTeleop
from modules_teleop.kinematics_utils import KinHelper
from gello.teleop_gello import GelloTeleop
from gello.teleop_gello_residual import GelloTeleopResidual
from camera.multi_realsense import MultiRealsense
from camera.single_realsense import SingleRealsense


class EnvEnum(Enum):
    NONE = 0
    INFO = 1
    DEBUG = 2
    VERBOSE = 3

class RobotTeleopEnv(mp.Process):

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
    
        # auto or manual
        # self.realsense.set_exposure(exposure=None)
        # self.realsense.set_white_balance(white_balance=None)
        self.realsense.set_exposure(exposure=100, gain=60)  # 100: bright, 60: dark
        self.realsense.set_white_balance(3800)

        # base calibration
        # self.calibrate_result_dir = calibrate_result_dir
        # with open(f'{self.calibrate_result_dir}/base.pkl', 'rb') as f:
        #     base = pickle.load(f)
        # if self.bimanual:
        #     R_leftbase2board = base['R_leftbase2world']
        #     t_leftbase2board = base['t_leftbase2world']
        #     R_rightbase2board = base['R_rightbase2world']
        #     t_rightbase2board = base['t_rightbase2world']
        #     leftbase2world_mat = np.eye(4)
        #     leftbase2world_mat[:3, :3] = R_leftbase2board
        #     leftbase2world_mat[:3, 3] = t_leftbase2board
        #     self.state["b2w_l"] = leftbase2world_mat
        #     rightbase2world_mat = np.eye(4)
        #     rightbase2world_mat[:3, :3] = R_rightbase2board
        #     rightbase2world_mat[:3, 3] = t_rightbase2board
        #     self.state["b2w_r"] = rightbase2world_mat
        # else:
        base2world_mat = np.eye(4)
        self.state["b2w"] = base2world_mat

        # # camera calibration
        # extr_list = []
        # with open(f'{self.calibrate_result_dir}/rvecs.pkl', 'rb') as f:
        #     rvecs = pickle.load(f)
        # with open(f'{self.calibrate_result_dir}/tvecs.pkl', 'rb') as f:
        #     tvecs = pickle.load(f)
        # for i in range(len(self.serial_numbers)):
        #     device = self.serial_numbers[i]
        #     R_world2cam = cv2.Rodrigues(rvecs[device])[0]
        #     t_world2cam = tvecs[device][:, 0]
        #     extr_mat = np.eye(4)
        #     extr_mat[:3, :3] = R_world2cam
        #     extr_mat[:3, 3] = t_world2cam
        #     extr_list.append(extr_mat)
        # self.state["extr"] = np.stack(extr_list)

        # save calibration
        # mkdir(root / "log" / self.data_dir / self.exp_name / "calibration", overwrite=False, resume=False)
        # subprocess.run(f'cp -r {self.calibrate_result_dir}/* {str(root)}/log/{self.data_dir}/{self.exp_name}/calibration', shell=True)

        # Perception setup
        if perception is not None:
            assert isinstance(perception, Perception)
            self.perception = perception
        else:
            self.perception = Perception(
                realsense=self.realsense,
                capture_fps=self.realsense.capture_fps,  # mush be the same as realsense capture fps 
                record_fps=record_fps,
                record_time=record_time,
                process_func=perception_process_func,
                exp_name=exp_name,
                data_dir=data_dir,
                verbose=self.debug > EnvEnum.VERBOSE.value)

        # Robot setup
        self.use_robot = use_robot
        self.use_gello = use_gello

        # use teleop without residual policy
        if use_robot and not use_residual_policy:
            # if xarm_controller is not None:
            #     assert isinstance(xarm_controller, XarmController)
            #     self.xarm_controller = xarm_controller
            # else:
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
        img_w, img_h = 848, 480
        col_num = 2
        self.screen_width, self.screen_height = img_w * col_num, img_h * len(self.realsense.serial_numbers)
        self.image_window = None

        # Shared memory for image data
        self.image_data = mp.Array('B', np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8).flatten())

        # record robot action
        # self.robot_record_restart = mp.Value('b', False)
        # self.robot_record_stop = mp.Value('b', False)

        # robot eef
        self.eef_point = np.array([[0.0, 0.0, 0.175]])  # the eef point in the gripper frame

        # residual policy inputs
        self.use_residual_policy = use_residual_policy
        self.model_path = load_model
        self.prev_time = 0
        self.update_flag = False
        self.kin_helper = KinHelper(robot_name='xarm7')

        if self.use_residual_policy:
            self.actor_critic = ActorCriticVisual(7*8+120*120,8*8+120*120,8) #TODO: hardcoded input dim
            self.policy = self.get_policy(self.actor_critic, self.model_path)
            assert(use_gello, "Residual policy requires gello")
            print("Residual policy loaded")

            self.visual_obs = np.zeros((120*120), dtype=np.float32)
            self.teleop_states_obs = np.zeros((8*4), dtype=np.float32) #mp.Array('d', [0.0]*(8*4))
            self.robot_states_obs = np.zeros((8*3), dtype=np.float32) #mp.Array('d', [0.0]*(8*3))
            self.teleop_command = mp.Array('d', [0.0]*8)

            if self.use_gello and not self.use_residual_policy:
                self.teleop = GelloTeleop(bimanual=self.bimanual)
            elif self.use_gello and self.use_residual_policy:
                self.teleop = GelloTeleopResidual(bimanual=self.bimanual)
            else:
                self.teleop = KeyboardTeleop()

    def get_policy(self, actor_critic, model_path) -> Callable: #TODO: add empirical normalizatiopn
        checkpoint = torch.load(model_path)
        actor_critic.load_state_dict(checkpoint['model_state_dict'])
        policy = actor_critic.act_inference
        return policy
    
    def get_observations(self):
        # visual obs
        raw_depth = self.state["perception_out"]["value"][0]["depth"] # 480*848
        cropped_depth = cv2.resize(raw_depth, (120, 120)).flatten()
        self.visual_obs = self.normalize_depth_01(cropped_depth)

        # print("robot time", self.state["robot_out"]["time"])
        # print("teleop time", self.state["teleop_command"]["time"])

        self._update_teleop_states_obs()
        self._update_robot_states_obs()

        obs = np.concatenate([self.visual_obs, self.teleop_states_obs, self.robot_states_obs], dtype=np.float32)
        # print("visual obs: ", np.mean(self.visual_obs))
        # print("teleop obs: ", self.teleop_states_obs[:8])
        # print("robot obs: ", self.robot_states_obs[:8])
        obs = torch.tensor(obs.reshape(1,-1), dtype=torch.float32)
        return obs

    def real_start(self, start_time) -> None:
        self._real_alive.value = True
        print("starting real env")
        
        # Realsense camera setup
        self.realsense.start()
        self.realsense.restart_put(start_time + 1)
        time.sleep(2)

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
        
        # get intrinsics
        intrs = self.get_intrinsics()
        intrs = np.array(intrs)
        np.save(root / "log" / self.data_dir / self.exp_name / "calibration" / "intrinsics.npy", intrs)
        
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

        self.image_display_thread.join()
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
                teleop_command_ee = np.zeros(8)
            else:
                teleop_command_np = np.frombuffer(self.teleop_command.get_obj(), dtype=np.float64)
                teleop_command_ee = self.fk(teleop_command_np)
            self.state["teleop_command"] = {
                "time": time.time(),
                "value": teleop_command_ee
            }
            # if self.update_flag == True:
            #     self._update_teleop_states_obs()
            #     self.update_flag = False
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
                # if self.use_residual_policy:
                #     if (time.time() - self.prev_time) > 0.01: 
                #         self._update_robot_states_obs()
                #         self.prev_time = time.time()
                #         self.update_flag = True
        return

    def _update_robot_states_obs(self) -> None: #TODO debug
        self.robot_states_obs = np.roll(self.robot_states_obs, shift=8, axis=0)
        self.robot_states_obs[:3] = self.state['robot_out']['value'][3,:3]
        self.robot_states_obs[3:7] = self.quat_from_matrix_np(self.state['robot_out']['value'][:3,:3]) 
        gripper_obs = self.gripper_real2sim(self.state['gripper_out']['value']) 
        self.robot_states_obs[7] = gripper_obs
        # print("most recent robot obs: ", self.robot_states_obs[:8])
        

    def _update_teleop_states_obs(self) -> None: 
        self.teleop_states_obs = np.roll(self.teleop_states_obs, shift=8, axis=0)
        self.teleop_states_obs[:7] = self.state['teleop_command']['value'][:7]
        gripper_obs = self.gripper_real2sim(self.state['teleop_command']['value'][-1]) 
        self.teleop_states_obs[7] = gripper_obs
        # print("most recent teleop obs: ",self.teleop_states_obs[:8])

    def update_real_state(self) -> None:
        while self.real_alive:
            try: # NOTE: by design, updates in rotation, i.e., don't need to synchronize time steps
                if self.use_robot:
                    self._update_robot()
                if self.perception is not None:
                    self._update_perception()
                if self.use_gello and self.use_residual_policy:
                    self._update_teleop_command()
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

    def run(self) -> None:
        # if self.use_robot:
            # if self.use_gello and not self.use_residual_policy:
            #     teleop = GelloTeleop(bimanual=self.bimanual)
            # elif self.use_gello and self.use_residual_policy:
            #     teleop = GelloTeleopResidual(bimanual=self.bimanual)
            # else:
            #     teleop = KeyboardTeleop()
            # self.teleop.start()
        time.sleep(1)

        robot_record_dir = root / "log" / self.data_dir / self.exp_name / "robot"
        os.makedirs(robot_record_dir, exist_ok=True)

        # initialize images
        rgbs = []
        depths = []
        resolution = self.realsense.resolution
        for i in range(len(self.realsense.serial_numbers)):
            rgbs.append(np.zeros((resolution[1], resolution[0], 3), np.uint8))
            depths.append(np.zeros((resolution[1], resolution[0]), np.uint16))

        fps = self.record_fps if self.record_fps > 0 else self.realsense.capture_fps  # visualization fps
        idx = 0
        while self.alive:
            try:
                tic = time.time()
                state = deepcopy(self.state)
                if self.bimanual:
                    b2w_l = state["b2w_l"]
                    b2w_r = state["b2w_r"]
                else:
                    b2w = state["b2w"]

                if self.teleop.record_start.value == True:
                    self.perception.set_record_start()
                    self.teleop.record_start.value = False

                if self.teleop.record_stop.value == True:
                    self.perception.set_record_stop()
                    self.teleop.record_stop.value = False

                idx += 1

                self.teleop_command[:] = list(self.teleop.command)

                if self.use_gello and self.use_residual_policy:
                    if not all(value == 0.0 for value in self.teleop_command):
                        obs = self.get_observations()
                        residual = self.policy(obs) # torch tensor (1, 8)
                        # print("ee residual: ", residual)
                        ee_goal = self.teleop_states_obs[:8] + residual.detach().numpy() # np array [1,8]
                        qpos_arm_goal = self.ik(np.array(self.teleop_command[:7]), ee_goal) # np array (7,)=
                        qpos_goal = np.append(qpos_arm_goal, (self.teleop_command[-1]+ee_goal[0,-1])) # np array (8,)
                        # print("joint residual", qpos_goal - self.teleop_command[:])
                        self.teleop.joint_residual[:] = np.zeros(8,)#0*#(qpos_goal - self.teleop_command[:]) # np array (8,)

                # update images from realsense to shared memory
                perception_out = state.get("perception_out", None)
                robot_out = state.get("robot_out", None)
                gripper_out = state.get("gripper_out", None)

                intrinsics = self.get_intrinsics()
                if perception_out is not None:
                    for k, v in perception_out['value'].items():
                        rgbs[k] = v["color"]
                        depths[k] = v["depth"]
                        intr = intrinsics[k]

                        l = 0.1
                        origin = np.ones((3,4)) @ np.array([0, 0, 0, 1])
                        x_axis = np.ones((3,4)) @ np.array([l, 0, 0, 1])
                        y_axis = np.ones((3,4)) @ np.array([0, l, 0, 1])
                        z_axis = np.ones((3,4)) @ np.array([0, 0, l, 1])
                        if origin[2] != 0:
                            origin = origin[:3] / origin[2]  # Shape (3,1)
                        if x_axis[2] != 0:
                            x_axis = x_axis[:3] / x_axis[2]
                        if y_axis[2] != 0:
                            y_axis = y_axis[:3] / y_axis[2]
                        if z_axis[2] != 0:
                            z_axis = z_axis[:3] / z_axis[2]
                        origin = intr @ origin
                        x_axis = intr @ x_axis
                        y_axis = intr @ y_axis
                        z_axis = intr @ z_axis
                        cv2.line(rgbs[k], (int(origin[0]), int(origin[1])), (int(x_axis[0]), int(x_axis[1])), (255, 0, 0), 2)
                        cv2.line(rgbs[k], (int(origin[0]), int(origin[1])), (int(y_axis[0]), int(y_axis[1])), (0, 255, 0), 2)
                        cv2.line(rgbs[k], (int(origin[0]), int(origin[1])), (int(z_axis[0]), int(z_axis[1])), (0, 0, 255), 2)
                        if self.use_robot:
                            eef_points = np.concatenate([self.eef_point, np.ones((self.eef_point.shape[0], 1))], axis=1)  # (n, 4)
                            eef_colors = [(0, 255, 255)]

                            eef_axis = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])  # (3, 4)
                            eef_axis_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]

                            if robot_out is not None:
                                assert gripper_out is not None
                                eef_points_world_vis = []
                                eef_points_vis = []
                                if self.bimanual:
                                    left_eef_world_list = []
                                    right_eef_world_list = []
                                    for val, b2w, eef_world_list in zip(["left_value", "right_value"], [b2w_l, b2w_r], [left_eef_world_list, right_eef_world_list]):
                                        e2b = robot_out[val]  # (4, 4)
                                        eef_points_world = (b2w @ e2b @ eef_points.T).T[:, :3]  # (n, 3)
                                        eef_points_vis.append(eef_points)
                                        eef_points_world_vis.append(eef_points_world)
                                        eef_orientation_world = (b2w[:3, :3] @ e2b[:3, :3] @ eef_axis[:, :3].T).T  # (3, 3)
                                        eef_world = np.concatenate([eef_points_world, eef_orientation_world], axis=0)  # (n+3, 3)
                                        eef_world_list.append(eef_world)
                                    left_eef_world = np.concatenate(left_eef_world_list, axis=0)  # (n+3, 3)
                                    right_eef_world = np.concatenate(right_eef_world_list, axis=0)  # (n+3, 3)
                                    eef_world = np.concatenate([left_eef_world, right_eef_world], axis=0)  # (2n+6, 3)
                                else:
                                    e2b = robot_out["value"]  # (4, 4)
                                    eef_points_world = (b2w @ e2b @ eef_points.T).T[:, :3]  # (n, 3)
                                    eef_points_vis.append(eef_points)
                                    eef_points_world_vis.append(eef_points_world)
                                    eef_orientation_world = (b2w[:3, :3] @ e2b[:3, :3] @ eef_axis[:, :3].T).T  # (3, 3)
                                    eef_world = np.concatenate([eef_points_world, eef_orientation_world], axis=0)  # (n+3, 3)
                                
                                # add gripper
                                if self.bimanual:
                                    left_gripper = gripper_out["left_value"]
                                    right_gripper = gripper_out["right_value"]
                                    gripper_world = np.array([left_gripper, right_gripper, 0.0])[None, :]  # (1, 3)
                                else:
                                    gripper = gripper_out["value"]
                                    gripper_world = np.array([gripper, 0.0, 0.0])[None, :]  # (1, 3)

                                eef_world = np.concatenate([eef_world, gripper_world], axis=0)  # (n+4, 3) or (2n+7, 3)
                                np.savetxt(robot_record_dir / f"{robot_out['time']:.3f}.txt", eef_world, fmt="%.6f")
                                
                                eef_points_vis = np.concatenate(eef_points_vis, axis=0)
                                eef_points_world_vis = np.concatenate(eef_points_world_vis, axis=0)
                                eef_points_world_vis = np.concatenate([eef_points_world_vis, np.ones((eef_points_world_vis.shape[0], 1))], axis=1)  # (n, 4)
                                eef_colors = eef_colors * eef_points_world_vis.shape[0]
                            
                                if self.bimanual:
                                    for point_orig, point, color, val, b2w in zip(eef_points_vis, eef_points_world_vis, eef_colors, ["left_value", "right_value"], [b2w_l, b2w_r]):
                                        e2b = robot_out[val]  # (4, 4)
                                        point = state["extr"][k] @ point
                                        point = point[:3] / point[2]
                                        point = intr @ point
                                        cv2.circle(rgbs[k], (int(point[0]), int(point[1])), 2, color, -1)
                                    
                                        # draw eef axis
                                        for axis, color in zip(eef_axis, eef_axis_colors):
                                            eef_point_axis = point_orig + 0.1 * axis
                                            eef_point_axis_world = (b2w @ e2b @ eef_point_axis).T
                                            eef_point_axis_world = state["extr"][k] @ eef_point_axis_world
                                            eef_point_axis_world = eef_point_axis_world[:3] / eef_point_axis_world[2]
                                            eef_point_axis_world = intr @ eef_point_axis_world
                                            cv2.line(rgbs[k], 
                                                (int(point[0]), int(point[1])), 
                                                (int(eef_point_axis_world[0]), int(eef_point_axis_world[1])), 
                                                color, 2)
                                else:
                                    point_orig = eef_points_vis[0]
                                    point = eef_points_world_vis[0]
                                    color = eef_colors[0]
                                    e2b = robot_out["value"]  # (4, 4)
                                    point = np.eye(4) @ point
                                    point = point[:3] / point[2]
                                    point = intr @ point
                                    cv2.circle(rgbs[k], (int(point[0]), int(point[1])), 2, color, -1)
                                
                                    # draw eef axis
                                    for axis, color in zip(eef_axis, eef_axis_colors):
                                        eef_point_axis = point_orig + 0.1 * axis
                                        eef_point_axis_world = (b2w @ e2b @ eef_point_axis).T
                                        eef_point_axis_world = np.eye(4) @ eef_point_axis_world
                                        eef_point_axis_world = eef_point_axis_world[:3] / eef_point_axis_world[2]
                                        eef_point_axis_world = intr @ eef_point_axis_world
                                        cv2.line(rgbs[k], 
                                            (int(point[0]), int(point[1])), 
                                            (int(eef_point_axis_world[0]), int(eef_point_axis_world[1])), 
                                            color, 2)

                row_imgs = []
                for row in range(len(self.realsense.serial_numbers)):
                    row_imgs.append(
                        np.hstack(
                            (cv2.cvtColor(rgbs[row], cv2.COLOR_BGR2RGB), 
                            cv2.applyColorMap(cv2.convertScaleAbs(depths[row], alpha=0.03), cv2.COLORMAP_JET))
                        )
                    )
                combined_img = np.vstack(row_imgs)
                combined_img = cv2.resize(combined_img, (self.screen_width,self.screen_height))
                np.copyto(
                    np.frombuffer(self.image_data.get_obj(), dtype=np.uint8).reshape((self.screen_height, self.screen_width, 3)), 
                    combined_img
                )

                time.sleep(max(0, 1 / fps - (time.time() - tic)))
            
            except BaseException as e:
                print(f"Error in robot teleop env: {e.with_traceback()}")
                break
        
        # if self.use_robot:
        #     self.teleop.stop()
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

        return np.array([[roll, pitch, yaw]])  # (φ, θ, ψ)

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
    
    def gripper_sim2real(self, sim_gripper_status: float) -> float:
        """
        Maps the simulator's gripper status (0.0 open, 0.84 closed) to the real robot's (800 open, 0 closed)
        using a quadratic function.

        Args:
            sim_gripper_status (float): Gripper status from the simulator (range: 0.0 to 0.84).

        Returns:
            float: Mapped gripper status for the real robot (range: 800 to 0).
        """
        a, b, c = -889.36, -205.32, 800
        return a * sim_gripper_status**2 + b * sim_gripper_status + c

    def gripper_real2sim(self, real_gripper_status: float) -> float:
        """
        Maps the real robot's gripper status (800 open, 0 closed) to the simulator's (0.0 open, 0.84 closed)
        using a quadratic function.

        Args:
            real_gripper_status (float): Gripper status from the real robot (range: 800 to 0).

        Returns:
            float: Mapped gripper status for the simulator (range: 0.0 to 0.84).
        """
        return (800 - real_gripper_status) / 800

    def mp2np(self, mp_array: mp.Array) -> np.array:
        return np.frombuffer(mp_array.get_obj())  

    def ik(self, curr_qpos: np.array, ee_goal: np.array) -> np.array:
        '''
        ee_goal: shape (1, 8)
        '''
        cartesian_goal = np.concatenate([ee_goal[:,:3], self.euler_from_quat_np(ee_goal[:, 3:7].reshape(-1,))])
        qpos = self.kin_helper.compute_ik_sapien(curr_qpos, cartesian_goal.reshape(-1,))
        return qpos.reshape(-1,)

    def fk(self, joints: np.array) -> np.array:
        fk = self.kin_helper.compute_fk_sapien_links(joints[:7], [self.kin_helper.sapien_eef_idx])[0]
        pos = fk[:3, 3]
        quat = self.quat_from_matrix_np(fk[:3, :3])
        gripper_status = joints[-1]
        return np.concatenate([pos, quat, [gripper_status]])

    def normalize_depth_01(self, depth_input, min_depth=0.0, max_depth=2.0):
        """Normalize depth to range [0, 1] for CNN input using NumPy."""
        depth_input = np.nan_to_num(depth_input, nan=0.0).astype(np.float32)  # Replace NaNs with 0.0 and ensure float32 dtype
        depth_input = depth_input.reshape(depth_input.shape[0], -1)  # Flatten each sample
        depth_input = np.clip(depth_input, min_depth, max_depth)  # Ensure values are within range
        depth_input = (depth_input - min_depth) / (max_depth - min_depth)  # Normalize to [0, 1]
        return depth_input.flatten().astype(np.float32)  # Ensure final dtype is float32


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
        self.start_image_display()
        super().start()

    def stop(self) -> None:
        self._alive.value = False
        self.real_stop()
