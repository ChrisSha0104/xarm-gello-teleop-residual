import threading
import multiprocess as mp
from multiprocess.managers import SharedMemoryManager
import time
import numpy as np
import copy
from pynput import keyboard
from pathlib import Path
from typing import Tuple, List
import logging

from modules_teleop.udp_util import udpReceiver, udpSender
from modules_teleop.common.communication import XARM_STATE_PORT, XARM_CONTROL_PORT, XARM_CONTROL_PORT_L, XARM_CONTROL_PORT_R
from modules_teleop.common.xarm import GRIPPER_OPEN_MIN, GRIPPER_OPEN_MAX, POSITION_UPDATE_INTERVAL, COMMAND_CHECK_INTERVAL
from camera.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer

from gello.agents.agent import BimanualAgent, DummyAgent
from gello.agents.gello_agent import GelloAgent
from gello.env import RobotEnv
from gello.zmq_core.robot_node import ZMQClientRobot
from gello.agents.gello_agent import DynamixelRobotConfig
from gello.dynamixel.driver import DynamixelDriver

np.set_printoptions(precision=5, suppress=True)


class GelloListener(mp.Process):

    def __init__(
        self, 
        # shm_manager: SharedMemoryManager, 
        bimanual: bool = False,
        gello_port: str = '/dev/ttyUSB1',
        bimanual_gello_port: List[str] = ['/dev/ttyUSB0', '/dev/ttyUSB1'],
    ):
        super().__init__()
        
        self.bimanual = bimanual
        self.bimanual_gello_port = bimanual_gello_port

        self.num_joints = 7
        self.gello_port = gello_port
        self.calibrate_offset = False  # whether to recalibrate the offset
        self.verbose = True

        if bimanual:
            examples = dict()
            examples['command'] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            # examples['left_timestamp'] = 0.0
            # examples['right_timestamp'] = 0.0
        else:
            examples = dict()
            examples['command'] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            # examples['timestamp'] = 0.0

        # ring_buffer = SharedMemoryRingBuffer.create_from_examples(
        #     shm_manager=shm_manager,
        #     examples=examples,
        #     get_max_k=30,
        #     get_time_budget=0.2,
        #     put_desired_frequency=100,
        # )
        self.command = mp.Array('d', examples['command'])
        # self.ring_buffer = ring_buffer
        self.stop_event = mp.Event()
        self.ready_event = mp.Event()

    # def start(self, wait=True):
    #     super().start()
    #     if wait:
    #         self.start_wait()
    
    def stop(self, wait=False):
        self.stop_event.set()
        if wait:
            self.end_wait()

    # def start_wait(self):
    #     self.ready_event.wait()
    
    def end_wait(self):
        self.join()

    @property
    def is_ready(self):
        return self.ready_event.is_set()

    def get(self): # , k=None, out=None):
        return copy.deepcopy(np.array(self.command[:]))
        # if k is None:
        #     return self.ring_buffer.get(out=out)
        # else:
        #     return self.ring_buffer.get_last_k(k, out=out)

    def init_gello(self):
        if self.bimanual:
            if self.calibrate_offset:
                assert len(self.bimanual_gello_port) == 2, "Please provide two ports for bimanual calibration"
                left_joint_offsets, left_gripper_config = self.calibrate_offset(port=self.bimanual_gello_port[0])
                right_joint_offsets, right_gripper_config = self.calibrate_offset(port=self.bimanual_gello_port[1])
                dynamixel_config_left = DynamixelRobotConfig(
                    joint_ids=(1, 2, 3, 4, 5, 6, 7),
                    joint_offsets=left_joint_offsets,
                    joint_signs=(1, 1, 1, 1, 1, 1, 1),
                    gripper_config=left_gripper_config,
                )
                dynamixel_config_right = DynamixelRobotConfig(
                    joint_ids=(1, 2, 3, 4, 5, 6, 7),
                    joint_offsets=right_joint_offsets,
                    joint_signs=(1, 1, 1, 1, 1, 1, 1),
                    gripper_config=right_gripper_config,
                )
            else:
                dynamixel_config_left = DynamixelRobotConfig(
                    joint_ids=(1, 2, 3, 4, 5, 6, 7),
                    joint_offsets=(
                        1 * np.pi / 2,
                        2 * np.pi / 2,
                        4 * np.pi / 2,
                        1 * np.pi / 2,
                        2 * np.pi / 2,
                        2 * np.pi / 2,
                        2 * np.pi / 2
                    ),
                    joint_signs=(1, 1, 1, 1, 1, 1, 1),
                    gripper_config=(8, 288, 246),
                )
                dynamixel_config_right = DynamixelRobotConfig(
                    joint_ids=(1, 2, 3, 4, 5, 6, 7),
                    joint_offsets=(
                        2 * np.pi / 2,
                        2 * np.pi / 2,
                        3 * np.pi / 2,
                        1 * np.pi / 2,
                        1 * np.pi / 2,
                        2 * np.pi / 2,
                        0 * np.pi / 2
                    ),
                    joint_signs=(1, 1, 1, 1, 1, 1, 1),
                    gripper_config=(8, 114, 72),
                )
            left_start_joints = np.deg2rad([0, -45, 0, 30, 0, 75, 0, 0])
            right_start_joints = np.deg2rad([0, -45, 0, 30, 0, 75, 0, 0])
            left_agent = GelloAgent(port=self.bimanual_gello_port[0], dynamixel_config=dynamixel_config_left, start_joints=left_start_joints)
            right_agent = GelloAgent(port=self.bimanual_gello_port[1], dynamixel_config=dynamixel_config_right, start_joints=right_start_joints)
            agent = BimanualAgent(left_agent, right_agent)
            self.agent = agent

        else:
            if self.calibrate_offset:
                joint_offsets, gripper_config = self.calibrate_offset(port=self.gello_port)
            
            else:
                joint_offsets = (
                    1 * np.pi / 2,
                    2 * np.pi / 2,
                    4 * np.pi / 2,
                    1 * np.pi / 2,
                    2 * np.pi / 2,
                    2 * np.pi / 2,
                    2 * np.pi / 2
                )
                gripper_config = (8, 288, 246)

            dynamixel_config = DynamixelRobotConfig(
                joint_ids=(1, 2, 3, 4, 5, 6, 7),
                joint_offsets=joint_offsets,
                joint_signs=(1, 1, 1, 1, 1, 1, 1),
                gripper_config=gripper_config,
            )
            gello_port = self.gello_port
            start_joints = np.deg2rad([0, -45, 0, 30, 0, 75, 0, 0])
            agent = GelloAgent(port=gello_port, dynamixel_config=dynamixel_config, start_joints=start_joints)
            self.agent = agent

        self.ready_event.set()
    
    def calibrate_offset(self, port, verbose=False):
        # MENAGERIE_ROOT = Path(__file__).parent / "third_party" / "mujoco_menagerie"
        
        start_joints = tuple(np.deg2rad(0, -45, 0, 30, 0, 75, 0))  # The joint angles that the GELLO is placed in at (in radians)
        joint_signs = (1, 1, -1, 1, 1, 1)  # The joint angles that the GELLO is placed in at (in radians)

        joint_ids = list(range(1, self.num_joints + 1))
        driver = DynamixelDriver(joint_ids, port=port, baudrate=57600)

        # assume that the joint state shouold be start_joints
        # find the offset, which is a multiple of np.pi/2 that minimizes the error between the current joint state and args.start_joints
        # this is done by brute force, we seach in a range of +/- 8pi

        def get_error(offset: float, index: int, joint_state: np.ndarray) -> float:
            joint_sign_i = joint_signs[index]
            joint_i = joint_sign_i * (joint_state[index] - offset)
            start_i = start_joints[index]
            return np.abs(joint_i - start_i)

        for _ in range(10):
            driver.get_joints()  # warmup

        for _ in range(1):
            best_offsets = []
            curr_joints = driver.get_joints()
            for i in range(self.num_joints):
                best_offset = 0
                best_error = 1e6
                for offset in np.linspace(
                    -8 * np.pi, 8 * np.pi, 8 * 4 + 1
                ):  # intervals of pi/2
                    error = get_error(offset, i, curr_joints)
                    if error < best_error:
                        best_error = error
                        best_offset = offset
                best_offsets.append(best_offset)

        gripper_open = np.rad2deg(driver.get_joints()[-1]) - 0.2
        gripper_close = np.rad2deg(driver.get_joints()[-1]) - 42
        if self.verbose:
            print()
            print("best offsets               : ", [f"{x:.3f}" for x in best_offsets])
            print(
                "best offsets function of pi: ["
                + ", ".join([f"{int(np.round(x/(np.pi/2)))}*np.pi/2" for x in best_offsets])
                + " ]",
            )
            print(
                "gripper open (degrees)       ",
                gripper_open,
            )
            print(
                "gripper close (degrees)      ",
                gripper_close,
            )

        joint_offsets = tuple(best_offsets)
        gripper_config = (8, gripper_open, gripper_close)
        return joint_offsets, gripper_config

    def run(self):
        self.init_gello()

        # start_time = time.time()
        while self.alive:
            try:
                # print('GelloListener alive')
                action = self.agent.get_action()
                self.command[:] = action
                # self.ring_buffer.put({'command': action, 'timestamp': time.time()})
            except:
                print(f"Error in GelloListener")
                break

        self.stop()
        print("GelloListener exit!")
        
    @property
    def alive(self):
        return not self.stop_event.is_set() and self.ready_event.is_set()



class GelloTeleopResidual(mp.Process):

    name="gello_teleop"

    def __init__(
        self, 
        gripper_enable=True,
        bimanual=False,
    ) -> None:
        super().__init__()
        self.gripper_enable = gripper_enable
        assert self.gripper_enable, "Gripper must be enabled for Gello teleop"

        self.bimanual = bimanual

        self.key_states = {
            "p": False,
            ",": False,
            ".": False,
        }

        # additional states
        self.init = True
        self.pause = False
        self.record_start = mp.Value('b', False)
        self.record_stop = mp.Value('b', False)

        # states for moving the arm
        # self.cur_joints = None
        self.command = mp.Array('d', [0.0] * 8)

        self._alive = mp.Value('b', True)
        self.controller_quit = True

        # self.state_receiver = None # udpReceiver(ports={'xarm_state':XARM_STATE_PORT})
        self.command_sender = None # udpSender(port=XARM_CONTROL_PORT)
        self.command_sender_left = None
        self.command_sender_right = None

        # self.shm_manager = SharedMemoryManager()
        # self.shm_manager.start()
        # self.gello_activated = False

        # self.ee_residual = mp.Array('d', [0.0] * 8)
        self.comm_with_residual = mp.Array('d', [0.0] * 8)
        self.start_residual_policy = mp.Value('b', False)
        self.start_recording = mp.Value('b', False)


    @staticmethod
    def log(msg):
        print(f"\033[94m{msg}\033[0m")

    def on_press(self,key):
        try:
            key_char = key.char.lower() if key.char else key.char
            if key_char in self.key_states:
                self.key_states[key_char] = True
        except AttributeError:
            pass

    def on_release(self,key):
        try:
            key_char = key.char.lower() if key.char else key.char
            if key_char in self.key_states:
                self.key_states[key_char] = False
        except AttributeError:
            if key == keyboard.Key.esc:
                return False

    def update_xarm_joints(self):
        raise NotImplementedError
        self.state_receiver = udpReceiver(ports={'xarm_state': XARM_STATE_PORT}, re_use_address=True)
        self.state_receiver.start()
        
        while self.alive:
            try:
                update_start_time = time.time()
                xarm_state = self.state_receiver.get("xarm_state", pop=True)
                if xarm_state is not None:
                    # self.cur_position = xarm_state["pos"]
                    cur_joints = xarm_state["qpos"]
                    if self.gripper_enable:
                        # self.cur_gripper_pos = xarm_state["gripper"]
                        gripper_pos = xarm_state["gripper"]
                        normalized_gripper_pos = (gripper_pos - GRIPPER_OPEN_MAX) / (GRIPPER_OPEN_MIN - GRIPPER_OPEN_MAX)
                        if isinstance(cur_joints, np.ndarray):
                            cur_joints = np.concatenate([cur_joints, [normalized_gripper_pos]])
                        else:
                            cur_joints = cur_joints + [normalized_gripper_pos]
                    self.cur_joints = cur_joints
                time.sleep(max(0, POSITION_UPDATE_INTERVAL - (time.time() - update_start_time)))
            except:
                print(f"Error in update_xarm_joints")
                break

        self.state_receiver.stop()
        self.log(f"update_xarm_joints exit!")

    def get_command(self):
        # if self.cur_joints is None:
        #     return

        if self.key_states["p"]:
            # abandon all other keyinputs
            self.pause = not self.pause
            self.log(f"keyboard teleop pause: {self.pause}")
            time.sleep(0.5)

        if self.key_states[","]:
            if self.start_residual_policy.value:
                self.start_residual_policy.value = False
                self.log(f" --------------- residual policy stopped --------------- ")
            else:
                self.start_residual_policy.value = True
                self.log(f" --------------- residual policy started --------------- ")
            time.sleep(0.5)
        
        if self.key_states["."]:
            self.start_recording.value = True
            self.log(f" --------------- start recording --------------- ")
            time.sleep(0.5)

        if self.pause:
            self.command = []
            return False
        else:
            # current_joints = copy.deepcopy(self.cur_joints)
            command_joints = self.gello_listener.get()
            # command = self.gello_listener.get()
            # command_joints = command['command']
            # command_timestamp = command['timestamp']

            """
            delta = command_joints - current_joints
            joint_delta_norm = np.linalg.norm(delta)
            max_joint_delta = np.abs(delta).max()
            # print('gello activated:', self.gello_activated, 'command latency:', time.time() - command_timestamp, 'command_joints:', command_joints, 'current_joints:', current_joints)

            max_activate_delta = 0.5
            max_delta_norm = 0.05
            if not self.gello_activated:
                if max_joint_delta < max_activate_delta:
                    self.gello_activated = True
                next_joints = current_joints
            else:
                if joint_delta_norm > max_delta_norm:
                    delta = delta / joint_delta_norm * max_delta_norm
                next_joints = current_joints + delta
            
            # denormalize gripper position
            gripper_pos = next_joints[-1]
            denormalized_gripper_pos = gripper_pos * (GRIPPER_OPEN_MIN - GRIPPER_OPEN_MAX) + GRIPPER_OPEN_MAX
            next_joints[-1] = denormalized_gripper_pos
            next_joints = next_joints.tolist()
            """
            # print('command_joints:', command_joints)
            next_joints = command_joints.tolist()
            # print('next_joints:', next_joints)
            self.command[:] = next_joints
            return True

    def run(self) -> None:
        self.gello_listener = GelloListener(
            # shm_manager=self.shm_manager,
            bimanual=self.bimanual,
            gello_port='/dev/ttyUSB0',
            bimanual_gello_port=['/dev/ttyUSB0', '/dev/ttyUSB1'],
        )
        self.gello_listener.start()

        self.keyboard_listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.keyboard_listener.start()
        
        # self.update_joints_t = threading.Thread(name="get_pos_from_XarmController", target=self.update_xarm_joints)
        # self.update_joints_t.start()

        if self.bimanual:
            self.command_sender_left = udpSender(port=XARM_CONTROL_PORT_L)
            self.command_sender_right = udpSender(port=XARM_CONTROL_PORT_R)
        else:
            self.command_sender = udpSender(port=XARM_CONTROL_PORT)

        time.sleep(1)
        print("Gello teleop start!")
        use_residual_policy = True

        while self.alive:
            try:
                # command_start_time = time.time()
                teleop_start_time = time.time()
                teleop_status = self.get_command()
                # print("------------------got command------------------")
                if teleop_status:
                    # print('command:', self.command[:])
                    # goal = np.array(self.command[:], dtype=np.float64) #+ np.array(self.joint_residual[:], dtype=np.float64)
                    goal = np.array(self.comm_with_residual, dtype=np.float64) # TODO: check frequency

                    # print("command:", self.command[:])
                    # self.command[:] = goal  # NOTE: comment this out to disable residual

                if self.bimanual:
                    self.command_sender_left.send([self.command[0][0:8]])
                    self.command_sender_right.send([self.command[0][8:16]])
                else:
                    if use_residual_policy:
                        if teleop_status: # Not yet paused
                            self.command_sender.send([list(self.comm_with_residual)])
                        else: # paused
                            self.command_sender.send([list(self.command)])
                    else: # regular teleop
                        self.command_sender.send([list(self.command)])
                # time.sleep(max(0, COMMAND_CHECK_INTERVAL / 2 - (time.time() - command_start_time)))
                # print(f"teleop freq: {1 / (time.time() - teleop_start_time)} Hz")
            except:
                print(f"Error in GelloTeleop")
                break
        
        self.stop()
        if self.bimanual:
            self.command_sender_left.close()
            self.command_sender_right.close()
        else:
            self.command_sender.close()
        self.gello_listener.stop()
        self.keyboard_listener.stop()
        # self.update_joints_t.join()
        print(f"{'='*20} keyboard + gello teleop exit!")
    
    @property
    def alive(self):
        alive = self._alive.value & self.keyboard_listener.is_alive()
        self._alive.value = alive
        return alive 

    def stop(self, stop_controller=False):
        if stop_controller:
            self.log("teleop stop controller")
            if self.command_sender is not None:
                self.command_sender.send(["quit"])
            if self.command_sender_left is not None:
                self.command_sender_left.send(["quit"])
            if self.command_sender_right is not None:
                self.command_sender_right.send(["quit"])
            time.sleep(1)
        self._alive.value = False
        self.log("teleop stop")


if __name__ == "__main__":
    gello_teleop = GelloTeleopResidual()
    gello_teleop.start()
    gello_teleop.join()

