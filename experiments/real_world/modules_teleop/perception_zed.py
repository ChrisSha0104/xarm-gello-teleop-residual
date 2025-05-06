from pathlib import Path
import os
import time
import numpy as np
import cv2
import open3d as o3d
from threadpoolctl import threadpool_limits
import multiprocess as mp
import threading
from threading import Lock

from utils import get_root
root: Path = get_root(__file__)

from camera.multi_realsense import MultiRealsense
from camera.single_realsense import SingleRealsense

import pyzed.sl as sl
import math

class PerceptionZED(mp.Process):
    name = "PerceptionZED"

    def __init__(
        self,
        capture_fps, 
        record_fps, 
        record_time, 
        process_func,
        exp_name=None,
        data_dir="data",
        verbose=False,
    ):
        super().__init__()
        self.verbose = verbose

        self.record_fps = record_fps
        self.record_time = record_time
        self.exp_name = exp_name
        self.data_dir = data_dir

        if self.exp_name is None:
            assert self.record_fps == 0

        self.process_func = process_func
        self.perception_q = mp.Queue(maxsize=1)
        self.alive = mp.Value('b', False)
        self.record_restart = mp.Value('b', False)
        self.record_stop = mp.Value('b', False)

    def log(self, msg):
        if self.verbose:
            print(f"\033[92m{self.name}: {msg}\033[0m")

    @property
    def can_record(self):
        return self.record_fps != 0

    def run(self):
        # limit threads
        threadpool_limits(1)
        cv2.setNumThreads(1)

        # initialize camera
        zed = sl.Camera()

        # zed camera initial params
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD720  # Use HD720 resolution
        init_params.depth_mode = sl.DEPTH_MODE.NEURAL  # Use ULTRA depth mode
        init_params.coordinate_units = sl.UNIT.MILLIMETER  # Use meter units (for depth measurements)
        init_params.depth_minimum_distance = 100  # Set minimum depth distance to 0.1 meters
        init_params.depth_maximum_distance = 1500  # Set maximum depth distance to 1.0 meters
        init_params.camera_fps = 30
        init_params.sdk_verbose = 1

        # Open the camera
        status = zed.open(init_params)
        if status != sl.ERROR_CODE.SUCCESS: #Ensure the camera has opened succesfully
            print("Camera Open : "+repr(status)+". Exit program.")
            self.close()
            exit(1)

        # print("ZED camera information: ", zed.get_camera_information().camera_configuration.calibration_parameters.left_cam.fx)
        # print("ZED camera information: ", zed.get_camera_information().camera_configuration.calibration_parameters.left_cam.fy)
        # print("ZED camera information: ", zed.get_camera_information().camera_configuration.calibration_parameters.left_cam.cx)
        # print("ZED camera information: ", zed.get_camera_information().camera_configuration.calibration_parameters.left_cam.cy)
        # print("-0--------------------------------")

        # print("ZED camera opened")

        # Create and set RuntimeParameters after opening the camera
        runtime_parameters = sl.RuntimeParameters()
        runtime_parameters.enable_fill_mode = True

        # print("ZED camera parameters set")

        # # i = self.index
        # capture_fps = self.capture_fps
        # record_fps = self.record_fps
        # record_time = self.record_time

        # cameras_output = None
        # recording_frame = float("inf")  # local record step index (since current record start), record fps
        # record_start_frame = 0  # global step index (since process start), capture fps
        # is_recording = False  # recording state flag
        # timestamps_f = None

        image_size = zed.get_camera_information().camera_configuration.resolution
        image_size.width = int(image_size.width / 2)
        image_size.height = int(image_size.height / 2)

        # print("ZED camera image size: ", image_size)

        # image_zed = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4, sl.MEM.CPU)
        # depth_image_zed = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4, sl.MEM.CPU)
        depth_zed = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4, sl.MEM.CPU)
        # point_cloud = sl.Mat(sl.MEM.CPU)

        # print("ZED camera image mat created")

        # zed.grab(runtime_parameters)

        # print("grabbed result: ", zed.grab(runtime_parameters))
        # print("ZED camera started grabbing")

        while self.alive.value:
            # print("is alive")
            try: 
                if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                    # print("started zed")
                    # Retrieve left image
                    # zed.retrieve_image(image_zed, sl.VIEW.LEFT, sl.MEM.CPU, image_size)
                    # zed.retrieve_image(depth_image_zed, sl.VIEW.DEPTH, sl.MEM.CPU, image_size)
                    # zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, image_size)
                    zed.retrieve_measure(depth_zed, sl.MEASURE.DEPTH, sl.MEM.CPU, image_size)

                    # image_ocv = image_zed.get_data()
                    # depth_image_ocv = depth_image_zed.get_data()
                    depth_ocv = depth_zed.get_data()

                    depth_vis = self.filter_depth_for_visualization(depth_ocv, unit="mm")
                    # cv2.imshow("depth_zed", depth_vis)
                    # cv2.imshow("depth_image", depth_image_ocv)
                    # cv2.imshow("Image", image_ocv)
                    # cv2.waitKey(1)

                    perception_start_time = time.time()
                    cameras_output = depth_ocv
                    get_time = time.time()
                    # timestamps = [cameras_output[i]['timestamp'].item() for i in range(self.num_cam)]

                    # treat captured time and record time as the same
                    process_start_time = get_time
                    process_out = self.process_func(cameras_output) if self.process_func is not None else cameras_output
                    self.log(f"process time: {time.time() - process_start_time}")
                
                    if not self.perception_q.full():
                        self.perception_q.put(process_out)             

            except BaseException as e:
                print("Perception error: ", e)
                break

        # cv2.destroyAllWindows()
        zed.close()
        self.stop()
        print("Perception process stopped")


    def start(self):
        self.alive.value = True
        super().start()

    def stop(self):
        self.alive.value = False
        self.perception_q.close()
    
    def set_record_start(self):
        if self.record_fps == 0:
            print("record disabled because record_fps is 0")
            assert self.record_restart.value == False
        else:
            self.record_restart.value = True
            print("record restart cmd received")

    def set_record_stop(self):
        if self.record_fps == 0:
            print("record disabled because record_fps is 0")
            assert self.record_stop.value == False
        else:
            self.record_stop.value = True
            print("record stop cmd received")

    def filter_depth_for_visualization(self, depth: np.ndarray, unit: str = "m") -> np.ndarray:
        """
        Visualize depth image using cv2.
        """

        if unit == "mm":
            depth = depth / 1000.0
        elif unit == "cm":
            depth = depth / 100.0

        max_depth = 1.0 # in meters
        min_depth = 0.1 # in meters

        # remove NaN and Inf values
        depth_filtered = np.nan_to_num(
            depth,
            nan=0.0,
            posinf=max_depth,
            neginf=min_depth,
        )
        
        # clip depth
        depth = np.clip(depth_filtered, min_depth, max_depth)

        # Normalize to [0, 255] and convert to uint8
        depth_uint8 = (depth / max_depth * 255.0).astype(np.uint8)
        
        depth_vis = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)
        # depth_vis = cv2.resize(depth_vis, (480, 480))
        
        return depth_vis


def perception_func_test():

    def process_perception_out(perception):
        # vis = o3d.visualization.Visualizer()
        # vis.create_window()
        # coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
        # vis.add_geometry(coordinate)

        # Set camera view
        # ctr = vis.get_view_control()
        # ctr.set_lookat([0, 0, 0])
        # ctr.set_front([0, -1, 0])
        # ctr.set_up([0, 0, -1])

        while perception.alive.value:
            print(f"[Vis thread] perception out {time.time()}")
            if not perception.perception_q.empty():
                output = perception.perception_q.get()

                # Visualize the merged point cloud
                # vis.clear_geometries()
                # vis.add_geometry(output["object_pcd"])
                # vis.poll_events()
                # vis.update_renderer()
            time.sleep(1)

    # Create an instance of MultiRealsense or SingleRealsense
    realsense = MultiRealsense(
        resolution=(848, 480),
        capture_fps=30,
        enable_color=True,
        enable_depth=True,
        verbose=False
    )
    realsense.start(wait=True)
    realsense.restart_put(start_time=time.time() + 2)
    time.sleep(10)

    # Create an instance of the Perception class
    perception = Perception(
        realsense=realsense, 
        capture_fps=30, # should be the same as the camera fps
        record_fps=30, # 0 for no record
        record_time=10, # in seconds
        process_func=None,
        verbose=False
    )

    # Start the perception process
    perception.start()

    # Start the echo thread
    vis_thread = threading.Thread(name="vis",target=process_perception_out, args=(perception,))
    vis_thread.start()

    # Start the record
    perception.set_record_start()

    time.sleep(12)

    # Stop the record
    perception.set_record_stop()

    time.sleep(2)

    # Stop the perception process
    perception.stop()
    vis_thread.join()
    realsense.stop()

    # exit
    print("Test finished")

if __name__ == "__main__":
    perception_func_test()
