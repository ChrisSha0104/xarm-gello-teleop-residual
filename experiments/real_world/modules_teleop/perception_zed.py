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
        zed: sl.Camera(), 
        capture_fps, 
        record_fps, 
        record_time, 
        # process_func,
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

        self.zed = zed
        self.perception_q = mp.Queue(maxsize=1)

        # self.process_func = process_func
        self.alive = mp.Value('b', False)
        self.record_restart = mp.Value('b', False)
        self.record_stop = mp.Value('b', False)

        # zed camera initial params
        self.init_params = sl.InitParameters()
        self.init_params.depth_mode = sl.DEPTH_MODE.ULTRA
        self.init_params.coordinate_units = sl.UNIT.MILLIMETER
        self.init_params.camera_resolution = sl.RESOLUTION.HD720
        self.init_params.camera_fps = capture_fps

    def log(self, msg):
        if self.verbose:
            print(f"\033[92m{self.name}: {msg}\033[0m")

    @property
    def can_record(self):
        return self.record_fps != 0

    def run(self):
        # # limit threads
        # threadpool_limits(1)
        # cv2.setNumThreads(1)

        zed = self.zed
        init_params = self.init_params

        # Open the camera
        status = zed.open(init_params)
        if status != sl.ERROR_CODE.SUCCESS: #Ensure the camera has opened succesfully
            print("Camera Open : "+repr(status)+". Exit program.")
            exit()

        # Create and set RuntimeParameters after opening the camera
        runtime_parameters = sl.RuntimeParameters()

        # i = self.index
        capture_fps = self.capture_fps
        record_fps = self.record_fps
        record_time = self.record_time

        cameras_output = None
        recording_frame = float("inf")  # local record step index (since current record start), record fps
        record_start_frame = 0  # global step index (since process start), capture fps
        is_recording = False  # recording state flag
        timestamps_f = None

        i = 0
        image = sl.Mat()
        depth = sl.Mat()
        point_cloud = sl.Mat()

        mirror_ref = sl.Transform()
        mirror_ref.set_translation(sl.Translation(2.75,4.0,0))

        while self.alive.value:
            try: 
                if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                    # Retrieve left image
                    zed.retrieve_image(image, sl.VIEW.LEFT)
                    # Retrieve depth map. Depth is aligned on the left image
                    zed.retrieve_measure(depth, sl.MEASURE.DEPTH)

                    print(f"depth size: {depth.get_width()} x {depth.get_height()}")
                    print("depth value: ", depth.get_value(0, 0))
                    # # Retrieve colored point cloud. Point cloud is aligned on the left image.
                    # zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)

                    # # Get and print distance value in mm at the center of the image
                    # # We measure the distance camera - object using Euclidean distance
                    # x = round(image.get_width() / 2)
                    # y = round(image.get_height() / 2)
                    # err, point_cloud_value = point_cloud.get_value(x, y)

                    # if math.isfinite(point_cloud_value[2]):
                    #     distance = math.sqrt(point_cloud_value[0] * point_cloud_value[0] +
                    #                         point_cloud_value[1] * point_cloud_value[1] +
                    #                         point_cloud_value[2] * point_cloud_value[2])
                    #     print(f"Distance to Camera at {{{x};{y}}}: {distance}")
                    # else : 
                    #     print(f"The distance can not be computed at {{{x};{y}}}")




                    perception_start_time = time.time()
                    cameras_output = depth
                    print("camera output type: ", type(cameras_output))
                    print("camera output", cameras_output.get_value(0, 0))
                    get_time = time.time()
                    timestamps = [cameras_output[i]['timestamp'].item() for i in range(self.num_cam)]

                    # treat captured time and record time as the same
                    process_start_time = get_time
                    process_out = self.process_func(cameras_output) if self.process_func is not None else cameras_output
                    self.log(f"process time: {time.time() - process_start_time}")
                
                    if not self.perception_q.full():
                        self.perception_q.put(process_out)             

            except BaseException as e:
                print("Perception error: ", e)
                break

        self.stop()
        print("Perception process stopped")


    def start(self):
        self.alive.value = True
        super().start()

    def stop(self):
        self.alive.value = False
        self.zed.close()
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
