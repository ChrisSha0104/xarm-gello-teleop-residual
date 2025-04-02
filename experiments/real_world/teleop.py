from pathlib import Path
import argparse
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils import get_root, mkdir
root: Path = get_root(__file__)
sys.path.append(str(root / "real_world"))

from experiments.real_world.modules_teleop.robot_env_teleop import RobotTeleopEnv


if __name__ == '__main__':
    # cv2 encounter error when using multi-threading, use tk instead
    # cv2.setNumThreads(cv2.getNumberOfCPUs())
    # cv2.namedWindow("real env monitor", cv2.WINDOW_NORMAL)

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--bimanual', action='store_true')
    parser.add_argument('--use_residual_policy', action='store_true')
    parser.add_argument('--load_model', type=str, default='')
    args = parser.parse_args()

    assert args.name != '', "Please provide a name for the experiment"

    env = RobotTeleopEnv(
        mode='3D',
        exp_name=args.name,
        resolution=(848, 480),
        capture_fps=30,
        record_fps=30,
        perception_process_func=None,
        use_robot=True,
        use_gello=True,
        use_residual_policy=args.use_residual_policy,
        load_model=args.load_model,
        bimanual=args.bimanual,
        gripper_enable=True,
        data_dir="data",
        debug=True,
    )

    env.start()
    env.join()
