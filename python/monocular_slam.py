import sys
import argparse
from pathlib import Path
import cv2
import numpy as np
import cvui

LIBRARY_PATH = str(Path(__file__).resolve().parent.parent.joinpath("build/python"))
sys.path.append(LIBRARY_PATH)

from scripts.camera import Camera
from scripts.camera_config import get_config
import openvslam_python as OpenVSLAM
from functools import partial


def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config-file-path", type=str, default="cfg/config.yaml")
    parser.add_argument("--vocab_file_path", type=str, default="cfg/orb_vocab.dbow2")
    parser.add_argument("--camera-parameter-file-path", type=str, default="cfg/camera_parameter.toml")
    return parser.parse_args()


def scaling_int(int_num, scale):
    return int(int_num * scale)


def main(config_file_path, vocab_file_path, camera_parameter_file_path):
    camera = Camera(get_config(camera_parameter_file_path))
    ovs_config = OpenVSLAM.config(config_file_path=config_file_path)
    SLAM = OpenVSLAM.system(cfg=ovs_config, vocab_file_path=vocab_file_path)

    SLAM.startup()
    scaling = partial(scaling_int, scale=2.0 / 3)
    WINDOW_NAME = "OpenVSLAM Python Sample Code"
    cvui.init(WINDOW_NAME)
    while True:
        key = cv2.waitKey(10)
        frame = np.zeros((scaling(1080), scaling(1920), 3), np.uint8)
        frame[:] = (49, 52, 49)
        status = camera.update()

        if status:
            # WARNING:If distortion correction is enabled, the rectangle on windows doesn't indicate actual RoI area for auto exposure.
            see3cam_rgb_image = camera.remap_image
            scaled_width = scaling(1920)
            scaled_height = scaling(1080)
            see3cam_rgb_image_resized = cv2.resize(see3cam_rgb_image, (scaled_width, scaled_height))
            frame[:scaled_height, :scaled_width, :] = see3cam_rgb_image_resized

        if key == 27 or key == ord("q"):
            break

        cvui.update()
        cvui.imshow(WINDOW_NAME, frame)
    cv2.destroyAllWindows()
    SLAM.shutdown()


if __name__ == "__main__":
    args = parse_args()
    main(args.config_file_path, args.vocab_file_path, args.camera_parameter_file_path)