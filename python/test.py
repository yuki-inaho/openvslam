import sys
import argparse
from pathlib import Path
import cv2

LIBRARY_PATH = str(Path(__file__).resolve().parent.parent.joinpath("build/python"))
sys.path.append(LIBRARY_PATH)

from scripts.utils import read_yaml
import openvslam_python as OpenVSLAM


def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config-file-path", type=str, default="config.yaml")
    parser.add_argument("--vocab_file_path", type=str, default="orb_vocab.dbow2")
    return parser.parse_args()


def main(config_file_path, vocab_file_path):
    config_dict = read_yaml(config_file_path)
    ovs_config = OpenVSLAM.config(config_file_path="./config.yaml")
    SLAM = OpenVSLAM.system(cfg=ovs_config, vocab_file_path="./orb_vocab.dbow2")
    SLAM.startup()

    SLAM.shutdown()


if __name__ == "__main__":
    args = parse_args()
    main(args.config_file_path, args.vocab_file_path)