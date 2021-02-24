import sys
from pathlib import Path

LIBRARY_PATH = str(Path(__file__).resolve().parent.parent.joinpath("build/python"))
sys.path.append(LIBRARY_PATH)

import openvslam_python

config = openvslam_python.config(config_file_path="./config.yaml")
SLAM = openvslam_python.system(cfg=config, vocab_file_path="./orb_vocab.dbow2")
SLAM.startup()
SLAM.shutdown()
