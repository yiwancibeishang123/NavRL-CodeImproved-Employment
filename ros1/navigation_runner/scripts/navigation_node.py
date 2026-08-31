#!/usr/bin/env python3

import rospy
import hydra
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
# Hydra treats ROS remap arguments (``__name:=...``, ``__log:=...``) as
# configuration overrides.  Remove those transport arguments before Hydra
# parses the deployment config.
sys.argv = [arg for arg in sys.argv if not arg.startswith("__")]

from navigation import Navigation

FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts/cfg")
@hydra.main(config_path=FILE_PATH, config_name="train", version_base=None)
def main(cfg):
    rospy.init_node("navigation_node", anonymous=True)
    nav = Navigation(cfg)
    nav.run()
    rospy.spin()


if __name__ == "__main__":
    main()
