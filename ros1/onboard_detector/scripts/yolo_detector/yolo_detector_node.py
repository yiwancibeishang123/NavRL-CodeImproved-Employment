#!/usr/bin/env python3

import rospy
import os
import sys
import rospkg

module_dir = os.path.join(rospkg.RosPack().get_path("onboard_detector"), "scripts", "yolo_detector")
if module_dir not in sys.path:
    sys.path.insert(0, module_dir)
from yolo_detector import yolo_detector


def main():
	rospy.init_node("yolo_detector_node")
	yolo_detector()
	rospy.spin()

if __name__=="__main__":
	main()
	
