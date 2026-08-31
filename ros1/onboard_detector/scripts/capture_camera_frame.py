#!/usr/bin/env python3
"""Save one live ROS color frame for deterministic simulator inspection."""

import argparse
import os
import sys

import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--topic", default="/camera/color/image_raw")
    parser.add_argument("--timeout", type=float, default=15.0)
    args = parser.parse_args(rospy.myargv(argv=sys.argv)[1:])
    rospy.init_node("capture_camera_frame", anonymous=True)
    message = rospy.wait_for_message(args.topic, Image, timeout=args.timeout)
    image = CvBridge().imgmsg_to_cv2(message, desired_encoding="bgr8")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    if not cv2.imwrite(args.output, image):
        raise SystemExit("failed to save {}".format(args.output))
    print("saved {}x{} {} to {}".format(message.width, message.height, message.encoding, args.output))


if __name__ == "__main__":
    main()
