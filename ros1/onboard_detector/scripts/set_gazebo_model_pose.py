#!/usr/bin/env python3
"""Place an existing Gazebo model for repeatable sensor validation scenarios."""

import argparse
import sys

import rospy
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--x", type=float, required=True)
    parser.add_argument("--y", type=float, required=True)
    parser.add_argument("--z", type=float, default=0.0)
    args = parser.parse_args(rospy.myargv(argv=sys.argv)[1:])
    rospy.init_node("set_gazebo_model_pose", anonymous=True)
    rospy.wait_for_service("/gazebo/set_model_state", timeout=15.0)
    state = ModelState()
    state.model_name = args.model
    state.reference_frame = "world"
    state.pose.position.x = args.x
    state.pose.position.y = args.y
    state.pose.position.z = args.z
    state.pose.orientation.w = 1.0
    response = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)(state)
    if not response.success:
        raise SystemExit(response.status_message)
    print("placed {} at ({:.3f}, {:.3f}, {:.3f})".format(args.model, args.x, args.y, args.z))


if __name__ == "__main__":
    main()
