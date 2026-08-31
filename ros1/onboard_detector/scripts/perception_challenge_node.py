#!/usr/bin/env python3
"""Apply reproducible camera-domain corruptions before the YOLO node.

The node sits between Gazebo's RGB camera and YOLO. Gazebo truth is used only
to place a local occluder around the rendered target; no truth labels, depth,
pose, or velocity are published to the detector.
"""

import json
import math
import random

import cv2
import numpy as np
import rospy
import tf
from cv_bridge import CvBridge
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String


class PerceptionChallengeNode:
    MODES = {"clean", "partial_occlusion", "fog", "rain", "backlight", "combined"}

    def __init__(self):
        self.input_topic = rospy.get_param("~input_topic", "/camera/color/image_raw")
        self.output_topic = rospy.get_param("~output_topic", "/camera/challenge/color/image_raw")
        self.mode = str(rospy.get_param("~mode", "combined")).lower()
        self.seed = int(rospy.get_param("~seed", 17))
        self.target_model = str(rospy.get_param("~target_model", "person1_0.5_0.5_1.8"))
        if self.mode not in self.MODES:
            raise ValueError("Unsupported challenge mode: {}".format(self.mode))
        self.bridge = CvBridge()
        self.rng = random.Random(self.seed)
        self.frame_index = 0
        self.camera_info = None
        self.target = None
        self.tf_listener = tf.TransformListener()
        self.image_pub = rospy.Publisher(self.output_topic, Image, queue_size=1)
        self.meta_pub = rospy.Publisher("/camera/challenge/metadata", String, queue_size=2, latch=True)
        rospy.Subscriber(self.input_topic, Image, self.on_image, queue_size=1)
        rospy.Subscriber("/camera/color/camera_info", CameraInfo, self.on_camera_info, queue_size=1)
        rospy.Subscriber("/gazebo/model_states", ModelStates, self.on_model_states, queue_size=5)
        self.mode_sub = rospy.Subscriber(
            "/camera/challenge/set_mode", String, self.on_mode, queue_size=2
        )
        self.publish_metadata()

    def publish_metadata(self):
        self.meta_pub.publish(String(data=json.dumps({
            "mode": self.mode,
            "seed": self.seed,
            "target_model": self.target_model,
            "input_topic": self.input_topic,
            "output_topic": self.output_topic,
            "truth_use": "occluder_placement_only; never published to detector",
        }, sort_keys=True)))

    def on_mode(self, message):
        mode = str(message.data).strip().lower()
        if mode not in self.MODES:
            rospy.logwarn("Ignoring unsupported challenge mode: %s", mode)
            return
        if mode != self.mode:
            rospy.loginfo("Camera challenge mode transition: %s -> %s", self.mode, mode)
            self.mode = mode
            self.frame_index = 0
            self.publish_metadata()

    def on_camera_info(self, message):
        self.camera_info = message

    def on_model_states(self, message):
        try:
            index = message.name.index(self.target_model)
        except ValueError:
            return
        position = message.pose[index].position
        self.target = (float(position.x), float(position.y), float(position.z) + 0.9)

    def target_pixel(self, width, height):
        if self.target is None or self.camera_info is None:
            return width // 2, height // 2
        try:
            stamped = PointStamped()
            stamped.header.frame_id = "map"
            stamped.header.stamp = rospy.Time(0)
            stamped.point.x, stamped.point.y, stamped.point.z = self.target
            camera = self.tf_listener.transformPoint(
                self.camera_info.header.frame_id or "camera_link", stamped
            )
            z = float(camera.point.z)
            if z > 0.05 and math.isfinite(z):
                u = self.camera_info.K[0] * camera.point.x / z + self.camera_info.K[2]
                v = self.camera_info.K[4] * camera.point.y / z + self.camera_info.K[5]
                if math.isfinite(u) and math.isfinite(v):
                    return int(np.clip(u, 0, width - 1)), int(np.clip(v, 0, height - 1))
        except (tf.Exception, tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException, ValueError):
            pass
        phase = self.frame_index * 0.09
        return int(width * (0.5 + 0.16 * math.sin(phase))), int(height * 0.48)

    def partial_occlusion(self, image):
        height, width = image.shape[:2]
        u, v = self.target_pixel(width, height)
        output = image.copy()
        occlusion_w = max(24, int(width * 0.075))
        occlusion_h = max(54, int(height * 0.28))
        x0 = int(np.clip(u - 0.62 * occlusion_w, 0, width - 1))
        y0 = int(np.clip(v - 0.20 * occlusion_h, 0, height - 1))
        x1 = int(np.clip(x0 + occlusion_w, 0, width))
        y1 = int(np.clip(y0 + occlusion_h, 0, height))
        cv2.rectangle(output, (x0, y0), (x1, y1), (34, 42, 47), thickness=-1)
        cv2.line(output, (x0, y0), (x1, y1), (92, 104, 110), thickness=3)
        return output

    def fog(self, image):
        height, width = image.shape[:2]
        haze = np.full_like(image, (210, 220, 225), dtype=np.uint8)
        result = cv2.addWeighted(image, 0.52, haze, 0.48, 0.0)
        result = cv2.GaussianBlur(result, (5, 5), 0.8)
        yy, xx = np.mgrid[0:height, 0:width]
        radial = np.sqrt(((xx - width / 2) / width) ** 2 + ((yy - height / 2) / height) ** 2)
        alpha = np.clip(0.12 + 0.32 * radial, 0.0, 0.42)[..., None]
        return np.clip(result * (1.0 - alpha) + haze * alpha, 0, 255).astype(np.uint8)

    def rain(self, image):
        height, width = image.shape[:2]
        overlay = image.copy()
        count = max(80, int(width * height / 2100))
        for _ in range(count):
            x = self.rng.randrange(width)
            y = self.rng.randrange(height)
            length = self.rng.randrange(8, 26)
            cv2.line(overlay, (x, y), (x - 3, min(height - 1, y + length)), (220, 228, 232), 1)
        return cv2.GaussianBlur(cv2.addWeighted(image, 0.76, overlay, 0.24, 0.0), (3, 3), 0.45)

    def backlight(self, image):
        height, width = image.shape[:2]
        yy, xx = np.mgrid[0:height, 0:width]
        cx, cy = width * 0.50, height * 0.18
        radius = max(width, height) * 0.52
        glare = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * radius ** 2))[..., None]
        arr = image.astype(np.float32)
        arr = arr * (0.52 + 0.24 * (1.0 - glare)) + 255.0 * 0.70 * glare
        return cv2.GaussianBlur(np.clip(arr, 0, 255).astype(np.uint8), (5, 5), 1.0)

    def corrupt(self, image):
        if self.mode == "clean":
            return image
        result = image
        if self.mode in {"partial_occlusion", "combined"}:
            result = self.partial_occlusion(result)
        if self.mode in {"fog", "combined"}:
            result = self.fog(result)
        if self.mode in {"rain", "combined"}:
            result = self.rain(result)
        if self.mode in {"backlight", "combined"}:
            result = self.backlight(result)
        return result

    def on_image(self, message):
        try:
            image = self.bridge.imgmsg_to_cv2(message, "bgr8")
            output = self.corrupt(image)
            result = self.bridge.cv2_to_imgmsg(output, "bgr8")
            result.header = message.header
            self.image_pub.publish(result)
            self.frame_index += 1
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "challenge image failed: %s", exc)


def main():
    rospy.init_node("perception_challenge_node")
    PerceptionChallengeNode()
    rospy.spin()


if __name__ == "__main__":
    main()
