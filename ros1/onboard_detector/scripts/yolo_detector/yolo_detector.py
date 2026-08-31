#!/usr/bin/env python
import rospy
import cv2
import numpy as np
import torch
import os
import time
import std_msgs
from sensor_msgs.msg import Image
# from sensor_msgs.msg import poin
from vision_msgs.msg import Detection2DArray
from vision_msgs.msg import Detection2D
from cv_bridge import CvBridge
from utils.tool import handle_preds
from module.detector import Detector

target_classes = ["person"]
path_curr = os.path.dirname(__file__)

class yolo_detector:
    def __init__(self):
        print("[onboardDetector]: yolo detector init...")

        self.img_topic = rospy.get_param("~image_topic", "/camera/color/image_raw")
        self.device = rospy.get_param("~device", "cuda" if torch.cuda.is_available() else "cpu")
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            rospy.logwarn("CUDA requested for YOLO but unavailable; using CPU")
            self.device = "cpu"
        self.weight = rospy.get_param(
            "~weight",
            os.path.join(path_curr, "weights", "weight_AP05_0.253207_280-epoch.pth"),
        )
        self.class_names = rospy.get_param("~class_names", os.path.join(path_curr, "config", "coco.names"))
        self.thresh = float(rospy.get_param("~threshold", 0.65))

        self.img_received = False
        self.img_detected = False
        self.img_stamp = rospy.Time(0)
        self.image_sequence = 0
        self.processed_sequence = -1


        # init and load
        self.model = Detector(80, True).to(self.device)
        if not os.path.isfile(self.weight):
            raise FileNotFoundError("YOLO weights not found: {}".format(self.weight))
        self.model.load_state_dict(torch.load(self.weight, map_location=self.device))
        self.model.eval()

        # subscriber
        self.br = CvBridge()
        self.img_sub = rospy.Subscriber(self.img_topic, Image, self.image_callback, queue_size=1)

        # publisher
        self.img_pub = rospy.Publisher("yolo_detector/detected_image", Image, queue_size=10)
        self.bbox_pub = rospy.Publisher("yolo_detector/detected_bounding_boxes", Detection2DArray, queue_size=10)
        self.time_pub = rospy.Publisher("yolo_detector/yolo_time", std_msgs.msg.Float64, queue_size=1)
        self.forward_time_pub = rospy.Publisher("yolo_detector/yolo_forward_time", std_msgs.msg.Float64, queue_size=1)
        self.device_time_pub = rospy.Publisher("yolo_detector/yolo_device_time", std_msgs.msg.Float64, queue_size=1)
        self.image_age_pub = rospy.Publisher("yolo_detector/yolo_image_age", std_msgs.msg.Float64, queue_size=1)

        # timer
        rospy.Timer(rospy.Duration(0.033), self.detect_callback)
        rospy.Timer(rospy.Duration(0.033), self.vis_callback)
        rospy.Timer(rospy.Duration(0.033), self.bbox_callback)
    
    def image_callback(self, msg):
        self.img = self.br.imgmsg_to_cv2(msg, "bgr8")
        self.img_stamp = msg.header.stamp
        self.image_sequence += 1
        self.img_received = True

    def detect_callback(self, event):
        start_time = time.perf_counter()
        if self.img_received and self.processed_sequence != self.image_sequence:
            output, forward_time_s, device_time_s = self.inference(self.img)
            self.detected_img, self.detected_bboxes = self.postprocess(self.img.copy(), output)
            self.img_detected = True
            self.processed_sequence = self.image_sequence
            self.forward_time_pub.publish(forward_time_s)
            # -1 identifies CPU timing; it is not a measured CUDA kernel duration.
            self.device_time_pub.publish(-1.0 if device_time_s is None else device_time_s)
            if self.img_stamp != rospy.Time(0):
                self.image_age_pub.publish(max(0.0, (rospy.Time.now() - self.img_stamp).to_sec()))
        self.time_pub.publish(time.perf_counter() - start_time)
        

    def vis_callback(self, event):
        if (self.img_detected == True):
            self.img_pub.publish(self.br.cv2_to_imgmsg(self.detected_img, "bgr8"))

    def bbox_callback(self, event):
        if (self.img_detected == True):
            bboxes_msg = Detection2DArray()
            bboxes_msg.header.stamp = self.img_stamp if self.img_stamp != rospy.Time(0) else rospy.Time.now()
            for detected_box in self.detected_bboxes:
                if (detected_box[4] in target_classes):
                    bbox_msg = Detection2D()
                    bbox_msg.bbox.center.x = int(detected_box[0])
                    bbox_msg.bbox.center.y = int(detected_box[1])
                    bbox_msg.bbox.size_x = abs(detected_box[2] - detected_box[0]) 
                    bbox_msg.bbox.size_y = abs(detected_box[3] - detected_box[1])
                    bboxes_msg.detections.append(bbox_msg)
            self.bbox_pub.publish(bboxes_msg)

    def inference(self, ori_img):
        # image pre-processing
        res_img = cv2.resize(ori_img, (352, 352), interpolation = cv2.INTER_LINEAR) 
        img = res_img.reshape(1, 352, 352, 3)
        img = torch.from_numpy(img.transpose(0, 3, 1, 2))
        img = img.to(self.device).float() / 255.0

        # CUDA is asynchronous. Synchronization makes forward_time_s the
        # latency available to the 30 Hz detector callback, rather than just
        # host launch overhead. CUDA events retain the device-only duration.
        cuda_enabled = self.device.startswith("cuda") and torch.cuda.is_available()
        start_event = end_event = None
        if cuda_enabled:
            torch.cuda.synchronize(self.device)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        start_time = time.perf_counter()
        with torch.no_grad():
            preds = self.model(img)
        forward_time_s = time.perf_counter() - start_time
        device_time_s = None
        if cuda_enabled:
            end_event.record()
            torch.cuda.synchronize(self.device)
            device_time_s = start_event.elapsed_time(end_event) / 1000.0
        output = handle_preds(preds, self.device, self.thresh)
        return output, forward_time_s, device_time_s

    def postprocess(self, ori_img, output):
        LABEL_NAMES = []
        with open(self.class_names, 'r') as f:
            for line in f.readlines():
                LABEL_NAMES.append(line.strip())
        
        H, W, _ = ori_img.shape
        scale_h, scale_w = H / 352., W / 352.

        detected_boxes = []
        for box in output[0]:
            box = box.tolist()
           
            obj_score = box[4]
            category = LABEL_NAMES[int(box[5])]
            x1, y1 = int(box[0] * W), int(box[1] * H)
            x2, y2 = int(box[2] * W), int(box[3] * H)
            detected_box = [x1, y1, x2, y2, category]
            detected_boxes.append(detected_box)

            cv2.rectangle(ori_img, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(ori_img, '%.2f' % obj_score, (x1, y1 - 5), 0, 0.7, (0, 255, 0), 2)  
            cv2.putText(ori_img, category, (x1, y1 - 25), 0, 0.7, (0, 255, 0), 2)
        return ori_img, detected_boxes
