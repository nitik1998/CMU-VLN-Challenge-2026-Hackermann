#!/usr/bin/env python3
"""Grab a synchronized pair: /camera/image and the oracle /camera/semantic_image.

Debug-only, off the scored path (semantic_image is not an allowed test-time
topic -- README FAQ #6). Run INSIDE the container.

usage: capture_semantic_compare.py <outdir>
"""
import sys, os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv_bridge, cv2

outdir = sys.argv[1]
os.makedirs(outdir, exist_ok=True)

rclpy.init()
node = Node("semantic_compare_capture")
bridge = cv_bridge.CvBridge()
state = {"rgb": None, "sem": None}


def rgb_cb(msg):
    state["rgb"] = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")


def sem_cb(msg):
    state["sem"] = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")


node.create_subscription(Image, "/camera/image", rgb_cb, 5)
node.create_subscription(Image, "/camera/semantic_image", sem_cb, 5)

import time
t0 = time.time()
while time.time() - t0 < 6.0 and (state["rgb"] is None or state["sem"] is None):
    rclpy.spin_once(node, timeout_sec=0.1)

if state["rgb"] is None or state["sem"] is None:
    print(f"FAILED: rgb={state['rgb'] is not None} sem={state['sem'] is not None}")
    rclpy.shutdown(); sys.exit(1)

cv2.imwrite(f"{outdir}/rgb.png", state["rgb"])
cv2.imwrite(f"{outdir}/semantic_oracle.png", state["sem"])
print(f"rgb shape: {state['rgb'].shape}")
print(f"semantic shape: {state['sem'].shape}")
print(f"saved -> {outdir}")
rclpy.shutdown()
