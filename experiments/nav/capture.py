#!/usr/bin/env python3
"""Capture a synchronized sensor snapshot using ONLY challenge-allowed topics.

Allowed at test time:
  /camera/image      sensor_msgs/Image      (360 equirect, 1920x640)
  /registered_scan   sensor_msgs/PointCloud2 (map frame)
  /state_estimation  nav_msgs/Odometry      (sensor pose in map)

Writes: frame.png, cloud_map.npy (Nx3, map frame), pose.npz
Run INSIDE iros2026_system.

usage: capture.py <outdir> [accumulate_seconds]
"""
import sys, os, time
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from nav_msgs.msg import Odometry
from sensor_msgs_py import point_cloud2
import cv_bridge, cv2

outdir = sys.argv[1]
accum_secs = float(sys.argv[2]) if len(sys.argv) > 2 else 4.0
os.makedirs(outdir, exist_ok=True)

rclpy.init()
node = Node("sensor_capture")
bridge = cv_bridge.CvBridge()

state = {"img": None, "pose": None, "clouds": [], "terrain": []}


def odom_cb(msg):
    p = msg.pose.pose.position
    q = msg.pose.pose.orientation
    state["pose"] = np.array([p.x, p.y, p.z, q.x, q.y, q.z, q.w])


def img_cb(msg):
    # always keep the freshest image + the pose valid at that moment
    if state["pose"] is None:
        return
    state["img"] = (bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8"),
                    state["pose"].copy())


def scan_cb(msg):
    pts = point_cloud2.read_points_numpy(msg, field_names=("x", "y", "z"), skip_nans=True)
    if pts.size:
        state["clouds"].append(np.asarray(pts, dtype=np.float32).reshape(-1, 3))


def terrain_cb(msg):
    # terrainAnalysis encodes height-above-ground in 'intensity';
    # intensity > obstacleHeightThre (0.2) means obstacle.
    pts = point_cloud2.read_points_numpy(
        msg, field_names=("x", "y", "z", "intensity"), skip_nans=True)
    if pts.size:
        state["terrain"].append(np.asarray(pts, dtype=np.float32).reshape(-1, 4))


node.create_subscription(Odometry, "/state_estimation", odom_cb, 20)
node.create_subscription(Image, "/camera/image", img_cb, 5)
node.create_subscription(PointCloud2, "/registered_scan", scan_cb, 20)
node.create_subscription(PointCloud2, "/terrain_map_ext", terrain_cb, 10)

print(f"accumulating {accum_secs}s of sensor data ...")
t0 = time.time()
while time.time() - t0 < accum_secs:
    rclpy.spin_once(node, timeout_sec=0.1)

if state["img"] is None or not state["clouds"]:
    print("FAILED: img=%s clouds=%d" % (state["img"] is not None, len(state["clouds"])))
    rclpy.shutdown(); sys.exit(1)

img, pose = state["img"]
cloud = np.concatenate(state["clouds"], axis=0)

# voxel downsample (5cm) to keep it manageable
key = np.floor(cloud / 0.05).astype(np.int64)
_, idx = np.unique(key, axis=0, return_index=True)
cloud = cloud[idx]

cv2.imwrite(f"{outdir}/frame.png", img)
np.save(f"{outdir}/cloud_map.npy", cloud)
np.savez(f"{outdir}/pose.npz", pose=pose)

if state["terrain"]:
    terr = np.concatenate(state["terrain"], axis=0)
    tk = np.floor(terr[:, :2] / 0.1).astype(np.int64)
    _, ti = np.unique(tk, axis=0, return_index=True)
    terr = terr[ti]
    np.save(f"{outdir}/terrain.npy", terr)
    print(f"terrain    : {terr.shape} (x,y,z,height_above_ground)")
else:
    print("terrain    : NONE received")

print(f"image      : {img.shape}")
print(f"cloud      : {cloud.shape} (map frame, 5cm voxels)")
print(f"sensor pos : {pose[:3]}")
print(f"sensor quat: {pose[3:]}")
print(f"saved -> {outdir}")

rclpy.shutdown()
