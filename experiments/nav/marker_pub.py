#!/usr/bin/env python3
"""Publish a selected object's 3D box on /selected_object_marker.

Marker.CUBE is a box with independent per-axis scale -- not a cube. A yaw-oriented
box is exactly the scored representation: ground truth is stored as
"x y z length width height orientation_of_length_edge", and README L174 scores the
answer on overlap with that box. Matching the object's true silhouette would score
WORSE, because IoU is computed box-to-box. The box centre doubles as the navigation
waypoint (README L53).

Optionally also publishes the mask-selected member points as a SPHERE_LIST in a
separate namespace. Those are diagnostic only, never the answer -- they reveal
whether the points really lie on the object or have bled onto the wall behind it,
which is what inflated one scroll's width to 0.25 m against a true 0.04 m.

Publishes continuously until killed; a short burst is fragile because the
evaluation node has to be listening at that exact moment.

Run INSIDE iros2026_system.
usage: marker_pub.py '<json>' [points.npy]
   json: {"center":[x,y,z],"length":L,"width":W,"height":H,"yaw":rad,"label":"..."}
"""
import json
import math
import sys

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker

raw = json.loads(sys.argv[1])
# accept a single box or a list. For object-reference questions the answer is ONE
# box (the referred object is unique); a list is for inspecting a whole count.
specs = raw if isinstance(raw, list) else [raw]

rclpy.init()
node = Node("selected_object_marker_pub")
pub = node.create_publisher(Marker, "/selected_object_marker", 5)

markers = []
for k, spec in enumerate(specs):
    cx, cy, cz = spec["center"]
    yaw = float(spec.get("yaw", 0.0))
    L = max(0.02, float(spec["length"]))
    W = max(0.02, float(spec["width"]))
    H = max(0.02, float(spec["height"]))
    label = spec.get("label", f"object_{k}")

    box = Marker()
    box.header.frame_id = "map"
    box.ns = "selected_object"
    box.id = 2 * k
    box.type = Marker.CUBE
    box.action = Marker.ADD
    box.pose.position.x, box.pose.position.y, box.pose.position.z = cx, cy, cz
    box.pose.orientation.z = math.sin(yaw / 2.0)
    box.pose.orientation.w = math.cos(yaw / 2.0)
    box.scale.x, box.scale.y, box.scale.z = L, W, H
    box.color.r, box.color.g, box.color.b, box.color.a = 0.0, 1.0, 0.3, 0.45
    box.text = label

    txt = Marker()
    txt.header.frame_id = "map"
    txt.ns = "selected_object_label"
    txt.id = 2 * k + 1
    txt.type = Marker.TEXT_VIEW_FACING
    txt.action = Marker.ADD
    txt.pose.position.x, txt.pose.position.y = cx, cy
    txt.pose.position.z = cz + H / 2 + 0.15
    txt.pose.orientation.w = 1.0
    txt.scale.z = 0.10
    txt.color.r = txt.color.g = txt.color.b = txt.color.a = 1.0
    txt.text = label
    markers += [box, txt]
    print(f"  {label}: c=({cx:.2f},{cy:.2f},{cz:.2f}) L={L:.3f} W={W:.3f} "
          f"H={H:.3f} yaw={math.degrees(yaw):.1f}deg")

pts_marker = None
if len(sys.argv) > 2:
    import numpy as np
    P = np.load(sys.argv[2])
    pts_marker = Marker()
    pts_marker.header.frame_id = "map"
    pts_marker.ns = "selected_object_points"
    pts_marker.id = 2
    pts_marker.type = Marker.SPHERE_LIST
    pts_marker.action = Marker.ADD
    pts_marker.pose.orientation.w = 1.0
    pts_marker.scale.x = pts_marker.scale.y = pts_marker.scale.z = 0.015
    pts_marker.color.r, pts_marker.color.g = 1.0, 0.35
    pts_marker.color.b, pts_marker.color.a = 0.0, 1.0
    step = max(1, len(P) // 3000)
    for q in P[::step]:
        pts_marker.points.append(Point(x=float(q[0]), y=float(q[1]), z=float(q[2])))
    print(f"also publishing {len(pts_marker.points)} member points (of {len(P)})")

print(f"publishing {len(specs)} box(es) on /selected_object_marker  (Ctrl-C to stop)")
try:
    while rclpy.ok():
        now = node.get_clock().now().to_msg()
        for mk in markers:
            mk.header.stamp = now
            pub.publish(mk)
        if pts_marker is not None:
            pts_marker.header.stamp = now
            pub.publish(pts_marker)
        rclpy.spin_once(node, timeout_sec=0.3)
except KeyboardInterrupt:
    pass
rclpy.shutdown()
