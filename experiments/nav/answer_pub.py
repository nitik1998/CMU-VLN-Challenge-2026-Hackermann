#!/usr/bin/env python3
"""Publish the numeric answer on /numerical_response. Run INSIDE iros2026_system.
usage: answer_pub.py <int> [seconds]
"""
import sys, time, rclpy
from rclpy.node import Node
from std_msgs.msg import Int32
val = int(sys.argv[1]); dur = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
rclpy.init(); n = Node("answer_pub")
p = n.create_publisher(Int32, "/numerical_response", 5)
t0 = time.time()
while time.time() - t0 < dur:
    p.publish(Int32(data=val)); rclpy.spin_once(n, timeout_sec=0.1); time.sleep(0.2)
print(f"published {val} on /numerical_response for {dur}s")
rclpy.shutdown()
