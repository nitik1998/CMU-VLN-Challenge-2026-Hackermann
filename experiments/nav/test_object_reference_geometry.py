import math
import unittest

import numpy as np

from object_reference_geometry import box_iou_3d, fit_upright_box, fuse_points


class ObjectReferenceGeometryTests(unittest.TestCase):
    def test_fused_box_recovers_rotated_cuboid_with_outliers(self):
        rng = np.random.default_rng(9)
        center = np.array([2.0, -1.0, 0.7])
        length, width, height, yaw = 0.8, 0.4, 1.0, math.radians(28)
        local = rng.uniform([-length / 2, -width / 2, -height / 2],
                            [length / 2, width / 2, height / 2], (8000, 3))
        # Keep surface-like samples from complementary views.
        local_a = local[(local[:, 0] > length * 0.40) | (local[:, 1] > width * 0.40)]
        local_b = local[(local[:, 0] < -length * 0.40) | (local[:, 1] < -width * 0.40)]
        rotation = np.array([[math.cos(yaw), -math.sin(yaw)],
                             [math.sin(yaw), math.cos(yaw)]])
        def world(values):
            result = values.copy()
            result[:, :2] = values[:, :2] @ rotation.T
            return result + center
        outliers = rng.uniform([-3, -3, -1], [3, 3, 2], (8, 3)) + center
        points = fuse_points([world(local_a), world(local_b), outliers])
        box = fit_upright_box(points)
        expected = {"center": center.tolist(), "length": length, "width": width,
                    "height": height, "yaw": yaw}
        self.assertGreater(box_iou_3d(box, expected), 0.83)

    def test_identical_box_iou(self):
        box = {"center": [1.0, 2.0, 0.5], "length": 0.8, "width": 0.3,
               "height": 1.0, "yaw": 0.4}
        self.assertAlmostEqual(box_iou_3d(box, box), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
