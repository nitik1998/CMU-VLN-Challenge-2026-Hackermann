#!/usr/bin/env python3
"""Regression tests for robot-footprint endpoint safety."""

import unittest
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from coverage import Coverage, RES, ROBOT_R


class CoverageSafetyTest(unittest.TestCase):
    def setUp(self):
        self.coverage = Coverage([0.0, 0.0])
        self.xy = np.array([1.0, 1.0])
        self.cell = self.coverage._ij(self.xy)[0]
        self.radius = int(np.ceil(ROBOT_R / RES))

    def footprint_cells(self):
        i, j = self.cell
        for di in range(-self.radius, self.radius + 1):
            for dj in range(-self.radius, self.radius + 1):
                if di * di + dj * dj <= self.radius * self.radius:
                    yield i + di, j + dj

    def test_endpoint_radius_includes_vehicle_diagonal_and_map_margin(self):
        configured_half_diagonal = np.hypot(0.50 / 2.0, 0.50 / 2.0)
        self.assertGreaterEqual(ROBOT_R - configured_half_diagonal, 0.15)

    def test_free_center_with_unknown_footprint_is_unsafe(self):
        self.coverage.free[tuple(self.cell)] = True
        self.assertFalse(self.coverage.is_safe_xy(self.xy))

    def test_fully_known_free_footprint_is_safe(self):
        for cell in self.footprint_cells():
            self.coverage.free[cell] = True
        self.assertTrue(self.coverage.is_safe_xy(self.xy))

    def test_one_blocked_footprint_cell_is_unsafe(self):
        cells = list(self.footprint_cells())
        for cell in cells:
            self.coverage.free[cell] = True
        blocked = cells[-1]
        self.coverage.free[blocked] = False
        self.coverage.block[blocked] = True
        self.assertFalse(self.coverage.is_safe_xy(self.xy))

    def test_observation_records_minimum_range_per_cell(self):
        self.coverage.block[:] = True
        center = self.coverage._ij([0.0, 0.0])[0]
        target = center + np.array([5, 0])
        for i in range(center[0] - 6, center[0] + 7):
            for j in range(center[1] - 2, center[1] + 3):
                self.coverage.block[i, j] = False
                self.coverage.free[i, j] = True
        self.coverage.mark_observed_from([0.0, 0.0])
        self.assertAlmostEqual(
            float(self.coverage.min_observation_range[tuple(target)]),
            1.0, places=4)
        self.assertFalse(self.coverage.enumerated_for(0.8)[tuple(target)])
        self.assertTrue(self.coverage.enumerated_for(1.2)[tuple(target)])

    def test_residual_fit_rejects_sliver_and_retires_by_cell_overlap(self):
        self.coverage.block[:] = True
        center = self.coverage._ij([0.0, 0.0])[0]
        # A one-cell-wide wall sliver cannot contain a 45 cm cushion.
        sliver = [tuple(center + np.array([i, 0])) for i in range(-3, 4)]
        for cell in sliver:
            self.coverage.block[cell] = False
            self.coverage.free[cell] = True
        self.assertEqual(self.coverage.residual_components(3.8, 0.45), [])

        # A 3x3 component can; retiring a majority of its exact cells removes
        # it even if a future proposal would choose another centroid.
        self.coverage.free[:] = False
        self.coverage.block[:] = True
        square = [tuple(center + np.array([i, j]))
                  for i in range(-1, 2) for j in range(-1, 2)]
        for cell in square:
            self.coverage.block[cell] = False
            self.coverage.free[cell] = True
        components = self.coverage.residual_components(3.8, 0.45)
        self.assertEqual(len(components), 1)
        self.assertGreaterEqual(components[0]["fit_diameter_m"], 0.45)
        retired = set(components[0]["cells"][:5])
        self.assertEqual(self.coverage.residual_components(
            3.8, 0.45, retired_cells=retired), [])

    def test_one_attempt_cannot_retire_a_room_sized_residual(self):
        self.coverage.block[:] = True
        center = self.coverage._ij([0.0, 0.0])[0]
        for i in range(center[0] - 10, center[0] + 11):
            for j in range(center[1] - 10, center[1] + 11):
                self.coverage.block[i, j] = False
                self.coverage.free[i, j] = True
        first = self.coverage.residual_components(3.8, 0.45)[0]
        self.assertLess(first["cell_count"], first["source_component_cells"])
        remaining = self.coverage.residual_components(
            3.8, 0.45, retired_cells=set(first["cells"]))
        self.assertTrue(remaining)

    def test_viewpoint_selection_rejects_disconnected_safe_room(self):
        self.coverage.free[:] = False
        self.coverage.block[:] = True
        origin = self.coverage._ij([0.0, 0.0])[0]
        remote = origin + np.array([30, 0])
        for center in (origin, remote):
            for i in range(center[0] - 5, center[0] + 6):
                for j in range(center[1] - 5, center[1] + 6):
                    self.coverage.block[i, j] = False
                    self.coverage.free[i, j] = True
        self.assertTrue(self.coverage.is_safe_xy([0.0, 0.0]))
        remote_xy = self.coverage._xy(remote)
        self.assertTrue(self.coverage.is_safe_xy(remote_xy))
        self.assertFalse(self.coverage.is_reachable_xy(
            [0.0, 0.0], remote_xy))


if __name__ == "__main__":
    unittest.main()
