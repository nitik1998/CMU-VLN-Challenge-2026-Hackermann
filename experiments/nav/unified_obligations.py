#!/usr/bin/env python3
"""Deterministic exploration-obligation policy for the unified runner.

Torch-free on purpose: the policy is pure geometry over the scene graph and
coverage grid, so it stays unit-testable without the perception stack.

Obligations must be DISCHARGEABLE. Residual regions are cell-set identities:
one attempted visit retires the exact space regardless of navigation outcome.
An unexplained clipped proposal is executable work, not a permanent veto.
"""

from __future__ import annotations

import numpy as np

from coverage import Coverage
from unified_scene_graph import SceneGraph


def complete_visual_audit(pending: dict, captured_pose: np.ndarray,
                          new_cells: int, min_displacement_m: float = 0.50,
                          min_new_cells: int = 8,
                          enumerated_fraction: float = 0.0) -> dict:
    """Convert a navigation attempt into evidence only after a new capture."""
    displacement = float(np.linalg.norm(
        np.asarray(captured_pose[:2], float) -
        np.asarray(pending["start_pose"], float)))
    visual = displacement >= min_displacement_m or new_cells >= min_new_cells
    outcome = pending["status"]
    if not pending["cells"]:
        discharge_state = "audit_only"
    elif outcome in {"arrived", "far_reports_goal_reached"}:
        discharge_state = ("enumerated" if enumerated_fraction >= 0.90
                           else "unobservable")
    else:
        discharge_state = "unreachable"
    return {
        "state": discharge_state,
        "goal": pending["goal"],
        "status": outcome,
        "displacement_m": round(displacement, 3),
        "new_cells": int(new_cells),
        "enumerated_fraction": round(float(enumerated_fraction), 3),
        "visual_audit": bool(visual),
        "cell_count": len(pending["cells"]),
    }


def _yaw(pose: np.ndarray) -> float:
    x, y, z, w = np.asarray(pose, float)[3:]
    return float(np.arctan2(2.0 * (w * z + x * y),
                            1.0 - 2.0 * (y * y + z * z)))


def truncated_recovery(proposal: dict, pose: np.ndarray, coverage: Coverage,
                       tried: list[tuple[float, float]], image_width: int,
                       backoff_m: float = 0.75) -> dict | None:
    """Find a safe back-away pose that brings a clipped object into view."""
    box = proposal["box"]
    centre_x = (float(box[0]) + float(box[2])) / 2.0
    panorama_azimuth = (centre_x / image_width - 0.5) * 2.0 * np.pi
    # Calibration: positive panorama azimuth is negative sensor/map yaw.
    map_bearing = _yaw(pose) - panorama_azimuth
    toward = np.array([np.cos(map_bearing), np.sin(map_bearing)])
    robot = np.asarray(pose[:2], float)
    lateral = np.array([-toward[1], toward[0]])
    for distance, side in ((backoff_m, 0.0), (1.0, 0.0),
                           (backoff_m, 0.45), (backoff_m, -0.45)):
        goal = robot - distance * toward + side * lateral
        if not coverage.is_safe_xy(goal):
            continue
        if any(np.linalg.norm(goal - np.asarray(old)) < 0.55 for old in tried):
            continue
        return {
            "kind": "resolve_truncated",
            "xy": goal.tolist(),
            "proposal": proposal,
            "reason": "back away so the image-boundary proposal enters the FOV",
        }
    return None


def choose_obligation(graph: SceneGraph, entity_id: str, coverage: Coverage,
                      robot_xy: np.ndarray, accumulated_cloud: np.ndarray,
                      tried: list[tuple[float, float]],
                      residuals: list[dict] | None = None,
                      unexplained_truncated: list[dict] = (),
                      pose: np.ndarray | None = None,
                      image_width: int = 1920,
                      surface_visits: list[dict] = ()) -> dict | None:
    unresolved = [node for node in graph.nodes_for(entity_id)
                  if node.facts.get("is_class") is None or
                  node.needs_corroboration() or
                  (node.facts.get("is_class") is True and
                   not node.visual_evidence()["semantic_verified"])]
    resolvable = [node for node in unresolved
                  if np.all(np.isfinite(node.position()))]
    if resolvable:
        from run_question import viewpoint  # lazy: pulls the torch stack
        node = min(resolvable,
                   key=lambda item: np.linalg.norm(item.position()[:2] - robot_xy))
        goal = viewpoint(node.position(), accumulated_cloud, robot_xy,
                         max(0.75, min(1.5, np.linalg.norm(
                             node.position()[:2] - robot_xy) * 0.55)))
        if coverage.is_safe_xy(goal) and all(
                np.linalg.norm(np.asarray(goal) - np.asarray(old)) >= 0.55
                for old in tried):
            return {"kind": "resolve", "node": node.id,
                    "xy": [float(goal[0]), float(goal[1])],
                    "reason": ("corroborate a weak single-view proposal" if
                               node.needs_corroboration() else
                               "obtain independent semantic visual evidence")}

    if pose is not None:
        for proposal in unexplained_truncated:
            recovery = truncated_recovery(
                proposal, pose, coverage, tried, image_width)
            if recovery is not None:
                return recovery

    # Support-surface domains come before free-space residuals: for an
    # "on the table" question, uninspected tabletops ARE the search space.
    for visit in surface_visits:
        goal = np.asarray(visit["xy"], float)
        if any(np.linalg.norm(goal - np.asarray(old)) < 0.55 for old in tried):
            continue
        return visit

    for component in residuals or []:
        if component.get("state") != "active" or component.get("xy") is None:
            continue
        goal = np.asarray(component["xy"], float)
        if any(np.linalg.norm(goal - np.asarray(old)) < 0.55 for old in tried):
            continue
        role = component.get("domain_role", "target")
        entity_class = component.get("entity_class", "queried entity")
        return {
            "kind": "enumerate_residual",
            "xy": goal.tolist(),
            "target_xy": list(component["target_xy"]),
            "cells": [list(cell) for cell in component["cells"]],
            "source_cells": [list(cell) for cell in
                             component.get("source_cells", component["cells"])],
            "domain_key": component.get("domain_key", "target"),
            "domain_role": role,
            "owner_entity_id": component.get("owner_entity_id", entity_id),
            "max_range_m": component.get("max_range_m"),
            "min_size_m": component.get("min_size_m"),
            "viewpoint_kind": component.get("viewpoint_kind"),
            "fit_diameter_m": component["fit_diameter_m"],
            "area_m2": component["area_m2"],
            "reason": (f"inspect fit-capable residual space for {role} "
                       f"entity {entity_class} at its evidence-derived "
                       "detectable range"),
        }
    return None
