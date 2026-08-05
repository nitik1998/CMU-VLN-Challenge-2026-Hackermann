#!/usr/bin/env python3
"""Instruction-following legs: region synthesis, avoid zones, leg automaton.

Torch-free. Implements SEARCH_DOMAIN_PIPELINE.md section 5's follow operator:
each leg's anchor is an object-reference sub-problem; this module owns the
geometry that turns grounded anchors into scored trajectory constraints.

The evaluator scores the ACTUAL driven trajectory against ordered constraints
and forbidden regions, so avoid zones apply to all motion from the moment
their anchors ground, and pass-between emits entry AND exit waypoints so the
base planner is forced through the corridor rather than merely near it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np


@dataclass
class Leg:
    kind: str                 # goto/stop/pass/avoid relation operator
    entity_ids: list[str]
    state: str = "pending"    # pending | grounded | executing | done | best_guess
    anchors_xy: list = field(default_factory=list)
    waypoints: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"kind": self.kind, "entities": self.entity_ids,
                "state": self.state,
                "anchors_xy": [list(map(float, a)) for a in self.anchors_xy],
                "waypoints": [list(map(float, w)) for w in self.waypoints]}


def parse_legs(program: dict) -> list[Leg]:
    legs = []
    for raw in program.get("answer", {}).get("legs", []) or []:
        entity_ids = raw.get("of")
        if isinstance(entity_ids, str):
            entity_ids = [entity_ids]
        kind = str(raw.get("kind", "goto_near"))
        if kind not in {"goto_near", "stop_at", "pass_between",
                        "avoid_between", "pass_near", "avoid_near"}:
            kind = "goto_near"
        legs.append(Leg(kind, [str(e) for e in (entity_ids or [])]))
    return legs


def annulus_waypoint(anchor_xy: np.ndarray, robot_xy: np.ndarray, coverage,
                     standoff: float = 0.9) -> list | None:
    """Safe pose near the anchor, preferring the robot's current side."""
    anchor = np.asarray(anchor_xy, float)
    robot = np.asarray(robot_xy, float)
    toward_robot = robot - anchor
    base = math.atan2(toward_robot[1], toward_robot[0]) \
        if np.linalg.norm(toward_robot) > 1e-6 else 0.0
    for offset in (0.0, 0.5, -0.5, 1.0, -1.0, 1.6, -1.6, math.pi):
        for radius in (standoff, standoff + 0.35, standoff + 0.7):
            goal = anchor + radius * np.array([math.cos(base + offset),
                                               math.sin(base + offset)])
            if coverage.is_safe_xy(goal):
                return [float(goal[0]), float(goal[1])]
    return None


def corridor_waypoints(anchor_a: np.ndarray, anchor_b: np.ndarray,
                       robot_xy: np.ndarray, coverage,
                       entry_offset: float = 1.0) -> list[list] | None:
    """Entry and exit poses through the gap between two anchors.

    Entry/exit lie on the perpendicular through the gap midpoint; ordering
    puts the robot-side point first so the driven path crosses the corridor.
    """
    first = np.asarray(anchor_a, float)
    second = np.asarray(anchor_b, float)
    middle = 0.5 * (first + second)
    gap = second - first
    norm = np.linalg.norm(gap)
    if norm < 1e-6:
        return None
    axis = np.array([-gap[1], gap[0]]) / norm
    robot = np.asarray(robot_xy, float)
    if (robot - middle) @ axis > 0:
        axis = -axis          # entry on the robot's side, exit beyond
    for offset in (entry_offset, entry_offset + 0.4, entry_offset - 0.3):
        entry = middle - offset * axis
        exit_ = middle + offset * axis
        if coverage.is_safe_xy(entry) and coverage.is_safe_xy(exit_):
            return [[float(entry[0]), float(entry[1])],
                    [float(exit_[0]), float(exit_[1])]]
    if coverage.is_safe_xy(middle):
        return [[float(middle[0]), float(middle[1])]]
    return None


def pass_near_waypoints(anchor_xy: np.ndarray, robot_xy: np.ndarray, coverage,
                        standoff: float = 0.9) -> list[list] | None:
    """Two safe poses that make the driven trajectory pass by an anchor.

    A single annulus point can satisfy a destination but not a path constraint;
    entry and exit are placed on opposite sides of the landmark.
    """
    anchor = np.asarray(anchor_xy, float)
    robot = np.asarray(robot_xy, float)
    direction = anchor - robot
    norm = float(np.linalg.norm(direction))
    axis = (np.array([1.0, 0.0]) if norm < 1e-6 else direction / norm)
    lateral = np.array([-axis[1], axis[0]])
    for radius in (standoff, standoff + 0.35, standoff + 0.7):
        for sign in (1.0, -1.0):
            offset = sign * radius * lateral
            entry = anchor - 0.9 * axis + offset
            exit_ = anchor + 0.9 * axis + offset
            if coverage.is_safe_xy(entry) and coverage.is_safe_xy(exit_):
                return [[float(entry[0]), float(entry[1])],
                        [float(exit_[0]), float(exit_[1])]]
    return None


def avoid_zone(anchor_a: np.ndarray, anchor_b: np.ndarray,
               half_width: float = 0.8) -> dict:
    """Forbidden corridor between two anchors, as a capsule (segment+radius)."""
    return {"a": [float(v) for v in np.asarray(anchor_a, float)[:2]],
            "b": [float(v) for v in np.asarray(anchor_b, float)[:2]],
            "radius": float(half_width)}


def _point_segment_distance(point: np.ndarray, seg_a: np.ndarray,
                            seg_b: np.ndarray) -> float:
    direction = seg_b - seg_a
    length_sq = float(direction @ direction)
    if length_sq < 1e-12:
        return float(np.linalg.norm(point - seg_a))
    t = float(np.clip((point - seg_a) @ direction / length_sq, 0.0, 1.0))
    return float(np.linalg.norm(point - (seg_a + t * direction)))


def segment_hits_zone(start_xy, goal_xy, zone: dict,
                      samples: int = 24) -> bool:
    start = np.asarray(start_xy, float)
    goal = np.asarray(goal_xy, float)
    seg_a = np.asarray(zone["a"], float)
    seg_b = np.asarray(zone["b"], float)
    for t in np.linspace(0.0, 1.0, samples):
        point = start + t * (goal - start)
        if _point_segment_distance(point, seg_a, seg_b) <= zone["radius"]:
            return True
    return False


def detour_waypoint(start_xy, goal_xy, zone: dict, coverage,
                    clearance: float = 0.6) -> list | None:
    """One guide waypoint routing a violating segment around the zone."""
    seg_a = np.asarray(zone["a"], float)
    seg_b = np.asarray(zone["b"], float)
    middle = 0.5 * (seg_a + seg_b)
    axis = seg_b - seg_a
    norm = float(np.linalg.norm(axis))
    # A forbidden corridor is a capsule: you get past it around an END CAP, so
    # the detour runs ALONG the anchor-to-anchor axis, never across it.
    direction = (np.array([1.0, 0.0]) if norm < 1e-6 else axis / norm)
    reach = zone["radius"] + clearance + 0.5 * norm
    for sign in (1.0, -1.0):
        for scale in (1.0, 1.4, 1.9):
            guide = middle + sign * scale * reach * direction
            if not coverage.is_safe_xy(guide):
                continue
            if (not segment_hits_zone(start_xy, guide, zone) and
                    not segment_hits_zone(guide, goal_xy, zone)):
                return [float(guide[0]), float(guide[1])]
    return None


def plan_leg_waypoints(leg: Leg, robot_xy, coverage,
                       zones: list[dict]) -> list[list]:
    """Waypoint sequence for one grounded leg, detouring around avoid zones."""
    if leg.kind in {"goto_near", "stop_at"}:
        if not leg.anchors_xy:
            return []
        goal = annulus_waypoint(leg.anchors_xy[0], robot_xy, coverage)
        raw = [goal] if goal is not None else []
    elif leg.kind == "pass_between":
        if len(leg.anchors_xy) < 2:
            return []
        raw = corridor_waypoints(leg.anchors_xy[0], leg.anchors_xy[1],
                                 robot_xy, coverage) or []
    elif leg.kind == "pass_near":
        if not leg.anchors_xy:
            return []
        raw = pass_near_waypoints(
            leg.anchors_xy[0], robot_xy, coverage) or []
    else:                       # avoidance contributes a zone, not motion
        return []
    guarded: list[list] = []
    position = np.asarray(robot_xy, float)
    for waypoint in raw:
        for zone in zones:
            if segment_hits_zone(position, waypoint, zone):
                guide = detour_waypoint(position, waypoint, zone, coverage)
                if guide is not None:
                    guarded.append(guide)
                    position = np.asarray(guide, float)
        guarded.append(waypoint)
        position = np.asarray(waypoint, float)
    return guarded
