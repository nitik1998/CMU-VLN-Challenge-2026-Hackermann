#!/usr/bin/env python3
"""Compile a typed question program into an explicit search domain.

Torch-free. Implements SEARCH_DOMAIN_PIPELINE.md section 2: the domain is a
finite set of surfaces and regions that bounds where answer evidence can live.
The floor is one support surface among many; frontier space matters only where
a domain-relevant class could physically fit.
"""

from __future__ import annotations

import numpy as np

from enumeration_range import (enumeration_range_m,
                               observed_entity_size_m)
from support_surfaces import SurfaceRegistry
from unified_scene_graph import appearance_compatible
from unified_program import entity_dependency_closure


# Geometry, rather than a released-scene object ontology, defines the domain.
# This is the conservative minimum width used only to range a sweep whose goal
# is discovering horizontal support planes (not recognizing a named class).
GENERIC_SUPPORT_MIN_WIDTH_M = 0.30


def entity_residual_profiles(spec: dict, program: dict, graph,
                             target_size_m: float,
                             target_range_m: float) -> list[dict]:
    """Compile independent physical-coverage profiles for query entities.

    Anchor discovery and target enumeration are different obligations. An
    observed large landmark is discoverable at room range; a small target may
    require close inspection, but only inside its relation-constrained domain.
    Class strings never select a scale or a support surface.
    """
    profiles = []
    named_supports = set(spec.get("target_named_support_ids", []))
    for entity_id in dict.fromkeys(spec.get("anchor_open_world_ids", [])):
        nodes = [node for node in graph.nodes_for(entity_id)
                 if node.facts.get("is_class") is True]
        size = observed_entity_size_m(nodes)
        profiles.append({
            "key": f"anchor:{entity_id}",
            "role": "anchor",
            "entity_id": entity_id,
            "class": program["entities"][entity_id]["class"],
            "min_size_m": size,
            "max_range_m": enumeration_range_m(size),
            "priority": 0 if not nodes else 1,
            "required_frontier_audits": int(np.clip(
                np.ceil(1.20 / max(size, 0.08)), 2, 5)),
            "relation_domain": ("named_support" if entity_id in named_supports
                                else "relation_anchor"),
        })

    # Exact named-support targets live on the grounded support instances, not
    # in every free-space cell. All other open-world or region relations retain
    # a target residual profile; region_filter narrows it once anchors ground.
    exact_named_support = bool(named_supports)
    target_needs_global = bool(
        spec.get("floor") or spec.get("wall") or
        spec.get("target_all_horizontal_surfaces") or spec.get("regions"))
    if target_needs_global and not exact_named_support:
        profiles.append({
            "key": f"target:{spec['target']}",
            "role": "target",
            "entity_id": spec["target"],
            "class": spec["target_class"],
            "min_size_m": float(target_size_m),
            "max_range_m": float(target_range_m),
            "priority": 1,
            "relation_domain": "open_or_relational",
        })
    return sorted(profiles, key=lambda item: (item["priority"], item["key"]))


def compile_domain(program: dict) -> dict:
    """Map the compiled program onto explicit domain elements."""
    target_id = program["answer"].get("of")
    entities = program["entities"]
    target = entities.get(target_id, {})
    spec = {
        "target": target_id,
        "target_class": target.get("class", "object"),
        "floor": False,
        "wall": False,
        "target_all_horizontal_surfaces": False,
        "target_named_support_ids": [],
        "anchor_open_world_ids": [],
        "all_horizontal_surfaces": False,
        "stated_support_classes": [],
        "anchor_entity_ids": [],
        "dependency_entity_ids": [],
        "regions": [],           # [{"kind":"disc","anchor":eid,"radius":m}]
        "prune_closest_to": None,
        "reason": [],
        "elements": [],
    }
    stated_support = False
    for predicate in program.get("filter", []):
        op, args = predicate["op"], predicate["args"]
        if args[0] != target_id:
            continue
        reference = entities.get(args[1], {}) if len(args) > 1 else {}
        if op == "on" and reference.get("structure") == "floor":
            spec["floor"] = True
            stated_support = True
            spec["reason"].append("stated: on the floor")
        elif op == "on" and reference.get("structure") == "wall":
            spec["wall"] = True
            stated_support = True
            spec["reason"].append("stated: on the wall")
        elif op == "on" and reference.get("class"):
            # The class name is useful for grounding the exact support entity,
            # but it may not exclude an unlabelled measured support plane.
            support_name = str(reference["class"]).lower().strip()
            spec["stated_support_classes"].append(support_name)
            spec["target_named_support_ids"].append(args[1])
            spec["anchor_entity_ids"].append(args[1])
            stated_support = True
            spec["reason"].append(f"stated: on {reference['class']}")
        elif op in {"above", "below", "under"} and reference.get("class"):
            spec["anchor_entity_ids"].append(args[1])
            spec["regions"].append(
                {"kind": "disc", "anchor": args[1], "radius": 2.0})
            spec["reason"].append(f"stated: {op} {reference['class']}")
        elif op == "near" and reference.get("class"):
            spec["anchor_entity_ids"].append(args[1])
            spec["regions"].append(
                {"kind": "disc", "anchor": args[1], "radius": 2.5})
            spec["reason"].append(f"stated: near {reference['class']}")
        elif op == "between" and len(args) == 3:
            spec["anchor_entity_ids"] += [a for a in args[1:]
                                          if entities.get(a, {}).get("class")]
            spec["regions"].append(
                {"kind": "corridor", "anchors": list(args[1:]),
                 "half_width": 1.2})
            spec["reason"].append("stated: between")
    answer = program.get("answer", {})
    if answer.get("op") == "argmin_dist" and answer.get("to"):
        spec["prune_closest_to"] = answer["to"]
        if answer["to"] not in spec["anchor_entity_ids"]:
            spec["anchor_entity_ids"].append(answer["to"])
    if answer.get("op") == "argmax_dist" and answer.get("to"):
        if answer["to"] not in spec["anchor_entity_ids"]:
            spec["anchor_entity_ids"].append(answer["to"])

    # Close over the relational query graph, not only predicates whose first
    # argument is answer.of. Nested supports and selectors are themselves
    # grounding obligations (sofa under pictures; table closest to a decal).
    dependencies = set(entity_dependency_closure(program, [target_id]))
    spec["dependency_entity_ids"] = sorted(dependencies)
    for dependency in sorted(dependencies - {target_id}):
        if entities.get(dependency, {}).get("class") and \
                dependency not in spec["anchor_entity_ids"]:
            spec["anchor_entity_ids"].append(dependency)

    # Every unresolved semantic reference is open-world too.  It may be on the
    # floor, a wall, or any measured support plane; its class name never closes
    # any of those possibilities.
    spec["anchor_open_world_ids"] = list(dict.fromkeys(
        anchor_id for anchor_id in spec["anchor_entity_ids"]
        if entities.get(anchor_id, {}).get("class")))
    if not stated_support:
        spec["floor"] = True
        spec["wall"] = True
        spec["target_all_horizontal_surfaces"] = True
        spec["reason"].append(
            "open-world support domain: floor, walls, and all measured "
            "horizontal surfaces")
    spec["all_horizontal_surfaces"] = bool(
        spec["target_all_horizontal_surfaces"] or
        spec["anchor_open_world_ids"])
    spec["stated_support_classes"] = list(dict.fromkeys(
        spec["stated_support_classes"]))
    spec["target_named_support_ids"] = list(dict.fromkeys(
        spec["target_named_support_ids"]))
    spec["elements"] = compile_domain_elements(program, spec)
    return spec


def compile_domain_elements(program: dict, spec: dict) -> list[dict]:
    """Canonical, orientation-neutral domain representation.

    Legacy booleans/lists remain during migration, but all new planning should
    consume these elements.  A wall, floor, tabletop, relation volume and
    unresolved landmark differ only in how visibility and candidate viewpoints
    are computed—not in the evidence loop that enumerates them.
    """
    entities = program["entities"]
    dependencies = set(spec.get("dependency_entity_ids", []))
    elements = []

    def add(value):
        key = tuple(sorted((name, repr(item)) for name, item in value.items()))
        if all(tuple(sorted((name, repr(item)) for name, item in old.items()))
               != key for old in elements):
            elements.append(value)

    for entity_id in spec.get("anchor_entity_ids", []):
        entity = entities.get(entity_id, {})
        if entity.get("class"):
            add({"kind": "ground_entity", "entity": entity_id,
                 "class": entity["class"], "state": "open"})

    for predicate in program.get("filter", []):
        op, args = predicate["op"], predicate["args"]
        if args[0] not in dependencies:
            continue
        if op == "on":
            reference = entities.get(args[1], {})
            if reference.get("structure"):
                add({"kind": "surface_family", "owner": args[0],
                     "surface": reference["structure"], "state": "open"})
            else:
                add({"kind": "support_instances", "owner": args[0],
                     "support_entity": args[1], "state": "open"})
        elif op in {"above", "below", "under", "near"}:
            add({"kind": "relation_volume", "owner": args[0],
                 "relation": op, "anchors": args[1:], "state": "open"})
        elif op == "between":
            add({"kind": "relation_volume", "owner": args[0],
                 "relation": "between", "anchors": args[1:],
                 "state": "open"})
        elif op == "with_on":
            add({"kind": "carried_object_relation", "owner": args[0],
                 "carried_entity": args[1], "state": "open"})

    target = spec["target"]
    if spec.get("floor"):
        add({"kind": "surface_family", "owner": target,
             "surface": "floor", "source": "stated_or_open_world",
             "state": "open"})
    if spec.get("wall"):
        add({"kind": "surface_family", "owner": target,
             "surface": "wall", "source": "stated_or_open_world",
             "state": "open"})
    if spec.get("target_all_horizontal_surfaces"):
        add({"kind": "surface_family", "owner": target,
             "surface": "horizontal_support", "source": "open_world",
             "state": "open"})
    if spec.get("anchor_open_world_ids"):
        add({"kind": "surface_family",
             "owner": list(spec["anchor_open_world_ids"]),
             "surface": "horizontal_support", "source": "open_world_anchor",
             "state": "open"})
    for entity_id, selector in program.get("selectors", {}).items():
        if entity_id in dependencies and selector.get("op") != "all":
            add({"kind": "entity_selector", "entity": entity_id,
                 "selector": selector["op"], "anchor": selector.get("to"),
                 "state": "open"})
    return elements


def anchor_positions(graph, entity_id: str) -> list[np.ndarray]:
    positions = []
    for node in graph.nodes_for(entity_id):
        # A raw/unknown SAM proposal is not a grounded semantic anchor.  Only
        # positive class evidence may shrink a domain or satisfy a relation.
        if node.facts.get("is_class") is not True:
            continue
        position = node.position()
        if np.all(np.isfinite(position)):
            positions.append(np.asarray(position[:2], float))
    return positions


def region_filter(spec: dict, graph, residuals: list[dict]) -> list[dict]:
    """Keep residual components only where they intersect stated regions.

    Until every region anchor is grounded the filter stays wide open: an
    unknown anchor position must not silently shrink the search space.
    """
    if not spec["regions"]:
        return residuals
    discs = []
    for region in spec["regions"]:
        if region["kind"] == "disc":
            for xy in anchor_positions(graph, region["anchor"]):
                discs.append((xy, float(region["radius"])))
        elif region["kind"] == "corridor":
            anchor_sets = [anchor_positions(graph, a)
                           for a in region["anchors"]]
            if all(anchor_sets):
                first, second = anchor_sets[0][0], anchor_sets[1][0]
                middle = 0.5 * (first + second)
                radius = 0.5 * float(np.linalg.norm(first - second)) + \
                    float(region["half_width"])
                discs.append((middle, radius))
    if not discs:
        return residuals
    kept = []
    for component in residuals:
        target = np.asarray(component["target_xy"], float)
        if any(np.linalg.norm(target - centre) <= radius
               for centre, radius in discs):
            kept.append(component)
    return kept


def closest_prune_radius(spec: dict, graph, program: dict) -> float | None:
    """For closest-questions, candidates beyond best-so-far cannot win."""
    anchor_id = spec.get("prune_closest_to")
    if not anchor_id:
        return None
    anchors = anchor_positions(graph, anchor_id)
    if not anchors:
        return None
    best = None
    for node in graph.matching_nodes(program):
        position = node.position()
        if not np.all(np.isfinite(position)):
            continue
        distance = min(float(np.linalg.norm(position[:2] - a))
                       for a in anchors)
        best = distance if best is None else min(best, distance)
    if best is None:
        return None
    return best + 0.75          # margin covers anchor/candidate position noise


def active_domain_surfaces(spec: dict, registry: SurfaceRegistry, graph,
                           prune_radius: float | None = None) -> list:
    """Canonical physical supports whose state can affect this question."""
    unresolved_anchor_ids = [
        entity_id for entity_id in spec.get("anchor_open_world_ids", [])
        if not anchor_positions(graph, entity_id)]
    target_open = spec.get("target_all_horizontal_surfaces", False)
    named_support_nodes = {
        node.id
        for entity_id in spec.get("target_named_support_ids", [])
        for node in graph.nodes_for(entity_id)
        if node.facts.get("is_class") is True}
    if target_open or unresolved_anchor_ids:
        candidates = registry.surfaces
    elif named_support_nodes:
        candidates = [surface for surface in registry.surfaces
                      if surface.bound_node in named_support_nodes]
    else:
        candidates = []
    selected = []
    for surface in candidates:
        if prune_radius is not None and spec.get("prune_closest_to"):
            anchors = anchor_positions(graph, spec["prune_closest_to"])
            cells = (np.array(sorted(surface.cells), float) + 0.5) * 0.05
            if anchors and len(cells) and min(
                    float(np.linalg.norm(cells - anchor, axis=1).min())
                    for anchor in anchors) > prune_radius:
                continue
        selected.append(surface)
    return selected


def surface_obligations(spec: dict, registry: SurfaceRegistry, graph, coverage,
                        robot_xy: np.ndarray, floor_z: float,
                        tried: list, prune_radius: float | None = None,
                        target_range_m: float | None = None
                        ) -> list[dict]:
    """Visit obligations for open in-domain surfaces, nearest first."""
    obligations = []
    unresolved_anchor_ids = [
        anchor_id for anchor_id in spec.get("anchor_open_world_ids", [])
        if not anchor_positions(graph, anchor_id)]
    target_open = spec.get("target_all_horizontal_surfaces", False)
    named_support_nodes = {
        node.id
        for entity_id in spec.get("target_named_support_ids", [])
        for node in graph.nodes_for(entity_id)
        if node.facts.get("is_class") is True}
    for surface in active_domain_surfaces(
            spec, registry, graph, prune_radius=prune_radius):
        bound_named_support = surface.bound_node in named_support_nodes
        anchor_search = bool(unresolved_anchor_ids)
        # A surface may already be enumerated for a furniture-sized target and
        # still require a closer view to ground a tiny reference on top of it.
        if anchor_search:
            if surface.attempts >= 2:
                continue
        elif surface.state != "open" or surface.attempts >= 2:
            continue
        goal = registry.viewpoint_for(
            surface, coverage, robot_xy, tried,
            standoff=(0.40 if anchor_search else max(
                0.30, min(1.10, float(target_range_m or 1.10) * 0.65))))
        if goal is None:
            continue
        obligations.append({
            "kind": "enumerate_surface",
            "surface": surface.id,
            "anchor_ids": unresolved_anchor_ids if anchor_search else [],
            "xy": goal,
            "reason": (f"inspect the top of {surface.klass or 'candidate'} "
                       f"surface {surface.id} at close range" +
                       (f" to ground {', '.join(unresolved_anchor_ids)}"
                        if anchor_search else "")),
            "_priority": 0 if bound_named_support else 1,
            "_distance": float(np.linalg.norm(
                np.asarray(goal) - np.asarray(robot_xy, float))),
        })
    obligations.sort(key=lambda item: (item["_priority"], item["_distance"]))
    for item in obligations:
        item.pop("_priority", None)
        item.pop("_distance", None)
    return obligations


def discharge_unactionable_surfaces(spec: dict, registry: SurfaceRegistry,
                                    graph, coverage, robot_xy: np.ndarray,
                                    tried: list, floor_z: float,
                                    prune_radius: float | None = None,
                                    target_range_m: float | None = None,
                                    residuals_remaining: bool = False
                                    ) -> list[str]:
    """Retire a blocker only when it has no safe executable inspection view."""
    if residuals_remaining:
        return []
    visits = surface_obligations(
        spec, registry, graph, coverage, robot_xy, floor_z, tried,
        prune_radius=prune_radius, target_range_m=target_range_m)
    actionable = {str(visit["surface"]) for visit in visits
                  if visit.get("kind") == "enumerate_surface"}
    retired = []
    for surface in active_domain_surfaces(
            spec, registry, graph, prune_radius=prune_radius):
        if surface.state != "open" or surface.id in actionable:
            continue
        surface.state = "unobservable"
        retired.append(surface.id)
    return retired


def anchor_inspection_obligations(spec: dict, registry: SurfaceRegistry,
                                  graph, coverage, robot_xy: np.ndarray,
                                  tried: list, focus_poses: dict,
                                  required_views: int = 2) -> list[dict]:
    """Reachable alternate bearings around grounded relation anchors."""
    obligations = []
    for entity_id in spec.get("target_named_support_ids", []):
        for node in graph.nodes_for(entity_id):
            if node.facts.get("is_class") is not True:
                continue
            prior = focus_poses.get(node.id, [])
            if len(prior) >= required_views:
                continue
            center = np.asarray(node.position()[:2], float)
            if not np.all(np.isfinite(center)):
                continue
            points = np.asarray(node.geometry_points(), float)
            object_radius = (float(np.percentile(
                np.linalg.norm(points[:, :2] - center, axis=1), 90))
                if len(points) >= 3 else 0.35)
            orbit_radius = float(np.clip(object_radius + 0.75, 0.90, 2.20))
            candidates = []
            for bearing in np.linspace(0.0, 2.0 * np.pi, 20, endpoint=False):
                goal = center + orbit_radius * np.array(
                    [np.cos(bearing), np.sin(bearing)])
                if not coverage.is_safe_xy(goal):
                    continue
                if (hasattr(coverage, "is_reachable_xy") and
                        not coverage.is_reachable_xy(robot_xy, goal)):
                    continue
                if any(np.linalg.norm(goal - np.asarray(old)) < 0.55
                       for old in tried):
                    continue
                travel = float(np.linalg.norm(goal - robot_xy))
                diversity = (min(float(np.linalg.norm(goal - np.asarray(old)))
                                 for old in prior) if prior else 0.0)
                candidates.append((travel - 0.35 * diversity, goal))
            if not candidates:
                continue
            goal = min(candidates, key=lambda pair: pair[0])[1]
            obligations.append({
                "kind": "inspect_relation_anchor",
                "anchor_node": node.id,
                "anchor_entity_id": entity_id,
                "xy": goal.tolist(),
                "reason": ("obtain an alternate undistorted close view of the "
                           "grounded relation anchor and its target-bearing "
                           "surface"),
            })
    return obligations


def grounding_range_m(spec: dict, target_range_m: float) -> float:
    """Residual enumeration range: furniture-sized when the domain is
    surface-first (finding all tables is the sweep), target-sized when the
    floor itself is in the domain."""
    if spec["floor"]:
        return target_range_m
    if (spec.get("target_all_horizontal_surfaces", False) or
            spec.get("target_named_support_ids")):
        return enumeration_range_m(GENERIC_SUPPORT_MIN_WIDTH_M)
    return target_range_m


def domain_reasons(spec: dict, registry: SurfaceRegistry, graph,
                   floor_z: float, prune_radius: float | None = None
                   ) -> list[str]:
    """Certificate blockers contributed by the surface/anchor domain."""
    reasons = []
    for surface in active_domain_surfaces(
            spec, registry, graph, prune_radius=prune_radius):
        if surface.state == "open":
            reasons.append(
                f"surface {surface.id} ({surface.klass or 'candidate'}) "
                "not yet enumerated")
    for entity_id in spec["anchor_entity_ids"]:
        if not anchor_positions(graph, entity_id):
            reasons.append(f"anchor entity {entity_id} not yet grounded")
    return reasons


def _visual_on_score(target_box, anchor_box) -> float:
    """Image evidence that one visible target lies on a visible anchor."""
    tx0, ty0, tx1, ty1 = map(float, target_box)
    ax0, ay0, ax1, ay1 = map(float, anchor_box)
    tw, th = max(tx1 - tx0, 1.0), max(ty1 - ty0, 1.0)
    aw, ah = max(ax1 - ax0, 1.0), max(ay1 - ay0, 1.0)
    horizontal = max(0.0, min(tx1, ax1) - max(tx0, ax0)) / tw
    vertical = max(0.0, min(ty1, ay1) - max(ty0, ay0)) / th
    bottom_to_top = abs(ty1 - ay0) / max(th, 0.20 * ah, 1.0)
    # Covers both containment (pillows on a bed/sofa) and contact at the top
    # image boundary (cup on table). The explicit relation supplies semantics;
    # this gate supplies visible physical compatibility.
    if horizontal < 0.70 or aw < 1.10 * tw:
        return 0.0
    if vertical >= 0.35:
        return 0.65 * horizontal + 0.35 * min(vertical, 1.0)
    if bottom_to_top <= 1.0:
        return 0.55 * horizontal + 0.30
    return 0.0


def assign_visible_support_relations(program: dict, best_detection: dict,
                                     graph, pose: np.ndarray,
                                     registry: SurfaceRegistry) -> list[dict]:
    """Ground explicit ``on`` relations from shared-view topology.

    Sparse panoramic LiDAR often samples a pillow but misses the mattress cell
    immediately below it. Same-view target/anchor topology plus metric vertical
    ordering supplies the missing relation, and anchor-relative tracks provide
    a viewpoint-invariant identity cue without a class synonym or centroid
    merge. Returns diagnostic events and rewrites ``best_detection`` after a
    proven track merge.
    """
    events = []
    current_ids = {item["node"].id for item in best_detection.values()}
    pose_key = [round(float(value), 2) for value in pose[:2]]
    merges = {}
    predicates = [predicate for predicate in program.get("filter", [])
                  if predicate.get("op") == "on" and
                  len(predicate.get("args", [])) == 2 and
                  program["entities"].get(predicate["args"][1], {}).get(
                      "class")]
    for predicate in predicates:
        target_entity, anchor_entity = predicate["args"]
        targets = [item for item in best_detection.values()
                   if item["entity"] == target_entity and
                   item["node"].facts.get("is_class") is not False]
        anchors = [item for item in best_detection.values()
                   if item["entity"] == anchor_entity and
                   item["node"].facts.get("is_class") is True]
        used_previous = set()
        for target_item in targets:
            ranked = sorted(((_visual_on_score(
                target_item["box"], anchor_item["box"]), anchor_item)
                for anchor_item in anchors), key=lambda pair: pair[0],
                reverse=True)
            if not ranked or ranked[0][0] < 0.72:
                continue
            visual_score, anchor_item = ranked[0]
            target_node, anchor_node = target_item["node"], anchor_item["node"]
            tx0, ty0, tx1, ty1 = map(float, target_item["box"])
            ax0, ay0, ax1, ay1 = map(float, anchor_item["box"])
            aw, ah = max(ax1 - ax0, 1.0), max(ay1 - ay0, 1.0)
            track = {
                "anchor_entity": anchor_entity,
                "anchor_node": anchor_node.id,
                "pose_xy": pose_key,
                "relative": [
                    round(((tx0 + tx1) / 2 - ax0) / aw, 4),
                    round(((ty0 + ty1) / 2 - ay0) / ah, 4),
                    round((tx1 - tx0) / aw, 4),
                    round((ty1 - ty0) / ah, 4),
                ],
            }
            previous = []
            for candidate in graph.nodes_for(target_entity):
                if candidate is target_node or candidate.id in current_ids:
                    continue
                if candidate.id in used_previous:
                    continue
                for old in candidate.facts.get("_anchor_tracks", []):
                    if (old.get("anchor_entity") != anchor_entity or
                            old.get("pose_xy") == pose_key):
                        continue
                    a, b = np.asarray(track["relative"], float), np.asarray(
                        old.get("relative", []), float)
                    if b.shape != (4,):
                        continue
                    delta = np.abs(a - b)
                    if (appearance_compatible(candidate.appearance(),
                                              target_node.appearance()) and
                            delta[0] <= 0.10 and delta[1] <= 0.45 and
                            delta[2] <= 0.10 and delta[3] <= 0.12):
                        previous.append((float(np.linalg.norm(
                            delta / np.array([0.10, 0.20, 0.10, 0.12]))),
                                         candidate))
                # Cross-view metric association is one-to-one and conditioned
                # on the same grounded relation anchor. Sparse LiDAR proposals
                # receive a wider sensor-uncertainty gate; dense proposals use
                # a tighter gate. Simultaneous current-view instances are
                # excluded above, so nearby layered objects cannot merge here.
                old_tracks = candidate.facts.get("_anchor_tracks", [])
                same_anchor = (candidate.facts.get("support_node") ==
                               anchor_node.id or any(
                                   old.get("anchor_entity") == anchor_entity
                                   for old in old_tracks))
                candidate_position = candidate.position()
                current_position = target_node.position()
                if (same_anchor and appearance_compatible(
                        candidate.appearance(), target_node.appearance()) and
                        np.all(np.isfinite(candidate_position)) and
                        np.all(np.isfinite(current_position))):
                    sparse = (len(candidate.geometry_points()) < 8 or
                              len(target_node.geometry_points()) < 8)
                    limit = 0.24 if sparse else 0.14
                    distance = float(np.linalg.norm(
                        candidate_position - current_position))
                    if distance <= limit:
                        previous.append((distance / limit, candidate))
            if previous:
                primary = min(previous, key=lambda pair: pair[0])[1]
                duplicate_id = target_node.id
                target_node = graph.merge_nodes(primary, target_node)
                merges[duplicate_id] = target_node
                events.append({"kind": "anchor_track_merge",
                               "primary": target_node.id,
                               "duplicate": duplicate_id,
                               "anchor": anchor_node.id})
                used_previous.add(target_node.id)
            tracks = list(target_node.facts.get("_anchor_tracks", []))
            if track not in tracks:
                tracks.append(track)
            target_node.facts["_anchor_tracks"] = tracks

            target_position, anchor_position = (target_node.position(),
                                                anchor_node.position())
            metric_order = (np.all(np.isfinite(target_position)) and
                            np.all(np.isfinite(anchor_position)) and
                            target_position[2] >= anchor_position[2] - 0.08)
            if metric_order:
                target_node.facts["support_node"] = anchor_node.id
                target_node.facts["support_relation_evidence"] = (
                    "explicit_on + same_view_topology + metric_vertical_order")
                top_surface = anchor_node.facts.get("top_surface")
                if top_surface:
                    target_node.support = f"surface:{top_surface}"
                    target_node.facts["support_surface"] = top_surface
                    surface = next((item for item in registry.surfaces
                                    if item.id == top_surface), None)
                    target_node.facts["support_class"] = (
                        surface.klass if surface and surface.klass else
                        program["entities"][anchor_entity]["class"])
                events.append({"kind": "visible_support_grounded",
                               "target": target_node.id,
                               "anchor": anchor_node.id,
                               "score": round(visual_score, 3)})

    if merges:
        rebuilt = {}
        for item in list(best_detection.values()):
            node = merges.get(item["node"].id, item["node"])
            item["node"] = node
            if (node.id not in rebuilt or item["pixel_width"] >
                    rebuilt[node.id]["pixel_width"]):
                rebuilt[node.id] = item
        best_detection.clear()
        best_detection.update(rebuilt)
    return events


def assign_supports(nodes, registry: SurfaceRegistry) -> None:
    """Record each node's supporting surface id/class as graph facts.

    Never downgrades an existing lidar-contact floor decision: registry
    surfaces cover elevated furniture tops, the floor path is owned by the
    scene graph's plane-contact/parallax logic.
    """
    for node in nodes:
        if node.support == "floor":
            continue
        if (node.facts.get("support_node") and
                str(node.facts.get("support_relation_evidence", "")).startswith(
                    "explicit_on")):
            # A generic plane-contact pass cannot overwrite an explicitly
            # grounded same-view relation with a nearby incidental plane.
            canonical = registry.canonical_id(
                node.facts.get("support_surface"))
            if canonical:
                node.facts["support_surface"] = canonical
                node.support = f"surface:{canonical}"
            continue
        surface = registry.support_of(node)
        if surface is None:
            continue
        node.support = f"surface:{surface.id}"
        node.facts["support_surface"] = surface.id
        node.facts["support_class"] = surface.klass or "candidate"
        if surface.bound_node:
            node.facts["support_node"] = surface.bound_node


def describe_program_relations(program: dict) -> str:
    """Plain-language restatement of the compiled filter, for a visual audit.

    Built from the compiled program rather than the raw question so an audit
    checks the same constraint the evaluator actually applied.
    """
    entity_id = program.get("answer", {}).get("of")
    entities = program.get("entities", {})
    phrases = []
    for predicate in program.get("filter", []):
        args = predicate.get("args") or []
        if not args or args[0] != entity_id:
            continue
        operator = str(predicate.get("op", "")).replace("_", " ").strip()
        named = []
        for other in args[1:]:
            spec = entities.get(other, {})
            named.append(str(spec.get("class") or spec.get("structure")
                             or other))
        if not (operator and named):
            continue
        phrases.append(f"{operator} {' and '.join(named)}")
    return "; ".join(phrases)
