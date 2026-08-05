"""Search-domain, surface-registry and instruction-leg tests.

Torch-free: everything here exercises the deterministic geometry layer that
decides WHERE to look and WHEN the domain is discharged.
"""

import math

import numpy as np
import pytest

from domain_inspection import inspection_bands, panorama_inspection_specs
from enumeration_range import observed_entity_size_m
from instruction_legs import (annulus_waypoint, avoid_zone, corridor_waypoints,
                              detour_waypoint, parse_legs,
                              pass_near_waypoints, plan_leg_waypoints,
                              segment_hits_zone)
from search_domain import (anchor_inspection_obligations, assign_supports,
                           assign_visible_support_relations,
                           closest_prune_radius, compile_domain,
                           discharge_unactionable_surfaces, domain_reasons,
                           entity_residual_profiles,
                           grounding_range_m, region_filter,
                           surface_obligations)
from support_surfaces import (Surface, SurfaceRegistry, cells_of,
                              nearby_cell_overlap)
from rectilinear import rectilinear_pixel_to_panorama
from unified_program import (entity_dependency_closure, fallback_program,
                             validate_program)
from unified_scene_graph import Observation, SceneGraph, SceneNode


class SafeCoverage:
    """Coverage stub where every pose is drivable."""

    def is_safe_xy(self, xy):
        return True


def make_node(node_id, entity_id, xy, z=0.0, is_class=True, points=None):
    node = SceneNode(node_id, entity_id, facts={"is_class": is_class})
    if points is None:
        points = np.array([[xy[0], xy[1], z],
                           [xy[0] + 0.04, xy[1], z],
                           [xy[0], xy[1] + 0.04, z]], np.float32)
    node.footprint_points = np.asarray(points, np.float32)
    node.observations.append(Observation(
        [0, 0, 0, 0, 0, 0, 1], 0.9, 100, "test", [0, 0, 1, 1], "test", 1.0))
    return node


# ---------------------------------------------------------------- domains
def test_stated_support_domain_targets_surface_class_not_floor():
    program = {
        "task": "count",
        "entities": {"E1": {"class": "cup", "attributes": [],
                            "sam_queries": ["cup"]},
                     "R1": {"class": "table", "attributes": [],
                            "sam_queries": ["table"]}},
        "filter": [{"op": "on", "args": ["E1", "R1"]}],
        "answer": {"op": "count", "of": "E1"},
    }
    domain = compile_domain(program)
    assert domain["floor"] is False
    assert domain["stated_support_classes"] == ["table"]
    assert domain["target_all_horizontal_surfaces"] is False
    assert domain["target_named_support_ids"] == ["R1"]
    assert domain["all_horizontal_surfaces"] is True
    assert domain["anchor_entity_ids"] == ["R1"]


def test_floor_question_keeps_the_floor_domain():
    question = "How many pillows are on the floor?"
    program = validate_program(fallback_program(question), question)
    domain = compile_domain(program)
    assert domain["floor"] is True
    assert domain["stated_support_classes"] == []


def test_wall_question_does_not_fall_back_to_furniture_surfaces():
    question = "How many calligraphy paintings are hanging on the wall?"
    program = {
        "task": "count",
        "entities": {
            "E1": {"class": "calligraphy painting", "attributes": [],
                   "sam_queries": ["calligraphy painting"]},
            "W": {"structure": "wall"},
        },
        "filter": [{"op": "on", "args": ["E1", "W"]}],
        "answer": {"op": "count", "of": "E1"},
    }
    domain = compile_domain(program)
    assert domain["wall"] is True
    assert domain["floor"] is False
    assert domain["stated_support_classes"] == []

    # Without that explicit relation, the target word may not imply a support.
    program["filter"] = []
    domain = compile_domain(program)
    assert domain["wall"] is True
    assert domain["floor"] is True
    assert domain["target_all_horizontal_surfaces"] is True
    assert domain["stated_support_classes"] == []


def test_above_relation_combines_anchor_volume_with_open_world_support():
    """A relation bounds the search only after its anchor is grounded; the
    target class itself must not choose a floor, wall, or furniture ontology."""
    program = {
        "task": "count",
        "entities": {
            "E1": {"class": "calligraphy painting", "attributes": [],
                   "sam_queries": ["calligraphy painting"]},
            "L": {"class": "display ledge", "attributes": [],
                  "sam_queries": ["display ledge"]},
        },
        "filter": [{"op": "above", "args": ["E1", "L"]}],
        "selectors": {},
        "answer": {"op": "count", "of": "E1"},
    }
    domain = compile_domain(program)
    assert domain["wall"] is True
    assert domain["floor"] is True
    assert domain["target_all_horizontal_surfaces"] is True
    assert domain["anchor_entity_ids"] == ["L"]
    assert domain["dependency_entity_ids"] == ["E1", "L"]
    assert domain["regions"] == [
        {"kind": "disc", "anchor": "L", "radius": 2.0}]
    kinds = {element["kind"] for element in domain["elements"]}
    assert {"ground_entity", "relation_volume", "surface_family"} <= kinds
    wall = next(element for element in domain["elements"]
                if element["kind"] == "surface_family"
                and element["surface"] == "wall")
    assert wall["owner"] == "E1"


def test_domain_closes_over_nested_selector_dependencies():
    program = {
        "task": "count",
        "entities": {
            "E1": {"class": "computer monitor", "attributes": [],
                   "sam_queries": ["computer monitor"]},
            "T": {"class": "table", "attributes": [],
                  "sam_queries": ["table"]},
            "M": {"class": "map wall decal", "attributes": [],
                  "sam_queries": ["map wall decal"]},
        },
        "filter": [{"op": "on", "args": ["E1", "T"]}],
        "selectors": {"T": {"op": "argmin_dist", "to": "M"}},
        "answer": {"op": "count", "of": "E1"},
    }
    domain = compile_domain(program)
    assert domain["dependency_entity_ids"] == ["E1", "M", "T"]
    assert set(domain["anchor_entity_ids"]) == {"T", "M"}
    assert any(element["kind"] == "support_instances"
               and element["owner"] == "E1"
               and element["support_entity"] == "T"
               for element in domain["elements"])
    assert any(element["kind"] == "entity_selector"
               and element["entity"] == "T"
               and element["selector"] == "argmin_dist"
               and element["anchor"] == "M"
               for element in domain["elements"])


def test_domain_elements_cover_carried_object_and_between_relations():
    program = {
        "task": "refer",
        "entities": {
            "C": {"class": "chair", "attributes": [],
                  "sam_queries": ["chair"]},
            "P": {"class": "pillow", "attributes": [],
                  "sam_queries": ["pillow"]},
            "V": {"class": "vase", "attributes": [],
                  "sam_queries": ["vase"]},
            "S": {"class": "stool", "attributes": [],
                  "sam_queries": ["stool"]},
        },
        "filter": [
            {"op": "with_on", "args": ["C", "P"]},
            {"op": "between", "args": ["C", "V", "S"]},
        ],
        "selectors": {},
        "answer": {"op": "unique", "of": "C"},
    }
    elements = compile_domain(program)["elements"]
    assert any(element["kind"] == "carried_object_relation"
               and element["owner"] == "C" for element in elements)
    assert any(element["kind"] == "relation_volume"
               and element["relation"] == "between"
               and element["anchors"] == ["V", "S"]
               for element in elements)


def test_follow_uses_recursive_dependencies_for_nested_leg_anchor():
    program = {
        "task": "follow",
        "entities": {
            "S": {"class": "stool", "attributes": [],
                  "sam_queries": ["stool"]},
            "P": {"class": "picture", "attributes": [],
                  "sam_queries": ["picture"]},
            "T": {"class": "table", "attributes": [],
                  "sam_queries": ["table"]},
            "C": {"class": "column", "attributes": [],
                  "sam_queries": ["column"]},
        },
        "filter": [{"op": "under", "args": ["S", "P"]}],
        "selectors": {"T": {"op": "argmax_dist", "to": "C"}},
        "answer": {"op": "path", "legs": [
            {"kind": "goto_near", "of": "S"},
            {"kind": "stop_at", "of": "T"},
        ]},
    }
    assert entity_dependency_closure(program, ["S", "T"]) == [
        "C", "P", "S", "T"]


def test_rectilinear_inventory_is_selected_by_domain_not_target_name():
    floor = {"floor": True, "wall": False, "elements": [
        {"kind": "surface_family", "surface": "floor"}]}
    wall = {"floor": False, "wall": True, "elements": [
        {"kind": "surface_family", "surface": "wall"}]}
    relation_only = {"floor": False, "wall": False, "elements": [
        {"kind": "relation_volume", "relation": "between"}]}
    assert [kind for kind, _v, _why in inspection_bands(floor)] == ["floor"]
    assert [kind for kind, _v, _why in inspection_bands(wall)] == ["wall"]
    assert [kind for kind, _v, _why in inspection_bands(relation_only)] == [
        "room"]
    specs = panorama_inspection_specs(floor, (2048, 640))
    assert len(specs) == 6
    assert {round(spec.bearing_deg) for spec in specs} == {
        -180.0, -120.0, -60.0, 0.0, 60.0, 120.0}
    assert all(spec.domain_kind == "floor" for spec in specs)

def test_bare_reference_uses_open_world_support_for_an_unseen_class():
    program = {
        "task": "refer",
        "entities": {"E1": {"class": "ceremonial samovar", "attributes": [],
                            "sam_queries": ["ceremonial samovar"]}},
        "filter": [],
        "answer": {"op": "unique", "of": "E1"},
    }
    domain = compile_domain(program)
    assert domain["floor"] is True
    assert domain["wall"] is True
    assert domain["target_all_horizontal_surfaces"] is True
    assert domain["all_horizontal_surfaces"] is True
    assert domain["stated_support_classes"] == []
    assert any("open-world" in reason for reason in domain["reason"])


def test_surface_first_domain_sweeps_at_furniture_range_not_cup_range():
    program = {
        "task": "count",
        "entities": {"E1": {"class": "paper cup", "attributes": [],
                            "sam_queries": ["paper cup"]},
                     "R1": {"class": "table", "attributes": [],
                            "sam_queries": ["table"]}},
        "filter": [{"op": "on", "args": ["E1", "R1"]}],
        "answer": {"op": "count", "of": "E1"},
    }
    domain = compile_domain(program)
    # Residual free space is swept looking for TABLES (visible far away), not
    # for cups; otherwise a cup-range sweep would demand standing everywhere.
    assert grounding_range_m(domain, target_range_m=0.8) > 2.0


def test_grounded_named_support_excludes_unrelated_open_planes():
    program = {
        "task": "count",
        "entities": {"E1": {"class": "pillow", "attributes": [],
                              "sam_queries": ["pillow"]},
                     "R1": {"class": "bed", "attributes": [],
                            "sam_queries": ["bed"]}},
        "filter": [{"op": "on", "args": ["E1", "R1"]}],
        "answer": {"op": "count", "of": "E1"},
    }
    domain = compile_domain(program)
    registry = SurfaceRegistry()
    bed_surface = Surface(
        "S1", "support", 0.6, np.array([0., 0., 1.]), -0.6,
        cells_of(_tabletop(height=0.6)[:, :2]), klass="bed", bound_node="B")
    unrelated = Surface(
        "S2", "support", 0.8, np.array([0., 0., 1.]), -0.8,
        cells_of(_tabletop(height=0.8, origin=(4., 0.))[:, :2]))
    registry.surfaces.extend([bed_surface, unrelated])
    graph = SceneGraph()
    bed = make_node("B", "R1", (0., 0.), z=0.6)
    graph.nodes.append(bed)

    visits = surface_obligations(
        domain, registry, graph, SafeCoverage(), np.zeros(2), 0.0, [],
        target_range_m=0.8)
    assert [visit["surface"] for visit in visits] == ["S1"]
    reasons = domain_reasons(domain, registry, graph, floor_z=0.0)
    assert any("S1" in reason for reason in reasons)
    assert all("S2" not in reason for reason in reasons)


def test_grounded_named_support_gets_two_local_orbit_views_first():
    program = {
        "task": "count",
        "entities": {"E1": {"class": "artifact"},
                     "R1": {"class": "platform"}},
        "filter": [{"op": "on", "args": ["E1", "R1"]}],
        "answer": {"op": "count", "of": "E1"},
    }
    domain = compile_domain(program)
    graph = SceneGraph()
    anchor = make_node("A", "R1", (2., 0.), z=0.7,
                       points=_tabletop(height=0.7, origin=(2., 0.)))
    graph.nodes.append(anchor)
    visits = anchor_inspection_obligations(
        domain, SurfaceRegistry(), graph, SafeCoverage(), np.zeros(2), [], {})
    assert visits and visits[0]["kind"] == "inspect_relation_anchor"
    completed = {"A": [[0., 0.], [1., 0.]]}
    assert anchor_inspection_obligations(
        domain, SurfaceRegistry(), graph, SafeCoverage(), np.zeros(2), [],
        completed) == []


def test_named_support_uses_anchor_profile_not_global_target_profile():
    program = {
        "task": "count",
        "entities": {"E1": {"class": "unseen small artifact"},
                     "R1": {"class": "unseen large machine"}},
        "filter": [{"op": "on", "args": ["E1", "R1"]}],
        "selectors": {},
        "answer": {"op": "count", "of": "E1"},
    }
    domain = compile_domain(program)
    graph = SceneGraph()
    anchor = make_node("A", "R1", (3., 0.), z=0.7)
    anchor.observations[0].pixel_width = 180.0
    graph.nodes.append(anchor)
    profiles = entity_residual_profiles(
        domain, program, graph, target_size_m=0.08, target_range_m=0.61)
    assert [(item["role"], item["entity_id"]) for item in profiles] == [
        ("anchor", "R1")]
    assert profiles[0]["min_size_m"] > 0.5
    assert profiles[0]["max_range_m"] > 3.5
    assert profiles[0]["required_frontier_audits"] == 2


def test_entity_scale_uses_strongest_observation_not_synthetic_ray():
    node = make_node("A", "R1", (3., 0.), z=0.7)
    node.observations[0].pixel_width = 240.0
    node.observations.append(Observation(
        [0, 0, 0, 0, 0, 0, 1], 0.9, 16.0, "synthetic_ray",
        [0, 0, 16, 16], "same_view", 1.0))
    with_weak_ray = observed_entity_size_m([node])
    node.observations.pop()
    assert with_weak_ray == pytest.approx(observed_entity_size_m([node]))


def test_bare_question_keeps_target_residual_profile():
    program = {
        "task": "count",
        "entities": {"E1": {"class": "unseen artifact"}},
        "filter": [], "selectors": {},
        "answer": {"op": "count", "of": "E1"},
    }
    domain = compile_domain(program)
    profiles = entity_residual_profiles(
        domain, program, SceneGraph(), 0.08, 0.61)
    assert [(item["role"], item["entity_id"]) for item in profiles] == [
        ("target", "E1")]


def test_closest_prune_radius_shrinks_with_a_better_candidate():
    program = {
        "task": "refer",
        "entities": {"E1": {"class": "pillow", "attributes": [],
                            "sam_queries": ["pillow"]},
                     "R1": {"class": "sushi", "attributes": [],
                            "sam_queries": ["sushi"]}},
        "filter": [],
        "answer": {"op": "argmin_dist", "of": "E1", "to": "R1"},
    }
    domain = compile_domain(program)
    assert domain["anchor_open_world_ids"] == ["R1"]
    assert domain["all_horizontal_surfaces"] is True
    graph = SceneGraph()
    graph.nodes.append(make_node("R1a", "R1", (0.0, 0.0)))
    graph.nodes.append(make_node("N1", "E1", (3.0, 0.0)))
    far = closest_prune_radius(domain, graph, program)
    graph.nodes.append(make_node("N2", "E1", (1.0, 0.0)))
    near = closest_prune_radius(domain, graph, program)
    assert far > near
    assert near == pytest.approx(1.75, abs=1e-6)


def test_anchor_only_support_retires_after_anchor_is_grounded():
    program = {
        "task": "refer",
        "entities": {"E1": {"class": "pillow", "attributes": [],
                              "sam_queries": ["pillow"]},
                     "R1": {"class": "sushi", "attributes": [],
                            "sam_queries": ["sushi"]},
                     "F": {"structure": "floor"}},
        "filter": [{"op": "on", "args": ["E1", "F"]}],
        "answer": {"op": "argmin_dist", "of": "E1", "to": "R1"},
    }
    domain = compile_domain(program)
    registry = SurfaceRegistry()
    table = Surface("S1", "support", 0.42, np.array([0., 0., 1.]), -0.42,
                    cells_of(_tabletop(height=0.42)[:, :2]), klass="table")
    registry.surfaces.append(table)
    graph = SceneGraph()
    assert any("S1" in reason for reason in
               domain_reasons(domain, registry, graph, 0.0))
    graph.nodes.append(make_node("A1", "R1", (2.0, 0.0), z=0.44))
    assert not any("S1" in reason for reason in
                   domain_reasons(domain, registry, graph, 0.0))


def test_region_filter_is_open_until_the_anchor_is_grounded():
    program = {
        "task": "count",
        "entities": {"E1": {"class": "chair", "attributes": [],
                            "sam_queries": ["chair"]},
                     "R1": {"class": "window", "attributes": [],
                            "sam_queries": ["window"]}},
        "filter": [{"op": "near", "args": ["E1", "R1"]}],
        "answer": {"op": "count", "of": "E1"},
    }
    domain = compile_domain(program)
    graph = SceneGraph()
    residuals = [{"target_xy": (0.5, 0.0)}, {"target_xy": (9.0, 9.0)}]
    # No anchor yet: an unknown anchor must not shrink the search space.
    assert region_filter(domain, graph, residuals) == residuals
    graph.nodes.append(make_node("R1a", "R1", (0.0, 0.0)))
    kept = region_filter(domain, graph, residuals)
    assert [c["target_xy"] for c in kept] == [(0.5, 0.0)]


# --------------------------------------------------------------- surfaces
def _tabletop(height=0.72, origin=(2.0, 0.0), half=0.4):
    xs = np.arange(origin[0] - half, origin[0] + half, 0.04)
    ys = np.arange(origin[1] - half, origin[1] + half, 0.04)
    grid = np.array([[x, y, height] for x in xs for y in ys], np.float32)
    return grid


def test_registry_binds_furniture_and_enumerates_only_at_close_range():
    registry = SurfaceRegistry()
    top = _tabletop()
    surface = Surface("S1", "support", 0.72, np.array([0., 0., 1.]), -0.72,
                      cells_of(top[:, :2]))
    registry.surfaces.append(surface)

    table = SceneNode("N9", "R1", facts={"is_class": True})
    table.points = top
    bound = registry.bind_class(table, "table")
    assert bound is surface and surface.klass == "table"

    far_pose = np.array([8.0, 0.0, 0.75, 0, 0, 0, 1.0])
    registry.mark_enumerated_from(far_pose, max_range_m=5.0)
    assert surface.enumerated_fraction() == 0.0
    assert surface.state == "open"

    near_pose = np.array([2.0, 1.2, 0.75, 0, 0, 0, 1.0])
    registry.mark_enumerated_from(near_pose, max_range_m=5.0)
    assert surface.enumerated_fraction() > 0.85
    assert surface.state == "enumerated"


def test_class_query_never_excludes_unbound_or_differently_named_surface():
    """A class word ranks exact bindings but cannot prune measured planes."""
    registry = SurfaceRegistry()
    registry.surfaces.append(Surface(
        "S1", "support", 0.72, np.array([0., 0., 1.]), -0.72,
        cells_of(_tabletop()[:, :2]), klass="altar"))
    registry.surfaces.append(Surface(
        "S2", "support", 1.85, np.array([0., 0., 1.]), -1.85,
        cells_of(_tabletop(height=1.85, origin=(4.0, 0.0))[:, :2])))
    matches = registry.surfaces_for_classes(["table"], floor_z=0.0)
    assert {s.id for s in matches} == {"S1", "S2"}


def test_surface_identity_tolerates_sparse_neighbouring_cells():
    first = {(10, 10), (11, 10), (12, 10), (13, 10)}
    second = {(11, 11), (12, 11), (13, 11), (14, 11)}
    assert not (first & second)
    assert nearby_cell_overlap(first, second) >= 0.75


def test_furniture_surface_binding_is_persistent_across_later_fragments():
    registry = SurfaceRegistry()
    first_cells = cells_of(_tabletop(height=0.6)[:, :2])
    second_cells = cells_of(
        _tabletop(height=0.9, origin=(4., 0.))[:, :2])
    first = Surface("S1", "support", 0.6, np.array([0., 0., 1.]), -0.6,
                    first_cells)
    second = Surface("S2", "support", 0.9, np.array([0., 0., 1.]), -0.9,
                     second_cells)
    registry.surfaces.extend([first, second])
    node = make_node("B", "R1", (0., 0.), z=0.6,
                     points=_tabletop(height=0.6))
    assert registry.bind_class(node, "bed") is first

    # A later sparse observation could overlap a different RANSAC fragment;
    # the physical furniture identity must retain its original top surface.
    node.points = _tabletop(height=0.9, origin=(4., 0.))
    node.footprint_points = node.points
    assert registry.bind_class(node, "bed") is first
    assert node.facts["top_surface"] == "S1"


def test_verified_anchor_extent_canonicalizes_disjoint_coplanar_fragments():
    """Two sparse strips inside one verified furniture mask are observations
    of one support; an equal-height strip outside that extent stays separate."""
    registry = SurfaceRegistry()
    first = Surface(
        "S13", "support", 0.393, np.array([0., 0., 1.]), -0.393,
        {(x, 0) for x in range(10)}, klass="bed", bound_node="B1",
        state="enumerated")
    second = Surface(
        "S23", "support", 0.410, np.array([0., 0., 1.]), -0.410,
        {(x, 6) for x in range(10)}, klass="bed", bound_node="B2")
    neighbour = Surface(
        "S24", "support", 0.405, np.array([0., 0., 1.]), -0.405,
        {(x, 20) for x in range(10)}, klass="bed", bound_node="B3")
    registry.surfaces.extend([first, second, neighbour])
    anchor = SceneNode("B1", "R1", facts={"is_class": True})
    anchor.footprint_set = {(x, y) for x in range(10) for y in range(7)}

    added = registry.observe_bound_extent(first, anchor)

    assert added > 0
    assert [surface.id for surface in registry.surfaces] == ["S13", "S24"]
    assert registry.canonical_id("S23") == "S13"
    assert len(first.cells) == 20
    assert len(first.identity_cells) == 70
    # Expanding an already-closed support reopens it until the new extent is
    # audited; later RANSAC strips inside that extent do not create new IDs.
    assert first.state == "open"
    assert anchor.facts["top_surface"] == "S13"


def test_surface_blocker_without_safe_view_is_explicitly_discharged():
    class UnsafeCoverage:
        def is_safe_xy(self, xy):
            return False

    program = {
        "task": "count",
        "entities": {"E1": {"class": "cup"},
                     "R1": {"class": "table"}},
        "filter": [{"op": "on", "args": ["E1", "R1"]}],
        "answer": {"op": "count", "of": "E1"},
    }
    domain = compile_domain(program)
    graph = SceneGraph()
    table = make_node("T1", "R1", (1.0, 0.0), z=0.7)
    graph.nodes.append(table)
    registry = SurfaceRegistry()
    surface = Surface(
        "S1", "support", 0.7, np.array([0., 0., 1.]), -0.7,
        {(20 + x, y) for x in range(5) for y in range(5)},
        klass="table", bound_node="T1")
    registry.surfaces.append(surface)

    assert domain_reasons(domain, registry, graph, 0.0)
    retired = discharge_unactionable_surfaces(
        domain, registry, graph, UnsafeCoverage(), np.zeros(2), [], 0.0,
        target_range_m=1.0, residuals_remaining=False)
    assert retired == ["S1"]
    assert surface.state == "unobservable"
    assert domain_reasons(domain, registry, graph, 0.0) == []


def test_generic_plane_assignment_cannot_override_explicit_on_relation():
    registry = SurfaceRegistry()
    explicit = Surface(
        "S1", "support", 0.6, np.array([0., 0., 1.]), -0.6,
        {(0, 0)}, klass="bed", bound_node="B1")
    incidental = Surface(
        "S2", "support", 0.8, np.array([0., 0., 1.]), -0.8,
        {(20, 20), (21, 20), (20, 21)}, klass=None)
    registry.surfaces.extend([explicit, incidental])
    node = make_node("P1", "E1", (1.02, 1.02), z=0.8)
    node.facts.update({
        "support_node": "B1", "support_surface": "S1",
        "support_relation_evidence":
            "explicit_on + same_view_topology + metric_vertical_order"})

    assign_supports([node], registry)

    assert node.facts["support_node"] == "B1"
    assert node.facts["support_surface"] == "S1"
    assert node.support == "surface:S1"


def test_visible_on_relation_merges_only_a_cross_view_anchor_track():
    program = {
        "task": "count",
        "entities": {"E1": {"class": "pillow"},
                     "R1": {"class": "bed"}},
        "filter": [{"op": "on", "args": ["E1", "R1"]}],
        "answer": {"op": "count", "of": "E1"},
    }
    graph = SceneGraph()
    old = make_node("P1", "E1", (0.1, 0.1), z=0.8)
    old.facts["_anchor_tracks"] = [{
        "anchor_entity": "R1", "anchor_node": "B",
        "pose_xy": [0., 0.], "relative": [0.25, 0.25, 0.1, 0.1]}]
    current = make_node("P2", "E1", (0.12, 0.1), z=0.8)
    other = make_node("P3", "E1", (0.6, 0.1), z=0.8)
    bed = make_node("B", "R1", (0., 0.), z=0.6)
    bed.facts["top_surface"] = "S1"
    graph.nodes.extend([old, current, other, bed])
    registry = SurfaceRegistry()
    registry.surfaces.append(Surface(
        "S1", "support", 0.6, np.array([0., 0., 1.]), -0.6,
        {(0, 0)}, klass="bed", bound_node="B"))
    mask = np.ones((200, 200), bool)
    best = {
        "P2": {"node": current, "box": [40, 40, 60, 60], "mask": mask,
               "concept": "pillow", "entity": "E1", "pixel_width": 20},
        "P3": {"node": other, "box": [120, 40, 140, 60], "mask": mask,
               "concept": "pillow", "entity": "E1", "pixel_width": 20},
        "B": {"node": bed, "box": [0, 0, 200, 200], "mask": mask,
              "concept": "bed", "entity": "R1", "pixel_width": 200},
    }
    events = assign_visible_support_relations(
        program, best, graph, np.array([1., 0., 0., 0., 0., 0., 1.]),
        registry)
    assert graph.nodes_for("E1") == [old, other]
    assert best["P1"]["node"] is old
    assert old.facts["support_node"] == "B"
    assert old.facts["support_surface"] == "S1"
    assert other.facts["support_node"] == "B"
    assert any(event["kind"] == "anchor_track_merge" for event in events)


def test_overlapping_verified_anchor_extents_reconcile_nonplanar_fragments():
    graph = SceneGraph()
    old = make_node("B1", "R1", (1.0, 1.0), z=0.4, is_class=True)
    fresh = make_node("B2", "R1", (1.1, 1.0), z=0.55, is_class=True)
    old.observations.append(Observation(
        [0, 0, 0, 0, 0, 0, 1], .9, 200, "test", [0, 0, 1, 1],
        "new", 0.0))
    fresh.observations.append(Observation(
        [1, 0, 0, 0, 0, 0, 1], .9, 200, "test", [0, 0, 1, 1],
        "new", 0.0))
    target = make_node("P1", "E1", (1.0, 1.0), z=0.8, is_class=True)
    target.facts.update({"support_node": "B2", "support_surface": "S2"})
    target.support = "surface:S2"
    graph.nodes.extend([old, fresh, target])
    registry = SurfaceRegistry()
    first = Surface("S1", "support", .32, np.array([0., 0., 1.]),
                    -.32, {(20, 20), (21, 20)}, klass="bed",
                    bound_node="B1", identity_cells={
                        (x, y) for x in range(20, 31) for y in range(20, 31)})
    second = Surface("S2", "support", .47, np.array([0., 0., 1.]),
                     -.47, {(24, 24), (25, 24)}, klass="bed",
                     bound_node="B2", identity_cells={
                         (x, y) for x in range(23, 34) for y in range(23, 34)})
    registry.surfaces.extend([first, second])

    events = registry.reconcile_bound_surfaces(graph)

    assert len(events) == 1
    assert len(registry.surfaces) == 1
    canonical = registry.surfaces[0]
    assert registry.canonical_id("S2") == canonical.id
    assert len(graph.nodes_for("R1")) == 1
    assert target.facts["support_node"] == graph.nodes_for("R1")[0].id
    assert target.facts["support_surface"] == canonical.id
    assert target.support == f"surface:{canonical.id}"


def test_anchor_relative_matching_survives_vertical_fragment_shift():
    """A partial anchor box may shift every target's normalized y together."""
    program = {
        "task": "count",
        "entities": {"E1": {"class": "pillow"},
                     "R1": {"class": "bed"}},
        "filter": [{"op": "on", "args": ["E1", "R1"]}],
        "answer": {"op": "count", "of": "E1"},
    }
    white = [0.75, 0.80, 0.84, 0.50, 0.52]
    dark = [0.28, 0.34, 0.39, 0.50, 0.52]
    old_specs = [(.229, .163, .227, white),
                 (.253, .181, .146, dark),
                 (.445, .159, .139, dark),
                 (.415, .138, .207, white)]
    current_specs = [(.187, .172, .217, white),
                     (.242, .198, .126, dark),
                     (.419, .156, .113, dark),
                     (.371, .131, .179, white)]
    graph = SceneGraph()
    for index, (x, width, height, appearance) in enumerate(old_specs, 1):
        node = make_node(f"P{index}", "E1", (index * 2., 0.), z=.8)
        node.facts.update({"support_node": "B", "_anchor_tracks": [{
            "anchor_entity": "R1", "anchor_node": "B",
            "pose_xy": [0., 0.],
            "relative": [x, -.08, width, height]}]})
        node.observations.append(Observation(
            [0, 0, 0, 0, 0, 0, 1], .9, 100, "test", [0, 0, 1, 1],
            "new", 0.0, appearance=appearance))
        graph.nodes.append(node)
    current = []
    for index, (x, width, height, appearance) in enumerate(current_specs, 5):
        node = make_node(f"P{index}", "E1", (index * 2., 5.), z=.8)
        node.observations.append(Observation(
            [1, 0, 0, 0, 0, 0, 1], .9, 100, "test", [0, 0, 1, 1],
            "new", 0.0, appearance=appearance))
        graph.nodes.append(node)
        current.append((node, x, width, height))
    bed = make_node("B", "R1", (0., 0.), z=.4, is_class=True)
    bed.facts["top_surface"] = "S1"
    graph.nodes.append(bed)
    registry = SurfaceRegistry()
    registry.surfaces.append(Surface(
        "S1", "support", .4, np.array([0., 0., 1.]), -.4,
        {(0, 0)}, klass="bed", bound_node="B"))
    mask = np.ones((1000, 1000), bool)
    best = {}
    for node, x, width, height in current:
        box = [(x - width / 2) * 1000, (0.265 - height / 2) * 1000,
               (x + width / 2) * 1000, (0.265 + height / 2) * 1000]
        best[node.id] = {"node": node, "box": box, "mask": mask,
                         "concept": "pillow", "entity": "E1",
                         "pixel_width": width * 1000}
    best["B"] = {"node": bed, "box": [0, 0, 1000, 1000], "mask": mask,
                 "concept": "bed", "entity": "R1", "pixel_width": 1000}

    events = assign_visible_support_relations(
        program, best, graph, np.array([1., 0., 0., 0., 0., 0., 1.]),
        registry)

    assert len(graph.nodes_for("E1")) == 4
    assert sum(event["kind"] == "anchor_track_merge"
               for event in events) == 4


def test_verified_quarantined_anchor_is_promoted_without_another_capture():
    graph = SceneGraph()
    anchor = make_node("B", "R1", (0., 0.), z=0.6, is_class=False)
    graph.nodes.append(anchor)
    assert graph.reject_nonmembers("R1") == [anchor]
    assert graph.nodes_for("R1") == []
    anchor.facts["is_class"] = True
    assert graph.promote_verified("R1") == [anchor]
    assert graph.nodes_for("R1") == [anchor]


def test_open_surface_blocks_the_certificate_and_yields_a_visit():
    registry = SurfaceRegistry()
    surface = Surface("S1", "support", 0.72, np.array([0., 0., 1.]), -0.72,
                      cells_of(_tabletop()[:, :2]))
    surface.klass = "table"
    registry.surfaces.append(surface)
    domain = {"stated_support_classes": [], "anchor_entity_ids": [],
              "target_all_horizontal_surfaces": True,
              "anchor_open_world_ids": [],
              "floor": False, "regions": [], "prune_closest_to": None,
              "target": "E1", "target_class": "cup", "reason": []}
    graph = SceneGraph()
    reasons = domain_reasons(domain, registry, graph, floor_z=0.0)
    assert any("not yet enumerated" in reason for reason in reasons)

    visits = surface_obligations(domain, registry, graph, SafeCoverage(),
                                 np.zeros(2), 0.0, [])
    assert visits and visits[0]["kind"] == "enumerate_surface"
    assert visits[0]["surface"] == "S1"

    surface.state = "enumerated"
    assert domain_reasons(domain, registry, graph, floor_z=0.0) == []
    assert surface_obligations(domain, registry, graph, SafeCoverage(),
                               np.zeros(2), 0.0, []) == []


def test_unresolved_tiny_anchor_reopens_enumerated_support_for_close_search():
    registry = SurfaceRegistry()
    surface = Surface("S1", "support", 0.42, np.array([0., 0., 1.]), -0.42,
                      cells_of(_tabletop(height=0.42)[:, :2]),
                      state="enumerated")
    registry.surfaces.append(surface)
    domain = {
        "stated_support_classes": [],
        "anchor_entity_ids": ["R1"],
        "anchor_open_world_ids": ["R1"],
        "target_all_horizontal_surfaces": False,
        "floor": True, "regions": [], "prune_closest_to": "R1",
        "target": "E1", "target_class": "pillow", "reason": [],
    }
    visits = surface_obligations(domain, registry, SceneGraph(), SafeCoverage(),
                                 np.zeros(2), 0.0, [])
    assert visits and visits[0]["anchor_ids"] == ["R1"]
    assert "ground R1" in visits[0]["reason"]


def test_rectilinear_view_centre_maps_back_to_requested_panorama_pixel():
    u, v = rectilinear_pixel_to_panorama(
        447.5, 319.5, (896, 640), center_u=1377.0, center_v=311.0,
        panorama_size=(1920, 640), hfov_deg=90.0)
    assert u == pytest.approx(1377.0, abs=1e-4)
    assert v == pytest.approx(311.0, abs=1e-4)


def test_support_of_matches_only_objects_resting_on_the_plane():
    registry = SurfaceRegistry()
    registry.surfaces.append(Surface(
        "S1", "support", 0.72, np.array([0., 0., 1.]), -0.72,
        cells_of(_tabletop()[:, :2])))
    on_table = SceneNode("N1", "E1")
    on_table.points = np.array([[2.0, 0.0, 0.74], [2.05, 0.0, 0.80],
                                [2.0, 0.05, 0.78], [2.02, 0.02, 0.76],
                                [2.03, 0.03, 0.79], [2.01, 0.01, 0.77],
                                [2.04, 0.04, 0.75], [2.02, 0.03, 0.81]],
                               np.float32)
    assert registry.support_of(on_table).id == "S1"

    on_floor = SceneNode("N2", "E1")
    on_floor.points = on_table.points.copy()
    on_floor.points[:, 2] -= 0.72
    assert registry.support_of(on_floor) is None


# ------------------------------------------------------------- predicates
def test_evaluate_resolves_closest_and_farthest_over_the_domain():
    program = {
        "task": "refer",
        "entities": {"E1": {"class": "pillow", "attributes": [],
                            "sam_queries": ["pillow"]},
                     "R1": {"class": "sushi", "attributes": [],
                            "sam_queries": ["sushi"]}},
        "filter": [],
        "answer": {"op": "argmin_dist", "of": "E1", "to": "R1"},
    }
    graph = SceneGraph()
    graph.nodes.append(make_node("R1a", "R1", (0.0, 0.0)))
    graph.nodes.append(make_node("N1", "E1", (3.0, 0.0)))
    graph.nodes.append(make_node("N2", "E1", (1.0, 0.0)))
    selected = graph.evaluate(program)
    assert [node.id for node in selected] == ["N2"]

    program["answer"]["op"] = "argmax_dist"
    assert [node.id for node in graph.evaluate(program)] == ["N1"]


def test_refer_always_resolves_to_exactly_one_box_candidate():
    """Object reference is scored on ONE marker: ties and ungrounded anchors
    must still produce a single deterministic candidate to publish."""
    program = {
        "task": "refer",
        "entities": {"E1": {"class": "vase", "attributes": [],
                            "sam_queries": ["vase"]},
                     "R1": {"class": "guitar", "attributes": [],
                            "sam_queries": ["guitar"]}},
        "filter": [],
        "answer": {"op": "argmin_dist", "of": "E1", "to": "R1"},
    }
    graph = SceneGraph()
    thin = make_node("N1", "E1", (1.0, 0.0))
    thin.best_px = 40.0
    solid = make_node("N2", "E1", (2.0, 0.0))
    solid.best_px = 180.0
    solid.observations.append(Observation(
        [4, 4, 0, 0, 0, 0, 1], 0.9, 180, "test", [0, 0, 1, 1], "test", 1.0))
    graph.nodes.extend([thin, solid])

    # Anchor not grounded yet -> still one candidate, the best-evidenced one.
    selected = graph.evaluate(program)
    assert [node.id for node in selected] == ["N2"]

    # Anchor grounded -> the geometric argmin wins over evidence quality.
    graph.nodes.append(make_node("R1a", "R1", (0.0, 0.0)))
    assert [node.id for node in graph.evaluate(program)] == ["N1"]

    program["answer"] = {"op": "unique", "of": "E1"}
    assert [node.id for node in graph.evaluate(program)] == ["N2"]


def test_evaluate_between_keeps_only_nodes_inside_the_corridor():
    program = {
        "task": "count",
        "entities": {"E1": {"class": "lantern", "attributes": [],
                            "sam_queries": ["lantern"]},
                     "A": {"class": "vase", "attributes": [],
                           "sam_queries": ["vase"]},
                     "B": {"class": "stone", "attributes": [],
                           "sam_queries": ["stone"]}},
        "filter": [{"op": "between", "args": ["E1", "A", "B"]}],
        "answer": {"op": "count", "of": "E1"},
    }
    graph = SceneGraph()
    graph.nodes.append(make_node("A1", "A", (0.0, 0.0)))
    graph.nodes.append(make_node("B1", "B", (4.0, 0.0)))
    graph.nodes.append(make_node("N1", "E1", (2.0, 0.2)))     # inside
    graph.nodes.append(make_node("N2", "E1", (2.0, 3.0)))     # too lateral
    graph.nodes.append(make_node("N3", "E1", (6.0, 0.0)))     # beyond B
    assert graph.evaluate(program) == 1


def test_evaluate_on_named_surface_requires_grounded_support_identity():
    program = {
        "task": "count",
        "entities": {"E1": {"class": "cup", "attributes": [],
                            "sam_queries": ["cup"]},
                     "R1": {"class": "table", "attributes": [],
                            "sam_queries": ["table"]}},
        "filter": [{"op": "on", "args": ["E1", "R1"]}],
        "answer": {"op": "count", "of": "E1"},
    }
    graph = SceneGraph()
    on_desk = make_node("N1", "E1", (2.0, 0.0))
    on_desk.facts["support_class"] = "desk"
    on_shelf = make_node("N2", "E1", (3.0, 0.0))
    on_shelf.facts["support_class"] = "shelf"
    graph.nodes.extend([on_desk, on_shelf])
    # Words alone are not contact evidence, even if somebody considers one a
    # synonym of the requested support.
    assert graph.evaluate(program) == 0
    graph.nodes.append(make_node("T1", "R1", (2.0, 0.0)))
    on_desk.facts["support_node"] = "T1"
    assert graph.evaluate(program) == 1


def test_nested_selector_counts_objects_on_only_the_closest_support():
    """"Monitors on the table closest to the map decal" selects the table
    inside the query graph; closest does not apply to the monitor answer."""
    program = {
        "task": "count",
        "entities": {
            "E1": {"class": "computer monitor", "attributes": [],
                   "sam_queries": ["computer monitor"]},
            "T": {"class": "table", "attributes": [],
                  "sam_queries": ["table"]},
            "M": {"class": "map wall decal", "attributes": [],
                  "sam_queries": ["map wall decal"]},
        },
        "filter": [{"op": "on", "args": ["E1", "T"]}],
        "selectors": {"T": {"op": "argmin_dist", "to": "M"}},
        "answer": {"op": "count", "of": "E1"},
    }
    graph = SceneGraph()
    graph.nodes += [
        make_node("MAP", "M", (0.0, 0.0), z=1.5),
        make_node("TN", "T", (1.0, 0.0), z=0.7),
        make_node("TF", "T", (4.0, 0.0), z=0.7),
        make_node("MN", "E1", (1.0, 0.0), z=0.9),
        make_node("MF", "E1", (4.0, 0.0), z=0.9),
    ]
    next(node for node in graph.nodes if node.id == "MN").facts[
        "support_node"] = "TN"
    next(node for node in graph.nodes if node.id == "MF").facts[
        "support_node"] = "TF"
    assert [node.id for node in graph.resolve_entity(program, "T")] == ["TN"]
    assert graph.evaluate(program) == 1


def test_nested_relation_filters_the_support_before_counting_targets():
    program = {
        "task": "count",
        "entities": {
            "E1": {"class": "pillow", "attributes": [],
                   "sam_queries": ["pillow"]},
            "S": {"class": "sofa", "attributes": [],
                  "sam_queries": ["sofa"]},
            "P": {"class": "picture", "attributes": [],
                  "sam_queries": ["picture"]},
        },
        "filter": [
            {"op": "on", "args": ["E1", "S"]},
            {"op": "below", "args": ["S", "P"]},
        ],
        "selectors": {},
        "answer": {"op": "count", "of": "E1"},
    }
    graph = SceneGraph()
    graph.nodes += [
        make_node("PIC", "P", (1.0, 0.0), z=2.0),
        make_node("S1", "S", (1.0, 0.0), z=0.5),
        make_node("S2", "S", (4.0, 0.0), z=0.5),
        make_node("P1", "E1", (0.8, 0.0), z=0.7),
        make_node("P2", "E1", (1.2, 0.0), z=0.7),
        make_node("P3", "E1", (4.0, 0.0), z=0.7),
    ]
    for node_id in ("P1", "P2"):
        next(node for node in graph.nodes if node.id == node_id).facts[
            "support_node"] = "S1"
    next(node for node in graph.nodes if node.id == "P3").facts[
        "support_node"] = "S2"
    assert [node.id for node in graph.resolve_entity(program, "S")] == ["S1"]
    assert graph.evaluate(program) == 2


def test_with_on_relation_counts_supports_that_have_a_carried_object():
    program = {
        "task": "count",
        "entities": {
            "C": {"class": "chair", "attributes": [],
                  "sam_queries": ["chair"]},
            "P": {"class": "pillow", "attributes": [],
                  "sam_queries": ["pillow"]},
        },
        "filter": [{"op": "with_on", "args": ["C", "P"]}],
        "selectors": {},
        "answer": {"op": "count", "of": "C"},
    }
    graph = SceneGraph()
    graph.nodes += [
        make_node("C1", "C", (1.0, 0.0), z=0.5),
        make_node("C2", "C", (3.0, 0.0), z=0.5),
        make_node("P1", "P", (1.0, 0.0), z=0.8),
    ]
    next(node for node in graph.nodes if node.id == "P1").facts[
        "support_node"] = "C1"
    assert graph.evaluate(program) == 1


def test_evaluate_above_requires_vertical_order_and_overlap():
    program = {
        "task": "count",
        "entities": {"E1": {"class": "picture", "attributes": [],
                            "sam_queries": ["picture"]},
                     "R1": {"class": "bed", "attributes": [],
                            "sam_queries": ["bed"]}},
        "filter": [{"op": "above", "args": ["E1", "R1"]}],
        "answer": {"op": "count", "of": "E1"},
    }
    graph = SceneGraph()
    graph.nodes.append(make_node("R1a", "R1", (0.0, 0.0), z=0.5))
    graph.nodes.append(make_node("N1", "E1", (0.01, 0.0), z=1.6))  # above
    graph.nodes.append(make_node("N2", "E1", (0.01, 0.0), z=0.1))  # below
    graph.nodes.append(make_node("N3", "E1", (5.0, 0.0), z=1.6))   # elsewhere
    assert graph.evaluate(program) == 1


# ------------------------------------------------------------------ legs
def test_parse_legs_reads_ordered_kinds_from_the_program():
    program = {"answer": {"op": "path", "legs": [
        {"kind": "goto_near", "of": "R1"},
        {"kind": "pass_between", "of": ["R2", "R3"]},
        {"kind": "avoid_between", "of": ["R4", "R5"]},
        {"kind": "stop_at", "of": "R6"}]}}
    legs = parse_legs(program)
    assert [leg.kind for leg in legs] == [
        "goto_near", "pass_between", "avoid_between", "stop_at"]
    assert legs[1].entity_ids == ["R2", "R3"]


def test_corridor_waypoints_cross_the_gap_from_the_robot_side():
    entry_exit = corridor_waypoints(np.array([0.0, 1.0]),
                                    np.array([0.0, -1.0]),
                                    robot_xy=np.array([-3.0, 0.0]),
                                    coverage=SafeCoverage())
    assert len(entry_exit) == 2
    entry, exit_ = entry_exit
    # Entry is on the robot's side (negative x), exit beyond the gap.
    assert entry[0] < 0.0 < exit_[0]
    assert entry[1] == pytest.approx(0.0)


def test_avoid_zone_blocks_a_crossing_path_and_detours_around_it():
    zone = avoid_zone(np.array([0.0, 1.0]), np.array([0.0, -1.0]),
                      half_width=0.8)
    assert segment_hits_zone([-2.0, 0.0], [2.0, 0.0], zone) is True
    guide = detour_waypoint([-2.0, 0.0], [2.0, 0.0], zone, SafeCoverage())
    assert guide is not None
    assert not segment_hits_zone([-2.0, 0.0], guide, zone)
    assert not segment_hits_zone(guide, [2.0, 0.0], zone)
    assert abs(guide[1]) > 0.8      # routed around an end, not through


def test_leg_plan_inserts_a_detour_when_the_direct_route_violates_avoid():
    from instruction_legs import Leg

    zone = avoid_zone(np.array([0.0, 1.0]), np.array([0.0, -1.0]),
                      half_width=0.8)
    leg = Leg("stop_at", ["R1"], anchors_xy=[np.array([3.0, 0.0])])
    waypoints = plan_leg_waypoints(leg, np.array([-3.0, 0.0]),
                                   SafeCoverage(), [zone])
    assert len(waypoints) >= 2      # guide first, then the goal
    position = np.array([-3.0, 0.0])
    for waypoint in waypoints:
        assert not segment_hits_zone(position, waypoint, zone)
        position = np.asarray(waypoint, float)


def test_annulus_waypoint_stands_off_the_anchor_on_the_robot_side():
    goal = annulus_waypoint(np.array([2.0, 0.0]), np.array([0.0, 0.0]),
                            SafeCoverage(), standoff=0.9)
    assert goal is not None
    distance = math.dist(goal, [2.0, 0.0])
    assert distance == pytest.approx(0.9, abs=1e-6)
    assert goal[0] < 2.0            # between the robot and the anchor


def test_support_filtered_question_runs_domain_cycle_to_a_clean_stop():
    """End-to-end decision cycle for "cups on a table" with no perception.

    This is the case the floor-only certificate could not certify: the domain
    must block on the uninspected tabletop, produce a visit that discharges it,
    and only then allow the stop certificate to fire.
    """
    from unified_scene_graph import confidence_stop_certificate

    program = {
        "task": "count",
        "entities": {"E1": {"class": "cup", "attributes": [],
                            "sam_queries": ["cup"]},
                     "R1": {"class": "table", "attributes": [],
                            "sam_queries": ["table"]}},
        "filter": [{"op": "on", "args": ["E1", "R1"]}],
        "answer": {"op": "count", "of": "E1"},
    }
    domain = compile_domain(program)
    registry = SurfaceRegistry()
    top = _tabletop(height=0.72, origin=(2.0, 0.0))
    surface = Surface("S1", "support", 0.72, np.array([0., 0., 1.]), -0.72,
                      cells_of(top[:, :2]))
    registry.surfaces.append(surface)

    graph = SceneGraph()
    table = make_node("T1", "R1", (2.0, 0.0), z=0.72, points=top)
    table.points = top
    graph.nodes.append(table)
    registry.bind_class(table, "table")

    cup = make_node("N1", "E1", (2.0, 0.0), z=0.74)
    cup.facts.update({"confidence": 0.95, "support_class": "table"})
    cup.best_px = 120.0
    cup.observations.append(Observation(
        [3, 1, 0, 0, 0, 0, 1], 0.8, 120, "lidar", [0, 0, 1, 1], "test", 1.0))
    graph.nodes.append(cup)
    assign_supports([cup], registry)
    assert graph.evaluate(program) == 1

    # Before inspection the tabletop blocks the certificate and yields a visit.
    blocked = confidence_stop_certificate(
        graph, program, [1, 1], [("N1",), ("N1",)], None, 0,
        frontier_attempts=1, frontier_visual_audits=1, residual_components=[],
        extra_reasons=domain_reasons(domain, registry, graph, 0.0))
    assert blocked["satisfied"] is False
    assert any("S1" in reason for reason in blocked["reasons_not_satisfied"])
    visits = surface_obligations(domain, registry, graph, SafeCoverage(),
                                 np.zeros(2), 0.0, [])
    assert visits[0]["surface"] == "S1"

    # Standing at the visit pose enumerates the top; the domain discharges.
    registry.mark_enumerated_from(
        np.array([visits[0]["xy"][0], visits[0]["xy"][1], 0.75, 0, 0, 0, 1.0]),
        max_range_m=5.0)
    assert surface.state == "enumerated"
    satisfied = confidence_stop_certificate(
        graph, program, [1, 1], [("N1",), ("N1",)], None, 0,
        frontier_attempts=1, frontier_visual_audits=1, residual_components=[],
        extra_reasons=domain_reasons(domain, registry, graph, 0.0))
    assert satisfied["satisfied"] is True, satisfied["reasons_not_satisfied"]


def test_every_released_question_routes_to_the_scored_output_type():
    """All 75 route correctly before Qwen performs the semantic compile.

    This intentionally does *not* call the regex fallback and claim the
    resulting program is semantically faithful.  That old test accepted e.g.
    ``sofas below a window`` as one detector class and empty follow legs.
    Runtime compilation is now audited by Qwen and fails visibly on drift.
    """
    import json
    from pathlib import Path
    from question_types import classify_question

    path = (Path(__file__).resolve().parents[2] / "questions" /
            "questions.json")
    scenes = json.loads(path.read_text())
    expected_type = {"numerical": "numerical",
                     "object_reference": "object_reference",
                     "instruction_following": "instruction_following"}
    seen = 0
    for scene in scenes:
        for kind, questions in scene["questions"].items():
            for question in questions:
                classified = classify_question(question)
                assert classified.question_type.value == expected_type[kind], question
                seen += 1
    assert seen == 75


def test_pass_near_generates_entry_and_exit_not_a_fake_destination():
    waypoints = pass_near_waypoints(
        np.array([0.0, 0.0]), np.array([-3.0, 0.0]), SafeCoverage())
    assert waypoints is not None and len(waypoints) == 2
    first, second = map(np.asarray, waypoints)
    assert first[0] < 0.0 < second[0]
    assert 0.75 <= np.linalg.norm(first) <= 1.9
    assert 0.75 <= np.linalg.norm(second) <= 1.9


def test_follow_validator_accepts_released_pass_and_avoid_near_operators():
    question = "Go to the cup, pass by the stairs, and avoid the path near the cabinet."
    program = {
        "task": "follow",
        "entities": {
            "C": {"class": "cup", "attributes": [], "sam_queries": ["cup"]},
            "S": {"class": "stairs", "attributes": [], "sam_queries": ["stairs"]},
            "B": {"class": "cabinet", "attributes": [],
                  "sam_queries": ["cabinet"]},
        },
        "filter": [], "selectors": {},
        "answer": {"op": "path", "legs": [
            {"kind": "goto_near", "of": "C"},
            {"kind": "pass_near", "of": "S"},
            {"kind": "avoid_near", "of": "B"},
        ]},
    }
    assert validate_program(program, question)["answer"]["legs"][1][
        "kind"] == "pass_near"


def test_distance_selector_can_use_floor_plane_without_fake_floor_node():
    program = {
        "task": "refer",
        "entities": {
            "P": {"class": "picture", "attributes": [],
                  "sam_queries": ["picture"]},
            "F": {"structure": "floor"},
        },
        "filter": [], "selectors": {},
        "answer": {"op": "argmax_dist", "of": "P", "to": "F"},
    }
    graph = SceneGraph()
    graph.nodes.append(make_node("LOW", "P", (0.0, 0.0), z=0.8))
    graph.nodes.append(make_node("HIGH", "P", (1.0, 0.0), z=2.0))
    assert [node.id for node in graph.evaluate(program)] == ["HIGH"]


def test_support_assignment_links_target_to_specific_furniture_node():
    registry = SurfaceRegistry()
    surface = Surface("S1", "support", 0.75, np.array([0., 0., 1.]), -0.75,
                      cells_of(_tabletop(height=0.75)[:, :2]), klass="table")
    surface.bound_node = "TABLE_1"
    registry.surfaces.append(surface)
    cup_points = np.array([[2.10, 0.10, 0.77], [2.12, 0.10, 0.78],
                           [2.10, 0.12, 0.80], [2.12, 0.12, 0.82],
                           [2.11, 0.11, 0.79], [2.13, 0.10, 0.81],
                           [2.10, 0.13, 0.83], [2.13, 0.13, 0.84]])
    cup = make_node("CUP", "C", (2.1, 0.1), z=0.8, points=cup_points)
    from search_domain import assign_supports
    assign_supports([cup], registry)
    assert cup.facts["support_node"] == "TABLE_1"
    assert cup.facts["support_surface"] == "S1"


def test_above_requires_xy_footprint_overlap_not_just_nearby_centres():
    graph = SceneGraph()
    anchor_points = np.array([[0.0, 0.0, 0.7], [1.0, 0.0, 0.7],
                              [0.0, 1.0, 0.8], [1.0, 1.0, 0.8]])
    above_points = np.array([[0.2, 0.2, 1.2], [0.8, 0.2, 1.2],
                             [0.2, 0.8, 1.4], [0.8, 0.8, 1.4]])
    diagonal_points = above_points + np.array([1.4, 0.0, 0.0])
    graph.nodes.extend([
        make_node("A", "A", (0.5, 0.5), points=anchor_points),
        make_node("GOOD", "T", (0.5, 0.5), points=above_points),
        make_node("DIAGONAL", "T", (1.9, 0.5), points=diagonal_points),
    ])
    program = {
        "task": "count",
        "entities": {
            "T": {"class": "picture", "attributes": [],
                  "sam_queries": ["picture"]},
            "A": {"class": "bed", "attributes": [], "sam_queries": ["bed"]},
        },
        "filter": [{"op": "above", "args": ["T", "A"]}],
        "selectors": {}, "answer": {"op": "count", "of": "T"},
    }
    assert [node.id for node in graph.matching_nodes(program)] == ["GOOD"]


def test_near_uses_box_distance_and_room_relative_scale():
    graph = SceneGraph()
    # A 5 x 5 x 3 m observed room gives a 0.75 m near threshold.
    room = np.array([[x, y, z] for x in (0.0, 5.0)
                     for y in (0.0, 5.0) for z in (0.0, 3.0)] * 20)
    assert graph.update_region_scale(room) == pytest.approx(0.75, abs=0.08)
    anchor = make_node("A", "A", (0.0, 0.0))
    close = make_node("C", "T", (0.5, 0.0))
    far = make_node("F", "T", (1.5, 0.0))
    graph.nodes.extend([anchor, close, far])
    program = {
        "task": "count",
        "entities": {
            "T": {"class": "chair", "attributes": [],
                  "sam_queries": ["chair"]},
            "A": {"class": "table", "attributes": [],
                  "sam_queries": ["table"]},
        },
        "filter": [{"op": "near", "args": ["T", "A"]}],
        "selectors": {}, "answer": {"op": "count", "of": "T"},
    }
    assert [node.id for node in graph.matching_nodes(program)] == ["C"]


def test_with_on_uses_support_contact_not_nearby_centroid():
    graph = SceneGraph()
    chair_points = np.array([[0.0, 0.0, 0.0], [0.8, 0.0, 0.0],
                             [0.0, 0.8, 0.7], [0.8, 0.8, 0.7]])
    on_points = np.array([[0.2, 0.2, 0.70], [0.6, 0.2, 0.70],
                          [0.2, 0.6, 0.82], [0.6, 0.6, 0.82]])
    beside_points = on_points + np.array([0.9, 0.0, 0.0])
    graph.nodes.extend([
        make_node("CHAIR", "C", (0.4, 0.4), points=chair_points),
        make_node("ON", "P", (0.4, 0.4), points=on_points),
        make_node("BESIDE", "P", (1.3, 0.4), points=beside_points),
    ])
    program = {
        "task": "count",
        "entities": {
            "C": {"class": "chair", "attributes": [],
                  "sam_queries": ["chair"]},
            "P": {"class": "pillow", "attributes": [],
                  "sam_queries": ["pillow"]},
        },
        "filter": [{"op": "with_on", "args": ["C", "P"]}],
        "selectors": {}, "answer": {"op": "count", "of": "C"},
    }
    assert [node.id for node in graph.matching_nodes(program)] == ["CHAIR"]


def test_refer_task_cannot_compile_to_a_numeric_answer():
    from question_types import classify_question

    question = "Find the pillow closest to the book on the stool."
    assert classify_question(question).question_type.value == "object_reference"
    bad = fallback_program(question)
    bad["answer"] = {"op": "count", "of": "E1"}
    with pytest.raises(ValueError, match="refer task requires"):
        validate_program(bad, question)


def test_nested_closest_reference_is_not_forced_onto_final_answer():
    question = "Find the bowl on the table closest to the folding screen."
    program = {
        "task": "refer",
        "entities": {
            "B": {"class": "bowl", "attributes": [], "sam_queries": ["bowl"]},
            "T": {"class": "table", "attributes": [], "sam_queries": ["table"]},
            "S": {"class": "folding screen", "attributes": [],
                  "sam_queries": ["folding screen"]},
        },
        "filter": [{"op": "on", "args": ["B", "T"]}],
        "selectors": {"T": {"op": "argmin_dist", "to": "S"}},
        "answer": {"op": "unique", "of": "B"},
    }
    assert validate_program(program, question)["answer"]["op"] == "unique"


def test_literal_reference_schema_slip_is_repaired_without_changing_semantics():
    from unified_program import repair_literal_entity_references

    question = "The red pillow closest to the sushi."
    raw = {
        "task": "refer",
        "entities": {"E1": {"class": "pillow", "attributes": ["red"],
                              "sam_queries": ["pillow", "red pillow"]}},
        "filter": [{"op": "near", "args": ["sushi"]}],
        "answer": {"op": "argmin_dist", "of": "E1", "to": "sushi"},
    }
    repaired = validate_program(
        repair_literal_entity_references(raw), question)
    assert repaired["answer"] == {
        "op": "argmin_dist", "of": "E1", "to": "R1"}
    assert repaired["entities"]["R1"]["class"] == "sushi"
    assert repaired["filter"] == []


def test_physical_support_in_structure_field_is_repaired_as_open_class():
    from unified_program import repair_literal_entity_references

    question = "How many pillows are on the bed?"
    raw = {
        "task": "count",
        "entities": {
            "E1": {"class": "pillow", "attributes": []},
            "F": {"structure": "bed"},
        },
        "filter": [{"op": "on", "args": ["E1", "F"]}],
        "answer": {"op": "count", "of": "E1", "to": "F"},
    }
    repaired = validate_program(
        repair_literal_entity_references(raw), question)
    assert repaired["entities"]["F"] == {
        "class": "bed", "attributes": [], "sam_queries": ["bed"]}
    assert repaired["answer"] == {"op": "count", "of": "E1"}


def test_refer_fallback_preserves_target_attribute_and_closest_anchor():
    question = "The red pillow closest to the sushi."
    program = validate_program(fallback_program(question), question)
    assert program["entities"]["E1"]["class"] == "pillow"
    assert program["entities"]["E1"]["attributes"] == ["red"]
    assert program["entities"]["R1"]["class"] == "sushi"
    assert program["answer"] == {
        "op": "argmin_dist", "of": "E1", "to": "R1"}


def test_unseen_explicit_support_keeps_exact_word_but_searches_all_geometry():
    program = {
        "task": "count",
        "entities": {
            "E1": {"class": "reliquary", "attributes": [],
                   "sam_queries": ["reliquary"]},
            "R1": {"class": "floating altar", "attributes": [],
                   "sam_queries": ["floating altar"]},
        },
        "filter": [{"op": "on", "args": ["E1", "R1"]}],
        "answer": {"op": "count", "of": "E1"},
    }
    domain = compile_domain(program)
    assert domain["stated_support_classes"] == ["floating altar"]
    assert domain["target_all_horizontal_surfaces"] is False
    assert domain["target_named_support_ids"] == ["R1"]
    assert domain["anchor_open_world_ids"] == ["R1"]

    registry = SurfaceRegistry()
    registry.surfaces.extend([
        Surface("S1", "support", 0.55, np.array([0., 0., 1.]), -0.55,
                cells_of(_tabletop(height=0.55)[:, :2]), klass="workbench"),
        Surface("S2", "support", 1.25, np.array([0., 0., 1.]), -1.25,
                cells_of(_tabletop(height=1.25,
                                   origin=(4.0, 0.0))[:, :2])),
    ])
    reasons = domain_reasons(domain, registry, SceneGraph(), floor_z=0.0)
    assert any("S1" in reason for reason in reasons)
    assert any("S2" in reason for reason in reasons)
    visits = surface_obligations(domain, registry, SceneGraph(), SafeCoverage(),
                                 np.zeros(2), 0.0, [])
    assert {visit["surface"] for visit in visits} == {"S1", "S2"}


# ------------------------------------------------- zero-answer safety net
def test_confirmed_target_without_geometry_blocks_a_certified_zero():
    """Regression for the office run that published a certified 0.

    Two potted plants were class-confirmed at 0.87/0.88 but localized only in
    a tabletop domain view, so they carried no lidar points and no support
    footprint. `_bounds` returns None for them, every `on` test answers False,
    and the count collapsed to zero -- which the certificate then passed
    vacuously because all its evidence rules quantify over the selected set.
    """
    from unified_scene_graph import confidence_stop_certificate

    program = {
        "task": "count",
        "entities": {"E1": {"class": "plant", "attributes": [],
                            "sam_queries": ["plant"]},
                     "E2": {"class": "table", "attributes": [],
                            "sam_queries": ["table"]}},
        "filter": [{"op": "on", "args": ["E1", "E2"]}],
        "answer": {"op": "count", "of": "E1"},
    }
    graph = SceneGraph()
    table = make_node("T1", "E2", (1.0, 1.8), z=0.76)
    table.points = np.array([[1.0, 1.8, 0.76], [1.4, 1.8, 0.76],
                             [1.0, 2.2, 0.76], [1.4, 2.2, 0.76]], np.float32)
    graph.nodes.append(table)

    # Class-confirmed, but no geometry at all: cannot be tested, not absent.
    blind = SceneNode("N7", "E1", facts={"is_class": True, "confidence": 0.87,
                                         "what_is_it": "small potted plant"})
    blind.observations.append(Observation(
        [0, 0, 0, 0, 0, 0, 1], .87, 50, "qwen_rectilinear_domain_localization",
        [0, 0, 1, 1], "new", 0.0))
    graph.nodes.append(blind)

    # Class-confirmed WITH geometry, genuinely far from the table: a correct
    # rejection that must NOT be flagged as unresolved.
    elsewhere = make_node("N18", "E1", (4.6, -0.6), z=0.92)
    elsewhere.points = np.array([[4.6, -0.6, 0.92], [4.65, -0.6, 0.92],
                                 [4.6, -0.55, 0.92], [4.62, -0.58, 0.95]],
                                np.float32)
    elsewhere.facts["confidence"] = 0.97
    graph.nodes.append(elsewhere)

    assert graph.evaluate(program) == 0
    flagged = [node.id for node in graph.discarded_class_members(program)]
    assert flagged == ["N7"]

    certificate = confidence_stop_certificate(
        graph, program, [0, 0], [(), ()], None, 0,
        frontier_attempts=3, frontier_visual_audits=2, residual_components=[])
    assert certificate["satisfied"] is False
    assert any("N7" in reason and "no metric geometry" in reason
               for reason in certificate["reasons_not_satisfied"])


def test_no_relation_question_is_not_blocked_by_the_geometry_veto():
    """With no relation to fail, a class-confirmed node already counts."""
    program = {
        "task": "count",
        "entities": {"E1": {"class": "plant", "attributes": [],
                            "sam_queries": ["plant"]}},
        "filter": [],
        "answer": {"op": "count", "of": "E1"},
    }
    graph = SceneGraph()
    blind = SceneNode("N1", "E1", facts={"is_class": True, "confidence": 0.9})
    blind.observations.append(Observation(
        [0, 0, 0, 0, 0, 0, 1], .9, 50, "test", [0, 0, 1, 1], "new", 0.0))
    graph.nodes.append(blind)
    assert graph.discarded_class_members(program) == []


def test_program_relations_render_for_the_visual_audit():
    from search_domain import describe_program_relations

    assert describe_program_relations({
        "answer": {"of": "E1"},
        "entities": {"E1": {"class": "plant"}, "E2": {"class": "table"}},
        "filter": [{"op": "on", "args": ["E1", "E2"]}]}) == "on table"
    assert describe_program_relations({
        "answer": {"of": "E1"},
        "entities": {"E1": {"class": "pillow"}, "F": {"structure": "floor"}},
        "filter": [{"op": "on", "args": ["E1", "F"]}]}) == "on floor"
    assert describe_program_relations({
        "answer": {"of": "E1"},
        "entities": {"E1": {"class": "x"}, "A": {"class": "vase"},
                     "B": {"class": "stool"}},
        "filter": [{"op": "between", "args": ["E1", "A", "B"]}]
    }) == "between vase and stool"
    assert describe_program_relations({
        "answer": {"of": "E1"}, "entities": {"E1": {"class": "x"}},
        "filter": []}) == ""
