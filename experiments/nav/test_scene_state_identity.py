import numpy as np

from scene_state import SceneState


def points_near(x, y, z=0.07):
    return np.array([
        [x - 0.03, y, z], [x, y - 0.03, z], [x + 0.03, y, z],
        [x, y + 0.03, z], [x + 0.02, y + 0.02, z],
    ], dtype=np.float32)


def test_floor_object_gaining_lidar_points_keeps_identity():
    state = SceneState("pillow")
    first = state.observe((-0.29, 1.43, -0.01), 0.61, 0.93, 124, (0, 0))
    closer = state.observe((-0.36, 1.58, 0.07), 0.72, 0.97, 163,
                           (-0.57, 0.48), points_near(-0.36, 1.58))
    assert closer.id == first.id
    assert len(state.hyps) == 1


def test_partial_floor_detections_merge_but_adjacent_instances_do_not():
    state = SceneState("pillow")
    a = state.observe((-0.41, 2.85, 0.07), 0.28, 0.75, 53, (1, 2),
                      points_near(-0.41, 2.85))
    b = state.observe((-0.27, 2.90, 0.06), 0.60, 0.84, 122, (1, 2),
                      points_near(-0.27, 2.90))
    c = state.observe((-0.93, 2.71, 0.07), 0.52, 0.81, 79, (1, 2),
                      points_near(-0.93, 2.71))
    state.merge_duplicates()
    assert a.id == b.id
    assert c.id != a.id
    assert len(state.hyps) == 2
