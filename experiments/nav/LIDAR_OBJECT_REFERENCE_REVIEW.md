# LiDAR-to-image review for object-reference boxes

## Conclusion

Use LiDAR for object-reference questions. Qwen and SAM can identify the intended
pixels, but the required answer is a metric, map-frame, oriented 3D cuboid. The
camera alone does not reliably provide its depth, world position, dimensions, or
yaw. LiDAR should be the geometry assistant, not the semantic decision-maker.

The current `flip_az=False, shift_pi=False` projection visually agrees with the
saved panorama. The core sensor/camera axis rotation is therefore in the right
direction for the current simulator capture. The remaining problems are in timing,
extrinsic translation, range association, and conversion of partial surface points
into a full object box.

## Findings, in priority order

### P0: the back-projected target ray starts at the wrong origin

`run_question.py` computes a point as `sensor_position + range * ray`. The ray is a
camera ray, so it must start at the camera origin:

`camera_origin_map = sensor_position + R_map_sensor @ T_sensor_camera`

With the current hard-coded translation this creates up to a 10 cm center error.
That is already half the width of many small challenge objects and can severely
reduce 3D box overlap.

### P0: camera/pose capture is not timestamp-synchronized

`capture.py` attaches whichever odometry sample happened to arrive immediately
before the image callback. It does not use either message's header timestamp. This
is tolerable only when the robot is fully stationary. During motion, even a small
time offset shifts every projected point and can mix adjacent objects.

Store a short timestamped odometry deque and interpolate the pose at the image
timestamp. Also reject captures when the nearest pose is outside a configured time
tolerance. Because `/registered_scan` is already in `map`, individual scan poses are
not needed for static scene geometry.

### P0: the camera-to-sensor translation is an unverified constant

`project.py` uses `T_SC = [0, 0, 0.1]`. The axis convention is supported by the
overlay, but the translation must come from the organizer's measured calibration
for the simulator/robot rather than a source constant. Make all six extrinsic
parameters configuration values and validate them using depth-edge alignment.

### P1: cone-based range can select a different foreground object

`range_along()` searches a 3-degree and then an 8-degree cone, sorts all ranges,
and takes the median of the nearest 40 percent. At 5 m, an 8-degree half-angle
covers roughly 1.4 m across, so furniture beside the requested object can win.

Project points first, restrict them to the eroded SAM mask, build a 1D depth
histogram, and select the coherent depth mode that covers the mask centre. Apply a
per-pixel z-buffer before associating points.

### P1: the generic ground-plane fallback fabricates positions

When no LiDAR return exists, every below-horizon detection is intersected with the
floor. That is valid only for an explicitly floor-contact point (normally the
bottom of a floor object's mask), not for the centre of a pillow, wall picture,
tabletop item, or chair. Gate this fallback by a confirmed support relation and use
the mask's contact pixel. Otherwise move closer or obtain another view.

### P1: mask association still admits background and stale-view geometry

`mask_points()` projects the accumulated world cloud through the current view and
accepts points within `estimated_range +/- 0.45 m`. Forty-five centimetres is much
thicker than most target objects, and an accumulated cloud contains surfaces seen
from other viewpoints that are occluded in the current image.

Use a z-buffer, a small adaptive depth band based on range noise, spatial clustering,
and multi-view intersection/consistency. Keep background points as debug evidence,
not members of the target.

### P1: PCA of visible LiDAR points is not the full object box

`Hypothesis.bbox()` fits min/max extents to the visible associated points. These are
usually only the front/side surfaces, so the cuboid is systematically too shallow
and may be too short. Raw min/max also lets one background outlier inflate the box.

Use robust percentiles after clustering, fuse several viewpoints, and complete the
unseen extent using the SAM mask's angular silhouette plus the selected depth. Use
PCA yaw only when the horizontal eigenvalues are sufficiently distinct; otherwise
prefer a relation-derived/support-derived yaw or mark yaw as low-confidence.

### P2: 5 cm voxelization is coarse for the scored small objects

`capture.py` downsamples to 5 cm while `SceneState` later uses 3 cm identity voxels.
Small cups, figurines, thin pictures, and bottles can retain too few points for a
box. Use roughly 1-2 cm for points inside candidate masks, while keeping a coarser
copy for navigation/coverage.

### P2: panorama seam handling needs an explicit test

An object can be split across pixel 0/1919. Detection merging sometimes recovers it
through shared 3D points, but range/mask logic should explicitly wrap horizontal
coordinates and regression-test seam-spanning masks.

## Recommended object-reference geometry flow

1. Qwen parses the unique target, attributes, anchors, and spatial relations.
2. Qwen describes/explores; SAM proposes masks only where Qwen needs localization.
3. Project synchronized, calibrated `/registered_scan` points into each SAM mask.
4. Z-buffer and depth-cluster the mask points; reject background bleed.
5. Convert the selected depth and camera ray from the camera origin into `map`.
6. Fuse the same physical object over at least two useful viewpoints when possible.
7. Fit a robust yaw-only cuboid and complete missing dimensions from the mask
   silhouette/support evidence.
8. Let Qwen select the unique candidate using the question's spatial relations.
9. Publish exactly one `Marker.CUBE` on `/selected_object_marker`; publish labels and
   evidence points only on `/selected_object_debug_marker`.

The new question agent implements steps 1's routing boundary and step 9's safe ROS
output contract. The existing perception loop still needs the geometry corrections
above before its boxes should be trusted for scoring.
