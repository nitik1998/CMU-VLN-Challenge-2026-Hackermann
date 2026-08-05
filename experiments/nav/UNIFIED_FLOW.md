# One chain of thought for all three question types

Design date: 2026-07-31. Grounded in the japanese_room pillow runs
(`japanese_numeric_pillows_restored_single_qwen_*_20260731`), `q_state.json`
(office_2 potted plants), the full 75-question set, and the validated
SAM3 / lidar / Qwen split.

## TL;DR

The chain of thought is not "kind of wrong" — it has one wrong axiom. It treats
the **answer as a sum of per-crop VLM verdicts**, glued together by free-form
deliberation. Everything observed failing follows from that. The fix is to make
the **scene graph the answer** and demote Qwen to exactly three jobs it is good
at: compiling the question into a typed program, stating atomic visual facts
about one node at a time, and breaking genuine semantic ties. Counting,
relations, identity, termination, and coordinates are all computed by code from
the lidar/camera fusion you already validated at 1–3 cm.

One pipeline serves all three question types because all 75 released questions
are the same problem with three different final operators:

```
COMPILE -> GROUND(explore) -> IDENTIFY -> MEASURE -> VERIFY -> EVALUATE -> PUBLISH
                                                                  |
                                          count(E)      -> Int32 -+   numerical
                                          argunique(E)  -> Marker -+  object ref
                                          path([legs])  -> Pose2D* -+ instruction
```

---

## 1. What actually happened in the runs (evidence)

**Run:** "How many pillows are on the floor?" (japanese_room, ground truth 4).
Trace: `japanese_numeric_pillows_restored_single_qwen_final_20260731/live_events.jsonl`.

- Confirmed hypotheses grew 2 → 3 → 5 → 6 → **8** across viewpoints. Every
  `judge_crop` said "red cushion", p=1.0. Exactly 2× the truth: every cushion
  was confirmed twice, once per viewing side.
- At the same time the model's own `reason_next_action` said, verbatim:
  *"From the current viewpoint, I see two red cushions … The panorama shows no
  other pillows on the floor"* — while echoing `"temporary_count": 8` because
  the prompt fed the running tally back to it.
- The run hit the iteration/budget wall mid-generation and **never published
  anything**. Scored outcome: 0, regardless of perception quality.

**Why the count doubled** (mechanism, from code reading):
floor cushions sit where the Livox is blind → `range_along()` falls back to
ground-plane intersection → the detection carries **0 lidar points** → the
voxel-set identity (the thing that actually works) can never fire → identity
falls back to a 0.40/0.45 m centroid gate → cross-view ground-plane ranging
error exceeds the gate → each new viewpoint mints fresh hypotheses → each is
independently, correctly, confirmed "red cushion". Every module did its job;
the *flow* has no identity path for lidar-blind objects and no cross-view
visual consistency check.

**Corroborating failure** (`q_state.json`, office_2 "potted plants on a table"):
H3 at (4.57,−0.57) **confirmed** "potted plant" p=0.95 with 2 points, H4 at
(4.68,−0.69) — 16 cm away, the same plant — **rejected** "potted plant on
shelf" p=0.90. One physical object, two nodes, opposite verdicts, because the
crop-level judgment bundles class + attribute + spatial relation into one bit.

**Parse loss:** "pillows **on the floor**" parsed to
`{relation: "on", reference: null}` — the floor is not an object, so the
constraint silently vanished from the strict description and any pillow
anywhere would have counted.

**Termination pathology:** the deliberation prompt forbids finalizing while
candidate viewpoints exist, and the frontier generator never runs dry
(`unexplored_edge_m2` never reaches 0). So the loop can only end by cap or
timeout, and the final answer is then renegotiated with the model over up to 4
transcript rounds in which the model can override the deterministic tally.

### The seven structural flaws

1. **Identity errors are count errors** — count = |confirmed nodes|, and
   identity has no path for 0-point objects (the majority of scored small
   objects: pillows, cups, bowls, books).
2. **Relations are judged per-crop by the VLM** despite cm-accurate 3D
   geometry ("on the floor", "on a table" are geometric predicates here).
3. **The parse drops constraints** whose reference is a structure (floor,
   wall, window) rather than a SAM-detectable object.
4. **Termination is negotiated with the model** under a rule that forbids
   answering; nothing guarantees an answer is ever published.
5. **The running count is fed back into prompts** — pure anchoring; the model
   parrots 8 while describing 2.
6. **The final transcript carries no dedup evidence** — eight identical
   "red cushion p=1.0" lines are indistinguishable; the model *cannot* dedupe
   from them even in principle.
7. **The flow only exists for counting** — 1 of 17 points per scene. Object
   reference (2×2) is prototype scripts; instruction following (2×6 = 71% of
   points) is a stub.

---

## 2. What the 75 questions actually require

Programmatic analysis of `questions/questions.json`:

| primitive | numerical (15) | object ref (30) | instruction (30) |
|---|---|---|---|
| `on` (support) | **11** | 15 | 17 |
| `closest` | 1 | **18** | 14 |
| `near` | 1 | 4 | **22** |
| `between` | – | 4 | 12 |
| `farthest` | – | 3 | 5 |
| `above`/`below`/`under` | 5 | 2 | 3 |
| `with X on it` (inverse support) | 2 | 2 | 5 |
| color attribute | 2 | 2 | 1 |
| `then` (ordering) | – | – | 19 |
| `avoid` | – | – | 3 |
| legs per instruction | – | – | 11×1, 11×2, 8×3 |

Three consequences:

- **The relation vocabulary is closed**: `on, above, below/under, near,
  between, closest, farthest, with_on` + color/size attributes + `then/avoid`
  for paths. Every one is computable from oriented boxes + a support relation.
  No open-ended spatial language exists in the released set.
- **`closest`/`farthest` (21/30 object-ref questions) require exhaustive
  enumeration of the target class before the argmin** — i.e., object reference
  *contains* the counting problem. They are not separate pipelines.
- **Every instruction leg names an object by reference** ("the nightstand with
  a clock on it", "the trash can closest to the refrigerator") — i.e.,
  instruction following *contains* the object-reference problem k times, plus
  path-region geometry. One grounding engine feeds all three answers.

This is why one chain of thought is the right ask: the tasks share everything
except the last operator.

---

## 3. The unified flow

### State: the scene graph is the only memory

Every other artifact (history strings, temporary counts, transcripts) goes
away. A node is:

```
Node {
  id
  fingerprint: voxel_set (3D, 5 cm)            # lidar-visible objects
           OR  footprint_set (2D, 5 cm, plane) # lidar-blind objects (§4)
  points: fused map-frame points               # -> bbox() oriented box
  support: node_id | "floor" | "wall" | None   # geometric, from planes
  facts: {what_is_it, is_class, color, marks}  # Qwen, atomic, cached, w/ px
  best_view: (pose, px_width, crop_path)
  observations: [(pose, mask, range_how), ...]
}
```

Structures (floor, walls, ceiling, and horizontal support planes) come from
`structural_lidar.extract_planes` — they are first-class graph entities so
"on the floor" and "above the display ledge" have something to bind to.

### Stage 1 — COMPILE (Qwen, text-only, once)

One call turns the question into a typed program over the closed vocabulary.
No image. Schema-validated by code; on violation, re-ask once with the
validation error; final fallback is the regex router + head-noun extraction.

```json
{
  "task": "count",
  "entities": {
    "E1": {"class": "pillow", "attributes": [], "sam_queries": ["pillow", "cushion"]},
    "F":  {"structure": "floor"}
  },
  "filter": [{"op": "on", "args": ["E1", "F"]}],
  "answer": {"op": "count", "of": "E1"}
}
```

Object reference: `"answer": {"op": "argmin_dist", "of": "E1", "to": "R1"}`
after the same filter chain. Instruction:
`"answer": {"op": "path", "legs": [{"kind": "goto_near", "of": "R1"}, {"kind":
"pass_between", "of": ["R2","R3"]}, {"kind": "stop_at", "of": "R4"},
{"kind": "avoid_between", "of": ["R5","R6"]}]}` — each `Rn` itself a
class+filter entity.

Rules that fix observed bugs:
- References may be **structures**; "on the floor" can never be dropped again.
- Nested references compile recursively ("pillows on the sofa **under the
  pictures**" → sofa filtered by under(pictures) first).
- Keep `_strip_counts` hygiene: no quantity, color, or location that the
  question didn't state may enter any downstream prompt.

### Stage 2 — GROUND (deterministic obligation loop, not deliberation)

The exploration loop's controller is an **obligation list computed from the
program**, refreshed every capture. The model never decides whether to stop.

Obligations, in fixed priority order:

1. `find(anchor)` — every reference entity/structure not yet instantiated.
   (Keep `survey_reference` + stand-at-1m behavior; it was correct.)
2. `enumerate(E)` — for `count`/`closest`/`farthest`: sweep until every
   reachable region has been imaged at a resolution where an instance of E's
   expected size subtends ≥ 60 px, and every support instance implied by the
   filter (each table for "on a table") has been scanned from ~1 m.
   Coverage frontier machinery stays, but as *this obligation's servant* —
   frontier value is weighted by whether the parsed query could hide there.
3. `resolve(node)` — nodes whose facts came from < 60 px crops → approach
   (existing `viewpoint()` + `MIN_PX` logic, unchanged).
4. `disambiguate(a, b)` — node pairs with uncertain identity (§4) → one
   viewpoint that sees both simultaneously.

Qwen still gets a deliberation call per iteration — but its output is
**advisory**: it may *propose* obligations ("cushions flank tables
symmetrically; the far side of the table is unviewed"), which are accepted only
if they map to a legal terrain-safe viewpoint. It cannot veto termination, and
the prompt never contains a running count — only the map, the obligations, and
the panorama. Ask for *enumeration* ("list the instances you can see, by
sector"), never for confirmation of a number.

**Termination is code, and query-conditioned.** `Coverage` stores the minimum
range from which every floor cell was observed. For target minimum width `s`,
the useful enumeration range is
`r_max = s * panorama_width / (2*pi*40px)`. Thus a cushion can be enumerated
from about 3.4 m while a paper cup requires roughly 0.8 m. Remaining reachable
or reachable-adjacent cells are split into connected components and a distance
transform rejects wall slivers in which the target cannot physically fit.

Each fit-capable residual component is a cell-set identity and must become one
of: enumerated, unreachable after a navigation attempt, or unobservable after
reaching a useful view. Attempts and evidence are separate counters: any
attempt discharges its exact cell set, while only the following captured frame
counts as a visual audit when pose displacement is at least 0.5 m or it adds
meaningful coverage. Centroid-radius retirement and controller-status-as-audit
are forbidden.

For counting, the certificate requires: no fit-capable residual component, no
unresolved or unexplained clipped proposal, stable answer and identity set over
the last two captures, each selected node settled from either two independent
poses or one score >= 0.5 observation with a >= 60 px semantic crop, and a real
visual audit after exploration. An unexplained image-boundary proposal creates
an executable back-away obligation; it cannot veto forever. Budget reserve
reached (§7) still publishes the best deterministic graph answer.

### Stage 3 — IDENTIFY (§4, the pillow fix)

### Stage 4 — MEASURE (pure geometry, existing validated math)

- Projection: `project.py` (`map_to_camera`, `cam_to_pixel`) with the three P0
  fixes from `LIDAR_OBJECT_REFERENCE_REVIEW.md`: rays originate at the *camera*
  origin (`sensor + R@T_SC`), pose interpolated to the image timestamp,
  extrinsics as config.
- Mask→points: standardize on `object_reference_geometry.associate_mask_points`
  (eroded-core depth mode, adaptive band, z-buffer, connected components) —
  strictly better than `run_question.mask_points`'s flat ±0.45 m gate, which is
  the known cause of the 0.25 m-wide "4 cm scroll".
- Boxes: `Hypothesis.bbox()` yaw-only oriented fit — already validated at
  1–2 cm. This *is* the object-reference answer payload.
- Support: nearest horizontal plane below the node's bottom face within τ, or
  floor. Computed once per node, never asked of the VLM.

### Stage 5 — VERIFY (Qwen, atomic facts only)

Keep `inspect_crop`, delete `judge_crop`. Per node, at its best available
resolution, one call: `{what_is_it, is_a_<class>, color, distinguishing_marks}`
— open questions, no relation clause, no compound verdict, no "does it
satisfy the description". Re-ask only when px width has ~doubled. Context-rich
crops (`crop_for`, ctx_frac ≈ 0.18) stay — that lesson was measured and real.

Synonym adjudication (desk-vs-table) becomes one cached **text-only** call at
evaluate time: given the set of `what_is_it` labels actually observed, "which
of these count as a ⟨table⟩ in ordinary language?" — applied deterministically
to all nodes at once, so one plant on a "desk" and one on a "cabinet" are
judged by the same rule in the same call.

### Stage 6 — EVALUATE (pure code; Qwen only for declared ties)

VLA-3D-style predicates over node boxes, thresholds calibrated offline on the
15 released scenes (they ship ground-truth object lists for exactly this):

- `on(A,B)`: bottom(A) within τ_on of top(B) ∧ footprint(A) mostly inside
  expanded footprint(B). `on(A, floor)`: bottom(A) within τ of floor plane.
- `above/below`: vertical order ∧ horizontal footprint overlap.
- `near(A,B)`: box-to-box distance < τ_near (calibrated, ~1–1.5 m indoors).
- `between(A,B,C)`: A inside the corridor around segment BC, projecting
  between the endpoints.
- `closest/farthest`: exact arg over the enumerated candidate set.
- `with_on(B,X)`: ∃ node X′ with on(X′,B).

Answers:
- **count** = |{n : is_class ∧ attributes ∧ filter}|. Qwen never does the
  arithmetic and cannot override it. Before publishing, run the **roll-call
  check** (§4.4); a discrepancy triggers targeted re-association, not
  renegotiation.
- **refer** = filter chain → argmin/argmax → exactly one node → publish its
  oriented `bbox()` as `Marker.CUBE` on `/selected_object_marker`. If the
  chain yields 0 or ≥2 nodes: relax the *weakest* constraint (near before
  closest before class), and if ≥2 survive, give Qwen a side-by-side panel of
  the finalists' best crops + the question and let it pick — semantic
  tie-breaking is the one legitimate use of a VLM verdict, and it is the
  decision it already made correctly 6/6 on the calligraphy set.
- **follow** = §5.

### Stage 7 — PUBLISH (always, unconditionally)

A deadline scheduler owns the clock. At T−60 s (numerical/refer) the current
best answer is published no matter what state the loop is in; for refer,
publish the current best marker *as soon as one exists* and refresh as it
improves. The pillow run scored 0 not because perception failed but because
nothing was ever sent. That may never be structurally possible again.

---

## 4. Identity for lidar-blind objects (the concrete fix)

Your core insight — *the surface points of an object don't move, so identity is
set overlap in the map frame* — is correct and validated. It just has no
representation for objects the lidar can't touch. Extend the same insight to
two more evidence classes; try in order:

### 4.1 Voxel-set identity (unchanged)
Exactly the current `SceneState` rules: 5 cm voxels, overlap ≥ 0.30 normalized
by the smaller set, cardinality ratio ≤ 1.7 to block superset/group merges,
group pruning by exact union-containment.

### 4.2 Support-plane footprint identity (new — this alone fixes the 4→8)
For a detection with no (or < 8) associated lidar points whose mask bottom
lies below the horizon:

1. Take the support plane: the floor plane (fit from the accumulated cloud,
   not assumed z=0) or a detected `horizontal_support` plane whose projected
   extent contains the mask's contact region.
2. Erode the SAM mask (boundary pixels smear far at grazing angles), then
   intersect **every remaining mask pixel's camera-origin ray** with the
   plane: `t = (h_cam − h_plane) / (−ray·n̂)`, point = origin + t·ray.
3. Quantize the hit points into 5 cm 2D cells on that plane →
   `footprint_set`.
4. Apply the *identical* overlap/cardinality rules as 4.1, in 2D.

Why it works where the centroid gate failed: two views of the same cushion
smear its footprint along their respective viewing directions, but both smears
are anchored on the true contact patch, so their cell sets intersect heavily —
while two *different* cushions 0.6 m apart share no cells. It is
viewpoint-invariant for the same reason the voxel set is: the contact patch is
a physical surface. And it converts "0-point object" from an identity hole
into a measured footprint that also gives the box's XY extent for free (height
then comes from the mask's angular height at the plane-derived range).

### 4.3 Reprojection identity (new — the cheap gate that runs first)
Before minting any new node: project every existing node (its points, or its
footprint, or just its box) into the *current* panorama through the current
pose. If the new detection's mask substantially overlaps an existing node's
projected silhouette (containment ≥ 0.5 with wraparound handling), it is that
node — no range needed at all. Bearing is the camera's dense, precise channel
(your own calibration notes say exactly this); using it only for ranging and
never for identity was the missed opportunity. This gate alone would have
absorbed H5/H9/H10/H11 in the pillow run.

### 4.4 Roll-call consistency check (pre-publish, catches what's left)
Deterministic half: in the single view where the most confirmed nodes of E
project, SAM's per-view instance masks are mutually exclusive — so the number
of distinct verified masks there is a hard **lower bound**, and if *every*
confirmed node projects into that view, it is also the expected count. Graph
count > distinct projected silhouettes ⇒ nodes sharing a silhouette are one
object; merge.
Semantic half: one Qwen call on that panorama — "list each distinct ⟨E⟩ you
can see, by sector" — cross-checked against the projections. This formalizes
the signal the failed run printed and ignored ("I see two red cushions" vs
count 8).

---

## 5. Instruction following (71% of the points, currently a stub)

Same spine; the answer operator is a path. Three specifics matter:

**The trajectory IS the answer, so exploration is not free.** The evaluator
scores the actual driven path against ordered constraints and forbidden
regions. Consequences, in order of importance:
1. `avoid` regions are compiled at parse time and become hard no-go cells for
   **all** motion — including exploration — as soon as their anchors are
   grounded; until they are grounded, route conservatively (don't thread
   unidentified furniture pairs).
2. Ground and execute **legs in order**: find leg-1's anchors (spin capture +
   targeted looks), commit leg 1, then ground leg 2 from the new pose.
   Opportunistic sightings of later-leg anchors are recorded but not driven to.
3. Partial credit is real: a leg whose anchor cannot be grounded in time still
   gets a best-guess waypoint toward its highest-probability region rather
   than aborting the sequence.

**Leg → region → waypoints** (all geometric):
- `goto_near(R)` / `stop_at(R)`: safest terrain cell in an annulus
  (0.7–1.2 m) around R's footprint, on the robot's side.
- `pass_between(A,B)`: entry and exit waypoints on the corridor axis through
  the midpoint of the A–B gap, so the base planner is *forced* through it, not
  merely near it.
- `avoid_between(A,B)`: block the corridor cells in the planning grid; if the
  direct route would cross it, insert a guide waypoint on the chosen side.
- Plan on the terrain grid (A*) over free cells minus forbidden cells;
  emit waypoints leg by leg on `/way_point_with_heading`.

**Monitor and advance by geometry:** watch `/state_estimation`; a leg is done
when the actual trajectory enters its region (not when far_planner claims goal
reached — known unreliable within `goal_adjust_radius`). If the base planner
shortcuts through a forbidden corridor, insert a stronger guide waypoint and
replan. Keep the far_planner topic remap discipline from `far_bridge.py`.

---

## 6. Exact division of labor (final)

| decision | owner | never |
|---|---|---|
| question → typed program | Qwen (text-only, schema-validated) | free-form parse into prose specs |
| where objects are, size, box, support | lidar + projection math | Qwen coordinates |
| what a thing is / its color | Qwen atomic facts on best crop | compound verdicts, relation clauses |
| whether relation holds | code (calibrated predicates) | per-crop VLM judgment |
| object identity | voxels / footprints / reprojection | centroid-radius gates, VLM opinion |
| where to look next | code obligations (+ Qwen *proposals*) | VLM-chosen termination |
| the count / the argmin / the path | code over the graph | VLM arithmetic or override |
| semantic tie among ≤3 finalists | Qwen, side-by-side panel | — (this is its one verdict) |
| synonym scope ("does desk count as table") | Qwen once, text-only, cached | per-node ad-hoc wording |
| when to answer | deadline scheduler + empty obligations | prompt rules ("not allowed to finalize") |

## 7. Budget (600 s, timing starts at system launch)

| phase | numerical / refer | instruction |
|---|---|---|
| boot: model load ∥ first capture+spin | 0–40 s | 0–40 s |
| compile + validate | ~10 s | ~10 s |
| ground/enumerate (obligation loop) | ≤ 380 s | folded into execution |
| execute legs | – | ≤ 460 s |
| verify + evaluate + roll-call | ~60 s | – |
| **hard publish reserve** | last 60 s | last 45 s (stop at best leg) |

Load SAM3 and Qwen concurrently with the first capture — loading is inside the
scored window. For refer, publish the best current marker as soon as any
candidate exists and refresh; for numerical, publish the current deterministic
count at the reserve line even mid-exploration.

## 8. Keep / delete list against the current code

**Keep verbatim (validated):**
`project.py` math; `associate_mask_points`; voxel identity + group pruning
(`scene_state.py`); `crop_for` context sizing; `survey_reference` +
reference-first viewpoints; `Coverage` frontier machinery (demoted to servant
of `enumerate`); `question_types.py` router; `_strip_counts` hygiene;
`far_bridge` remaps; verbatim-transcript principle (now applied as: prompts
contain only measurements and the model's own enumerations, never your
summaries or running counts).

**Delete:**
`judge_crop` and every compound "does it satisfy the description" verdict;
relation words in any crop prompt; `temporary_count` in any prompt;
`reason_next_action` as routing/termination authority (becomes advisory
proposal call); the 4-round `final_count` transcript negotiation (replaced by
deterministic evaluate + roll-call + obligation-driven rechecks); the
ground-plane *centroid* fallback as an identity path (replaced by §4.2
footprints — the ranging fallback itself survives as measurement).

**Fix while porting:** camera-origin ray offset, timestamp-interpolated pose,
extrinsics from config (the three P0s); 1–2 cm voxel downsampling for points
inside candidate masks; panorama seam wraparound in mask/identity logic.

## 9. Validation before trusting it

1. Replay the japanese pillow captures (`q_snap0..8`) through §4 offline — the
   8 nodes must collapse to 4 with no other parameter changes. This is the
   acceptance test for the identity fix.
2. Calibrate predicate thresholds (`τ_on`, `τ_near`, corridor width) against
   the 15 released scenes' ground-truth object lists; evaluate the compiled
   programs against ground truth *before* adding perception noise.
3. Re-run each released question ≥3 times from clean restarts; variance is a
   failing metric on its own (earlier trace: six runs of one question gave
   1, 0, 1, 1, 2, 0).
4. Score object-reference boxes by 3D IoU vs ground truth, not center error —
   that is what the 2 points are paid on.
