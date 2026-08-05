# Generalized Evidence Architecture

## Decision

Do not build a Japanese-room policy, a calligraphy policy, or separate
per-question agents. Use one recursive query graph, one persistent metric scene
graph, one domain-driven active-perception loop, and three final answer
operators.

The released corpus contains 75 questions in 15 rooms:

- 15 numerical questions -> `Int32`
- 30 object-reference questions -> one metric 3D marker
- 30 instruction-following questions -> an ordered driven trajectory

Across those outputs, the language composes the same physical operations:
`on`, `above/below/under`, `near`, `between`, `with X on it`, nested
closest/farthest selectors, and ordered go/stop/pass/avoid path constraints.
The hard problem is therefore not three unrelated agents. It is maintaining a
faithful belief about entities, supports, relations, visibility and unexplored
evidence, then applying a different output operator.

## Why this architecture is evidence-backed

- [EmbodiedQA](https://arxiv.org/abs/1711.11543) defines the task as active
  perception: navigation, first-person evidence gathering, language grounding
  and answering are coupled. A single-frame answerer cannot establish absence.
- [Explore Until Confident](https://arxiv.org/abs/2403.15941) identifies two
  exact failure modes we observed: VLMs lack persistent map memory and their
  confidence is miscalibrated, causing premature stopping or over-exploration.
  It uses a depth-backed semantic map and calibrated stopping.
- [ConceptGraphs](https://arxiv.org/abs/2309.16650) fuses open-vocabulary 2D
  foundation-model outputs into a compact multi-view 3D scene graph rather than
  treating every frame as a new story.
- [SayNav](https://arxiv.org/abs/2309.04077) feeds an incrementally built 3D
  scene graph to an LLM for high-level decisions while a low-level point-goal
  planner executes feasible motion. This supports our division between Qwen's
  semantic choices and deterministic navigation.
- [PanoSwin](https://arxiv.org/abs/2308.14726) documents ERP seam discontinuity
  and spatial distortion. The panorama should remain the canonical sensor
  record, but planar vision models should inspect overlapping tangent views and
  map their pixels back to the panorama.
- [VLA-3D](https://arxiv.org/abs/2411.03540), released by the challenge group,
  represents objects, boxes, scene graphs, navigable free space and
  view-independent spatial relations. Its
  [official generator](https://github.com/HaochenZ11/VLA-3D/blob/main/scene_graph/generate_scene_info.py)
  computes relations from 3D geometry: `on` is vertical contact, `near` uses
  object/room geometry, and closest/farthest are ordered class relations. The
  benchmark language is minimal and intended to uniquely disambiguate a
  target. Our metric predicate evaluator should mirror this formulation rather
  than ask a VLM to estimate metres from an image.
- [Active visual search with uncertain detections](https://arxiv.org/abs/2303.03155)
  explicitly models detector failure. A negative SAM result is not proof of
  absence; it updates evidence for one observable domain element.

## End-to-end flow

### 1. Compile language to a recursive typed query

Every physical noun phrase receives an entity. Every relation constrains the
entity it grammatically modifies. Closest/farthest attaches either to the final
answer or to a nested entity selector.

Example:

`monitors on the table closest to the map wall decal`

becomes:

`count(MONITOR where on(MONITOR, select_closest(TABLE, MAP_DECAL)))`

This is not equivalent to counting monitors on every table. Syntax validation
alone cannot detect this loss, so compilation uses a second text-only semantic
audit. Production fails visibly after repeated unfaithful parses instead of
silently running a regex approximation of a different question.

### 2. Compile the query to explicit physical domain elements

The domain is a finite collection of obligations, not a generic room sweep:

- `ground_entity`: find a relation/selector anchor
- `surface_family`: enumerate visible floor or wall evidence
- `support_instances`: enumerate the tops of every relevant table/shelf/etc.
- `relation_volume`: inspect above/below/near/between an anchor
- `carried_object_relation`: inspect every candidate chair/table/etc. for the
  required object on it
- `entity_selector`: ground all candidates and the comparator before pruning

An ungrounded anchor never narrows the domain. A cup-on-table question first
enumerates plausible tables at furniture detection range, then inspects their
tops at cup range. It does not demand that the robot stand cup-distance from
every square metre of floor.

### 3. Observe once, preserve both forms

- Save one complete 360 panorama as the canonical observation and append a
  plain, verbose Qwen scene story to memory.
- Derive overlapping rectilinear views from the active domain: downward for a
  floor family, vertical for a wall family, room-height for relation landmarks,
  and geometry-centred views for registered support tops.
- Map every Qwen/SAM localization back through the canonical panorama to LiDAR
  points and the map frame. Never treat a tangent-view index as a world
  location.

The story supplies semantic context and plausible hidden locations. It is not
the count, the metric map, or a certificate of completeness.

### 4. Divide model responsibilities

- Qwen: detailed story, language compilation, literal crop classification,
  semantic synonym boundary, and ranking *why* an unresolved domain element is
  likely useful.
- SAM: subordinate proposal/mask tool used for broad recall and precise pixels.
  A high-confidence Qwen localization does not need a ceremonial SAM call.
- LiDAR/odometry: identity across views, support contact, 3D extents,
  view-independent relations, coverage, and candidate viewpoint geometry.
- Controller: chooses and reaches only collision-safe metric goals. Qwen never
  invents raw coordinates.

### 5. Fuse evidence in one persistent graph

One physical object has one node. Association uses registered voxel overlap,
support-plane footprint overlap, and reprojection—not centroid-radius merging.
Each node records observations, semantic facts, supporting surface and the
specific furniture node bound to that surface.

That last link is essential. `cup on table closest to X` requires
`cup.support_node == selected_table.id`; a generic `support_class == table`
would accept cups from all tables.

### 6. Pick actions by expected answer change

Generate safe candidates from open domain elements. Rank each using:

`utility = P(answer changes | observation) * information gain - travel cost - risk - redundancy`

Qwen may rank semantic likelihood from the story (for example, an occluded far
side of a table), while geometry supplies what a candidate can actually see,
whether the base can reach it, and the travel cost. Detector misses remain
uncertain observations rather than closed facts.

### 7. Stop on evidence, not model confidence alone

Stop only when:

- the typed parse passed its semantic audit;
- every domain element is enumerated, disproved, or retired after a bounded
  unreachable attempt;
- no unexplained clipped/new proposal can alter the result;
- the resolved answer identity set is stable across a genuinely displaced
  observation;
- no available action has material probability of changing the answer.

Attempted-but-unreachable and visually-audited are separate counters. An
unreachable move can retire that exact goal, but must not masquerade as visual
confirmation. Explore Until Confident's conformal stopping is the principled
next step once held-out runs exist; it cannot be honestly added without a
calibration set.

## Output operators

### Count

Resolve the target query recursively and publish its cardinality. Completeness
is over all relevant support/region elements, not over an arbitrary number of
robot moves.

### Object reference

Resolve exactly one node using nested predicates/selectors. Spend remaining
budget on a target-specific orbit, fusing only re-observations that reproject
to that node, then fit and continuously publish the 3D box.

### Instruction following

Treat every leg anchor as the same recursive reference problem. Execute the
ordered automaton:

- `goto_near`, `stop_at`
- `pass_between`, `avoid_between`
- `pass_near` for “take the path near” and “pass by”
- `avoid_near` for “avoid the path near”

Passing emits entry and exit poses; it is not approximated by one nearby goal.
Avoidance creates a persistent forbidden capsule/circle for subsequent legs.

## Implemented now

- recursive entity filters and nested closest/farthest selectors;
- semantic compile audit and visible failure instead of silent regex drift;
- explicit domain elements and dependency closure;
- domain-driven rectilinear panorama inventory, replacing the wall-target gate;
- persistent 3D identity graph and support-surface registry;
- target-to-specific-support-node links;
- floor-plane distance for “farthest from the floor”;
- VLA-3D-style above/below footprint overlap and room-scaled box distance for
  `near`, replacing fixed centre-radius guesses;
- follow dependency closure plus pass-near and avoid-near operators;
- one final evaluator shared by count/reference, with follow using the same
  recursive resolver for each leg anchor.

## Remaining validation gates

Passing unit tests proves internal invariants, not challenge accuracy. Before
calling this solved, run a stratified live matrix:

1. floor count with occlusion;
2. support-surface count;
3. nested selected-support count;
4. above/below and with-on count;
5. closest/farthest reference;
6. between reference;
7. reference box orbit and ROS marker;
8. pass-between plus avoid-near follow path.

Record answer, domain discharge reasons, arrival accuracy, runtime and—where
ground truth is available—count correctness, target selection, box IoU and
trajectory constraint success. Do not tune a threshold from one room; change
it only when a failure category repeats across the stratified matrix.
