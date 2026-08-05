# The closed pipeline: search domains, one enumerator, one certificate

Design date: 2026-08-01. This supersedes the "floor-only" caveat in
UNIFIED_FLOW.md §3–4. Goal: every question type, every room, one mechanism —
no relation form or room topology left as a special case.

## The organizing idea

Everything so far (floor enumeration, hidden regions, frontier audits,
support-first search) becomes one concept: **the compiled question defines a
SEARCH DOMAIN — an explicit, finite set of surfaces and regions that provably
bounds where answer evidence can live.** One enumerator visits domain elements;
one certificate checks that every element reached a terminal state. The floor
was never special: it is simply one support surface among many.

This is where the strongest related work points:

- **Semantic Linking Maps** (Zeng et al., ICRA 2020): search for a target via
  landmark objects and inter-object spatial relations — "seeing a table aids
  the discovery of a cup." Our questions are easier: the relation is *stated*
  ("cups **on the** coffee table"), so the probabilistic prior collapses to a
  deterministic domain.
- **Clio** (Maggio et al., RA-L 2024): map only what the task needs
  (information-bottleneck task relevance). Our domain program is exactly that
  filter, computed from the compiled query instead of a CLIP embedding.
- **VLFM** (Yokoyama et al., ICRA 2024): score frontiers by task relevance
  rather than raw area. Ours is the deterministic analogue: a frontier matters
  only if the unknown space behind it could extend the domain.
- **GraphEQA** (Saxena et al., 2025) and **RATE-Nav** (ACL Findings 2025):
  explore-to-answer with explicit stopping — confidence-gated in GraphEQA,
  region-wise exploration accounting in RATE-Nav. Both validate region-level
  (not global) termination bookkeeping.
- **SORT3D** (Zantout et al., 2025 — from the VLA-3D group that authors this
  challenge): deterministic spatial-relation toolbox + LLM sequencing for
  referential grounding. Confirms the predicate engine belongs in code.
- **Lang2LTL** (Liu et al., CoRL 2023): commands → temporal-logic specs over
  grounded landmarks ("visit X after Y, always avoid Z"). Our instruction
  grammar is a small fragment of this: a linear leg sequence plus global avoid
  invariants — no LTL solver needed, just an automaton.
- **2025 4th-place team** (public write-up): hierarchical scene graph,
  geometric + semantic + contour exploration, multi-VLM entropy-aggregated
  counting, confidence-gated answers. Validates the overall shape; our edge is
  metric lidar identity and deterministic evaluation.
- The NBV literature's honest result (Border, DPhil 2019; 2025 survey):
  guaranteed complete coverage is impossible in general — coverage depends on
  scene structure and sensor reach. This is why the certificate proves
  "every domain element reached a **terminal state**" (enumerated / unreachable
  / unobservable-after-attempt), never "the room is fully seen."

## 1. The Surface Registry

New first-class scene structure, built continuously from what already exists:

```
Surface {
  id, class            # "floor" | "table" | "cabinet" | "shelf" | "bed" | "sofa" | "wall" ...
  plane                # normal, offset  (structural_lidar RANSAC / floor fit)
  boundary             # in-plane hull of inlier + footprint points
  cells                # 5 cm grid over the boundary
  obs_range[cell]      # best (minimum) range this cell was imaged from
  state                # open | enumerated(E) | unreachable | unobservable
  node_id              # link to the furniture instance in the scene graph
}
```

Sources, all existing machinery:
- `fit_floor_plane` → the floor surface.
- `structural_lidar.extract_planes` → `horizontal_support` planes (tabletops,
  shelves, bed tops) and `wall` planes.
- SAM furniture detections ("table", "cabinet", …) fused with lidar via the
  existing mask→points association → assigns class + scene-graph identity to
  each plane. Identity uses the same voxel-set rules as any node.

**Wall surfaces get the footprint-identity treatment too.**
`mask_plane_footprint` already accepts an arbitrary plane: intersecting a
picture/window/scroll mask with its wall plane yields a wall-cell footprint —
the same viewpoint-invariant identity and metric position that fixed the floor
cushions, now covering every lidar-thin wall object (pictures above beds,
calligraphy above ledges, windows, curtains). Nothing scored in the question
set remains without a metric identity mechanism.

**Elevated surfaces are enumerated incidence-aware.** A tabletop cell counts
as observed only from poses whose ray to the cell clears the table rim —
camera at ~0.75 m vs tabletops at ~0.7 m means the surface only "opens up"
within roughly 1–1.5 m. The old "stand ~1 m from each table" heuristic is now
derived geometry: the viewpoint set of a tabletop is computed, not guessed.

## 2. The domain compiler

Each compiled program maps to domain elements. This closes over every relation
form in the released 75 questions (counts: on×43, closest×33, near×27,
then×19, between×16, above/below×10, with-X-on-it×9, avoid×3):

| relation in program | domain contribution |
|---|---|
| `on(E, floor)` | floor surface (already live) |
| `on(E, S)`, S a class | ground the exact support instances S, but inspect **every measured horizontal support plane** until that grounding is complete. The word S ranks semantic checks; it never excludes an unlabelled or differently named plane. |
| `above/below(E, A)` | wall/vertical band adjacent to each A instance |
| `near(E, A)` | disc of calibrated τ_near (+margin) around each A |
| `between(E, A, B)` | corridor region around segment AB |
| `with_on(S, X)` | surfaces of class S, each requiring its top enumerated for X |
| `closest(E, anchor)` | disc around anchor of radius r* = distance to current best candidate — **shrinks as candidates are found**, a geometric prune that lets closest-questions stop before the room is finished |
| `farthest(E, anchor)` | E's full default domain (bounded by mapped extent) |
| no stated relation | open-world physical domain: floor + walls + every measured horizontal support plane. No object→support table and no released-scene vocabulary participates in correctness or stopping. |
| `follow` legs | each leg's anchor is a refer-subdomain; `pass_between` adds a corridor; `avoid_between` adds a **forbidden region applied to all motion** from the moment its anchors ground |

**Frontier exploration stops being a separate phase.** A reachable unknown
component is a domain element iff it can contain the generic physical element
currently being sought (a target, an anchor, or a horizontal support patch).
Class names may order candidate visits, but cannot reject a component or make a
stop certificate fire. Thus an unentered room still forces entry without
assuming which objects or furniture names the hidden test contains.

## 3. One enumerator

Obligation priority, generated from domain state (replaces the current
resolve/hidden/frontier triple, which it strictly generalizes):

1. **ground-domain**: ground explicit support/relational anchors while keeping
   all physical support families open; semantic likelihood only orders work.
2. **enumerate-element**: visit the computed viewpoint set of each open
   surface/region; a tabletop's set is its ~1 m ring, a floor region's is any
   pose with line of sight within r_max(E), a wall band's is a frontal pose.
3. **resolve / corroborate**: existing node-level obligations, unchanged.
4. **extend-domain**: enter any unknown component that passes the fit-test for
   a domain-relevant class.

Every attempt discharges (cell-set retirement, as now); attempts and visual
audits stay separate currencies (displacement/new-cells evidence, as now).

**Per-surface recall backstop.** SAM's residual risk is small-object recall on
cluttered tabletops. When a surface is enumerated up close, one extra Qwen call
on the surface crop — "list every ⟨cup⟩ visible on THIS table" — cross-checks
SAM's instances on that surface only. This is the deterministic, per-element
version of the 2025 4th-place team's multi-VLM count aggregation, and it stays
inside the atomic-facts rule: it is a perception query about one surface, never
a room total. Disagreement spawns a resolve obligation, not a number.

## 4. One certificate

Publish-with-confidence when:
1. every domain element is terminal: enumerated(E) | unreachable (attempt
   ended stuck) | unobservable (viewpoint reached, cells still occluded);
2. every counted/selected node is evidence-adequate (≥2 independent poses or
   one ≥0.5-score view; ≥60 px facts; class settled; per-surface roll-call
   consistent);
3. graph stable: identity set unchanged since the last visual audit.

Time pressure never changes the rule — the deadline watchdog publishes the
best current answer regardless, with the certificate written as diagnostics.
The certificate is a claim about *discharged uncertainty*, not about complete
coverage, which the NBV literature shows is unprovable anyway.

## 5. Answer operators on top (unchanged in spirit)

- **count**: |matching nodes| once the domain is terminal.
- **refer**: filter → argmin/argmax over the (pruned) domain; then spend the
  saved time on the winner's **box** — a 60–90 s orbit for multi-view point
  fusion, because the 2 points are paid on 3D IoU, not on the choice. Marker
  published from the first plausible candidate onward.
- **follow**: leg automaton (the Lang2LTL fragment): ground leg k's anchor via
  the refer machinery time-boxed to `remaining/(legs left) − drive estimate`,
  emit region waypoints (annulus / corridor entry+exit), advance on trajectory
  entry, insert detour guides on violation; avoid regions are global no-go
  cells for exploration and execution alike; on time-box expiry emit the
  best-guess waypoint and continue — partial credit per constraint.

## 6. Build order

1. Surface Registry (floor + horizontal_support + wall planes, class binding,
   per-cell obs_range). ~1 module, reuses extract_planes + mask fusion.
2. Domain compiler table above (pure function of the program + registry).
3. Swap the obligation generator to domain state; keep discharge/audit logic.
4. Wall-plane footprints in `observe` (one parameter: pass the matched wall
   plane instead of floor when the mask's depth mode sits on a wall).
5. Per-surface roll-call call in the verify stage.
6. Refer operator + box-refinement phase (reuses everything above).
7. Leg automaton for follow (corridor/annulus waypoint synthesis + monitor).

Steps 1–5 close the numerical family completely (all 15 released forms).
Step 6 closes object reference (30 forms). Step 7 closes instruction
following (30 forms). Nothing in the released set falls outside the table in
§2, and the support-prior default covers unseen phrasings in the held-out
scenes.
