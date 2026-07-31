# CMU VLN Challenge 2026: accuracy-first solution research and current-system review

Research date: 2026-07-30

## Executive verdict

The current direction is fundamentally right, but the current implementation is not yet pointed at the full scoring problem.

Keep these ideas:

- SAM 3 for high-recall, open-vocabulary instance proposals and masks.
- LiDAR and odometry as the authority for geometry.
- Persistent 3D instance identity across views.
- Qwen for question parsing, object attributes, semantic ambiguity, and high-level choice among geometrically valid actions.
- Query-aware exploration, especially visiting named support/anchor objects.

Change the center of the architecture:

- Build one persistent object-centric 3D scene graph and solve all three tasks from it.
- Express language as a small executable constraint program. Compute `on`, `near`, `between`, `closest`, `farthest`, ordering, and forbidden regions with geometry. Do not ask Qwen to perform metric reasoning or the final count.
- Use SAM 3 tracking and multi-view association instead of independent detections plus a fixed distance merge.
- Make active perception optimize expected information gain about the parsed query, not just unseen floor area.
- Implement instruction following first-class. It represents about 70.6% of the available score.

In one sentence: **SAM 3 + Qwen is a strong perception/reasoning pair, but the winning system is the deterministic 3D world model and constraint executor around them.**

## What the challenge actually rewards

The official specification gives each test scene one numerical question worth 1 point, two object-reference questions worth 2 points each, and two instruction-following questions worth 6 points each. That is 17 points per scene and 51 points over three hidden scenes:

| Task | Hidden-test points | Share |
|---|---:|---:|
| Numerical | 3 | 5.9% |
| Object reference | 12 | 23.5% |
| Instruction following | 36 | 70.6% |

The system restarts for every question, retains no previous map, and gets 10 minutes from system startup—including model loading, exploration, reasoning, and action. Only the 360° image, registered/sensor scans, terrain maps, and state estimate are allowed. Ground-truth semantics and traversable-area truth are not available. See the [official 2026 task and evaluation specification](https://github.com/Yuxin916/CMU-VLN-Challenge-2026#evaluation).

The released 75 questions are also very structured:

- 18/30 object-reference questions use `closest`; three more use `farthest/furthest`.
- 22/30 instruction questions use `near`, 12 use `between`, 19 use `then`, and three explicitly use `avoid`.
- 11/15 numerical questions use `on`; several use chained anchors such as “on the sofa under the pictures.”

This is closer to online 3D scene-graph construction plus constrained path planning than conventional end-to-end VLN.

The released instruction trajectories contain 155–2,015 sampled poses and are often much longer than the straight-line start-to-end distance. The evaluator scores the actual robot trajectory, order, achieved constraints, and forbidden-region violations—not merely whether the final object was reached. Endpoint-only navigation therefore cannot score reliably.

## Evidence from related systems

The strongest public evidence supports an explicit 3D world model:

- The 2025 challenge winner reported a visual-language autonomous agent with **scene-graph construction and multimodal frontier-exploration scoring**, and won both simulation and real-robot rounds with a reported 30% lead over second place. The announcement is high level rather than a reproducible paper, but its architecture is directly relevant. [HIT Shenzhen report](https://www.hitsz.edu.cn/news/2025/1107/c6a11217/page.htm)
- The public 2025 fourth-place system used YOLO-World/SAM, projected LiDAR, CLIP features, a hierarchical detection/object/keyframe/place/room/scene graph, geometric and semantic frontiers, contour exploration, multi-view object observations, and confidence aggregation across VLMs. [Project page](https://alvinjinsung.github.io/Vision-Language-Autonomy/)
- SORT3D, from researchers behind VLA-3D, combines 2D object captions with a deterministic spatial-reasoning toolbox and an LLM that invokes the tools sequentially. Its authors explicitly identify direct LLM coordinate reasoning as unreliable. On their VLA-3D subset, the GPT-4o version achieved 71.8% overall and 75.0% on hard statements; object captions materially improved view-dependent grounding. [SORT3D paper](https://arxiv.org/abs/2504.18684) and [code](https://github.com/nzantout/SORT3D)
- ConceptGraphs shows the correct mapping abstraction: fuse posed multi-view masks and visual descriptors into persistent 3D object nodes, then add inter-object relations for downstream planning. [ConceptGraphs](https://concept-graphs.github.io/)
- SG-Nav reports more than 10 percentage points of success-rate improvement on several zero-shot ObjectNav benchmarks from an online 3D scene graph and re-perception mechanism. [SG-Nav](https://papers.nips.cc/paper_files/paper/2024/hash/098491b37deebbe6c007e69815729e09-Abstract-Conference.html)
- Frontier-object maps improve exploration by maintaining frontiers, obstacle/exploration maps, and fine-grained 3D object information together. FOM-Nav reports large SR/SPL gains over VLFM on HM3D v2. [FOM-Nav](https://arxiv.org/abs/2512.01009)
- A June 2026 paper specifically about this challenge independently reports that a single-view VLM and random exploration failed, then moves to a semantic voxel map and structured world model. It reduced one mapping phase from 8:42 to 4:17 through exploration optimization. [Modular VLA framework](https://arxiv.org/abs/2606.31144)

The proposed model allocation is also supported by the model authors:

- SAM 3 accepts noun-phrase concepts and/or image exemplars, returns all matching instance masks with identities, and includes a memory-based video tracker. It should be used as a multi-view proposal/tracking engine, not only as an independent still-image detector. [Meta SAM 3](https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/)
- Qwen3-VL adds stronger fine-detail alignment, spatial perception, multi-image reasoning, grounding, and optional Thinking variants. These are useful for object attributes and ambiguous language, but do not replace calibrated 3D geometry. [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)
- VLA-3D defines exactly the relation family seen in this challenge: `above`, `below`, `closest`, `farthest`, `between`, `near`, `in`, and `on`. Its official generator code and Unity subset should be treated as the starting definition of the predicates. [VLA-3D repository](https://github.com/HaochenZ11/VLA-3D)

## Review of the current repository

### What is good

The comments and traces show real failure analysis rather than random prompt tuning. Several decisions should survive into production:

1. **Geometry is not delegated to the VLM.** `experiments/nav/project.py` and `run_question.py` establish camera/LiDAR projection and accumulate registered scans in the map frame.
2. **Instance state persists across views.** `scene_state.py` keeps accepted, rejected, and unresolved hypotheses and attempts voxel-overlap association.
3. **Masks, not boxes, select LiDAR points.** `mask_points()` range-gates points and reduces background leakage.
4. **The code recognized completeness as the hard part of counting.** `coverage.py` was added after a 2-versus-4 undercount caused by occlusion.
5. **It learned to preserve context.** `crop_for()` records a concrete towel experiment in which a context-rich crop changed Qwen from a wrong “rack” classification to the correct object.
6. **It targets named support objects.** `survey_reference()` and `candidate_viewpoints()` correctly infer that small objects “on a table” require approaching tables rather than generic room coverage.
7. **The traces retain evidence.** Captured images, prompts, raw generations, and scene states are exactly what an ablation/debugging harness needs.

The Japanese-room image is a useful example of the right division of labor: SAM 3 finds the three true calligraphy panels but also proposes the crane screen and figurative print. A semantic verifier is necessary, while their 3D positions and grouping should remain geometric.

### Blocking gaps

1. **Only one task family is implemented.** `run_question.py`, `run_sweep.py`, and the Qwen system prompt are counting-specific. `marker_pub.py` is a manual publisher, not an object-reference solver, and there is no language-constrained trajectory generator. Even perfect counting caps this approach at 3/51 hidden-test points.
2. **It is outside the deliverable.** The work lives under untracked `experiments/`; `ai_module` still contains the dummy C++ node. The scripts hard-code a host-side container name, use `docker exec`, copy helpers at runtime, and assume host model access. The rules say the submitted implementation must be under `ai_module`.
3. **The primary closed-loop script is currently API-inconsistent.** `run_question.py` calls `VLMAgent.judge_crop()` at lines 419, 643, and 653, but the current `agent.py` exposes `inspect_crop()` and no `judge_crop()`. Static compilation passes because Python does not check attributes, but the first judgeable candidate will fail at runtime.
4. **The current output evidence cannot support object-reference scoring.** In `q_state.json`, most hypotheses have only 1–4 LiDAR points and no box. The confirmed object has two points and `bbox: null`. Object reference is scored by overlap with a full 3D ground-truth box.
5. **Independent equirectangular detection is fragile.** SAM 3 is run directly on a 1920×640 panorama. Objects near the seam can split or duplicate, polar regions are distorted, and horizontal box width is converted to physical size as though every box were a normal perspective projection.
6. **Association is not robust enough.** With sparse points, the fallback merges anything within 0.4 m. That can merge adjacent pillows, bottles, monitors, or decorations. Conversely, partial masks from different views may fail voxel-overlap and duplicate an object.
7. **“Observed floor” is not “searched for the query.”** The coverage map marks known free cells as seen from 5 m through a 2D occupancy line of sight. It does not encode whether a wall, shelf, tabletop, or tiny target was imaged with sufficient pixels from a useful angle.
8. **Relations are still judged in images.** Asking Qwen “is this potted plant resting on a table?” produced contradictory classifications across nearby views. Once object and support boxes exist, `on` should be a geometric support relation; Qwen should only decide ambiguous class/attribute semantics.
9. **The final count can override the evidence.** `run_question.py` starts with `state.count()` and then replaces it with Qwen’s generated count. The traces show unsupported inferences and reversals. Counting unique scene-graph nodes that satisfy predicates should be deterministic.
10. **Confidence is uncalibrated.** Qwen’s self-reported 0.9/0.99 values are treated as probabilities, despite contradictions in the trace. They are not calibrated likelihoods.
11. **Planning success is not verified geometrically.** The logs include repeated `stuck` and `far_reports_goal_reached` events. A status string is not proof that the intended view was reached or that a required path relation was satisfied.
12. **The default four iterations are too rigid.** A hard iteration cap is unrelated to scene size, remaining ambiguity, or expected score gain.

### Direction assessment

The direction is approximately **70% correct at the architecture-idea level and far below submission readiness**. The perception/geometry split, persistent state, and active-view idea are good. The main correction is to stop extending the counting loop and instead promote the accumulated state into a reusable scene graph with a deterministic relation and path-constraint layer.

## Recommended end-to-end architecture

### 1. ROS episode manager

Create one ROS 2 node under `ai_module` with explicit states:

`LOAD -> WAIT_FOR_QUESTION -> PARSE -> MAP/EXPLORE -> VERIFY -> SOLVE -> EXECUTE/PUBLISH -> DONE`

Load weights as early as possible and measure startup-to-ready time. The official text says timing begins at system startup, so model loading is already inside the budget. Cache no scene data across restarts, but package weights locally unless simulation hardware/network is confirmed.

Synchronize or timestamp-associate image, registered scan, and pose. Record every episode to a compact replay format so the same sensor sequence can be tested without rerunning Unity.

### 2. Compile language to a constrained DSL

Use Qwen once to produce schema-validated JSON, then validate nouns and operators with deterministic code. Example:

```json
{
  "task": "instruction",
  "entities": {
    "p1": {"class": "potted plant", "selector": {"argmax_distance_from": "hookah"}},
    "cols": {"class": "column", "count": 2},
    "tray": {"class": "tray", "relations": [{"on": "table"}]}
  },
  "ordered_constraints": [
    {"near": "p1"},
    {"between": ["cols[0]", "cols[1]"]},
    {"stop_near": "tray"}
  ],
  "forbidden": []
}
```

Limit the operator vocabulary to the challenge’s known predicates and reject/repair malformed outputs. Qwen can interpret synonyms and nesting; code executes the result.

### 3. Panorama-aware, multi-view perception

At each selected keyframe:

1. Convert the equirectangular image into six or eight overlapping rectilinear views, retaining an exact pixel-to-ray transform. Include horizontal overlap and wrap-around so seam objects appear whole in at least one view.
2. Run SAM 3 with a prompt ensemble: the exact head noun, useful synonyms, and a broader superclass where recall is weak. Do not put `closest`, `between`, or support relations into the detector prompt.
3. Once a clean instance is found, use its crop as an optional SAM 3 visual exemplar. Use SAM 3’s tracker between nearby frames/keyframes.
4. Give Qwen an object-centric multi-view panel: the best two or three views, mask outline, and enough surroundings to identify support and attributes. Ask for atomic fields (`class`, `color`, `material`, `object description`, `support candidate`) rather than one compound yes/no verdict.
5. Retain all plausible hypotheses until multi-view evidence or geometry rejects them. Optimize recall first; exactness comes from scene-graph constraints.

For maximum stability, add a small metric image encoder such as SigLIP 2/DINO features for association and retrieval. SAM masks and Qwen text are not substitutes for a stable appearance embedding.

### 4. Persistent object-centric 3D scene graph

Each object node should store:

- stable ID and SAM track IDs;
- accumulated object points and robust oriented box;
- class probability distribution and aliases;
- color/shape/material attributes;
- visual embedding and best multi-view crops;
- observation poses, resolution, visibility, and uncertainty;
- support surface and relation edges;
- whether the object has been observed from enough angles.

Associate detections with a scored Hungarian assignment using track identity, 3D box/voxel overlap, center distance normalized by size, and appearance similarity. A fixed-radius first-match rule is insufficient.

For sparse or thin objects, do not require dense LiDAR hits:

- Fit local wall/table/support planes from accumulated LiDAR.
- Intersect mask pixel rays with the plane to reconstruct a dense metric mask/box.
- For small supported objects, infer depth from the support plane and nearby LiDAR, then accumulate multiple views.
- Use robust trimming before oriented-box fitting so a few background points do not inflate box dimensions.
- Learn class-conditioned size priors only as a fallback, using the released training object lists; never replace measured geometry when it is available.

This plane-intersection path is particularly important for pictures, windows, curtains, monitors, and tiny tabletop objects—the dominant classes in the released questions.

### 5. Deterministic spatial-relation toolbox

Port or adapt the official VLA-3D relation-generation functions and the SORT3D toolbox, then calibrate thresholds on the 15 released Unity scenes. At minimum implement:

- `on(A,B)`: A’s bottom is close to B’s top and A’s horizontal footprint overlaps B.
- `above/below`: vertical ordering plus the dataset’s horizontal-overlap rule.
- `near`: metric separation normalized by object/room scale, calibrated from VLA-3D.
- `closest/farthest`: exact argmin/argmax over every credible candidate of the requested class.
- `between(A,B,C)`: target lies within a calibrated corridor around segment BC and projects between its endpoints.
- `in`: target box is substantially enclosed by anchor box/region.

Evaluate nested descriptions recursively. “Pillows on the sofa under the pictures” first resolves sofas under pictures, then pillows supported by those sofas. Qwen chooses the symbolic parse; code evaluates it.

### 6. Query-conditioned active exploration

Maintain geometric frontiers, but score candidate viewpoints by expected reduction in answer uncertainty:

`score = new visible space + likely target/anchor discovery + unresolved-object information gain + expected pixel resolution - travel time - repeat/stuck risk`

Use the parsed query to focus exploration:

- `on table/cabinet/shelf/bed/sofa`: find all support instances first, then visit viewpoints around each support.
- `above/below window/picture`: inspect relevant wall sectors and their occluded sides.
- `closest/farthest`: completeness requires finding all target-class alternatives and the anchor, not merely one good match.
- `between`: find both anchors and inspect the corridor between them.
- Multi-room scenes: retain frontier probability until every reachable room/region has been entered or its target probability is negligible.

Stop only when a task-specific completeness criterion is met. For counting, every relevant support/region should be observed at sufficient resolution, all high-recall detections should be resolved, and remaining frontiers should have low expected probability of changing the answer.

### 7. Task solvers

#### Numerical

Evaluate the compiled predicate over unique scene-graph nodes and publish the integer. Qwen never performs the addition. Revisit a viewpoint only when a specific unresolved node could change the count.

#### Object reference

Resolve the unique target with the same predicate engine. Publish a continuously refreshed `Marker.CUBE` with the full oriented 3D box. Also publish the target waypoint if the 2026 evaluator expects the dummy-node behavior; this detail should be confirmed with organizers.

Optimize box IoU explicitly. A correct label with a two-point or centroid-only box will lose most of the two points.

#### Instruction following

Represent the command as an ordered finite-state machine. Convert semantic constraints to feasible free-space regions:

- `near object`: safe cells in a calibrated annulus around its footprint.
- `between A and B`: safe cells in the corridor connecting the two object footprints.
- `pass by`: a near-region waypoint that does not terminate the episode.
- `stop at object`: the safest reachable cell near the object, not its occupied center.
- `avoid between A and B`: mark the between-corridor as forbidden and force guide waypoints around one side.

Plan with A* or Dijkstra over `(free-space cell, constraint-state)`, so a route is valid only when constraints are satisfied in order. Send intermediate waypoints that force the base planner through required corridors or around forbidden ones. Monitor `/state_estimation`; advance the automaton only when the **actual trajectory** satisfies the predicate. If the base planner chooses a violating shortcut, replan with a stronger guide waypoint.

The public issue asking how instruction completion is signaled remains unanswered as of this research date. Do not optimize the tie-break bonus until the completion protocol is clarified. [Open organizer issue](https://github.com/Yuxin916/CMU-VLN-Challenge-2026/issues/3)

## How to use SAM 3 and Qwen

### SAM 3

Use it for:

- high-recall concept masks;
- image-exemplar refinement;
- video/keyframe tracking;
- precise mask boundaries for LiDAR/plane fusion.

Do not trust its concept score as final semantic truth. The existing crane-screen/calligraphy result is a clear counterexample.

### Qwen

Use it for:

- DSL parsing and synonym normalization;
- fine-grained object attributes from multi-view crops;
- resolving true semantic ambiguity after geometry has filtered candidates;
- ranking a short list of geometrically valid next-best views.

Do not use it for:

- metric coordinates or distances;
- instance identity;
- support/near/between decisions when geometry exists;
- counting;
- unconstrained waypoint generation;
- self-reported probability without calibration.

An 8B Instruct model is adequate for structured captioning and parsing if the surrounding system is strong. A larger/Thinking Qwen can improve rare semantic decisions, but it should be invoked sparsely. Accuracy per second matters because loading and inference consume the same 10-minute budget. On a 24 GB 4090, profile SAM 3 and Qwen together; sequential inference or model offload may be necessary. The simulation GPU/VRAM is still unspecified in the repository, so retain a smaller local fallback.

## Training and validation plan

### Data

1. Download all 15 released Unity scene packages and the Unity subset of VLA-3D, not only the three currently extracted scenes.
2. Render many allowed-sensor episodes with varied start poses, lighting, occlusions, and routes. Store panoramas, registered scans, pose, and terrain maps. Use ground truth only in the offline scorer.
3. Use VLA-3D’s official relation generator to label relation edges and to test the DSL executor.
4. Use the provided q4/q5 trajectories to fit a surrogate path-constraint evaluator and calibrate near/between/avoid regions.
5. Replay the provided real-robot bag to measure the sim-to-real loss before submission.

### Evaluation split

Use five-fold leave-three-scenes-out validation, matching the hidden-test scene count. Never calibrate relation thresholds or class priors on the validation fold.

### Metrics

Track each layer separately:

- 2D proposal recall at multiple pixel-size bands;
- 3D instance recall/precision and ID switches;
- 3D oriented-box IoU by class and point density;
- attribute accuracy and relation F1;
- numerical exact match;
- object-reference target accuracy and box IoU;
- instruction constraint completion, order accuracy, forbidden-region violations, path length, and wall-clock time;
- failure rate from model loading, ROS startup, stuck goals, and malformed generations.

### Required ablations

- panorama directly vs rectilinear multi-view;
- independent SAM detections vs SAM tracking;
- distance-only association vs geometry-plus-appearance assignment;
- Qwen compound verdict vs atomic attributes plus geometric relations;
- geometric frontier vs query-conditioned frontier;
- one view vs two/three object views;
- Qwen final answer vs deterministic executor;
- 8B Instruct vs the larger/Thinking model on only the ambiguous cases.

Run every released question multiple times from clean restarts. The current trace already reports six identical runs producing `1, 0, 1, 1, 2, 0`; variance is itself a failing metric even when mean accuracy looks acceptable.

## Implementation order for maximum score

1. **Submission-safe ROS skeleton and replay harness.** Move into `ai_module`; load models, subscribe to allowed topics, log timestamps, and publish all three output types.
2. **Panorama-to-perspective projection and synchronized sensor fusion.** Verify pixel-ray-LiDAR alignment quantitatively.
3. **Persistent 3D object nodes and robust box fitting.** This unlocks both counting and object reference.
4. **VLA-3D/SORT3D relation toolbox and DSL parser.** Validate on ground-truth object lists before adding perception noise.
5. **Object-reference end to end.** It is simpler than instruction following and exposes box/grounding defects early.
6. **Query-conditioned exploration and deterministic counting.** Replace the three experimental counting controllers with one scene-graph solver.
7. **Constraint-state path planner for instruction following.** Prioritize `near`, ordered `then`, `between`, `stop`, then `avoid`, matching released frequency.
8. **Multi-view Qwen verification and uncertainty-driven revisit.** Add only after the deterministic backbone is stable.
9. **Full cross-scene and real-bag validation, packaging, startup profiling, and failure recovery.**

Given the scoring, do not spend another week squeezing a few percent from counting while instruction following is absent. A partially correct instruction planner can earn partial points; a perfect numerical solver contributes only 5.9% of the hidden total.

## Organizer questions that materially block design

The existing `experiments/organizer_question.md` is directionally good. It should be posted, with two corrections/additions:

1. The README already answers model-load timing: timing starts immediately at system startup, so loading is counted.
2. Add questions about:
   - exact 3D box-overlap metric and whether orientation is included;
   - the tolerance/definition used for `near`, `between`, `avoid`, and ordered constraint completion;
   - instruction “done” signaling (already an unanswered public issue);
   - whether object-reference episodes require both marker and waypoint publication;
   - whether the AI container shares the simulation GPU and exact available VRAM;
   - outbound network availability and Docker image limits.

Until those answers arrive, build and score a local surrogate evaluator, publish outputs continuously/robustly, and make the local 8B path the guaranteed fallback.

## Bottom line

Do not throw away the experiments. Promote their best ideas into a production scene-graph system and remove the VLM from decisions that geometry can settle. The most accurate plausible approach is:

**query DSL -> query-conditioned exploration -> SAM 3 multi-view masks/tracks -> LiDAR-backed object-centric 3D scene graph -> deterministic VLA-3D/SORT3D relations -> Qwen semantic arbitration -> deterministic count/box/constraint-state trajectory.**

That design matches the structure of the benchmark, the scoring distribution, the public architecture of strong prior entrants, and the failure evidence in this repository.


ROS DOMAIN ID TO 0 if not already later hackathon