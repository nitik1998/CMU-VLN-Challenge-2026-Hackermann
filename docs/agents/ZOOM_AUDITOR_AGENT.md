# Qwen Zoom Auditor Agent

## Responsibility

The zoom auditor re-examines pixels already present in the current panorama. It
does not move the robot and does not create new evidence. Cropping increases the
share of visual tokens allocated to a small or crowded region.

The audit is intentionally independent: it is not shown the investigator's
previous numeric answer, preventing confirmation bias. A separate text-only
reconciliation call then merges the new visual evidence with room-completeness
hypotheses.

## Crop tool contract

Qwen requests up to three semantic crops:

```json
{
  "sectors": "S5-S7",
  "vertical": "upper|middle|lower|full",
  "target": "complete dining table and surrounding chairs",
  "purpose": "separate overlapping chair instances"
}
```

The deterministic tool:

- finds the shortest circular panorama arc containing the requested sectors;
- correctly handles seam-crossing ranges such as `S11-S1`;
- adds context padding;
- applies a vertical band without changing image content;
- enlarges the crop with Lanczos interpolation for visual-token allocation;
- retains the full sector-labelled panorama as image 0;
- saves the crop and metadata for auditability;
- rejects repeated requests for the same sectors, band, and target.

## Independent visual-audit prompt

```text
Image 0 is the complete panorama for context. The remaining images are enlarged
crops from that same capture:
{image guide containing target, sectors, vertical band, and purpose}

EXACT QUESTION:
{question}

Perform an INDEPENDENT fresh visual audit. You are intentionally not shown any
previous numeric answer because it may be wrong. Count physical instances, not
boxes, parts, shadows, or repeated views of one instance. For a structured
group, trace around the complete anchor clockwise and give every distinct
instance a stable ID. State its position relative to the anchor and the visible
pixels that distinguish it from overlapping neighbors. Inspect all visible
sides and partially occluded backs or legs. Symmetry may direct attention but
may not create an instance unsupported by pixels.

Keep the audit under 700 words. End with exactly one JSON object:
{
  "visible_answer": <integer|string|null>,
  "instances": [
    {
      "id":"I1",
      "description":"<one physical instance>",
      "sector":"S#",
      "relation_to_anchor":"<side/position>",
      "distinguishing_pixels":"<how it is separate>",
      "confidence":"high|medium|low"
    }
  ],
  "anchor":"<support/group or none>",
  "visible_sides_audited":"<sides/regions>",
  "remaining_visible_ambiguity":"<specific ambiguity or none>",
  "crop_contains_complete_group":true,
  "confidence":0.0
}
```

## Evidence-reconciliation prompt

```text
Reconcile an independent high-resolution visual audit with the
active-perception state.

QUESTION: {question}

PRE-ZOOM DECISION (its numeric count may be wrong):
{previous_decision}

INDEPENDENT ZOOM AUDIT (newer visual evidence; prefer it for visible instances):
{zoom_audit}

SAFE MOVEMENT VIEWPOINTS:
{candidate_viewpoints}

Update the stable instance ledger from the independent audit. A zoom resolves
only visible overlap/resolution uncertainty; it does not resolve a genuinely
hidden region. Keep any still-credible unseen hypothesis. Choose answer, zoom,
ask_sam, verify, or explore using the normal action rules.

Return one JSON object only with the exact investigator decision keys. Use empty
arrays for unused requests. Every counted object must have exactly one unique
ledger entry.
```

## Verified dining-chair example

The panorama-only reasoning undercounted the table arrangement. The independent
zoom audit found six distinct instances and separated the overlapping chairs by
visible legs, seats, and backrests. The investigator then returned count `6`
with bounds `6-6` and six unique ledger entries.

