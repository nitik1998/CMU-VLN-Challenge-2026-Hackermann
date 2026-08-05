# Agent Architecture and Prompt Index

This directory documents the perception and reasoning agents built for the CMU
VLN Challenge pipeline. Each agent has a deliberately narrow responsibility so
that visual descriptions, semantic decisions, pixel localization, navigation,
and stopping criteria do not become conflated.

## Current pipeline

```text
360-degree ROS capture
        |
        v
Qwen Panorama Observer
        |
        v
Qwen Active-Perception Investigator
        |
        +--> Qwen Zoom Auditor (visible pixels need more resolution)
        |
        +--> SAM3 Localization Assistant (specific visible ambiguity)
        |
        +--> Safe geometric viewpoint + robot movement (evidence is hidden)
        |
        v
Final answer with evidence ledger and completeness checks
```

## Agent documents

- [Panorama Observer](PANORAMA_OBSERVER_AGENT.md): produces a verbose,
  sector-grounded semantic account of the complete panorama.
- [Active-Perception Investigator](ACTIVE_PERCEPTION_INVESTIGATOR_AGENT.md):
  determines answerability, manages hypotheses, chooses tools or movement, and
  enforces counting bounds and stopping consistency.
- [Zoom Auditor](ZOOM_AUDITOR_AGENT.md): independently re-examines visible
  regions at higher visual-token density without seeing the previous count.
- [SAM Localization Assistant](SAM_LOCALIZATION_ASSISTANT.md): returns masks,
  boxes, overlays, and crops for Qwen-authored noun-phrase queries; it never
  answers or navigates.

## Latest verified regression

Question: `How many dining table chairs are in the room?`

- Initial panorama reasoning undercounted the overlapping chairs.
- The structural gate rejected the answer.
- The investigator selected sectors around the dining table for zoom.
- The independent zoom audit identified six distinct chairs.
- The final investigator produced bounds `6-6`, six unique ledger entries, and
  answer `6`.
- The robot did not move and SAM was not consulted.

The implementation is primarily in
`experiments/nav/run_story_explorer.py`, with model execution in
`experiments/nav/agent.py` and SAM support in
`experiments/nav/sam_assistant.py`.

