# SAM3 Localization Assistant

## Responsibility

SAM3 is a subordinate open-vocabulary localization tool. Qwen decides whether
to call it and authors short noun-phrase queries. SAM returns proposals; it does
not understand the room, decide semantic identity, count final objects, select
navigation actions, or determine completeness.

## When Qwen may call SAM

SAM is appropriate only when a specific visible candidate or support region is:

- too small to identify;
- visually ambiguous;
- partially occluded;
- a possible look-alike;
- difficult to localize precisely after Qwen's own visual audit.

SAM is not used to certify a clear Qwen answer, re-check the whole room, or
search a completely hidden region. Hidden evidence requires a new viewpoint.

## Qwen-authored request contract

```json
{
  "query": "short noun phrase such as dining chair or golden statue",
  "sector": "S6-S7 or all",
  "purpose": "which visible uncertainty this localization will resolve"
}
```

Up to six queries may be submitted. Queries longer than 120 characters or
missing a semantic purpose are rejected. Yes/no questions are discouraged
because SAM is a segmenter, not a conversational reasoner.

## SAM processing

For each query, the assistant:

1. Reuses one cached panorama vision embedding.
2. Computes query text features.
3. Produces instance masks, boxes, and similarity scores.
4. Filters proposals to the requested panorama sectors.
5. Clusters strongly overlapping boxes.
6. Saves a colored overlay and one contextual crop per cluster.
7. Returns proposal metadata with a warning that scores are not semantic truth.

The returned warning is:

```text
SAM scores are not semantic truth. Qwen must inspect crops, reject false
positives, and decide the next action.
```

## Qwen follow-up prompt after SAM

```text
You asked the SAM visual-localization assistant questions about the same
panorama. Image 0 is the marked full panorama. The remaining images are
contextual crops, one for each SAM cluster.

EXACT QUESTION:
{question}

YOUR PRE-SAM DECISION:
{decision}

SAM ASSISTANT RESULT:
{proposal metadata}

SAFE MOVEMENT VIEWPOINTS, if physical confirmation is still required:
{candidate_viewpoints}

SAM scores mean only text-mask similarity. They are not probabilities that an
object truly has the requested identity. Inspect the pixels and surrounding
context yourself. Merge overlapping prompt hits that refer to the same physical
object, reject look-alikes, and do not count a proposal merely because SAM
marked it.

Analyze each cluster exactly once. Never restart or repeat an inventory. Keep
reasoning under 900 words so that the required JSON is always completed.

Now decide one of:
- answer, only if the exact answer and completeness are supported;
- zoom, only for a materially different visible region not covered by the SAM
  crops;
- ask_sam, only for a materially different localization question;
- verify/explore if the necessary evidence requires another robot viewpoint.

Emit the complete investigator JSON schema, including hypotheses,
instance_ledger, structural_completeness, zoom_requests, sam_requests, counting
bounds, selected viewpoint, selected hypothesis, and action utility.
```

## Observed behavior

- SAM successfully localized the golden Buddha candidate when Qwen initially
  overlooked it.
- Direct SAM chair localization generated duplicate raw boxes, demonstrating why
  proposal count must not be treated as object count.
- After overlap clustering, the chair test contained six strong distinct chair
  proposals plus low-score fragments.
- The final dining-chair crop regression did not call SAM because Qwen's zoom
  audit resolved all six chairs independently.

