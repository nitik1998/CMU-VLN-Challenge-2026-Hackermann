**Title:** Simulation-round evaluation machine: GPU/VRAM, Docker image size ceiling, and network access

Hi — thanks for organizing this. We're building our AI module around locally-hosted
models (an open-vocabulary segmenter plus a vision-language model for reasoning), and
three details about the **simulation** evaluation environment materially change our
design. We'd rather size it correctly now than discover a mismatch at submission.

**1. GPU and VRAM on the simulation-evaluation machine**

FAQ #4 points to the [NUC13RNGi9](https://snuc.com/product/nuc13rngi9-full/) for the
evaluation machine. That platform ships with integrated Intel UHD Graphics 770 by
default and has a PCIe x16 slot with optional discrete cards (Arc A770, RTX 3060 Ti /
4070 / 4070 Ti, etc.), so the product page alone doesn't tell us what's actually
installed.

Could you confirm, for the **simulation** round:
- Which GPU is fitted (if any), and how much VRAM is available to the AI module container?
- Is that GPU shared with the Unity simulator and the autonomy stack, or does the AI
  module get it to itself?

This is the difference between running a VLM locally and having to restructure around
a hosted API. For reference, README L141 specifies the **real-robot** machine as
"16x i9 CPU cores, 32GB RAM, RTX 4090 GPU" — it's only the simulation round that's
unclear to us.

**2. Docker image size ceiling**

FAQ #4 says the limit "depends on the machine we use to run evaluation" without a
figure. A submission carrying model weights realistically lands in the 20–30 GB range
on top of the provided ROS base image. Is there a concrete cap (or a soft guideline)
we should stay under? If there's a pull-time limit as well, that would be useful to
know.

**3. Network access during evaluation**

FAQ #3 says online APIs are permitted provided we embed our access token. That implies
outbound internet from the evaluation container — could you confirm that's available in
the **simulation** round, and whether there are restrictions (allowed hosts, latency
expectations, proxy)? If the machine has limited VRAM, an API-backed model is our
fallback, so we'd like to know it's viable before committing to it.

**4. Minor, if convenient:** does the 10-minute per-question budget start at system
launch including our module's model-loading time, or from when the question is first
published on `/challenge_question`? Loading weights is a fixed cost we'd plan around
differently in each case.

Happy to move any of this to email if you'd prefer. Thanks!
