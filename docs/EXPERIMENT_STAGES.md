# Experiment-stage logic review

This document is a logic check for the experiment pipeline, with special attention to the last stages where interpretability claims can easily overreach.

## Stage 1 — Capture

**Purpose:** build a reproducible cache of generated images, attention maps, hidden activations, and latents.

**SD1.5 / SDXL:** full UNet path. Cross-attention is stored only at selected early/middle/late denoising steps to avoid uncontrolled storage growth.

**SD3 Medium / SANA:** shortened transformer path. Only selected block activations are cached. Raw attention probability extraction is intentionally not part of the default pipeline because transformer attention internals differ across implementations and Diffusers versions.

## Stage 2 — Attention localization or transformer diagnostics

**SD1.5 / SDXL:** token-level cross-attention is evaluated against masks. Metrics are IoU, centroid distance, and entropy. Missing masks or token matches are reported separately.

**SD3 Medium / SANA:** no raw attention-localization claim. The stage summarizes activation norms and variance across selected blocks and timesteps.

## Stage 3 — SAE training

**Purpose:** decompose dense hidden activations into sparse features.

**SD1.5:** main SAE training.

**SDXL:** reduced transfer SAE.

**SD3 Medium / SANA:** small pilot SAE only. This checks whether sparse feature analysis is promising in transformer backbones but does not become the main claim.

## Stage 4 — Concept dictionary

This stage is run for SD1.5 and SDXL only. For each annotated object mask, the code ranks SAE features by inside-vs-outside activation contrast.

Transformer concept dictionaries are disabled in the default configs. This prevents the thesis from claiming transformer-level concept dictionaries before the transformer pipeline is validated.

## Stage 5 — Causal interventions

### UNet models

Two intervention types are used:

1. **Map-level reweighting:** boost or suppress a target token's cross-attention in a target spatial region.
2. **SAE feature steering:** alter selected SAE feature activations at a chosen timestep and module.

These are valid causal probes only when compared against same-seed baselines and measured for target success and off-target drift.

### Transformer models

Transformer models use only **block-output ablation/scaling**. This is a weaker but safer causal probe. It asks whether a selected block contributes to generation without pretending to localize token-level attention.

## Stage 6 — Comparison

**Allowed comparison:** SD1.5 full baseline vs SDXL transfer.

**Allowed summary:** all models, with SD3 Medium and SANA explicitly marked as `pilot_only`.

**Not allowed in default pipeline:** direct raw-attention localization comparison between UNet and transformer models unless transformer attention capture is later validated.
