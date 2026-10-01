# SD3 causal mediation v2

This experiment is the next iteration after notebook 08. It is designed as causal model criticism rather than a search for a positive steering result.

## Motivation

Notebook 08 found weak natural-value SAE sufficiency despite stronger additive steering. Three confounders prevent interpreting that gap as a faithfulness result:

1. natural-value patching was performed at one anchor timestep, while successful steering acted over temporal windows;
2. the "full residual" positive control was RMS-clipped, so it was not a true causal ceiling;
3. target SAE features were selected primarily from mask-aligned dictionaries and were often reused/polysemantic.

The v2 design removes those confounders before scaling sample size.

## Hypotheses

**H1 — temporal mediation.** If a residual site carries object identity causally, source-to-destination full-residual interchange over the steering window should recover more target evidence than anchor-only interchange.

**H2 — counterfactual feature discovery.** SAE features discovered from minimal prompt counterfactuals on a discovery split should show stronger held-out sufficiency and necessity than mask-dictionary features and dose-matched non-target controls.

**H3 — SAE completeness.** Decompose the residual difference as

`x_s - x_d = (xhat_s - xhat_d) + [(x_s-xhat_s) - (x_d-xhat_d)]`

with `xhat = D(E(x))`. Intervening separately on the SAE reconstruction and reconstruction-error paths tests whether recoverable causal information lives in the SAE code or in information omitted by the SAE.

**H4 — steerability vs faithfulness.** Compare natural SAE-coordinate interchange with an additive decoder-direction intervention matched to the RMS energy of the natural intervention. A persistent gap cannot then be explained only by intervention magnitude.

## Data split

The 150-prompt SD3 set is split deterministically and globally without source-prompt reuse across labels:

- 3 discovery prompts per target label;
- 2 diagnostic evaluation prompts per target label;
- 2 disjoint confirmatory prompts per target label.

Target labels: dog, cat, car, book, bicycle, person.

Prompts where a label is the designated `target_word` are allocated before prompts where it is the second entity. The diagnostic run uses one seed. Only after the causal gates pass should the disjoint confirmatory split be run with seeds 0, 1, 2.

## Intervention sites

- early: `transformer_blocks.0`, anchor step 0, window 0–4;
- mid: `transformer_blocks.11`, anchor step 13, window 11–15;
- late: disabled by default; available as a later negative/control site.

Only the conditional residual branch is intervened on.

## Phase A

For every held-out pair and site:

- source clean;
- destination clean;
- exact full-residual interchange at the anchor;
- exact full-residual interchange over the full temporal window;
- counterfactually discovered SAE-coordinate reconstruction interchange at the anchor;
- the same SAE-coordinate interchange over the temporal window;
- mask-dictionary feature baseline;
- matched non-target feature control;
- reverse interchange: destination counterfactual feature values inserted into the source trajectory.

Full-residual replacement is exact and unclipped. SAE-coordinate interventions replace selected coordinates of the decoder reconstruction while preserving the current reconstruction residual; because the SAE is not perfectly invertible, these are not described as exact do-operations on encoder latents.

Controls are matched per pair/site on decoder-vector norm, observed latent-change energy, and reuse.

## Feature discovery

Feature discovery is performed only on the discovery split.

For each source/destination minimal-prompt pair:

1. capture clean residual updates over the site window;
2. encode both trajectories with the existing time-specific SAE;
3. compute source-minus-destination latent effects;
4. use a source-object CLIPSeg mask only for discovery/localization;
5. keep features with positive counterfactual effect, cross-pair sign consistency, and positive inside-vs-outside contrast;
6. rank by a paired t-like signal.

The held-out causal test therefore does not select features from the evaluation outcomes.

## Evaluation

Primary semantic outcome: Grounding DINO open-vocabulary object evidence.

Secondary outcome: CLIPSeg, retained for continuity with the thesis but not used as the primary causal evaluator.

Also record SSIM/L1 preservation, replacement-concept score, and unchanged-context drift.

Primary signed estimands include:

- full-window minus full-anchor effect;
- held-out CF-feature effect minus matched-control effect;
- CF-feature effect minus mask-dictionary feature effect;
- reverse-interchange necessity.

Inference uses prompt-clustered bootstrap CIs, paired sign-randomization tests, and Holm correction. Positive-only ratios and SSIM are not significance-tested.

## Positive-control gate

Phase B is run only if exact full-residual window interchange passes the preregistered site gate on valid pairs:

- source target score >= 0.15;
- source-minus-destination total effect >= 0.05;
- mean full-window recovery effect >= 0.02;
- aggregate recovery / total effect >= 0.10.

A failed full-residual gate means the site/window is not a sufficient causal handle for the claimed concept and SAE faithfulness should not be inferred there.

## Phase B

For sites that pass Phase A:

- broader counterfactual feature set;
- whole SAE-reconstruction interchange;
- SAE reconstruction-error interchange;
- SAE reconstruction excluding the selected counterfactual features;
- spatially permuted counterfactual-feature values;
- decoder-direction steering matched to natural-interchange RMS energy.

This distinguishes: semantic feature-selection failure, SAE-basis incompleteness, reconstruction-error mediation, spatial dependence, and steerability without natural causal mediation.

## Interpretation

Possible result profiles are deliberately separated:

- **full window > anchor and CF window > matched control:** temporally distributed sparse-feature mediation;
- **whole SAE reconstruction strong, selected CF features weak:** the SAE basis carries causal information but current semantic feature selection is incomplete/distributed;
- **reconstruction error strong:** causally important information is omitted by the current SAE representation;
- **matched-energy steering strong while natural interchange is weak:** controllability and natural causal faithfulness separate empirically;
- **full residual weak:** move to another block/window or multi-block path patching instead of making an SAE-faithfulness claim.

This validation target is different from TIDE-style reconstruction/interpretability/editing evidence: the central object is held-out counterfactual mediation and decomposition of the recoverable residual causal effect.


## Added validation: SAE fidelity across the temporal window

The early and middle SAEs are time-specific models trained at anchor steps 0 and 13. Reusing them over steps 0–4 and 11–15 is deliberate because it matches the temporal steering setup, but it introduces a possible distribution-shift confound. Before interpreting a weak window-level SAE mediation effect, the notebook therefore reports reconstruction cosine similarity, normalized RMSE, and active fraction at every step of the intervention window.

This diagnostic does not gate the full-residual H1 test, which is independent of the SAE. It qualifies H2–H4: if reconstruction fidelity collapses away from the anchor, a weak SAE-window effect can reflect temporal SAE mismatch rather than absence of a sparse causal mediator. In that case a temporal-aware SAE/TIDE-style representation becomes a follow-up hypothesis instead of being silently conflated with causal failure.

## Primary evaluator separation

Grounding DINO is the primary held-out semantic evaluator. CLIPSeg is used for discovery masks and as a secondary outcome only. This separation prevents the same vision model from both selecting sparse features and supplying the main causal endpoint. Target evidence and replacement evidence are retained separately so that apparent target recovery can be distinguished from simple destruction of the counterfactual replacement.
