# Diffusion Attention Analysis — v2 experiment repository

This repository supports the thesis project **attention mechanism analysis in text-to-image diffusion models**.

The repo is organized around **stage-separated YAML configs**. Each model has its own config for each stage, while the current thesis extension is intentionally **SD3 Medium-only**.

## Target models

| Model | Role |
|---|---|
| Stable Diffusion 1.5 | legacy U-Net attention baseline |
| SD3 Medium | active transformer/MM-DiT experiment target |

## Stages

1. **Capture**: generate images and cache internal states.
   - SD1.5: cross-attention maps, selected activation hooks, latents.
   - SD3 Medium: full joint-attention probability maps from selected transformer attention blocks, plus selected image-token residual updates for SAE training.
2. **Attention localization / diagnostics**:
   - SD1.5: compare token attention maps with object masks.
   - SD3 Medium: evaluate saved full joint-attention maps through the `image queries → text keys` slice.
3. **SAE training**:
   - Top-K SAE over selected image-token residual updates.
4. **Concept dictionary**:
   - mask-aligned SAE feature ranking using CLIPSeg pseudo-masks.
5. **Causal interventions**:
   - SD3 Medium time-window block scaling and active-token SAE steering.

## Notebook execution

Use the notebooks in `notebooks/` for execution. They start configs through the same CLI and display live progress, tables, visualizations and generated-image grids.

```text
notebooks/05_sd3_medium_time_adaptive_sae_circuits.ipynb
```

## Config layout

```text
configs/
  01_capture/
  02_attention_localization/
  03_sae_training/
  04_concept_dictionary/
  05_interventions/
```

## SD3 Medium-only time-adaptive SAE circuit run

The current thesis experiment extension is implemented as an SD3 Medium-only notebook-first pipeline. The artifact decision is:

- **Full SD3 joint-attention probability maps** are captured for localization and circuit-edge diagnostics. They are saved as head-averaged full query-by-key matrices with token order `[image tokens, text/context tokens]`.
- **Image-token residual updates** are captured separately for Top-K SAE training, mask-aligned concept dictionaries, and SAE decoder-direction steering.

New SD3 configs:

```text
configs/01_capture/sd3_medium_sae_circuit_150.yaml
configs/02_attention_localization/sd3_medium_sae_circuit_masks_clipseg.yaml
configs/02_attention_localization/sd3_medium_sae_circuit_diagnostics.yaml
configs/03_sae_training/sd3_medium_sae_circuit_base.yaml
configs/04_concept_dictionary/sd3_medium_sae_circuit_base.yaml
configs/05_interventions/sd3_medium_time_adaptive_block_scaling.yaml
configs/05_interventions/sd3_medium_time_adaptive_sae_steering.yaml
```

The capture config sets `track_step_on_denoiser_pre_hook: true`, `strict_activation_count: true`, and `strict_attention_count: true`. For the default 150 prompts, 1 seed, 3 denoising steps and 3 selected transformer attention/residual locations, the expected counts are:

- **1350 residual activation files**;
- **1350 full joint-attention probability-map files**.

If any file is missing, the stage writes `activation_missing.json` or `attention_missing.json` and fails early instead of allowing downstream artifacts to mix old and new runs.

The DataSphere notebook targets `container4` for model artifacts, pip cache, temporary files, outputs and annotations. The storage estimator in `sd3_notebook_utils.estimate_sd3_storage()` gives a conservative estimate of about **217 GiB** and rounds the minimum allocation to **220 GiB**. A practical DataSphere allocation is **240 GiB** if available, because head-averaged full SD3 joint-attention maps alone take about **99 GiB** for this 150-prompt run.
