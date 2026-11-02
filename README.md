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


## SD3 causal interpretability: notebooks 11–14 (November 2026)

The notebook sequence is designed for DataSphere with CUDA and the existing \`container4\` storage link. Each notebook contains a **pilot/full** switch and writes to a new output directory. Never delete or overwrite artifacts from notebooks 05–10.

**Execution order**

1. \`notebooks/11_sd3_sae_validated_controls_and_activity.ipynb\`: exact mid-block residual interchange, validated nonzero active controls, sign-flip/permutation tests, activity and top-K coverage diagnostics. The repository runner supplements the previously distributed full notebook 11.
2. \`notebooks/12_sd3_sae_fidelity_rescue.ipynb\`: train TopK x2/k16, x4/k32, x8/k64, and temporal-conditioned x4/k32 (pilot subset first). An **entire image/capture sample** is held out; no per-token random leakage. Save checkpoint index and validation metrics.
3. \`notebooks/13_sd3_comparative_causal_mediation.ipynb\`: **re-discover each candidate SAE's feature IDs on the notebook 09 discovery split**; never reuse old feature IDs across models. Compare full/whole/error/semantic-K/active-control/sign-flip/permutation interventions and downstream probe.
4. \`notebooks/14_sd3_independent_confirmatory.ipynb\`: generate disjoint held-out source/destination prompts, freeze the manifest, re-discover on separate examples and evaluate on other prompt templates and seeds with label-cluster bootstrap.

**Required existing artifacts**

- Stage 01: \`outputs/sd3_medium_sae_circuit_150/01_capture/captures/\` for notebook 12 (step 13; steps 0,13,27 for the temporal model).
- Notebook 09: \`06_causal_mediation_v2/split_manifest.csv\`, and preferably its \`clean_cache/\` on \`container4\`.
- Historical mid SAE: \`03_sae/mid_block11_step13/sae.pt\` for notebook 11 and baseline comparison.

Notebook 13 requires \`07_sae_fidelity_rescue/checkpoint_index.json\` from notebook 12. Notebook 14 uses a selected checkpoint from the same index. Notebook 14 **does not reuse** notebook 09 source/destination prompts. Full mode requires SD3 inference for many new pair/seed/arm combinations; run pilot first and check available GPU space.

**Validation and dependencies**

Install PyTorch with the correct CUDA build, Diffusers, Transformers, Accelerate, Pandas, NumPy, Pillow, scikit-image, PyYAML and pytest in your DataSphere environment. Ensure access to the gated Stable Diffusion 3 Medium model and to Grounding DINO Tiny. Validate repository changes using:

\`\`\`bash
pip install -e .
pytest -q tests/test_causal_controls.py tests/test_sae_causal_v4.py
\`\`\`

The tests are CPU-only and do not download SD3. They verify source/destination algebra, nonzero controls, sample-wise splits, temporal checkpoint loading and prompt disjointness. **CPU tests do not establish end-to-end GPU correctness.** A zero-dose negative control is an explicit error. A positive object score without preserved image structure and downstream recovery must be interpreted cautiously.

Important: residual-patch sufficiency at \`transformer_blocks.11\` is **not** identical to isolating the joint-attention contribution. An attention-versus-MLP decomposition would require additional hooks and matched tests.
