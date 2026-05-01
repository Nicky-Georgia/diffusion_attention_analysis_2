# Diffusion Attention Analysis — v2 experiment repository

This is the second, cleaner variant of the repository for the thesis project **attention mechanism analysis in text-to-image diffusion models**.

The repo is organized around **stage-separated YAML configs**. Each model has its own config for each stage. The only intentional exception is the comparison stage, where reports from several models are combined.

## Target models

| Model | Role | Pipeline depth |
|---|---|---|
| Stable Diffusion 1.5 | Main statistical baseline | Full pipeline |
| SDXL | Transfer check | Reduced full pipeline |
| SD3 Medium | Transformer architecture pilot | Shortened pipeline |
| SANA | Transformer architecture pilot | Shortened pipeline |

The transformer models are intentionally shortened. They use selected block-output hooks, small pilot SAEs, and block ablations. They do **not** claim full raw-attention localization unless a validated transformer attention-probability recorder is added later.

## Experiment stages

1. **Capture**: generate images and cache internal states.
   - SD1.5 / SDXL: cross-attention maps, selected activation hooks, latents.
   - SD3 Medium / SANA: selected transformer-block activations only.

2. **Attention localization / diagnostics**:
   - SD1.5 / SDXL: compare token attention maps with object masks.
   - SD3 Medium / SANA: report activation statistics by block and timestep.

3. **SAE training**:
   - SD1.5: main Top-K SAE.
   - SDXL: reduced transfer SAE.
   - SD3 Medium / SANA: very small pilot SAE only.

4. **Concept dictionary**:
   - SD1.5 / SDXL: build mask-aligned SAE concept dictionaries.
   - SD3 Medium / SANA: disabled by design in default configs.

5. **Causal interventions**:
   - SD1.5 / SDXL: map-level attention reweighting and SAE feature steering.
   - SD3 Medium / SANA: block-output ablation / scaling only.

6. **Comparison**:
   - UNet quantitative comparison: SD1.5 vs SDXL.
   - All-model summary: SD3 Medium and SANA are marked as `pilot_only`.

## Quick start

```bash
pip install -r requirements.txt
export PYTHONPATH=$PWD/src

# Check configs without loading any model
python -m diffusion_attention_analysis_v2.cli.run_stage --config configs/01_capture/sd15_full.yaml --dry-run
python -m diffusion_attention_analysis_v2.cli.run_suite --suite suites/smoke_test_dry_run.yaml --dry-run

# Example actual small GPU run
python -m diffusion_attention_analysis_v2.cli.run_stage \
  --config configs/01_capture/sd15_full.yaml \
  --override data.max_prompts=8 \
  --override runtime.num_seeds=1
```

## Notebook execution

Use the notebooks in `notebooks/` for cluster-friendly execution. They start configs through the same CLI and display live progress bars by parsing the generated JSON progress stream.

Recommended order:

1. `notebooks/00_environment_and_config_check.ipynb`
2. `notebooks/01_sd15_full_pipeline.ipynb`
3. `notebooks/02_sdxl_transfer_pipeline.ipynb`
4. `notebooks/03_transformer_short_pilots.ipynb`
5. `notebooks/04_comparison_and_reports.ipynb`

## Config layout

```text
configs/
  01_capture/
  02_attention_localization/
  03_sae_training/
  04_concept_dictionary/
  05_interventions/
  06_comparison/
```

## Main thesis-safe logic

The last stages are deliberately conservative:

- Attention heatmaps alone are not treated as explanations.
- SAE features must be checked against masks before being called concepts.
- Causal claims require intervention outputs and later off-target drift metrics.
- Transformer models are not forced into the UNet attention pipeline; they remain shortened architecture pilots.

## GPU recommendation

Preferred: **2×A100**. Minimum practical: **1×A100**. T4/V100 are acceptable only for smoke tests or SD1.5-only debugging.
