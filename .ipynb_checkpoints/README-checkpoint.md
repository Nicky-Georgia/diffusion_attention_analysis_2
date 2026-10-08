# Diffusion Attention Analysis — v2 experiment repository

This is the repository for the thesis project **attention mechanism analysis in text-to-image diffusion models**.

The repo is organized around **stage-separated YAML configs**. Each model has its own config for each stage. 

## Target models

| Model |Role|
|---|---|
| Stable Diffusion 1.5| U-Net
| SD3 Medium | Transformer 


## stages

1. **Capture**: generate images and cache internal states.
   - SD1.5: cross-attention maps, selected activation hooks, latents.
   - SD3 Medium: selected transformer-block activations.

2. **Attention localization / diagnostics**:
   - SD1.5: compare token attention maps with object masks.
   - SD3 Medium: report activation statistics by block and timestep.

3. **SAE training**:
   - SD1.5: main Top-K SAE.
   - SD3 Medium SAE.

4. **Concept dictionary**:
   - SD1.5: build mask-aligned SAE concept dictionaries.
   - SD3 Medium

5. **Causal interventions**:
   - SD1.5: map-level attention reweighting and SAE feature steering.
   - SD3 Medium: block-output ablation / scaling.


## Notebook execution

Use the notebooks in `notebooks/` for execution. They start configs through the same CLI and display live progress.



## Config layout

```text
configs/
  01_capture/
  02_attention_localization/
  03_sae_training/
  04_concept_dictionary/
  05_interventions/
```
