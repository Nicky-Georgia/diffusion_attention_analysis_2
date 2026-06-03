from __future__ import annotations

from typing import Any, Dict, Iterable, List


def _common_setup(cfg):
    from ..config import deep_get, output_dir, resolve_path
    from ..io_utils import ensure_dir, read_jsonl

    out_dir = output_dir(cfg)
    prompts_path = resolve_path(cfg, "data.prompts_path", required=True)
    max_prompts = deep_get(cfg, "data.max_prompts")
    prompts = read_jsonl(prompts_path, max_items=max_prompts)
    ensure_dir(out_dir / "samples")
    return out_dir, prompts


def _pipe_run_with_step_callback(cfg, pipe, prompt: str, seed: int, step_setter):
    import torch
    from ..config import deep_get
    from ..pipelines import generation_kwargs

    device = str(deep_get(cfg, "model.device", "cuda"))
    gen = torch.Generator(device=device if device == "cuda" and torch.cuda.is_available() else "cpu").manual_seed(int(seed))

    def callback_on_step_end(pipe_, step_index: int, timestep, callback_kwargs):
        step_setter(int(step_index))
        return callback_kwargs

    kwargs = generation_kwargs(cfg, prompt, gen)
    kwargs["callback_on_step_end"] = callback_on_step_end
    kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]
    try:
        return pipe(**kwargs)
    except TypeError:
        kwargs.pop("callback_on_step_end_tensor_inputs", None)
        return pipe(**kwargs)


def _map_reweight(cfg: Dict[str, Any], progress, *, dry_run: bool = False):
    from ..config import deep_get, output_dir
    from ..io_utils import ensure_dir, safe_name, save_json
    from ..text import get_token_map, prompt_targets
    from .common import dry_run_response

    logic = {
        "question": "Does changing target-token cross-attention at selected steps alter the intended image region?",
        "guardrail": "same-seed intervention outputs must be compared against baseline outputs and off-target drift later",
    }
    if dry_run:
        return dry_run_response(cfg, progress, logic_check=logic)

    import torch

    from ..attention import AttentionStore, install_unet_attention_processors, restore_attention_processors
    from ..interventions import SpatialTokenReweighter, quadrant_mask
    from ..pipelines import first_tokenizer, get_denoiser_module, load_pipeline

    out_dir, prompts = _common_setup(cfg)
    if not prompts:
        progress.skip(message="no intervention prompts")
        return {"status": "skipped", "reason": "no_prompts"}

    pipe = load_pipeline(cfg)
    denoiser = get_denoiser_module(pipe)
    tokenizer = first_tokenizer(pipe)
    int_cfg = cfg.get("intervention", {})
    strengths = [float(x) for x in int_cfg.get("strengths", [2.0])]
    target_steps = [int(x) for x in int_cfg.get("target_steps", [])]
    res = int_cfg.get("attention_resolution", [16, 16])
    qmask_resolution = (int(res[0]), int(res[1]))
    sample_root = ensure_dir(out_dir / "samples")
    capture_root = ensure_dir(out_dir / "captures")
    seed_start = int(deep_get(cfg, "runtime.seed_start", 0))
    num_seeds = int(deep_get(cfg, "runtime.num_seeds", 1))
    total = len(prompts) * num_seeds * len(strengths)
    progress.start(total=total, message="map intervention started")
    rows = []

    try:
        for p_idx, row in enumerate(prompts):
            prompt = str(row["prompt"])
            prompt_id = str(row.get("id", f"prompt_{p_idx:05d}"))
            target_word = str(row.get("target_word") or row.get("target") or (prompt_targets(row)[0] if prompt_targets(row) else ""))
            if not target_word:
                continue
            token_indices = get_token_map(tokenizer, prompt, [target_word]).get(target_word.lower(), [])
            if not token_indices:
                progress.skip(message="no target token indices", prompt_id=prompt_id, target_word=target_word)
                continue
            quadrant = str(row.get("quadrant") or int_cfg.get("quadrant", "top_left"))
            qmask = quadrant_mask(qmask_resolution[0], qmask_resolution[1], quadrant)
            for s in range(num_seeds):
                seed = seed_start + s
                for strength in strengths:
                    sample_id = f"{safe_name(prompt_id)}_seed_{seed:04d}_x{strength:g}"
                    current = {"step": -1}
                    store = AttentionStore(capture_root, save_headwise=False)
                    store.capture_steps = set(target_steps) if target_steps else None
                    store.begin_sample(sample_id, {"prompt": prompt, "seed": seed, "target_word": target_word, "token_indices": token_indices, "strength": strength})
                    intervention = SpatialTokenReweighter(
                        token_indices=token_indices,
                        query_mask=qmask,
                        multiplier=strength,
                        outside_multiplier=float(int_cfg.get("outside_multiplier", 1.0)),
                        target_layers=int_cfg.get("target_layer_fragments", []),
                        target_steps=target_steps,
                    )
                    original, installed = install_unet_attention_processors(
                        denoiser,
                        store,
                        layer_whitelist=deep_get(cfg, "capture.attention.layer_whitelist", int_cfg.get("target_layer_fragments", [])),
                        include_self=False,
                        include_cross=True,
                        intervention=intervention,
                    )
                    def set_step(i):
                        current["step"] = i
                        store.set_step(i)
                    try:
                        out = _pipe_run_with_step_callback(cfg, pipe, prompt, seed, set_step)
                    finally:
                        restore_attention_processors(denoiser, original)
                    sample_dir = ensure_dir(sample_root / sample_id)
                    out.images[0].save(sample_dir / "intervention.png")
                    meta = {"sample_id": sample_id, "prompt": prompt, "seed": seed, "target_word": target_word, "token_indices": token_indices, "strength": strength, "installed_layers": installed}
                    save_json(sample_dir / "metadata.json", meta)
                    rows.append(meta)
                    progress.update(message="map intervention generated", sample_id=sample_id)
    finally:
        pass
    report = {"status": "ok", "samples": len(rows), "kind": "map_reweight"}
    save_json(out_dir / "intervention_rows.json", rows)
    save_json(out_dir / "report.json", report)
    progress.end(message="map intervention finished", samples=len(rows))
    return report


def _sae_steering(cfg: Dict[str, Any], progress, *, dry_run: bool = False):
    from ..config import deep_get, resolve_path
    from ..io_utils import ensure_dir, safe_name, save_json
    from .common import dry_run_response

    logic = {
        "question": "Do SAE-discovered features causally steer generated content when modified at a specific denoising block/step?",
        "guardrail": "concept IDs should come from the concept dictionary, not arbitrary feature selection",
    }
    if dry_run:
        return dry_run_response(cfg, progress, logic_check=logic)

    from ..activations import remove_hooks, resolve_module_names
    from ..interventions import ResidualConceptSteerer
    from ..pipelines import get_denoiser_module, load_pipeline
    from ..sae import load_sae_checkpoint

    out_dir, prompts = _common_setup(cfg)
    sae_path = resolve_path(cfg, "data.sae_checkpoint")
    if sae_path is None or not sae_path.exists():
        report = {"status": "skipped", "reason": "missing_sae_checkpoint", "sae_checkpoint": str(sae_path)}
        save_json(out_dir / "report.json", report)
        progress.skip(message="missing SAE checkpoint", sae_checkpoint=str(sae_path))
        return report
    pipe = load_pipeline(cfg)
    denoiser = get_denoiser_module(pipe)
    int_cfg = cfg.get("intervention", {})
    module = str(int_cfg["module"])
    target_steps = [int(x) for x in int_cfg.get("target_steps", [])]
    concept_ids = [int(x) for x in int_cfg.get("concept_ids", [])]
    betas = [float(x) for x in int_cfg.get("betas", [1.0])]

    # ResidualConceptSteerer historically accepted these SD3 controls through
    # environment variables so notebooks could sweep them quickly. Mirror config
    # values into the same variables to make YAML-driven runs reproducible.
    import os

    steerer_env = {
        "branch_mode": "SD3_SAE_BRANCH_MODE",
        "direction_mode": "SD3_SAE_DIRECTION_MODE",
        "spatial_mode": "SD3_SAE_SPATIAL_MODE",
        "top_frac": "SD3_SAE_TOP_FRAC",
        "active_quantile": "SD3_SAE_ACTIVE_QUANTILE",
        "outside_scale": "SD3_SAE_OUTSIDE_SCALE",
        "normalize_direction": "SD3_SAE_NORMALIZE_DIRECTION",
        "beta_scale_mode": "SD3_SAE_BETA_SCALE_MODE",
    }
    for cfg_key, env_key in steerer_env.items():
        if cfg_key in int_cfg and int_cfg[cfg_key] is not None:
            os.environ[env_key] = str(int_cfg[cfg_key])

    seed_start = int(deep_get(cfg, "runtime.seed_start", 0))
    num_seeds = int(deep_get(cfg, "runtime.num_seeds", 1))
    device = str(deep_get(cfg, "model.device", "cuda"))
    sae = load_sae_checkpoint(sae_path, device=device)
    modules = dict(denoiser.named_modules())
    resolved = resolve_module_names(denoiser, [module])
    if not resolved:
        report = {"status": "skipped", "reason": "module_not_found", "module": module}
        save_json(out_dir / "report.json", report)
        progress.skip(message="module not found", module=module)
        return report
    current = {"step": -1}
    denoiser_call = {"idx": -1}

    def set_step(i):
        current["step"] = i

    def get_step():
        return current["step"]

    def _denoiser_step_pre_hook(module_, inputs):
        # Stage callbacks in Diffusers run after the denoiser forward, while
        # forward hooks on transformer blocks run inside it. Track the root
        # denoiser forward so intervention hooks see the actual step index.
        denoiser_call["idx"] += 1
        current["step"] = int(denoiser_call["idx"])

    step_tracker_handle = denoiser.register_forward_pre_hook(_denoiser_step_pre_hook)
    sample_root = ensure_dir(out_dir / "samples")
    total = len(prompts) * num_seeds * len(betas)
    progress.start(total=total, message="SAE steering started", resolved_modules=resolved)
    rows = []
    for p_idx, row in enumerate(prompts):
        prompt = str(row["prompt"])
        prompt_id = str(row.get("id", f"prompt_{p_idx:05d}"))
        for s in range(num_seeds):
            seed = seed_start + s
            for beta in betas:
                sample_id = f"{safe_name(prompt_id)}_seed_{seed:04d}_beta_{beta:g}"
                denoiser_call["idx"] = -1
                current["step"] = -1
                steerer = ResidualConceptSteerer(sae, concept_ids, beta, step_getter=get_step, active_steps=target_steps, device=device)
                handles = [modules[name].register_forward_hook(steerer) for name in resolved]
                try:
                    out = _pipe_run_with_step_callback(cfg, pipe, prompt, seed, set_step)
                finally:
                    remove_hooks(handles)
                sample_dir = ensure_dir(sample_root / sample_id)
                out.images[0].save(sample_dir / "steered.png")
                meta = {"sample_id": sample_id, "prompt": prompt, "seed": seed, "beta": beta, "concept_ids": concept_ids, "module": module, "resolved_modules": resolved}
                save_json(sample_dir / "metadata.json", meta)
                rows.append(meta)
                progress.update(message="SAE steering generated", sample_id=sample_id)
    remove_hooks([step_tracker_handle])
    report = {
        "status": "ok",
        "samples": len(rows),
        "kind": "sae_steering",
        "resolved_modules": resolved,
        "target_steps": target_steps,
        "concept_ids": concept_ids,
        "betas": betas,
        "steerer_config": {k: int_cfg.get(k) for k in steerer_env.keys() if k in int_cfg},
    }
    save_json(out_dir / "intervention_rows.json", rows)
    save_json(out_dir / "report.json", report)
    progress.end(message="SAE steering finished", samples=len(rows))
    return report


def _block_ablation(cfg: Dict[str, Any], progress, *, dry_run: bool = False):
    from ..config import deep_get
    from ..io_utils import ensure_dir, safe_name, save_json
    from .common import dry_run_response

    logic = {
        "question": "Do selected transformer blocks causally affect generation when their residual output is scaled?",
        "scope": "SD3 time-window causal diagnostic; this is a residual-stream intervention, not an attention-probability localization claim.",
    }
    if dry_run:
        return dry_run_response(cfg, progress, logic_check=logic)

    from ..activations import remove_hooks, resolve_module_names
    from ..interventions import BlockOutputScaler
    from ..pipelines import get_denoiser_module, load_pipeline

    out_dir, prompts = _common_setup(cfg)
    pipe = load_pipeline(cfg)
    denoiser = get_denoiser_module(pipe)
    int_cfg = cfg.get("intervention", {})
    requested = int_cfg.get("modules", [])
    resolved = resolve_module_names(denoiser, requested)
    if not resolved:
        report = {"status": "skipped", "reason": "modules_not_found", "modules": requested}
        save_json(out_dir / "report.json", report)
        progress.skip(message="modules not found", modules=requested)
        return report
    modules = dict(denoiser.named_modules())
    target_steps = [int(x) for x in int_cfg.get("target_steps", [])]
    scales = [float(x) for x in int_cfg.get("scales", [0.0])]
    seed_start = int(deep_get(cfg, "runtime.seed_start", 0))
    num_seeds = int(deep_get(cfg, "runtime.num_seeds", 1))
    current = {"step": -1}
    denoiser_call = {"idx": -1}

    def set_step(i):
        current["step"] = i

    def get_step():
        return current["step"]

    def _denoiser_step_pre_hook(module_, inputs):
        denoiser_call["idx"] += 1
        current["step"] = int(denoiser_call["idx"])

    step_tracker_handle = denoiser.register_forward_pre_hook(_denoiser_step_pre_hook)
    sample_root = ensure_dir(out_dir / "samples")
    total = len(prompts) * num_seeds * len(scales)
    progress.start(total=total, message="block ablation started", resolved_modules=resolved)
    rows = []
    for p_idx, row in enumerate(prompts):
        prompt = str(row["prompt"])
        prompt_id = str(row.get("id", f"prompt_{p_idx:05d}"))
        for s in range(num_seeds):
            seed = seed_start + s
            for scale in scales:
                sample_id = f"{safe_name(prompt_id)}_seed_{seed:04d}_scale_{scale:g}"
                denoiser_call["idx"] = -1
                current["step"] = -1
                scaler = BlockOutputScaler(scale, step_getter=get_step, active_steps=target_steps)
                handles = [modules[name].register_forward_hook(scaler) for name in resolved]
                try:
                    out = _pipe_run_with_step_callback(cfg, pipe, prompt, seed, set_step)
                finally:
                    remove_hooks(handles)
                sample_dir = ensure_dir(sample_root / sample_id)
                out.images[0].save(sample_dir / "ablation.png")
                meta = {"sample_id": sample_id, "prompt": prompt, "seed": seed, "scale": scale, "requested_modules": requested, "resolved_modules": resolved}
                save_json(sample_dir / "metadata.json", meta)
                rows.append(meta)
                progress.update(message="block ablation generated", sample_id=sample_id)
    remove_hooks([step_tracker_handle])
    report = {
        "status": "ok",
        "samples": len(rows),
        "kind": "transformer_block_ablation",
        "resolved_modules": resolved,
        "target_steps": target_steps,
        "scales": scales,
    }
    save_json(out_dir / "intervention_rows.json", rows)
    save_json(out_dir / "report.json", report)
    progress.end(message="block ablation finished", samples=len(rows))
    return report


def run(cfg: Dict[str, Any], progress, *, dry_run: bool = False) -> Dict[str, Any]:
    from .common import maybe_disabled
    disabled = maybe_disabled(cfg, progress)
    if disabled is not None:
        return disabled
    kind = str(cfg.get("intervention", {}).get("kind", ""))
    if kind == "map_reweight":
        return _map_reweight(cfg, progress, dry_run=dry_run)
    if kind == "sae_steering":
        return _sae_steering(cfg, progress, dry_run=dry_run)
    if kind == "transformer_block_ablation":
        return _block_ablation(cfg, progress, dry_run=dry_run)
    raise ValueError(f"Unsupported intervention kind: {kind}")
