from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def run(cfg: Dict[str, Any], progress, *, dry_run: bool = False) -> Dict[str, Any]:
    from ..config import deep_get, output_dir, resolve_path
    from ..io_utils import ensure_dir, read_jsonl, safe_name, save_json
    from ..text import get_token_map, prompt_targets
    from .common import dry_run_response, maybe_disabled

    disabled = maybe_disabled(cfg, progress)
    if disabled is not None:
        return disabled

    logic = {
        "main_question": "Which internal states appear at early/middle/late denoising steps?",
        "unet_models": "record cross-attention maps, latents, and selected residual/activation updates",
        "transformer_models": "for SD3/MM-DiT record selected full joint-attention probability maps plus selected image-token residual updates",
    }
    if dry_run:
        return dry_run_response(cfg, progress, logic_check=logic)

    import torch

    from ..activations import ActivationStore, register_activation_hooks, remove_hooks
    from ..attention import AttentionStore, install_sd3_joint_attention_processors, install_unet_attention_processors, restore_attention_processors
    from ..pipelines import first_tokenizer, generation_kwargs, get_denoiser_module, load_pipeline

    out_dir = output_dir(cfg)
    prompts_path = resolve_path(cfg, "data.prompts_path", required=True)
    assert prompts_path is not None
    max_prompts = deep_get(cfg, "data.max_prompts")
    prompts = read_jsonl(prompts_path, max_items=max_prompts)
    if not prompts:
        progress.skip(message="no prompts found", prompts_path=str(prompts_path))
        return {"status": "skipped", "reason": "no_prompts"}

    capture_root = ensure_dir(out_dir / "captures")
    sample_root = ensure_dir(out_dir / "samples")
    latency_root = ensure_dir(out_dir / "latents")

    capture_steps = [int(x) for x in deep_get(cfg, "capture.step_indices", [])]
    save_latents = bool(deep_get(cfg, "capture.save_latents", False))
    attention_cfg = deep_get(cfg, "capture.attention", {}) or {}
    activation_modules = deep_get(cfg, "capture.activation_modules", []) or []

    pipe = load_pipeline(cfg)
    denoiser = get_denoiser_module(pipe)
    tokenizer = first_tokenizer(pipe)

    current_step = {"idx": -1}
    denoiser_call = {"idx": -1}

    def step_getter() -> int:
        return int(current_step["idx"])

    def _denoiser_step_pre_hook(module, inputs):
        # Diffusers exposes callback_on_step_end after the denoiser forward.
        # Activation hooks fire inside the denoiser forward, so relying only on
        # callback_on_step_end shifts capture by one step and can miss the final
        # timestep. Track the denoiser forward itself to align activations with
        # the actual denoising step.
        denoiser_call["idx"] += 1
        current_step["idx"] = int(denoiser_call["idx"])
        attn_store.set_step(int(current_step["idx"]))

    attn_store = AttentionStore(
        capture_root,
        save_headwise=bool(attention_cfg.get("save_headwise", False)),
        row_chunk_size=int(attention_cfg.get("row_chunk_size", 512)),
        save_dtype=str(attention_cfg.get("save_dtype", "float16")),
    )
    attn_store.capture_steps = set(capture_steps)
    original_processors = None
    installed_attention_layers = []
    if bool(attention_cfg.get("enabled", False)) and hasattr(denoiser, "attn_processors"):
        is_sd3_joint = str(cfg.get("model", {}).get("pipeline_class", "")).lower() == "sd3" or str(cfg.get("model", {}).get("family", "")).lower() == "transformer"
        if is_sd3_joint and str(attention_cfg.get("processor", "sd3_joint")).lower() in {"sd3_joint", "joint"}:
            original_processors, installed_attention_layers = install_sd3_joint_attention_processors(
                denoiser,
                attn_store,
                layer_whitelist=attention_cfg.get("layer_whitelist", []),
                include_attn2=bool(attention_cfg.get("include_attn2", False)),
            )
        else:
            original_processors, installed_attention_layers = install_unet_attention_processors(
                denoiser,
                attn_store,
                layer_whitelist=attention_cfg.get("layer_whitelist", []),
                include_self=bool(attention_cfg.get("include_self", False)),
                include_cross=bool(attention_cfg.get("include_cross", True)),
            )

    act_store = ActivationStore(capture_root)
    track_step_on_denoiser = bool(deep_get(cfg, "capture.track_step_on_denoiser_pre_hook", True))
    step_tracker_handle = denoiser.register_forward_pre_hook(_denoiser_step_pre_hook) if track_step_on_denoiser else None
    act_handles, resolved_activation_modules = register_activation_hooks(
        denoiser,
        activation_modules,
        act_store,
        step_getter=step_getter,
        capture_steps=capture_steps,
    ) if activation_modules else ([], [])

    total = len(prompts) * int(deep_get(cfg, "runtime.num_seeds", 1))
    progress.start(total=total, message="capture started", prompts=len(prompts), capture_steps=capture_steps)
    manifest = []
    seed_start = int(deep_get(cfg, "runtime.seed_start", 0))
    num_seeds = int(deep_get(cfg, "runtime.num_seeds", 1))
    device = str(deep_get(cfg, "model.device", "cuda"))

    try:
        for p_idx, row in enumerate(prompts):
            prompt = str(row["prompt"])
            prompt_id = str(row.get("id", f"prompt_{p_idx:05d}"))
            targets = prompt_targets(row)
            token_indices_by_word = get_token_map(tokenizer, prompt, targets)
            for s in range(num_seeds):
                seed = seed_start + s
                sample_id = f"{safe_name(prompt_id)}_seed_{seed:04d}"
                metadata = {
                    "sample_id": sample_id,
                    "prompt_id": prompt_id,
                    "prompt": prompt,
                    "prompt_record": row,
                    "seed": seed,
                    "model": cfg.get("model", {}),
                    "generation": cfg.get("generation", {}),
                    "capture_steps": capture_steps,
                    "token_indices_by_word": token_indices_by_word,
                }
                attn_store.begin_sample(sample_id, metadata)
                act_store.begin_sample(sample_id)
                denoiser_call["idx"] = -1
                current_step["idx"] = -1

                def callback_on_step_end(pipe_, step_index: int, timestep, callback_kwargs):
                    current_step["idx"] = int(step_index)
                    attn_store.set_step(int(step_index))
                    if save_latents and int(step_index) in set(capture_steps):
                        latents = callback_kwargs.get("latents")
                        if latents is not None:
                            latent_dir = ensure_dir(latency_root / sample_id)
                            torch.save({"step_index": int(step_index), "timestep": int(timestep) if hasattr(timestep, "item") else str(timestep), "latents": latents.detach().cpu().half()}, latent_dir / f"step_{int(step_index):03d}.pt")
                    return callback_kwargs

                generator = torch.Generator(device=device if device == "cuda" and torch.cuda.is_available() else "cpu").manual_seed(seed)
                kwargs = generation_kwargs(cfg, prompt, generator)
                kwargs["callback_on_step_end"] = callback_on_step_end
                kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]
                try:
                    output = pipe(**kwargs)
                except TypeError:
                    # Older Diffusers fallback: no callback_on_step_end_tensor_inputs support.
                    kwargs.pop("callback_on_step_end_tensor_inputs", None)
                    output = pipe(**kwargs)
                image = output.images[0]
                sample_dir = ensure_dir(sample_root / sample_id)
                image.save(sample_dir / "image.png")
                save_json(sample_dir / "metadata.json", metadata)
                manifest.append({"sample_id": sample_id, "prompt_id": prompt_id, "seed": seed, "image_path": str(sample_dir / "image.png")})
                progress.update(message="sample captured", sample_id=sample_id)
    finally:
        remove_hooks(act_handles)
        if step_tracker_handle is not None:
            remove_hooks([step_tracker_handle])
        if original_processors is not None:
            restore_attention_processors(denoiser, original_processors)

    from ..io_utils import module_to_filename

    activation_expected = len(manifest) * len(capture_steps) * len(resolved_activation_modules)
    activation_actual = 0
    activation_missing = []
    activation_counts = {}
    for item in manifest:
        sample_id = item["sample_id"]
        for step in capture_steps:
            step_key = f"step_{int(step):03d}"
            for module_name in resolved_activation_modules:
                rel = Path(sample_id) / "activations" / step_key / module_to_filename(module_name)
                exists = (capture_root / rel).exists()
                activation_counts.setdefault(module_name, {}).setdefault(str(step), 0)
                if exists:
                    activation_actual += 1
                    activation_counts[module_name][str(step)] += 1
                else:
                    activation_missing.append(str(rel))
    strict_activation_count = bool(deep_get(cfg, "capture.strict_activation_count", False))
    if strict_activation_count and activation_missing:
        save_json(out_dir / "activation_missing.json", {"missing": activation_missing[:1000], "missing_count": len(activation_missing)})
        raise RuntimeError(
            f"Missing {len(activation_missing)} activation files out of expected {activation_expected}. "
            f"See {out_dir / 'activation_missing.json'}"
        )

    attention_expected = len(manifest) * len(capture_steps) * len(installed_attention_layers) if installed_attention_layers else 0
    attention_actual = 0
    attention_missing = []
    attention_counts = {}
    for item in manifest:
        sample_id = item["sample_id"]
        for step in capture_steps:
            step_key = f"step_{int(step):03d}"
            for layer_name in installed_attention_layers:
                rel = Path(sample_id) / "attention" / step_key / module_to_filename(layer_name)
                exists = (capture_root / rel).exists()
                attention_counts.setdefault(layer_name, {}).setdefault(str(step), 0)
                if exists:
                    attention_actual += 1
                    attention_counts[layer_name][str(step)] += 1
                else:
                    attention_missing.append(str(rel))
    strict_attention_count = bool(deep_get(cfg, "capture.strict_attention_count", False))
    if strict_attention_count and attention_missing:
        save_json(out_dir / "attention_missing.json", {"missing": attention_missing[:1000], "missing_count": len(attention_missing)})
        raise RuntimeError(
            f"Missing {len(attention_missing)} attention files out of expected {attention_expected}. "
            f"See {out_dir / 'attention_missing.json'}"
        )

    report = {
        "status": "ok",
        "samples": len(manifest),
        "capture_steps": capture_steps,
        "installed_attention_layers": installed_attention_layers,
        "resolved_activation_modules": resolved_activation_modules,
        "output_dir": str(out_dir),
        "activation_expected_files": activation_expected,
        "activation_actual_files": activation_actual,
        "activation_missing_files": len(activation_missing),
        "activation_counts": activation_counts,
        "attention_expected_files": attention_expected,
        "attention_actual_files": attention_actual,
        "attention_missing_files": len(attention_missing),
        "attention_counts": attention_counts,
        "attention_save_headwise": bool(attention_cfg.get("save_headwise", False)),
        "attention_row_chunk_size": int(attention_cfg.get("row_chunk_size", 512)),
        "track_step_on_denoiser_pre_hook": track_step_on_denoiser,
    }
    save_json(out_dir / "manifest.json", {"samples": manifest})
    save_json(out_dir / "activation_missing.json", {"missing": activation_missing, "missing_count": len(activation_missing)})
    save_json(out_dir / "attention_missing.json", {"missing": attention_missing if 'attention_missing' in locals() else [], "missing_count": len(attention_missing) if 'attention_missing' in locals() else 0})
    save_json(out_dir / "report.json", report)
    progress.end(message="capture finished", samples=len(manifest))
    return report
