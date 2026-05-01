from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict


def _ns(d: Dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(**(d or {}))


def resolve_dtype(dtype_name: str):
    import torch

    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    key = str(dtype_name or "float16").lower()
    if key not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return mapping[key]


def load_pipeline(cfg: Dict[str, Any]):
    """Load a Diffusers text-to-image pipeline from config.

    The model IDs are intentionally config-level values so the user can swap gated/local
    checkpoints without touching code.
    """
    import torch

    model_cfg = cfg.get("model", {})
    runtime_cfg = cfg.get("runtime", {})
    dtype = resolve_dtype(model_cfg.get("dtype", "float16"))
    pipeline_class = model_cfg.get("pipeline_class", "auto_t2i")
    model_id = model_cfg["model_id"]

    if pipeline_class == "auto_t2i":
        from diffusers import AutoPipelineForText2Image

        pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=dtype)
    elif pipeline_class == "sd3":
        from diffusers import StableDiffusion3Pipeline

        pipe = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=dtype)
    elif pipeline_class == "sana":
        from diffusers import SanaPipeline

        pipe = SanaPipeline.from_pretrained(model_id, torch_dtype=dtype)
    else:
        raise ValueError(f"Unsupported pipeline_class={pipeline_class!r}. Expected auto_t2i, sd3, or sana.")

    if runtime_cfg.get("enable_model_cpu_offload", False) and hasattr(pipe, "enable_model_cpu_offload"):
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(model_cfg.get("device", "cuda"))

    if runtime_cfg.get("enable_xformers_memory_efficient_attention", False) and hasattr(pipe, "enable_xformers_memory_efficient_attention"):
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception as exc:  # noqa: BLE001
            print(f"[warning] could not enable xformers: {exc}")

    if runtime_cfg.get("torch_compile", False):
        if hasattr(pipe, "unet"):
            pipe.unet = torch.compile(pipe.unet)
        if hasattr(pipe, "transformer"):
            pipe.transformer = torch.compile(pipe.transformer)
    return pipe


def get_denoiser_module(pipe):
    if hasattr(pipe, "unet"):
        return pipe.unet
    if hasattr(pipe, "transformer"):
        return pipe.transformer
    raise AttributeError("Pipeline has neither `.unet` nor `.transformer`.")


def first_tokenizer(pipe):
    for name in ("tokenizer", "tokenizer_2", "tokenizer_3"):
        tok = getattr(pipe, name, None)
        if tok is not None:
            return tok
    return None


def generation_kwargs(cfg: Dict[str, Any], prompt: str, generator):
    gen = cfg.get("generation", {})
    model = cfg.get("model", {})
    kwargs = {
        "prompt": prompt,
        "num_inference_steps": int(gen.get("num_inference_steps", 50)),
        "guidance_scale": float(gen.get("guidance_scale", 7.5)),
        "generator": generator,
    }
    for key in ("height", "width"):
        if gen.get(key) is not None:
            kwargs[key] = int(gen[key])
    neg = gen.get("negative_prompt", model.get("negative_prompt"))
    if neg is not None:
        kwargs["negative_prompt"] = neg
    return kwargs
