from __future__ import annotations

from typing import Any, Dict


def run(cfg: Dict[str, Any], progress, *, dry_run: bool = False) -> Dict[str, Any]:
    from ..config import deep_get, output_dir, resolve_path
    from ..io_utils import save_json
    from .common import dry_run_response, list_activation_paths, maybe_disabled

    disabled = maybe_disabled(cfg, progress)
    if disabled is not None:
        return disabled

    logic = {
        "question": "Can dense denoising activations be decomposed into sparse reusable SAE features?",
        "main_path": "Train a Top-K SAE over the selected residual/image-token activations specified by layer and step.",
        "transformer_path": "For SD3 Medium, SAE training uses image-token residual updates; full joint-attention probability maps are captured and evaluated separately as localization/circuit-edge evidence.",
    }
    if dry_run:
        return dry_run_response(cfg, progress, logic_check=logic)

    from dataclasses import asdict

    from ..sae import TopKSparseAutoencoder, infer_d_model, save_sae_checkpoint, train_sae

    out_dir = output_dir(cfg)
    capture_dir = resolve_path(cfg, "data.capture_dir")
    if capture_dir is None or not capture_dir.exists():
        report = {"status": "skipped", "reason": "missing_capture_dir", "capture_dir": str(capture_dir)}
        save_json(out_dir / "report.json", report)
        progress.skip(message="missing capture dir", capture_dir=str(capture_dir))
        return report

    sae_cfg = cfg.get("sae", {})
    layer = str(sae_cfg["layer"])
    step = int(sae_cfg["step"])
    max_files = sae_cfg.get("max_activation_files")
    paths = list_activation_paths(capture_dir, layer=layer, step=step, max_files=max_files)
    if not paths:
        report = {"status": "skipped", "reason": "no_activation_files", "layer": layer, "step": step}
        save_json(out_dir / "report.json", report)
        progress.skip(message="no activation files", layer=layer, step=step)
        return report

    d_model = infer_d_model(paths)
    sae = TopKSparseAutoencoder(d_model=d_model, expansion_factor=int(sae_cfg.get("expansion_factor", 4)), k=int(sae_cfg.get("k", 20)))
    device = str(deep_get(cfg, "model.device", "cuda"))
    epochs = int(sae_cfg.get("num_epochs", 1))
    progress.start(total=epochs, message="SAE training started", activation_files=len(paths), d_model=d_model)
    history = train_sae(
        sae,
        paths,
        batch_size=int(sae_cfg.get("batch_size", 4096)),
        learning_rate=float(sae_cfg.get("learning_rate", 1e-4)),
        num_epochs=epochs,
        weight_decay=float(sae_cfg.get("weight_decay", 0.0)),
        device=device,
        max_tokens_per_file=sae_cfg.get("max_tokens_per_file"),
        progress=progress,
    )
    metadata = {
        "model": cfg.get("model", {}),
        "layer": layer,
        "step": step,
        "activation_files": len(paths),
        "d_model": d_model,
        "sae": sae_cfg,
        "claim_level": deep_get(cfg, "stage.claim_level", "unknown"),
    }
    ckpt_path = out_dir / "sae.pt"
    save_sae_checkpoint(ckpt_path, sae, metadata)
    history_rows = [asdict(x) for x in history]
    save_json(out_dir / "history.json", {"history": history_rows})
    report = {"status": "ok", "checkpoint": str(ckpt_path), "history": history_rows, "metadata": metadata}
    save_json(out_dir / "report.json", report)
    progress.end(message="SAE training finished", checkpoint=str(ckpt_path))
    return report
