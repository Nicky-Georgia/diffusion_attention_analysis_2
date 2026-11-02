from __future__ import annotations

import ast
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import structural_similarity

from .activations import extract_first_tensor, flatten_feature_map, unflatten_feature_map, replace_first_tensor, remove_hooks
from .causal_controls import active_nonsemantic_features, match_dose, signflip_spatial_control
from .io_utils import module_to_filename
from .pipelines import generation_kwargs, get_denoiser_module, load_pipeline
from .sae_fidelity_v4 import encode_at, load_research_sae

MID = "transformer_blocks.11"
PROBE = "transformer_blocks.23"
STEPS = (11, 12, 13, 14, 15)
PROBE_STEP = 27
ARMS = ("full", "whole_sae", "error", "semantic_4", "semantic_16", "semantic_32",
        "active_control_16", "signflip_16", "permutation_16")


def _tensor_path(root, pair_id, seed, role, module, step):
    return Path(root) / "clean_cache" / str(pair_id) / f"seed_{seed:04d}" / role / module_to_filename(module).removesuffix(".pt") / f"step_{step:03d}.pt"


def _image_path(root, pair_id, seed, role):
    return Path(root) / "clean_cache" / str(pair_id) / f"seed_{seed:04d}" / role / "image.png"


def tensor_ref(roots, pair_id, seed, role, module, step):
    for root in roots:
        path = _tensor_path(root, pair_id, seed, role, module, step)
        if path.is_file():
            d = torch.load(path, map_location="cpu", weights_only=False)["features"]
            x = d.float().reshape(-1, d.shape[-1])
            if x.shape != (4096, 1536):
                raise ValueError(f"Expected conditional image tokens [4096,1536] at {path}; got {x.shape}")
            return x
    raise FileNotFoundError(f"No cached activation for {pair_id}/{seed}/{role}/{module}/{step}")


def image_ref(roots, pair_id, seed, role):
    for root in roots:
        path = _image_path(root, pair_id, seed, role)
        if path.is_file():
            return path
    raise FileNotFoundError(f"No cached image for {pair_id}/{seed}/{role}")


def _conditional_tensor(update):
    flat, shape = flatten_feature_map(update.detach().float())
    if len(shape) != 3:
        raise ValueError(f"Expected B,N,D: {shape}")
    b, n, d = map(int, shape)
    if b not in (1, 2) or n != 4096 or d != 1536:
        raise ValueError(f"Unexpected SD3 activation dimensions: {shape}")
    x = flat.reshape(b, n, d)
    return x[b//2:].reshape(4096, 1536), shape


def run_sd3(pipe, cfg, prompt, seed, install=None):
    denoiser = get_denoiser_module(pipe)
    tick = {"value": -1}
    pre = denoiser.register_forward_pre_hook(lambda *args: tick.__setitem__("value", tick["value"] + 1))
    hooks = []
    try:
        if install is not None:
            hooks = list(install(denoiser, lambda: tick["value"]) or [])
        device = "cuda" if torch.cuda.is_available() else "cpu"
        g = torch.Generator(device=device).manual_seed(int(seed))
        with torch.inference_mode():
            img = pipe(**generation_kwargs(cfg, prompt, g)).images[0]
        if tick["value"] != int(cfg["generation"]["num_inference_steps"]) - 1:
            raise RuntimeError("Denoising step counter mismatch")
        return img
    finally:
        remove_hooks(hooks + [pre])


def capture_clean(pipe, cfg, root, pair, seed, role):
    pid = str(pair["pair_id"])
    target_image = _image_path(root, pid, seed, role)
    required = [(MID, s) for s in STEPS] + [(PROBE, PROBE_STEP)]
    if target_image.exists() and all(_tensor_path(root, pid, seed, role, m, s).is_file() for m, s in required):
        return
    cache = {}
    def install(denoiser, get_step):
        modules = dict(denoiser.named_modules())
        handles = []
        for name, steps in ((MID, STEPS), (PROBE, (PROBE_STEP,))):
            if name not in modules:
                raise KeyError(f"Module {name} missing in model")
            def make_hook(name, steps):
                def hook(module, inputs, output):
                    step = get_step()
                    if step not in steps:
                        return output
                    xi, xo = extract_first_tensor(inputs), extract_first_tensor(output)
                    if xo is None:
                        raise RuntimeError("Missing image token stream")
                    delta = xo - xi if xi is not None and xi.shape == xo.shape else xo
                    x, _ = _conditional_tensor(delta)
                    cache[(name, step)] = x.cpu().to(torch.float16)
                    return output
                return hook
            handles.append(modules[name].register_forward_hook(make_hook(name, steps)))
        return handles
    prompt = str(pair[f"{role}_prompt"])
    img = run_sd3(pipe, cfg, prompt, seed, install)
    if set(cache) != set(required):
        raise RuntimeError(f"Capture incomplete for {pid}/{role}: {set(required)-set(cache)}")
    for (module, step), data in cache.items():
        path = _tensor_path(root, pid, seed, role, module, step)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"features": data}, path)
    target_image.parent.mkdir(parents=True, exist_ok=True)
    img.save(target_image)


def feature_rankings(sae, discovery, roots, output, step=13, seed=0, sample_tokens=256):
    """Discover features exclusively from discovery pairs; no evaluation image scores."""
    labels = sorted(discovery.label.unique())
    norms = sae.decoder.weight.detach().float().norm(dim=0).cpu().numpy()
    score_by_label = {}
    for label in labels:
        deltas = []
        for _, pair in discovery[discovery.label == label].iterrows():
            src = tensor_ref(roots, pair.pair_id, seed, "source", MID, step)
            dst = tensor_ref(roots, pair.pair_id, seed, "destination", MID, step)
            rg = torch.Generator().manual_seed(2601 + int(hashlib.sha256(str(pair.pair_id).encode()).hexdigest()[:7], 16))
            idx = torch.randperm(len(src), generator=rg)[:sample_tokens]
            with torch.no_grad():
                device = next(sae.parameters()).device
                zs = encode_at(sae, src[idx].to(device), step).float().cpu()
                zd = encode_at(sae, dst[idx].to(device), step).float().cpu()
            # Natural counterfactual magnitudes weighted by decoder norm, rather than evaluator-induced scores.
            deltas.append((zs - zd).abs().mean(dim=0).numpy() * norms)
        score_by_label[label] = np.mean(deltas, axis=0)
    scores = np.stack([score_by_label[label] for label in labels])
    ranks = {}
    for i, label in enumerate(labels):
        other = np.max(np.delete(scores, i, axis=0), axis=0)
        specificity = scores[i] / (other + 1e-8)
        priority = scores[i] * np.minimum(specificity, 10.)
        order = np.argsort(-priority, kind="stable").astype(int).tolist()
        ranks[label] = order
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "frozen_feature_ranks.json").write_text(json.dumps(ranks, indent=2))
    pd.DataFrame([{"label": y, "feature_id": int(fid), "rank": j + 1, "raw_energy": float(score_by_label[y][fid])}
                  for y, order in ranks.items() for j, fid in enumerate(order[:64])]).to_csv(output / "discovery_features.csv", index=False)
    return ranks


@torch.no_grad()
def sae_delta(sae, destination, source, step, ids=None):
    device = next(sae.parameters()).device
    output, whole, error = [], [], []
    for xd_cpu, xs_cpu in zip(destination.split(256), source.split(256)):
        xd, xs = xd_cpu.to(device), xs_cpu.to(device)
        zd, zs = encode_at(sae, xd, step), encode_at(sae, xs, step)
        hd, hs = sae.decode(zd), sae.decode(zs)
        whole.append((hs-hd).float().cpu())
        error.append(((xs-hs)-(xd-hd)).float().cpu())
        if ids is not None:
            output.append(((zs[:, ids]-zd[:, ids]) @ sae.decoder.weight[:, ids].T).float().cpu())
    return (torch.cat(output) if ids is not None else None, torch.cat(whole), torch.cat(error))


def make_intervention_delta(sae, destination, source, *, mode, semantic_ids, step, seed):
    full = source - destination
    if mode == "full":
        return full
    if mode == "identity":
        return torch.zeros_like(full)
    if mode in ("whole_sae", "error"):
        _, whole, error = sae_delta(sae, destination, source, step)
        return whole if mode == "whole_sae" else error
    ids = semantic_ids
    if mode == "active_control_16":
        device = next(sae.parameters()).device
        with torch.no_grad():
            zs = torch.cat([encode_at(sae, x.to(device), step).float().cpu() for x in source.split(256)])
            zd = torch.cat([encode_at(sae, x.to(device), step).float().cpu() for x in destination.split(256)])
        ids = active_nonsemantic_features(zs, zd, sae.decoder.weight.cpu(), forbidden=semantic_ids, count=len(semantic_ids))
    delta, _, _ = sae_delta(sae, destination, source, step, ids=ids)
    if mode == "active_control_16":
        reference, _, _ = sae_delta(sae, destination, source, step, ids=semantic_ids)
        delta = match_dose(delta, reference)
    elif mode == "signflip_16":
        delta = signflip_spatial_control(delta, seed=seed, step=step)
    elif mode == "permutation_16":
        g = torch.Generator().manual_seed(2601 + seed * 41 + step)
        delta = delta[torch.randperm(len(delta), generator=g)]
    elif not mode.startswith("semantic_"):
        raise ValueError(f"Unknown intervention {mode}")
    if not torch.isfinite(delta).all() or delta.square().mean() < 1e-14:
        raise RuntimeError(f"Zero or invalid intervention: {mode}")
    return delta


def intervention_image(pipe, cfg, source_roots, pair, seed, model, ranks, arm):
    pid = str(pair["pair_id"])
    size = int(arm.rsplit("_", 1)[-1]) if arm.rsplit("_", 1)[-1].isdigit() else 16
    ids = ranks[str(pair["label"])][:size]
    trace = {"applied": [], "probe": None}
    ref_by_step = {s: tensor_ref(source_roots, pid, seed, "source", MID, s) for s in STEPS}
    def install(denoiser, get_step):
        mods = dict(denoiser.named_modules())
        def mid_hook(module, inputs, output):
            step = get_step()
            if step not in STEPS:
                return output
            xi, xo = extract_first_tensor(inputs), extract_first_tensor(output)
            if xo is None:
                raise RuntimeError("Cannot patch missing image-token output")
            residual = xi is not None and xi.shape == xo.shape
            original = xo - xi if residual else xo
            _, shape = _conditional_tensor(original)
            flat, _ = flatten_feature_map(original.detach().float())
            b, n, d = map(int, shape)
            tokens = flat.reshape(b, n, d)
            dest = tokens[b//2:].reshape(4096, 1536).cpu()
            delta = make_intervention_delta(model, dest, ref_by_step[step], mode=arm, semantic_ids=ids, step=step, seed=seed)
            full = ref_by_step[step] - dest
            trace["applied"].append({"step": step, "dose": float(delta.square().mean().sqrt()),
                                     "full_dose": float(full.square().mean().sqrt())})
            tokens[b//2:] = (dest + delta).to(tokens.device, dtype=tokens.dtype).reshape_as(tokens[b//2:])
            new = unflatten_feature_map(tokens.reshape(-1, d), shape, like=xo)
            updated = xi + new if residual else new
            return replace_first_tensor(output, updated, xo.shape)
        def late_hook(module, inputs, output):
            if get_step() == PROBE_STEP:
                xi, xo = extract_first_tensor(inputs), extract_first_tensor(output)
                if xo is not None:
                    delta = xo - xi if xi is not None and xi.shape == xo.shape else xo
                    trace["probe"] = _conditional_tensor(delta)[0].cpu()
            return output
        return [mods[MID].register_forward_hook(mid_hook), mods[PROBE].register_forward_hook(late_hook)]
    image = run_sd3(pipe, cfg, str(pair["destination_prompt"]), seed, install)
    if len(trace["applied"]) != len(STEPS) or trace["probe"] is None:
        raise RuntimeError("Incomplete intervention/probe hooks")
    src = tensor_ref(source_roots, pid, seed, "source", PROBE, PROBE_STEP)
    dst = tensor_ref(source_roots, pid, seed, "destination", PROBE, PROBE_STEP)
    r0 = (dst-src).square().mean().sqrt().item()
    r1 = (trace["probe"]-src).square().mean().sqrt().item()
    info = {"probe_recovery": float(1-r1/(r0+1e-8)),
            "mean_dose": float(np.mean([x["dose"] for x in trace["applied"]])),
            "dose_vs_full": float(np.mean([x["dose"]/(x["full_dose"]+1e-8) for x in trace["applied"]]))}
    return image, info


class DINO:
    def __init__(self, model_id="IDEA-Research/grounding-dino-tiny", threshold=.05):
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to("cuda").eval()
        self.threshold = threshold

    @torch.no_grad()
    def score(self, image_path, label):
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, text=str(label) + ".", return_tensors="pt")
        inputs = {k: v.to("cuda") if hasattr(v, "to") else v for k, v in inputs.items()}
        output = self.model(**inputs)
        kwargs = {"outputs": output, "target_sizes": torch.tensor([[image.height, image.width]], device="cuda"), "text_threshold": self.threshold}
        if "input_ids" in inputs:
            kwargs["input_ids"] = inputs["input_ids"]
        try:
            found = self.processor.post_process_grounded_object_detection(**kwargs, threshold=self.threshold)[0]
        except TypeError:
            found = self.processor.post_process_grounded_object_detection(**kwargs, box_threshold=self.threshold)[0]
        scores = found.get("scores", [])
        return float(scores.detach().float().max()) if len(scores) else 0.


def _ssim(reference, modified):
    a = np.asarray(Image.open(reference).convert("RGB").resize((256, 256)), dtype=np.float32)/255.
    b = np.asarray(Image.open(modified).convert("RGB").resize((256, 256)), dtype=np.float32)/255.
    return float(structural_similarity(a, b, data_range=1.0, channel_axis=2))


def score_campaign(result_dir, cases, reference_roots, *, evaluator=None):
    result_dir = Path(result_dir)
    evaluator = evaluator or DINO()
    cache = {}
    def score(path, label):
        key = (str(path), label)
        if key not in cache:
            cache[key] = evaluator.score(path, label)
        return cache[key]
    rows = []
    for case in cases:
        pair_id, seed, label, replacement = case["pair_id"], int(case["seed"]), case["label"], case["replacement_label"]
        context = case.get("context_label") or ""
        if not isinstance(context, str):
            context = ""
        src = image_ref(reference_roots, pair_id, seed, "source")
        dst = image_ref(reference_roots, pair_id, seed, "destination")
        paths = {"source": src, "destination": dst}
        paths.update({a: result_dir/"images"/str(pair_id)/f"seed_{seed:04d}"/(a+".png") for a in ARMS})
        for arm, path in paths.items():
            if not path.is_file():
                raise FileNotFoundError(path)
            scores = {"target": score(path, label), "replacement": score(path, replacement),
                      "context": score(path, context) if context else np.nan}
            rows.append({"pair_id": pair_id, "seed": seed, "label": label, "arm": arm, **scores,
                         "margin": scores["target"]-scores["replacement"],
                         "ssim": _ssim(dst, path)})
    df = pd.DataFrame(rows)
    df.to_csv(result_dir/"scores.csv", index=False)
    pairs = []
    for (pair_id, seed), sub in df.groupby(["pair_id", "seed"]):
        m = sub.set_index("arm")
        src, dst, full = m.loc["source"], m.loc["destination"], m.loc["full"]
        valid = bool(src.target >= .15 and src.margin-dst.margin >= .10 and full.margin-dst.margin >= .02)
        for arm in ARMS:
            v = m.loc[arm]
            pairs.append({"pair_id": pair_id, "seed": seed, "label": dst.label, "arm": arm,
                          "site_valid": valid, "margin_effect": v.margin-dst.margin,
                          "target_effect": v.target-dst.target,
                          "replacement_suppression": dst.replacement-v.replacement,
                          "context_shift": v.context-dst.context,
                          "ssim": v.ssim})
    effects = pd.DataFrame(pairs)
    effects.to_csv(result_dir/"paired_effects.csv", index=False)
    summary = effects.groupby("arm", as_index=False).agg(n=("margin_effect", "size"), margin=("margin_effect", "mean"),
            target=("target_effect", "mean"), replacement=("replacement_suppression", "mean"),
            context=("context_shift", "mean"), ssim=("ssim", "mean"))
    summary.to_csv(result_dir/"summary.csv", index=False)
    return effects, summary


def cluster_bootstrap_primary(effects, target="semantic_16", control="active_control_16", seed=2601, n=4000):
    effects = effects.loc[effects.site_valid].copy()
    if effects.empty:
        raise RuntimeError("No valid full-site positive-control pairs; cannot test sparse mediation")
    pivot = effects.pivot_table(index=["label", "pair_id", "seed"], columns="arm", values="margin_effect")
    paired = (pivot[target]-pivot[control]).dropna().rename("effect").reset_index()
    means = paired.groupby("label").effect.mean()
    labels = means.index.to_numpy()
    if len(labels) < 2:
        raise ValueError("At least two independent labels required for cluster inference")
    rng = np.random.default_rng(seed)
    bs = [np.mean(rng.choice(means.to_numpy(), len(means), replace=True)) for _ in range(n)]
    result = {"test": f"{target} - {control}", "n_labels": len(labels), "n_pair_seeds": len(paired),
              "mean_of_label_means": float(means.mean()), "bootstrap_ci95": list(map(float,np.quantile(bs,[.025,.975])))}
    return result


def run_campaign(cfg, manifest, *, output, reference_roots, checkpoint, discovery_seed=0,
                 evaluation_seeds=(0,), max_eval_pairs=None, force=False):
    output = Path(output)
    (output/"images").mkdir(parents=True, exist_ok=True)
    checkpoint = Path(checkpoint)
    with checkpoint.open("rb") as fp:
        digest = hashlib.sha256()
        for chunk in iter(lambda: fp.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    fingerprint = {"checkpoint_sha256": digest.hexdigest(),
                   "manifest_sha256": hashlib.sha256(manifest.to_csv(index=False).encode()).hexdigest(),
                   "generation_sha256": hashlib.sha256(json.dumps(cfg["generation"],sort_keys=True).encode()).hexdigest(),
                   "arms": list(ARMS), "site": MID, "steps": list(STEPS)}
    meta_path = output/"protocol_fingerprint.json"
    if meta_path.exists():
        old = json.loads(meta_path.read_text())
        if old != fingerprint:
            raise RuntimeError("Cached experiment protocol/checkpoint changed; use a new output directory")
    else:
        meta_path.write_text(json.dumps(fingerprint,indent=2))
    discovery = manifest[manifest.split == "discovery"].copy()
    evaluation = manifest[manifest.split.isin(["evaluation", "confirmatory"])].copy()
    if max_eval_pairs is not None:
        evaluation = evaluation.groupby("label", group_keys=False).head(int(max_eval_pairs))
    if discovery.empty or evaluation.empty:
        raise ValueError("Missing independent discovery or evaluation pairs")
    if set(discovery.pair_id) & set(evaluation.pair_id):
        raise ValueError("Discovery/evaluation pair overlap")
    if not torch.cuda.is_available():
        raise RuntimeError("SD3 causal experiments require CUDA")
    model, meta = load_research_sae(checkpoint, device="cuda")
    pipe = load_pipeline(cfg)
    rank_path = output/"frozen_feature_ranks.json"
    try:
        needed = pd.concat([discovery, evaluation]).drop_duplicates(subset="pair_id")
        for _, pair in needed.iterrows():
            seeds = [discovery_seed] if pair["split"] == "discovery" else list(evaluation_seeds)
            for seed in seeds:
                for role in ("source", "destination"):
                    try:
                        image_ref(reference_roots, pair.pair_id, seed, role)
                        for s in STEPS:
                            tensor_ref(reference_roots, pair.pair_id, seed, role, MID, s)
                        tensor_ref(reference_roots, pair.pair_id, seed, role, PROBE, PROBE_STEP)
                    except FileNotFoundError:
                        capture_clean(pipe, cfg, output, pair, seed, role)
        roots = [output, *reference_roots]
        if rank_path.exists() and not force:
            ranks = json.loads(rank_path.read_text())
        else:
            ranks = feature_rankings(model, discovery, roots, output, seed=discovery_seed)
        rows, cases = [], []
        for _, pair in evaluation.iterrows():
            for seed in evaluation_seeds:
                pid = str(pair.pair_id)
                for arm in ARMS:
                    target = output/"images"/pid/f"seed_{seed:04d}"/(arm+".png")
                    telemetry = target.with_suffix(".json")
                    if target.exists() and telemetry.exists() and not force:
                        info = json.loads(telemetry.read_text())
                    else:
                        image, info = intervention_image(pipe, cfg, roots, pair, int(seed), model, ranks, arm)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        image.save(target)
                        telemetry.write_text(json.dumps(info, indent=2))
                    rows.append({"pair_id": pid, "seed": seed, "label": pair.label, "arm": arm, **info})
                c = pair.to_dict()
                c["seed"] = seed
                if not c.get("context_label"):
                    obj = c.get("source_entities", "[]")
                    if isinstance(obj, str):
                        try: obj = ast.literal_eval(obj)
                        except (ValueError, SyntaxError): obj = []
                    if isinstance(obj, list):
                        contexts = [str(x.get("label", "")) for x in obj if isinstance(x, dict)
                                    and str(x.get("label", "")) not in (c["label"], c["replacement_label"])]
                        c["context_label"] = contexts[0] if contexts else ""
                cases.append(c)
        pd.DataFrame(rows).to_csv(output/"intervention_telemetry.csv", index=False)
        del model, pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        effects, summary = score_campaign(output, cases, roots)
        primary = cluster_bootstrap_primary(effects)
        (output/"primary_inference.json").write_text(json.dumps(primary, indent=2))
        return summary, primary
    finally:
        if "model" in locals():
            del model
        if "pipe" in locals():
            del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def new_heldout_manifest(previous_csv, *, labels=None, discovery_per_label=3, confirm_per_label=5, seed=4107):
    labels = labels or ["dog","cat","car","bicycle","book","person","apple","chair","lamp",
                        "boat","horse","flower","umbrella","window","banana","bed"]
    replacements = ["clock","guitar","hat","shoe","camera","suitcase","vase","bottle",
                    "plant","backpack","cup","ball","pillow","bus","couch","kite"]
    contexts = ["wooden table", "park bench", "stone wall", "city sidewalk",
                "sunny garden", "living room", "library shelf", "market stall"]
    styles = ["documentary photo", "high-detail studio photo", "natural-light photograph",
              "realistic wide-angle photograph", "editorial photograph"]
    rng = random.Random(seed)
    rows = []
    prior = pd.read_csv(previous_csv)
    old_prompts = set(prior.source_prompt) | set(prior.destination_prompt)
    used_prompts = set()
    for i, target in enumerate(labels):
        replacement = replacements[i % len(replacements)]
        for j in range(discovery_per_label + confirm_per_label):
            context = contexts[(j + i) % len(contexts)]
            style = styles[(2*j + i) % len(styles)]
            tag = f"scene {i+1}-{j+1}, background with a {context}"
            src = f"{style} of a {target} beside a {context}, {tag}"
            dst = f"{style} of a {replacement} beside a {context}, {tag}"
            if src in old_prompts or dst in old_prompts or src in used_prompts or dst in used_prompts:
                raise AssertionError("Held-out prompt leakage")
            used_prompts.update((src,dst))
            rows.append({"pair_id": f"holdout_{i:02d}_{j:02d}", "label": target,
                         "replacement_label": replacement, "context_label": context.split()[-1],
                         "source_prompt": src, "destination_prompt": dst,
                         "split": "discovery" if j < discovery_per_label else "confirmatory"})
    return pd.DataFrame(rows)
