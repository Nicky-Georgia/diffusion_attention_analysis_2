from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

from .io_utils import ensure_dir, module_to_filename, save_json


@dataclass
class AttentionRecordMeta:
    layer_name: str
    step_index: int
    is_cross: bool
    height: int | None
    width: int | None
    batch_size: int
    num_heads: int
    num_queries: int
    num_keys: int


class AttentionStore:
    def __init__(self, root_dir: str | Path, save_headwise: bool = False) -> None:
        self.root_dir = ensure_dir(root_dir)
        self.save_headwise = save_headwise
        self.current_step = 0
        self.capture_steps: set[int] | None = None
        self.current_sample_id: str | None = None

    def begin_sample(self, sample_id: str, metadata: Dict[str, Any]) -> None:
        self.current_sample_id = sample_id
        sample_dir = ensure_dir(self.root_dir / sample_id)
        save_json(sample_dir / "metadata.json", metadata)

    def set_step(self, step_index: int) -> None:
        self.current_step = int(step_index)

    def should_record(self) -> bool:
        return self.capture_steps is None or self.current_step in self.capture_steps

    def record(self, layer_name: str, attention_probs: torch.Tensor, *, is_cross: bool, height: int | None, width: int | None, num_heads: int) -> None:
        if not self.should_record():
            return
        if self.current_sample_id is None:
            raise RuntimeError("Call begin_sample before recording attention.")
        step_dir = ensure_dir(self.root_dir / self.current_sample_id / "attention" / f"step_{self.current_step:03d}")
        batch_size = attention_probs.shape[0] // int(num_heads)
        probs = attention_probs.detach().cpu().float().reshape(batch_size, int(num_heads), attention_probs.shape[1], attention_probs.shape[2])
        to_save = probs if self.save_headwise else probs.mean(dim=1)
        meta = AttentionRecordMeta(
            layer_name=layer_name,
            step_index=int(self.current_step),
            is_cross=bool(is_cross),
            height=height,
            width=width,
            batch_size=int(batch_size),
            num_heads=int(num_heads),
            num_queries=int(probs.shape[-2]),
            num_keys=int(probs.shape[-1]),
        )
        torch.save({"meta": meta.__dict__, "attention": to_save.half()}, step_dir / module_to_filename(layer_name))


class RecordingAttnProcessor:
    """Transparent Diffusers attention processor that materializes attention probabilities.

    Intended for UNet-family Diffusers pipelines. Transformer pipelines are kept in a shortened
    block-output path by default.
    """

    def __init__(self, store: AttentionStore, layer_name: str, is_cross: bool, intervention=None) -> None:
        self.store = store
        self.layer_name = layer_name
        self.is_cross = is_cross
        self.intervention = intervention

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, *args, **kwargs):
        residual = hidden_states
        input_ndim = hidden_states.ndim
        height, width = None, None

        if getattr(attn, "spatial_norm", None) is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        else:
            batch_size = hidden_states.shape[0]

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif getattr(attn, "norm_cross", None) is not None:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        sequence_length = encoder_hidden_states.shape[1]
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if getattr(attn, "group_norm", None) is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        if self.intervention is not None:
            attention_probs = self.intervention(
                layer_name=self.layer_name,
                step_index=self.store.current_step,
                attention_probs=attention_probs,
                batch_size=batch_size,
                num_heads=attn.heads,
                height=height,
                width=width,
            )

        self.store.record(self.layer_name, attention_probs, is_cross=self.is_cross, height=height, width=width, num_heads=attn.heads)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if getattr(attn, "residual_connection", False):
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


def install_unet_attention_processors(
    unet,
    store: AttentionStore,
    *,
    layer_whitelist: Optional[Sequence[str]] = None,
    include_self: bool = False,
    include_cross: bool = True,
    intervention=None,
):
    original = dict(unet.attn_processors)
    new_processors = {}

    def keep(name: str) -> bool:
        if not layer_whitelist:
            return True
        return any(fragment in name for fragment in layer_whitelist)

    installed: List[str] = []
    for name, old in original.items():
        is_self = ".attn1.processor" in name
        is_cross = ".attn2.processor" in name
        if not keep(name):
            new_processors[name] = old
        elif (is_self and include_self) or (is_cross and include_cross):
            new_processors[name] = RecordingAttnProcessor(store=store, layer_name=name, is_cross=is_cross, intervention=intervention)
            installed.append(name)
        else:
            new_processors[name] = old
    unet.set_attn_processor(new_processors)
    return original, installed


def restore_attention_processors(unet, original: Dict[str, Any]) -> None:
    unet.set_attn_processor(original)
