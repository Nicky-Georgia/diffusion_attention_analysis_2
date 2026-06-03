from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
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
    attention_kind: str = "unet"
    stored: str = "head_mean"
    query_image_tokens: int | None = None
    query_text_tokens: int | None = None
    key_image_tokens: int | None = None
    key_text_tokens: int | None = None
    processor: str | None = None
    row_chunk_size: int | None = None


class AttentionStore:
    def __init__(
        self,
        root_dir: str | Path,
        save_headwise: bool = False,
        *,
        row_chunk_size: int = 512,
        save_dtype: str = "float16",
    ) -> None:
        self.root_dir = ensure_dir(root_dir)
        self.save_headwise = save_headwise
        self.row_chunk_size = int(row_chunk_size)
        self.save_dtype = str(save_dtype)
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

    def _save_dtype(self):
        if self.save_dtype.lower() in {"float32", "fp32"}:
            return torch.float32
        if self.save_dtype.lower() in {"bfloat16", "bf16"}:
            return torch.bfloat16
        return torch.float16

    def _step_dir(self) -> Path:
        if self.current_sample_id is None:
            raise RuntimeError("Call begin_sample before recording attention.")
        return ensure_dir(self.root_dir / self.current_sample_id / "attention" / f"step_{self.current_step:03d}")

    def record(
        self,
        layer_name: str,
        attention_probs: torch.Tensor,
        *,
        is_cross: bool,
        height: int | None,
        width: int | None,
        num_heads: int,
    ) -> None:
        """Record UNet-style probabilities with shape [batch*heads, query, key]."""
        if not self.should_record():
            return
        step_dir = self._step_dir()
        batch_size = attention_probs.shape[0] // int(num_heads)
        probs = attention_probs.detach().cpu().float().reshape(
            batch_size, int(num_heads), attention_probs.shape[1], attention_probs.shape[2]
        )
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
            attention_kind="unet_cross" if is_cross else "unet_self",
            stored="headwise" if self.save_headwise else "head_mean",
            processor="RecordingAttnProcessor",
        )
        torch.save({"meta": meta.__dict__, "attention": to_save.to(self._save_dtype())}, step_dir / module_to_filename(layer_name))

    def record_precomputed(
        self,
        layer_name: str,
        attention: torch.Tensor,
        *,
        is_cross: bool,
        height: int | None = None,
        width: int | None = None,
        num_heads: int = 1,
        attention_kind: str = "joint",
        meta_extra: Dict[str, Any] | None = None,
    ) -> None:
        """Record probabilities already shaped as [batch, query, key] or [batch, heads, query, key]."""
        if not self.should_record():
            return
        step_dir = self._step_dir()
        attn_cpu = attention.detach().cpu()
        if attn_cpu.ndim == 3:
            batch_size, num_queries, num_keys = attn_cpu.shape
            stored = "head_mean"
        elif attn_cpu.ndim == 4:
            batch_size, heads, num_queries, num_keys = attn_cpu.shape
            num_heads = int(heads)
            stored = "headwise"
        else:
            raise ValueError(f"Unsupported precomputed attention shape: {tuple(attn_cpu.shape)}")
        meta = AttentionRecordMeta(
            layer_name=layer_name,
            step_index=int(self.current_step),
            is_cross=bool(is_cross),
            height=height,
            width=width,
            batch_size=int(batch_size),
            num_heads=int(num_heads),
            num_queries=int(num_queries),
            num_keys=int(num_keys),
            attention_kind=attention_kind,
            stored=stored,
            row_chunk_size=int(self.row_chunk_size),
        )
        meta_dict = meta.__dict__
        if meta_extra:
            meta_dict.update(meta_extra)
        torch.save({"meta": meta_dict, "attention": attn_cpu.to(self._save_dtype())}, step_dir / module_to_filename(layer_name))

    @torch.no_grad()
    def record_joint_qk(
        self,
        layer_name: str,
        query: torch.Tensor,
        key: torch.Tensor,
        *,
        image_tokens: int,
        text_tokens: int,
        scale: float | None = None,
        height: int | None = None,
        width: int | None = None,
        attention_kind: str = "sd3_joint_full",
    ) -> None:
        """Materialize SD3/MM-DiT full joint-attention probabilities in chunks.

        `query` and `key` must have shape [batch, heads, tokens, head_dim] with
        token order [image tokens, text/context tokens], matching Diffusers' SD3
        JointAttnProcessor2_0.  The saved artifact is the full query-by-key
        probability matrix, head-averaged by default. This is intentionally
        separate from residual-stream capture used for SAE training/steering.
        """
        if not self.should_record():
            return
        if query.ndim != 4 or key.ndim != 4:
            raise ValueError(f"Expected q/k as [B,H,N,D], got {tuple(query.shape)} and {tuple(key.shape)}")
        batch, heads, q_tokens, head_dim = query.shape
        _, key_heads, k_tokens, key_dim = key.shape
        if heads != key_heads or head_dim != key_dim:
            raise ValueError(f"Incompatible q/k shapes: {tuple(query.shape)} and {tuple(key.shape)}")

        row_chunk = max(1, int(self.row_chunk_size))
        scale = float(scale if scale is not None else 1.0 / sqrt(float(head_dim)))
        save_dtype = self._save_dtype()

        if self.save_headwise:
            # Usually disabled for SD3 Medium because it is roughly 24x larger
            # than the default head-averaged full maps.
            out = torch.empty((batch, heads, q_tokens, k_tokens), dtype=save_dtype, device="cpu")
            for b in range(batch):
                for h in range(heads):
                    q_bh = query[b : b + 1, h : h + 1].detach().float()
                    k_bh_t = key[b : b + 1, h : h + 1].detach().float().transpose(-1, -2)
                    for start in range(0, q_tokens, row_chunk):
                        end = min(start + row_chunk, q_tokens)
                        scores = torch.matmul(q_bh[:, :, start:end, :], k_bh_t) * scale
                        probs = torch.softmax(scores, dim=-1)[0, 0].to(save_dtype).cpu()
                        out[b, h, start:end] = probs
            attention = out
        else:
            # Accumulate the full probability matrix as a head mean. Keep the
            # accumulator on CPU to avoid retaining O(heads*tokens^2) on GPU.
            out = torch.zeros((batch, q_tokens, k_tokens), dtype=torch.float32, device="cpu")
            inv_heads = 1.0 / float(heads)
            for b in range(batch):
                for h in range(heads):
                    q_bh = query[b : b + 1, h : h + 1].detach().float()
                    k_bh_t = key[b : b + 1, h : h + 1].detach().float().transpose(-1, -2)
                    for start in range(0, q_tokens, row_chunk):
                        end = min(start + row_chunk, q_tokens)
                        scores = torch.matmul(q_bh[:, :, start:end, :], k_bh_t) * scale
                        probs = torch.softmax(scores, dim=-1)[0, 0].cpu().float()
                        out[b, start:end] += probs * inv_heads
            attention = out.to(save_dtype)

        self.record_precomputed(
            layer_name,
            attention,
            is_cross=True,
            height=height,
            width=width,
            num_heads=int(heads),
            attention_kind=attention_kind,
            meta_extra={
                "query_image_tokens": int(image_tokens),
                "query_text_tokens": int(text_tokens),
                "key_image_tokens": int(image_tokens),
                "key_text_tokens": int(text_tokens),
                "processor": "RecordingSD3JointAttnProcessor2_0",
                "scale": scale,
            },
        )


class RecordingAttnProcessor:
    """Transparent Diffusers attention processor that materializes attention probabilities.

    Intended for UNet-family Diffusers pipelines. SD3/MM-DiT uses
    `RecordingSD3JointAttnProcessor2_0` below.
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


class RecordingSD3JointAttnProcessor2_0:
    """SD3/MM-DiT attention processor that records full joint probability maps.

    Diffusers' default SD3 processor uses `torch.nn.functional.scaled_dot_product_attention`,
    which does not expose probabilities. This drop-in processor keeps the normal
    SDPA output path for generation, and separately materializes full q-by-k
    probability maps in row/head chunks only at configured capture steps.
    """

    def __init__(self, store: AttentionStore, layer_name: str) -> None:
        if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            raise ImportError("RecordingSD3JointAttnProcessor2_0 requires PyTorch 2.0+.")
        self.store = store
        self.layer_name = layer_name

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        *args,
        **kwargs,
    ):
        import torch.nn.functional as F

        residual = hidden_states
        batch_size = hidden_states.shape[0]

        # Sample/image projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        image_tokens = int(hidden_states.shape[1])

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if getattr(attn, "norm_q", None) is not None:
            query = attn.norm_q(query)
        if getattr(attn, "norm_k", None) is not None:
            key = attn.norm_k(key)

        text_tokens = 0
        if encoder_hidden_states is not None:
            text_tokens = int(encoder_hidden_states.shape[1])
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)
            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            if getattr(attn, "norm_added_q", None) is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if getattr(attn, "norm_added_k", None) is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)
            query = torch.cat([query, encoder_hidden_states_query_proj], dim=2)
            key = torch.cat([key, encoder_hidden_states_key_proj], dim=2)
            value = torch.cat([value, encoder_hidden_states_value_proj], dim=2)

        # Normal generation path. Keep SDPA so generation remains as close as
        # possible to the default Diffusers implementation.
        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)

        # Auxiliary recording path. This is intentionally after SDPA, so even if
        # recording is slow it does not change the numerical output path.
        if self.store.should_record():
            scale = getattr(attn, "scale", None)
            self.store.record_joint_qk(
                self.layer_name,
                query,
                key,
                image_tokens=image_tokens,
                text_tokens=text_tokens,
                scale=scale,
                attention_kind="sd3_joint_full",
            )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : residual.shape[1]],
                hidden_states[:, residual.shape[1] :],
            )
            if not getattr(attn, "context_pre_only", False):
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
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


def install_sd3_joint_attention_processors(
    transformer,
    store: AttentionStore,
    *,
    layer_whitelist: Optional[Sequence[str]] = None,
    include_attn2: bool = False,
):
    """Install probability-recording processors on SD3 transformer attention modules."""
    original = dict(transformer.attn_processors)
    new_processors = {}

    def keep(name: str) -> bool:
        if not layer_whitelist:
            return True
        return any(fragment in name for fragment in layer_whitelist)

    installed: List[str] = []
    for name, old in original.items():
        is_primary_joint = name.endswith(".attn.processor")
        is_secondary = name.endswith(".attn2.processor")
        if keep(name) and (is_primary_joint or (include_attn2 and is_secondary)):
            new_processors[name] = RecordingSD3JointAttnProcessor2_0(store=store, layer_name=name)
            installed.append(name)
        else:
            new_processors[name] = old
    transformer.set_attn_processor(new_processors)
    return original, installed


def restore_attention_processors(model, original: Dict[str, Any]) -> None:
    model.set_attn_processor(original)
