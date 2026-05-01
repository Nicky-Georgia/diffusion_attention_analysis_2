from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Sequence


def _clean_token(token: str) -> str:
    token = token.lower()
    token = token.replace("</w>", "")
    token = token.replace("Ġ", "")
    token = token.replace("▁", "")
    token = re.sub(r"[^a-z0-9]+", "", token)
    return token


def get_token_map(tokenizer: Any, prompt: str, targets: Sequence[str]) -> Dict[str, List[int]]:
    """Best-effort target-word to tokenizer-index map for CLIP/T5-like tokenizers.

    This is deliberately permissive. For formal reporting, manually inspect token maps for
    the curated intervention/localization subset.
    """
    out: Dict[str, List[int]] = {str(t).lower(): [] for t in targets}
    if tokenizer is None:
        words = re.findall(r"[a-zA-Z0-9]+", prompt.lower())
        for target in out:
            target_clean = _clean_token(target)
            out[target] = [i + 1 for i, word in enumerate(words) if target_clean in word]
        return out

    try:
        encoded = tokenizer(prompt, return_tensors=None, add_special_tokens=True)
        ids = encoded["input_ids"]
        if isinstance(ids[0], list):
            ids = ids[0]
        tokens = tokenizer.convert_ids_to_tokens(ids)
    except Exception:
        return get_token_map(None, prompt, targets)

    cleaned = [_clean_token(str(tok)) for tok in tokens]
    for target in out:
        target_clean = _clean_token(target)
        if not target_clean:
            continue
        indices = [i for i, tok in enumerate(cleaned) if target_clean in tok or tok in target_clean]
        out[target] = indices
    return out


def prompt_targets(row: Dict[str, Any]) -> List[str]:
    labels: List[str] = []
    for key in ("target_word", "target", "label"):
        if row.get(key):
            labels.append(str(row[key]))
    for ent in row.get("entities", []) or []:
        if isinstance(ent, dict):
            labels.append(str(ent.get("token") or ent.get("label") or ent.get("name")))
        else:
            labels.append(str(ent))
    return sorted({x.lower() for x in labels if x and x != "None"})
