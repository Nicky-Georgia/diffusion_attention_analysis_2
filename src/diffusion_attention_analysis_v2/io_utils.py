from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: str | Path, max_items: int | None = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_items is not None and len(rows) >= max_items:
                break
    return rows


def safe_name(text: str) -> str:
    allowed = []
    for ch in str(text):
        if ch.isalnum() or ch in {"-", "_"}:
            allowed.append(ch)
        else:
            allowed.append("_")
    out = "".join(allowed).strip("_")
    while "__" in out:
        out = out.replace("__", "_")
    return out or "item"


def module_to_filename(module_name: str) -> str:
    return module_name.replace("/", "__").replace(".", "__") + ".pt"


def filename_to_module(filename: str) -> str:
    stem = Path(filename).stem
    return stem.replace("__", ".")
