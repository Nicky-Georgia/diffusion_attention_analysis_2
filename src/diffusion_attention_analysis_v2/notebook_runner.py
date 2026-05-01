from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List


def _project_root() -> Path:
    p = Path.cwd().resolve()
    if p.name == "notebooks":
        p = p.parent
    return p


def run_config(config_path: str, *, overrides: Iterable[str] | None = None, dry_run: bool = False):
    """Run a config through the CLI and display JSON progress in notebooks."""
    from IPython.display import JSON, display
    from tqdm.notebook import tqdm

    root = _project_root()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [sys.executable, "-m", "diffusion_attention_analysis_v2.cli.run_stage", "--config", config_path]
    if dry_run:
        cmd.append("--dry-run")
    for ov in overrides or []:
        cmd.extend(["--override", str(ov)])
    print("$", " ".join(cmd))
    proc = subprocess.Popen(cmd, cwd=root, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    bar = None
    final = None
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            print(line)
            continue
        if event.get("event") == "start":
            total = event.get("total")
            bar = tqdm(total=total, desc=event.get("message", config_path)) if total is not None else None
        elif event.get("event") == "progress":
            if bar is not None:
                current = int(event.get("current", 0))
                bar.update(max(0, current - bar.n))
            print(event.get("message", "progress"), {k: v for k, v in event.items() if k not in {"event", "message"}})
        elif event.get("event") == "end":
            if bar is not None:
                total = event.get("total") or bar.total or bar.n
                bar.update(max(0, int(total) - bar.n))
                bar.close()
            print(event.get("message", "finished"))
        elif event.get("event") == "skip":
            print("SKIP:", event)
        elif event.get("event") == "result":
            final = event.get("result")
        else:
            print(event)
    code = proc.wait()
    if code != 0:
        raise RuntimeError(f"Config failed with exit code {code}: {config_path}")
    if final is not None:
        display(JSON(final))
    return final


def run_suite(configs: List[str], *, overrides: Iterable[str] | None = None, dry_run: bool = False):
    results = []
    for config in configs:
        print("\n===", config, "===")
        results.append(run_config(config, overrides=overrides, dry_run=dry_run))
    return results


def show_report(report_path: str):
    from IPython.display import JSON, display

    root = _project_root()
    path = Path(report_path)
    if not path.is_absolute():
        path = root / path
    if not path.exists():
        print(f"Report not found: {path}")
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    display(JSON(data))
    return data
