from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from ..config import project_root
from ..runner import run_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a suite of configured stages.")
    parser.add_argument("--suite", required=True, help="Path to suite YAML file.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--override", action="append", default=[], help="Global override applied to every config.")
    args = parser.parse_args()
    suite_path = Path(args.suite)
    if not suite_path.is_absolute():
        suite_path = project_root() / suite_path
    suite = yaml.safe_load(suite_path.read_text(encoding="utf-8")) or {}
    for cfg_path in suite.get("configs", []):
        run_config(cfg_path, dry_run=args.dry_run, overrides=args.override, echo=True)


if __name__ == "__main__":
    main()
