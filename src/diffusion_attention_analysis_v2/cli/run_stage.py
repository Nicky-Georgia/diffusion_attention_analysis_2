from __future__ import annotations

import argparse

from ..runner import run_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one configured experiment stage.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--dry-run", action="store_true", help="Print the execution plan without loading a model.")
    parser.add_argument("--override", action="append", default=[], help="Override config value, e.g. data.max_prompts=8. Can be repeated.")
    args = parser.parse_args()
    run_config(args.config, dry_run=args.dry_run, overrides=args.override, echo=True)


if __name__ == "__main__":
    main()
