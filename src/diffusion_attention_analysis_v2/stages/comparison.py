from __future__ import annotations

from typing import Any, Dict


def run(cfg: Dict[str, Any], progress, *, dry_run: bool = False) -> Dict[str, Any]:
    from ..config import output_dir, resolve_path
    from ..io_utils import load_json, save_json
    from .common import dry_run_response, maybe_disabled

    disabled = maybe_disabled(cfg, progress)
    if disabled is not None:
        return disabled
    logic = {
        "question": "What can be compared fairly across models and what must stay marked as pilot-only?",
        "rule": "SD3 Medium and SANA are not treated as symmetric raw-attention benchmarks in default configs.",
    }
    if dry_run:
        return dry_run_response(cfg, progress, logic_check=logic)

    import pandas as pd

    out_dir = output_dir(cfg)
    inputs = cfg.get("comparison", {}).get("inputs", [])
    progress.start(total=len(inputs), message="comparison started")
    rows = []
    for i, item in enumerate(inputs, 1):
        raw_path = item.get("report_path")
        path = resolve_path({**cfg, "data": {"output_dir": cfg.get("data", {}).get("output_dir"), "tmp": raw_path}}, "data.tmp") if raw_path else None
        row = {
            "model": item.get("model"),
            "stage": item.get("stage"),
            "claim_level": item.get("claim_level"),
            "report_path": raw_path,
            "exists": bool(path and path.exists()),
            "status": "missing",
        }
        if path and path.exists():
            try:
                report = load_json(path)
                row["status"] = report.get("status", "unknown")
                row["rows"] = report.get("rows")
                row["samples"] = report.get("samples")
                row["summary"] = report.get("summary")
                if isinstance(report.get("metadata"), dict):
                    row["activation_files"] = report["metadata"].get("activation_files")
                    row["d_model"] = report["metadata"].get("d_model")
            except Exception as exc:  # noqa: BLE001
                row["status"] = "read_error"
                row["error"] = str(exc)
        rows.append(row)
        progress.update(current=i, message="comparison input processed", model=row["model"], stage=row["stage"], status=row["status"])
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "comparison_summary.csv", index=False)
    report = {
        "status": "ok",
        "rows": len(rows),
        "interpretation_rules": cfg.get("comparison", {}).get("interpretation_rules", []),
        "missing_reports": int((~df["exists"]).sum()) if len(df) else 0,
    }
    save_json(out_dir / "report.json", report)
    progress.end(message="comparison finished", rows=len(rows), missing_reports=report["missing_reports"])
    return report
