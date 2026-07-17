#!/usr/bin/env python3
"""Fail if RDNA2 decode coverage regresses below baseline."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "tools" / "rdna2_opcode_matrix.py"
BASELINE = ROOT / "tools" / "rdna2_coverage_baseline.json"
REPORT = ROOT / "tools" / "rdna2_coverage.json"


def main() -> int:
    subprocess.check_call([sys.executable, str(MATRIX)], cwd=ROOT)
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    current = report["summary"]["decode_coverage_pct"]

    if not BASELINE.exists():
        BASELINE.write_text(json.dumps({"decode_coverage_pct": current}, indent=2), encoding="utf-8")
        print(f"Created baseline coverage: {current}%")
        return 0

    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    minimum = baseline.get("decode_coverage_pct", 0.0)
    print(f"RDNA2 decode coverage: {current}% (baseline {minimum}%)")
    if current + 1e-6 < minimum:
        print("Coverage regression detected", file=sys.stderr)
        return 1
    enum_count = report["summary"].get("decoder_enum_count", 0)
    lower_count = report["summary"].get("lower_ops_count", 0)
    doc_count = report["summary"].get("doc_opcode_entries", 0)
    lower_doc_hits = report["summary"].get("lower_ops_doc_hits", lower_count)
    lower_doc_pct = report["summary"].get(
        "lower_ops_doc_coverage_pct",
        100.0 * lower_doc_hits / doc_count if doc_count > 0 else 0.0,
    )
    alias_extra = report["summary"].get("lower_ops_alias_extra", 0)
    if doc_count > 0:
        lower_minimum = baseline.get("lower_ops_doc_coverage_pct", baseline.get("lower_ops_pct", 68.0))
        print(
            f"IR lower_ops: {lower_doc_hits}/{doc_count} doc entries ({lower_doc_pct:.1f}%, "
            f"minimum {lower_minimum}%)"
        )
        if alias_extra:
            print(
                f"IR lower_ops alias extras: {lower_count} decoder mappings "
                f"({alias_extra} beyond doc hits)"
            )
        if lower_doc_pct + 1e-6 < lower_minimum:
            print("IR lower_ops coverage below minimum", file=sys.stderr)
            return 1
        if lower_doc_pct > lower_minimum:
            baseline["lower_ops_doc_coverage_pct"] = round(lower_doc_pct, 1)
            baseline.pop("lower_ops_pct", None)
    elif enum_count > 0:
        lower_pct = 100.0 * lower_count / enum_count
        lower_minimum = baseline.get("lower_ops_pct", 95.0)
        print(f"IR lower_ops: {lower_count}/{enum_count} ({lower_pct:.1f}%, minimum {lower_minimum}%)")
        if lower_pct + 1e-6 < lower_minimum:
            print("IR lower_ops coverage below minimum", file=sys.stderr)
            return 1
        if lower_pct > lower_minimum:
            baseline["lower_ops_pct"] = round(lower_pct, 1)
    if current > minimum:
        baseline["decode_coverage_pct"] = current
        BASELINE.write_text(json.dumps(baseline, indent=2), encoding="utf-8")
        print(f"Updated baseline to {current}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
