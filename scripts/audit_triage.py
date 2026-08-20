#!/usr/bin/env python3
"""Compare pip-audit findings to the accepted-advisory register. Never hide findings."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description="Triage locked dependency advisories.")
    parser.add_argument("--requirements", type=Path, required=True)
    parser.add_argument("--accepted", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    audit = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip_audit",
            "-r",
            str(args.requirements),
            "--no-deps",
            "--disable-pip",
            "--format",
            "json",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if audit.returncode not in {0, 1}:
        sys.stderr.write(audit.stderr or audit.stdout)
        return audit.returncode
    payload = json.loads(audit.stdout or "{\"dependencies\": []}")
    findings = _findings(payload)
    accepted_doc = json.loads(args.accepted.read_text(encoding="utf-8"))
    accepted = {
        (str(item["package"]).lower(), str(item["id"])): item
        for item in accepted_doc.get("advisories") or []
    }
    unexpected: list[dict[str, Any]] = []
    accepted_hits: list[dict[str, Any]] = []
    for item in findings:
        key = (item["package"].lower(), item["id"])
        if key in accepted:
            merged = dict(accepted[key])
            merged.update(item)
            accepted_hits.append(merged)
        else:
            unexpected.append(item)
    args.out.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "blocked" if unexpected else ("accepted_advisories" if accepted_hits else "clean"),
        "finding_count": len(findings),
        "accepted_count": len(accepted_hits),
        "unexpected_count": len(unexpected),
        "accepted": accepted_hits,
        "unexpected": unexpected,
        "review_deadline": accepted_doc.get("review_deadline"),
    }
    (args.out / "advisory-triage.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.out / "advisory-triage.md").write_text(_markdown(report), encoding="utf-8")
    print(f"dependency audit status={report['status']} findings={report['finding_count']}")
    if unexpected:
        print("new advisories require an explicit accepted-risk update:")
        for item in unexpected:
            print(f"- {item['package']} {item['version']} {item['id']}")
        return 1
    return 0


def _findings(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dep in payload.get("dependencies") or []:
        name = str(dep.get("name") or "")
        version = str(dep.get("version") or "")
        for vuln in dep.get("vulns") or []:
            rows.append(
                {
                    "package": name,
                    "version": version,
                    "id": str(vuln.get("id") or ""),
                    "fixed_version": ",".join(str(item) for item in (vuln.get("fix_versions") or [])),
                    "description": str(vuln.get("description") or "")[:240],
                }
            )
    return rows


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Dependency advisory triage",
        "",
        f"- status: `{report['status']}`",
        f"- findings: `{report['finding_count']}`",
        f"- accepted: `{report['accepted_count']}`",
        f"- unexpected: `{report['unexpected_count']}`",
        f"- review deadline: `{report.get('review_deadline')}`",
        "",
        "| package | version | id | fixed | extra | reachable | action |",
        "|---|---|---|---|---|---|---|",
    ]
    for item in report["accepted"]:
        lines.append(
            f"| {item.get('package')} | {item.get('version')} | {item.get('id')} | "
            f"{item.get('fixed_version')} | {item.get('extra')} | {item.get('reachable')} | "
            f"{item.get('action')} |"
        )
    for item in report["unexpected"]:
        lines.append(
            f"| {item.get('package')} | {item.get('version')} | {item.get('id')} | "
            f"{item.get('fixed_version')} | unknown | unknown | BLOCK |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
