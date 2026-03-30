from __future__ import annotations

import json
import time
from pathlib import Path
import sys
import shutil

from fastapi import HTTPException

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server import FsToolGuard


def _assert(condition: bool, label: str) -> None:
    if not condition:
        raise AssertionError(label)


def run_allowlist_case(base_dir: Path) -> None:
    guard = FsToolGuard(
        base_dir=base_dir,
        fs_cfg={
            "allowlist": ["safe"],
            "budget": {"max_calls": 10, "window_seconds": 60},
            "audit_log": "logs/fs_audit.jsonl",
        },
    )
    safe_path = (base_dir / "safe" / "ok.txt").resolve()
    outside_path = (base_dir / "other" / "no.txt").resolve()
    _assert(guard.is_allowed(safe_path), "allowlist should permit safe path")
    _assert(not guard.is_allowed(outside_path), "allowlist should block non-listed path")


def run_budget_case(base_dir: Path) -> None:
    guard = FsToolGuard(
        base_dir=base_dir,
        fs_cfg={
            "allowlist": ["."],
            "budget": {"max_calls": 2, "window_seconds": 60},
            "audit_log": "logs/fs_audit.jsonl",
        },
    )
    guard.enforce_budget("client-A")
    guard.enforce_budget("client-A")
    try:
        guard.enforce_budget("client-A")
        raise AssertionError("expected HTTPException 429 on budget exceed")
    except HTTPException as e:
        _assert(e.status_code == 429, "budget exceed must return 429")


def run_audit_case(base_dir: Path) -> None:
    guard = FsToolGuard(
        base_dir=base_dir,
        fs_cfg={
            "allowlist": ["."],
            "budget": {"max_calls": 10, "window_seconds": 60},
            "audit_log": "logs/fs_audit.jsonl",
        },
    )
    guard.audit({"ts": time.time(), "op": "fs_list", "status": "ok", "detail": "listed"})
    audit_file = (base_dir / "logs" / "fs_audit.jsonl").resolve()
    _assert(audit_file.exists(), "audit log file should be created")
    row = audit_file.read_text(encoding="utf-8").strip().splitlines()[-1]
    payload = json.loads(row)
    _assert(payload.get("op") == "fs_list", "audit payload should contain op")


def main() -> None:
    base_dir = (ROOT / ".tmp_fs_guard").resolve()
    if base_dir.exists():
        shutil.rmtree(base_dir, ignore_errors=True)
    (base_dir / "safe").mkdir(parents=True, exist_ok=True)
    run_allowlist_case(base_dir)
    run_budget_case(base_dir)
    run_audit_case(base_dir)
    shutil.rmtree(base_dir, ignore_errors=True)
    print("FS_TOOLS_GUARD_OK")


if __name__ == "__main__":
    main()
