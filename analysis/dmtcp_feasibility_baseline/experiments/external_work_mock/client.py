#!/usr/bin/env python3
"""Checkpointed mock client with optional application-level idempotency."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from urllib.request import Request, urlopen


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _json_request(url: str, payload: dict[str, object] | None = None) -> dict[str, object]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(url, data=data, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=10) as response:  # noqa: S310 - loopback-only test service
        return json.loads(response.read())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--mode", choices=("non_idempotent", "idempotent"), required=True)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pre-submit-pause", type=float, default=2.0)
    parser.add_argument("--post-submit-pause", type=float, default=5.0)
    args = parser.parse_args()

    request_id = "resq-dmtcp-c3-request-0001" if args.mode == "idempotent" else None
    _write_json(args.progress, {"phase": "ready_to_submit", "request_id": request_id})
    time.sleep(args.pre_submit_pause)
    job = _json_request(args.base_url + "/jobs", {"request_id": request_id})
    _write_json(
        args.progress,
        {"job_id": job["job_id"], "phase": "server_ack_received", "request_id": request_id},
    )
    time.sleep(args.post_submit_pause)
    result = _json_request(args.base_url + f"/jobs/{job['job_id']}")
    _write_json(
        args.output,
        {
            "job_id": result["job_id"],
            "mode": args.mode,
            "request_id": request_id,
            "result": result["result"],
            "status": result["status"],
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
