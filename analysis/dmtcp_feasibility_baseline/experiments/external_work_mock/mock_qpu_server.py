#!/usr/bin/env python3
"""Local independent service for testing remote-work reconciliation semantics."""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import threading
from urllib.parse import urlparse


class Store:
    def __init__(self, state_path: Path) -> None:
        self.state_path = state_path
        self.lock = threading.Lock()
        self.attempts = 0
        self.jobs: dict[str, dict[str, object]] = {}
        self.requests: dict[str, str] = {}

    def submit(self, request_id: str | None) -> dict[str, object]:
        with self.lock:
            self.attempts += 1
            if request_id and request_id in self.requests:
                job = self.jobs[self.requests[request_id]]
                self._persist()
                return {**job, "deduplicated": True}
            job_id = f"mock-job-{len(self.jobs) + 1:04d}"
            job = {
                "job_id": job_id,
                "request_id": request_id,
                "result": {"counts": {"0": 384, "1": 128}, "shots": 512},
                "status": "COMPLETED",
            }
            self.jobs[job_id] = job
            if request_id:
                self.requests[request_id] = job_id
            self._persist()
            return {**job, "deduplicated": False}

    def snapshot(self) -> dict[str, object]:
        with self.lock:
            return self._payload()

    def _payload(self) -> dict[str, object]:
        return {
            "accepted_jobs": len(self.jobs),
            "attempts": self.attempts,
            "jobs": self.jobs,
            "request_index": self.requests,
        }

    def _persist(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_name(f".{self.state_path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(self._payload(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, self.state_path)


def handler_for(store: Store):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, payload: dict[str, object]) -> None:
            encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/jobs":
                self._send(404, {"error": "not_found"})
                return
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            self._send(200, store.submit(payload.get("request_id")))

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/state":
                self._send(200, store.snapshot())
                return
            if parsed.path.startswith("/jobs/"):
                job_id = parsed.path.rsplit("/", maxsplit=1)[-1]
                snapshot = store.snapshot()
                job = snapshot["jobs"].get(job_id)
                self._send(200 if job else 404, job or {"error": "unknown_job"})
                return
            self._send(404, {"error": "not_found"})

        def log_message(self, _format: str, *_args: object) -> None:
            return

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port-file", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    args = parser.parse_args()
    store = Store(args.state)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(store))
    args.port_file.parent.mkdir(parents=True, exist_ok=True)
    args.port_file.write_text(str(server.server_port) + "\n", encoding="utf-8")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
