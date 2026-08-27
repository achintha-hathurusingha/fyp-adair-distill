"""Unified dashboard — local aggregator.

Runs on this Windows machine (not on either training host), so it can SSH
into BOTH devon and qbits and merge their status without either host needing
trust in the other. Each poll: two ssh calls (one per host), each running
remote_status.py there and returning one JSON line; this process merges
them and serves the combined result. No tunnel needed — this listens
directly on localhost.

Usage: python local_server.py
"""
from __future__ import annotations

import json
import subprocess
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PORT = 8091
STATIC_DIR = Path(__file__).parent
SSH_KEY = r"C:\Users\User\Documents\FYP\Achintha"

# host -> (user@host, repo_root on that host, log path on that host, [arms])
HOSTS = {
    "devon": {
        "target": "minura@192.248.10.68",
        "repo_root": "/home/minura/fyp-adair-distill",
        "log_path": "/tmp/kd_freq_3seed.log",
        "arms": ["M-DEHAZE", "M-DEHAZE-KD", "M-DEHAZE-KD-FREQ"],
    },
    "qbits": {
        "target": "minura@192.248.10.67",
        "repo_root": "/home/minura/fyp-adair-distill",
        "log_path": "/home/minura/qbits_arms.log",
        "arms": ["M-DEHAZE-KD-FEAT", "M-DEHAZE-ECA", "M-DEHAZE-GROUPNORM"],
    },
}

REMOTE_SCRIPT_PATH = "/tmp/remote_status.py"


def _fetch_host(name: str, cfg: dict) -> dict:
    # Plain per-call SSH -- ControlMaster multiplexing was tried first but
    # Windows OpenSSH's mux support proved unreliable here ("Connection
    # reset by peer" setting up the control socket). A fresh handshake every
    # poll costs ~200-500ms, fine at a 6s interval.
    cmd = [
        "ssh", "-i", SSH_KEY, "-o", "ConnectTimeout=6",
        cfg["target"],
        f"python3 {REMOTE_SCRIPT_PATH} {cfg['repo_root']} {cfg['log_path']} "
        + " ".join(cfg["arms"]),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if result.returncode != 0:
            return {"error": f"ssh exit {result.returncode}: {result.stderr[-500:]}",
                    "arms": {}, "log_tail": [], "crash_detected": False}
        return json.loads(result.stdout.strip().splitlines()[-1])
    except (subprocess.TimeoutExpired, json.JSONDecodeError, IndexError) as e:
        return {"error": str(e), "arms": {}, "log_tail": [], "crash_detected": False}


def build_status() -> dict:
    hosts_out = {}
    for name, cfg in HOSTS.items():
        hosts_out[name] = _fetch_host(name, cfg)
    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "teacher_psnr": 34.5056,
        "hosts": hosts_out,
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        if self.path.startswith("/api/status"):
            body = json.dumps(build_status()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path in ("/", "/index.html"):
            fp = STATIC_DIR / "unified.html"
            body = fp.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store, must-revalidate")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.end_headers()


def main():
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"unified dashboard on http://127.0.0.1:{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    main()
