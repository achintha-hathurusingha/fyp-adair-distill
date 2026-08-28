"""kd_feature_multitask dashboard v2 -- local aggregator.

Runs on THIS Windows machine (not on either training host), so it can SSH
into both devon and qbits and merge their status without either host
needing trust in the other. Each poll: one ssh call per configured host,
running remote_status.py there and returning one JSON line; this process
merges them and serves the combined result. No tunnel needed -- listens
directly on localhost.

Replaces the old dashboard/ (both the kd_freq-era single-page one and the
unified/ one) entirely, rebuilt for the current phase: B0V2-KD-FEAT
(control) vs B0V2-KD-FEAT-COND (treatment), tracking per-task PSNR
(denoise/derain/dehaze) now that the B0V2 eval-gap fix makes those numbers
real for the first time.

Usage: python local_server.py
"""
from __future__ import annotations

import json
import subprocess
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PORT = 8092
STATIC_DIR = Path(__file__).parent
SSH_KEY = r"C:\Users\User\Documents\FYP\Achintha"

# host -> (ssh target, arms to track there as "out_root/ARM_NAME")
HOSTS = {
    "devon": {
        "target": "minura@192.248.10.68",
        "arms": ["b0v2_kd_feat/B0V2-KD-FEAT", "b0v2_kd_feat_cond/B0V2-KD-FEAT-COND"],
    },
    "qbits": {
        "target": "minura@192.248.10.67",
        "arms": [],
    },
}

REMOTE_SCRIPT_PATH = "/tmp/remote_status.py"
POLL_INTERVAL_S = 6

_cache = {"hosts": {}, "polled_at": 0}


def _fetch_host(name: str, cfg: dict) -> dict:
    if not cfg["arms"]:
        return {"idle": True, "arms": []}
    env = f"ARMS={','.join(cfg['arms'])}"
    cmd = [
        "ssh", "-i", SSH_KEY, "-o", "ConnectTimeout=6",
        "-o", "StrictHostKeyChecking=accept-new",
        cfg["target"], f"{env} python3 {REMOTE_SCRIPT_PATH}",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if result.returncode != 0:
            return {"error": f"ssh exit {result.returncode}: {result.stderr[-500:]}"}
        return json.loads(result.stdout.strip().splitlines()[-1])
    except subprocess.TimeoutExpired:
        return {"error": "ssh timed out"}
    except (json.JSONDecodeError, IndexError) as e:
        return {"error": f"bad response: {e}"}


def poll_all() -> dict:
    hosts_out = {}
    for name, cfg in HOSTS.items():
        hosts_out[name] = _fetch_host(name, cfg)
    return {"hosts": hosts_out, "polled_at": time.time()}


def _refresh_loop():
    while True:
        try:
            _cache.update(poll_all())
        except Exception as e:  # noqa -- the server must keep serving stale data
            _cache["error"] = str(e)
        time.sleep(POLL_INTERVAL_S)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # quiet -- polling every 6s would flood the console otherwise

    def do_GET(self):
        if self.path == "/api/status":
            body = json.dumps(_cache).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        path = "index.html" if self.path in ("/", "") else self.path.lstrip("/")
        f = STATIC_DIR / path
        if not f.exists() or not f.is_file():
            self.send_response(404)
            self.end_headers()
            return
        ctype = "text/html" if f.suffix == ".html" else \
                "application/javascript" if f.suffix == ".js" else \
                "text/css" if f.suffix == ".css" else "application/octet-stream"
        body = f.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    import threading
    threading.Thread(target=_refresh_loop, daemon=True).start()
    print(f"kd_feature_multitask dashboard v2 -> http://127.0.0.1:{PORT}")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
