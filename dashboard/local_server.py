"""Training board -- local aggregator.

Runs on THIS Windows machine (not on a training host), so it can SSH into both
devon and qbits and merge their status without either host needing to trust the
other. Each poll: one ssh call per host running remote_status.py there, which
reads history.json and tails train.log straight off disk and never touches the
training process. Serves the merged JSON at /api/status plus index.html.

  python local_server.py     ->  http://127.0.0.1:8092

Deploy the remote reader once per host first:

    scp -i <key> remote_status.py <user>@<host>:/tmp/remote_status.py

TRACKED ARMS -- the S3.3 block ablation, plus the three finished arms it is
measured against:

  B0V3-KD-K11      plain 11x11 depthwise. The RECEPTIVE-FIELD CONTROL: the
                   decoder is 3x3 and the block is 11x11, so block-vs-no-block
                   would confound orientation with kernel size.
  B0V3-KD-ORI      the reparameterizable oriented block (S3.1).
  B0V3-KD-ORI-MID  same block, plus the middle placement.
  B0V3-KD-FEAT     finished at 90k. Current best arm and the no-block control.
  B0V3             finished at 90k. GT-only, isolates KD.
  B0V2-KD-FEAT     finished at 90k. NAFNet, isolates architecture.

TWO EVALUATION REGIMES ARE ON THIS BOARD, and the page labels every card with
which one it belongs to. The three finished arms were validated on
test/{derain,dehaze}/demo, which had been carved out of the TRAINING corpora --
their curves run ~1.9 dB high. The S3.3 arms validate on
BSD68 / Rain100L-100 / SOTS-clean-417 and are leak-free. Curves are comparable
within a regime only; the finished arms' honest numbers are the re-scored ones
in reports/clean_eval_rescore.json, not the values their own history shows.

Update HOSTS when an arm moves host or a new one is added -- a stale arm list
is the failure mode that bit every previous version of this dashboard.
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
        # S3.3 sequence (leak-free eval) first, then the finished references
        "arms": ["s33_b0v3_kd_k11/B0V3-KD-K11",
                 "s33_b0v3_kd_ori/B0V3-KD-ORI",
                 "s33_b0v3_kd_ori_mid/B0V3-KD-ORI-MID",
                 "b0v3_kd_feat/B0V3-KD-FEAT",
                 "b0v3/B0V3",
                 "b0v2_kd_feat/B0V2-KD-FEAT"],
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
    # Always poll, even with no arms configured -- GPU/VRAM status (used to
    # judge free capacity for placing new workload, e.g. on idle qbits) is
    # host-level, not tied to whether anything is running there.
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
    print(f"Training board -> http://127.0.0.1:{PORT}")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
