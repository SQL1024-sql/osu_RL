"""Black-and-white status page for training / evaluation runs, served on the LAN.

    python dashboard.py            # http://<this machine's LAN IP>:8000
Reads runs/*/progress.csv, results/*/results.md and runs/status.txt (a free-text note of what is going on now).
"""
import csv
import glob
import html
import json
import os
import socket
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = 8000
STYLE = """body{background:#fff;color:#000;font:14px/1.5 ui-monospace,Consolas,monospace;margin:16px;max-width:1000px}
h1{font-size:18px;margin:0 0 4px}h2{font-size:15px;border-bottom:1px solid #000;margin:28px 0 8px}
table{border-collapse:collapse;margin:6px 0}td,th{border:1px solid #000;padding:2px 8px;text-align:right}th{background:#000;color:#fff}
pre{border:1px solid #000;padding:8px;white-space:pre-wrap}.bar{border:1px solid #000;height:14px;width:100%;max-width:500px}
.bar div{background:#000;height:100%}.muted{color:#666}"""


def fmt_time(s):
    s = int(s)
    return f"{s // 3600}h{s // 60 % 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


def run_section(run_dir):
    name = os.path.basename(run_dir)
    rows = list(csv.DictReader(open(os.path.join(run_dir, "progress.csv"))))
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    a = cfg["args"]
    target = float(a["steps"])
    last = rows[-1]
    steps, elapsed, fps = float(last["time/total_timesteps"]), float(last["time/time_elapsed"]), float(last["time/fps"])
    age = time.time() - os.path.getmtime(os.path.join(run_dir, "progress.csv"))
    finished = os.path.exists(os.path.join(run_dir, "model.zip"))
    results = os.path.join("results", name, "results.md")
    eval_log = os.path.join(run_dir, "evaluate.log")
    if os.path.exists(results) and os.path.getmtime(results) > os.path.getmtime(os.path.join(run_dir, "progress.csv")):
        state = "done (evaluated)"
    elif finished and os.path.exists(eval_log) and time.time() - os.path.getmtime(eval_log) < 1800:
        state = "evaluating"
    elif finished:
        state = "trained, not evaluated"
    elif age < 300:
        state = f"training — ETA {fmt_time((target - steps) / max(fps, 1))}"
    else:
        state = f"stopped? (no update for {fmt_time(age)})"
    pct = min(1.0, steps / target)
    val = [r for r in rows if r.get("val/acc")]
    best = max(val, key=lambda r: float(r["val/acc"]), default=None)
    out = [f"<h2>{html.escape(name)}</h2><p><b>{state}</b> · diff={a.get('diff', 'narrow')} envs={a['envs']} workers={a.get('workers') or a['envs']}"
           f" timing={a.get('timing', 0)} seed={a['seed']}</p>",
           f"<div class=bar><div style='width:{pct * 100:.1f}%'></div></div>",
           f"<p>{steps / 1e6:.2f}M / {target / 1e6:.0f}M steps ({pct * 100:.1f}%) · elapsed {fmt_time(elapsed)} · {fps:.0f} fps"
           f" · {elapsed / steps * 1e6:.0f} µs/step ({fmt_time(elapsed / steps * 1e6)} per 1M steps)"
           + (f" · best val {float(best['val/acc']) * 100:.2f}% at {float(best['time/total_timesteps']) / 1e6:.2f}M" if best else "") + "</p>"]
    if val:
        # time per step over each interval since the previous evaluation, evaluation time included
        per_step = {id(r): (float(r["time/time_elapsed"]) - float(p["time/time_elapsed"])) / (float(r["time/total_timesteps"]) - float(p["time/total_timesteps"])) * 1e6
                    for p, r in zip(val, val[1:])}
        out.append("<table><tr><th>steps</th><th>val acc</th><th>val miss</th><th>BAKENOHANA acc</th><th>µs/step</th></tr>" + "".join(
            f"<tr><td>{float(r['time/total_timesteps']) / 1e6:.2f}M</td><td>{float(r['val/acc']) * 100:.2f}%</td>"
            f"<td>{float(r['val/miss']):.0f}</td><td>{float(r['eval/acc']) * 100:.2f}%</td>"
            + (f"<td>{per_step[id(r)]:.0f}</td></tr>" if id(r) in per_step else "<td>—</td></tr>") for r in val[-8:][::-1]) + "</table>")
    if os.path.exists(results):
        md = open(results, encoding="utf-8").read()
        out.append("<pre>" + html.escape(md[:md.find("## Real maps")].strip()) + "</pre>")
    return "".join(out)


def page():
    status = open("runs/status.txt", encoding="utf-8").read() if os.path.exists("runs/status.txt") else "(no status note)"
    runs = sorted((d for d in glob.glob("runs/*/") if os.path.exists(os.path.join(d, "progress.csv"))), key=os.path.getmtime, reverse=True)
    body = "".join(run_section(d.rstrip("/\\")) for d in runs[:4])
    return (f"<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
            f"<meta http-equiv=refresh content=15><title>osu_rl status</title><style>{STYLE}</style>"
            f"<h1>osu_rl status</h1><p class=muted>{time.strftime('%Y-%m-%d %H:%M:%S')} · refreshes every 15 s</p>"
            f"<h2>now</h2><pre>{html.escape(status)}</pre>{body}")


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        data = page().encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("10.255.255.255", 1))  # no packet is sent; picks the LAN interface
    print(f"http://localhost:{PORT}  http://{s.getsockname()[0]}:{PORT}")
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
