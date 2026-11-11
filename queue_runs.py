"""Experiment Queue Runner for osu_rl.

Allows scheduling and running multiple training experiments sequentially (overnight or in the background).
Reads the queue file once at start (append jobs before launching),
and automatically updates runs/status.txt for the live web dashboard (dashboard.py).

Usage examples:
    # 1. Run multiple seeds of the same experiment:
    python queue_runs.py experiments/016-lookahead.json --seeds 1 2 3

    # 2. Run multiple different experiments:
    python queue_runs.py experiments/015-tail-miss.json experiments/016-lookahead.json --seeds 1

    # 3. Run from a queue text file:
    python queue_runs.py --file queue.txt

    # 4. Preview queue without executing:
    python queue_runs.py experiments/016-lookahead.json --seeds 1 2 --dry-run
"""
import argparse
import datetime
import json
import os
import shlex
import subprocess
import sys
import time


def update_status(msg):
    """Updates runs/status.txt so dashboard.py shows live status on the web."""
    try:
        os.makedirs("runs", exist_ok=True)
        with open("runs/status.txt", "w", encoding="utf-8") as f:
            f.write(msg + "\n")
    except Exception as e:
        print(f"[queue] Warning: could not update status.txt: {e}")


def format_duration(seconds):
    s = int(seconds)
    if s >= 3600:
        return f"{s // 3600}h {s // 60 % 60:02d}m"
    return f"{s // 60}m {s % 60:02d}s"


def run_single_job(config_path, seed, device="hybrid", extra_args=None):
    """Runs one experiment seed via run.py."""
    cfg_name = os.path.splitext(os.path.basename(config_path))[0]
    job_name = f"{cfg_name}-s{seed}"
    cmd = [sys.executable, "-m", "state.run", config_path, "--seed", str(seed), "--device", device]
    if extra_args:
        cmd.extend(extra_args)

    print(f"\n{'='*70}")
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting Job: {job_name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*70}\n")

    t0 = time.time()
    try:
        res = subprocess.run(cmd, check=True)
        elapsed = time.time() - t0
        print(f"\n[OK] Job {job_name} finished successfully in {format_duration(elapsed)}.")

        # Read summary if available
        summary_path = os.path.join("results", "experiments", cfg_name, f"seed{seed}", "summary.json")
        summary_info = ""
        if os.path.exists(summary_path):
            try:
                with open(summary_path, encoding="utf-8") as f:
                    data = json.load(f)
                    real_acc = data.get("real", {}).get("acc", 0) * 100
                    real_fc = data.get("real", {}).get("fc", 0)
                    summary_info = f"Acc: {real_acc:.2f}%, FC: {real_fc}/26"
            except Exception:
                pass

        return True, elapsed, summary_info
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - t0
        print(f"\n[ERROR] Job {job_name} failed with returncode {e.returncode} after {format_duration(elapsed)}.")
        return False, elapsed, f"Exit code {e.returncode}"
    except Exception as e:
        elapsed = time.time() - t0
        print(f"\n[ERROR] Job {job_name} encountered error: {e}")
        return False, elapsed, str(e)


def main():
    parser = argparse.ArgumentParser(description="Queue and run multiple training experiments.")
    parser.add_argument("configs", nargs="*", help="Experiment config JSON paths")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0], help="List of seeds to run (default: [0])")
    parser.add_argument("--device", default="hybrid", choices=["cpu", "hybrid"], help="Compute device (default: hybrid)")
    parser.add_argument("--file", "-f", help="Queue file containing commands/configs (one per line)")
    parser.add_argument("--dry-run", action="store_true", help="Print the queue list without running")
    args = parser.parse_args()

    jobs = []

    # 1. Parse jobs from CLI configs and seeds
    if args.configs:
        for cfg in args.configs:
            for s in args.seeds:
                jobs.append((cfg, s, args.device, []))

    # 2. Parse jobs from file
    if args.file and os.path.exists(args.file):
        with open(args.file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = shlex.split(line)
                # Parse like: experiments/016-lookahead.json --seed 1 ...
                cfg = parts[0]
                s = 0
                dev = args.device
                rest = []
                idx = 1
                while idx < len(parts):
                    if parts[idx] == "--seed" and idx + 1 < len(parts):
                        s = int(parts[idx + 1])
                        idx += 2
                    elif parts[idx] == "--device" and idx + 1 < len(parts):
                        dev = parts[idx + 1]
                        idx += 2
                    else:
                        rest.append(parts[idx])
                        idx += 1
                jobs.append((cfg, s, dev, rest))

    if not jobs:
        print("No jobs found in queue! Provide config paths or --file <queue.txt>.")
        print("Example: python queue_runs.py experiments/016-lookahead.json --seeds 1 2")
        return

    print(f"\n=== Experiment Queue Loaded ({len(jobs)} jobs) ===")
    for i, (cfg, s, dev, rest) in enumerate(jobs, 1):
        extra = f" (extra: {' '.join(rest)})" if rest else ""
        print(f"  {i}. {cfg} [seed={s}, device={dev}]{extra}")
    print("=" * 45 + "\n")

    if args.dry_run:
        print("[Dry Run] Commands that would be executed:")
        for cfg, s, dev, rest in jobs:
            print(f"  python -m state.run {cfg} --seed {s} --device {dev} {' '.join(rest)}")
        return

    results_table = []
    total_start = time.time()

    for idx, (cfg, s, dev, rest) in enumerate(jobs, 1):
        cfg_name = os.path.splitext(os.path.basename(cfg))[0]
        curr_title = f"{cfg_name}-s{s}"

        status_msg = (
            f"Queue: Job [{idx}/{len(jobs)}] in progress: {curr_title}\n"
            f"Started: {datetime.datetime.now().strftime('%H:%M:%S')} | Device: {dev}\n"
            f"Remaining in queue: {len(jobs) - idx} job(s)"
        )
        update_status(status_msg)

        success, duration, info = run_single_job(cfg, s, dev, rest)
        status_str = "PASS" if success else "FAIL"
        results_table.append((curr_title, status_str, format_duration(duration), info))

    total_time = time.time() - total_start
    final_msg = f"Queue Completed: {len(jobs)} jobs finished in {format_duration(total_time)}."
    update_status(final_msg)

    # Print final summary table
    print("\n" + "=" * 70)
    print(f"=== Queue Execution Summary ({format_duration(total_time)}) ===")
    print("=" * 70)
    print(f"{'Job Name':<30} | {'Status':<6} | {'Duration':<10} | {'Details'}")
    print("-" * 70)
    for name, st, dur, detail in results_table:
        print(f"{name:<30} | {st:<6} | {dur:<10} | {detail}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
