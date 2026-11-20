"""Run the benchmark v1 run list (B-8). Restart-safe: a run
whose metrics.json exists is skipped. Seed 0 of every configuration runs
first, so a complete comparison exists early; seeds 1 and 2 follow."""

from __future__ import annotations

import time
import traceback

from .run import RESULTS, run

CONFIGS_N7 = [
    ("L0", "M0", "core"), ("L1", "M0", "core"),
    ("L4", "M0", "core"), ("L4", "M2", "core"), ("L4", "M3", "core"), ("L4", "M1", "core"),
    ("L2", "M0", "core"), ("L2", "M2", "core"), ("L2", "M1", "core"),
    ("L2", "M3", "core"),  # B-8 (revised): added after the first results
    ("L3", "M0", "core"),
    ("L4", "M0", "full"),
]
CONFIGS_N30 = [("L1", "M0", "core"), ("L4", "M0", "core"), ("L2", "M0", "core")]
CONFIGS_N14 = CONFIGS_N30  # R-5: added after the 7- and 30-day runs
DETERMINISTIC = {"L0", "L1"}


def run_list() -> list[tuple[int, str, str, str, int]]:
    out = []
    for seed in (0, 1, 2):
        for n, configs in ((7, CONFIGS_N7), (30, CONFIGS_N30), (14, CONFIGS_N14)):
            for level, method, fs in configs:
                if level in DETERMINISTIC and seed > 0:
                    continue
                out.append((n, level, method, fs, seed))
    return out


def done(n, level, method, fs, seed) -> bool:
    return (RESULTS / f"N{n}" / f"{level}-{method}-{fs}-seed{seed}" / "metrics.json").exists()


def main() -> None:
    for item in run_list():
        if done(*item):
            print("done   ", item, flush=True)
            continue
        print("running", item, time.strftime("%H:%M:%S"), flush=True)
        try:
            r = run(*item)
            v = r["metrics"]["valid_threshold_1pct"]
            print(f"  -> detection {v['detection_rate']:.3f} at FAR {v['false_alarm_rate']:.4f}, "
                  f"{r['run']['seconds']:.0f}s", flush=True)
        except Exception:
            traceback.print_exc()
            print("  !! failed, continuing", flush=True)
