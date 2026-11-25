"""Closed-loop trips on the local route (upper block of the closed-loop table, Section 5).

Reads one row per trip and prints, per configuration: trips that reached the end of the
route without a takeover, where each takeover happened, the median plan age, and the
steering-wheel rate (95th percentile within a trip, median over trips).

    python closed_loop_table.py
"""
import csv, os
from collections import OrderedDict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ORDER = ["DriveVLA-W0", "LTF", "LTF, plans held 0.5 s longer", "LTF, plans held 1.0 s longer",
         "LTF, plans held 2.0 s longer", "LTF, side views blacked out", "LTF fine-tuned on local data"]


def main():
    arms = OrderedDict((k, []) for k in ORDER)
    with open(os.path.join(HERE, "..", "data", "closed_loop", "trips.csv")) as f:
        for r in csv.DictReader(f):
            arms[r["configuration"]].append(r)
    print(f"{'configuration':30s} {'reached':>8s} {'takeover at (m)':>16s} {'plan age (s)':>12s} {'steer (deg/s)':>13s} {'expired':>8s}")
    for k, T in arms.items():
        ok = sum(int(t["reached_end"]) for t in T)
        tk = ", ".join(f"{float(t['takeover_at_m']):.1f}" for t in T if t["takeover_at_m"]) or "--"
        age = np.median([float(t["plan_age_p50_s"]) for t in T])
        rate = np.median([float(t["steer_rate_p95_deg_s"]) for t in T])
        exp = np.mean([float(t["expired_plan_share"]) for t in T])
        print(f"{k:30s} {f'{ok} of {len(T)}':>8s} {tk:>16s} {age:12.2f} {rate:13.1f} {exp:8.0%}")


if __name__ == "__main__":
    main()
