"""Stage 2 (latency gate) and Stage 4 (gate-restricted ranking) on the vehicle.

Reads the per-plan timing logs of the six candidates (vehicle parked, full input path
running) and reproduces the onboard columns of the latency table in Appendix A.5 and the
ranking that selects DriveVLA-W0 with LTF as challenger.

End-to-end latency of a plan = time the plan is ready for the controller minus the
exposure time of the newest frame the plan used. For models that read several views at
one instant, the earliest of those simultaneous exposures is used.
The gate passes a candidate when the 95th percentile is at most the deadline.

    python latency_gate.py
"""
import csv, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "latency")
ORDER = ["LTF", "Latent-WAM", "DriveVLA-W0", "ReCogDrive", "DriveVA", "Alpamayo-1.5"]


def load(name):
    with open(os.path.join(DATA, name + ".csv")) as f:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]


def main():
    cfg = json.load(open(os.path.join(DATA, "gate_config.json")))
    deadline, warmup = cfg["deadline_s"], cfg["warmup_plans_dropped"]
    print(f"{'candidate':14s} {'plans':>6s} {'p50':>6s} {'p95':>6s} {'>dl':>5s} {'infer p50':>9s} {'gate value':>10s}  gate")
    feasible = []
    for name in ORDER:
        rows = load(name)[warmup:]
        e2e = np.array([r["plan_ready_s"] - (r["oldest_frame_exposure_s"] if r["simultaneous_views"]
                                             else r["newest_frame_exposure_s"]) for r in rows])
        p50, p95 = np.percentile(e2e, 50), np.percentile(e2e, 95)
        infer = np.percentile([r["inference_s"] for r in rows], 50)
        value = max(p95, cfg["driving_p95_s"].get(name, 0.0))
        ok = value <= deadline
        if ok and name in cfg["benchmark_pdms"]:
            feasible.append(name)
        print(f"{name:14s} {len(rows):6d} {p50:6.3f} {p95:6.3f} {int((e2e > deadline).sum()):5d} {infer:9.3f} "
              f"{value:10.3f}  {'pass' if ok else 'fail'}")
    pdms = cfg["benchmark_pdms"]
    by_score = sorted(pdms, key=pdms.get, reverse=True)
    ranked = sorted(feasible, key=pdms.get, reverse=True)
    print("\nranking by benchmark score alone :", ", ".join(f"{a} ({pdms[a]:.2f})" for a in by_score))
    print("gate-restricted ranking (Stage 4):", ", ".join(f"{a} ({pdms[a]:.2f})" for a in ranked))
    print(f"selected: {ranked[0]}; challenger: {ranked[1]}")


if __name__ == "__main__":
    main()
