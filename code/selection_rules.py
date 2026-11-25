"""Selection rules under leave-one-city-out (the selection-rule table of the appendix).

For each deployment profile (camera set and deadline) and each held-out city, a rule picks
one candidate; its regret is the PDMS of the best candidate that passes both gates in the
held-out city minus the PDMS of the pick, averaged over the four folds. A negative regret
means the pick violates a gate and scores higher open loop than any feasible candidate.

Reproduces the rows that need only benchmark scores and gate outcomes: top benchmark score
without gates, top benchmark score among the candidates that pass both gates (Stage 4),
the expected regret of a uniform pick among them, and the oracle. The probe-predictor rows
need the probe measurements and are not part of this package.

    python selection_rules.py
"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "..", "data", "selection", "candidates.json")))
CITIES, PC, GLOB, CON = D["cities"], D["pdms_per_city"], D["pdms_global"], D["contracts"]


def gates(a, prof):
    c, p = CON[a], D["profiles"][prof]
    sensing = c["cameras"] in p["rig_cams"] and (not c["lidar"] or p["rig_lidar"])
    return sensing, c["latency_ms"] <= p["budget_ms"]


def mean_regret(pick, feasible):
    return float(np.mean([max(PC[a][h] for a in feasible) - PC[pick(h)][h] for h in CITIES]))


def main():
    allc = list(PC)
    top = max(allc, key=GLOB.get)
    print(f"{'profile':13s} {'feasible set':52s} {'score-first':>20s} {'Stage 4':>8s} {'uniform':>8s} {'oracle':>7s}")
    for prof in D["profiles"]:
        G = [a for a in allc if not CON[a]["blind"] and all(gates(a, prof))]
        s, t = gates(top, prof)
        viol = "" if (s and t) else (" (sensing)" if not s else " (latency)")
        if not G:
            print(f"{prof:13s} {'(empty)':52s} {top + viol:>20s} {'--':>8s} {'--':>8s} {'--':>7s}")
            continue
        sf = mean_regret(lambda h: top, G)
        st4 = mean_regret(lambda h: max(G, key=GLOB.get), G)
        uni = float(np.mean([np.mean([max(PC[a][h] for a in G) - PC[b][h] for b in G]) for h in CITIES]))
        orc = mean_regret(lambda h: max(G, key=lambda a: PC[a][h]), G)
        print(f"{prof:13s} {', '.join(G):52s} {f'{sf:+.2f}{viol}':>20s} {st4:8.2f} {uni:8.2f} {orc:7.2f}")


if __name__ == "__main__":
    main()
