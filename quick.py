"""Combining rules scored fast, from the base scores aggregate.py saved (SAVE_SCORES=1 writes data/scores/).

The same comparison as aggregate.py (each rule's cutoff set to find a share of moved ground at the held-out
digging sites, then unchanged ground lit up by kind) for rules that need no model taught, so a new rule is a
line in RULES and a run takes about a minute. Each digging and quiet site's scores came from models taught
without it, the test-only places' from models taught on every teaching site.

Run: uv run python quick.py [catch ...]   (default 0.6)
"""

import os
import sys

import numpy as np
from sklearn.metrics import average_precision_score

from aggregate import SCORES, cutoff

DESERT = ("place-sahara-dz", "place-riyadh-edge-sa", "place-cairo-desert-eg")


def gate_a(s, lo=0.05, w=0.10):
    return np.clip((s["aplain"] - lo) / w, 0, 1)


def gate_t(s, b, scale=1.0):
    lo, hi = b
    return np.clip((s["tplain"] - lo * scale) / max((hi - lo) * scale, 1e-6), 0, 1)


def either(s, b, lo=0.05, w=0.10, scale=1.0):
    return np.maximum(gate_a(s, lo, w), gate_t(s, b, scale))


RULES = {
    "aef_gated": lambda s, b: s["aef"] * gate_a(s),
    "mean": lambda s, b: (s["aef"] * gate_a(s) + s["tes"]) / 2,
    "both": lambda s, b: s["both"],
    "both_either": lambda s, b: s["both"] * either(s, b),
    # a gentler either gate: opens sooner, so digging with little change keeps more of its score
    "either_soft": lambda s, b: s["both"] * either(s, b, 0.03, 0.05, 0.6),
    "either_sqrt": lambda s, b: s["both"] * np.sqrt(either(s, b)),
    # the joint model held under the mean: quiet where the joint model is (crops, burns), and where the mean
    # is (deserts, held down by AlphaEarth's gate)
    "min_both_mean": lambda s, b: np.minimum(s["both"], (s["aef"] * gate_a(s) + s["tes"]) / 2),
    "geo_both_mean": lambda s, b: np.sqrt(s["both"] * (s["aef"] * gate_a(s) + s["tes"]) / 2),
    "min_both_either_mean": lambda s, b: np.minimum(s["both"] * either(s, b), (s["aef"] * gate_a(s) + s["tes"]) / 2),
}


def main():
    catches = [float(v) for v in sys.argv[1:]] or [0.6]
    Z = {f[:-4]: dict(np.load(f"{SCORES}/{f}", allow_pickle=True)) for f in sorted(os.listdir(SCORES)) if f.endswith(".npz")}
    teach = [n for n, z in Z.items() if str(z["role"]) == "teach"]
    kinds = sorted({str(z["kind"]) for n, z in Z.items() if n not in teach})
    S = {n: {w: f(z, tuple(z["bounds"])).astype(np.float32) for w, f in RULES.items()} for n, z in Z.items()}
    for c in catches:
        print(f"\nfinding {c:.0%} of moved ground at held-out digging sites: AP, share of unchanged ground lit up, moved found")
        head = ["dig"] + kinds
        print(f"  {'rule':22s} {'AP':>5s}  " + " ".join(f"{k[:8]:>8s}" for k in head) + "  | found: island  space  | desert")
        for w in RULES:
            t = cutoff([S[n][w][Z[n]["y"] == 1] for n in teach], c)
            ap = np.mean([average_precision_score(Z[n]["y"], S[n][w]) for n in teach])
            fa, fd = {}, {}
            for n, z in Z.items():
                k = "dig" if n in teach else str(z["kind"])
                if (z["y"] == 0).sum() >= 10:
                    fa.setdefault(k, []).append((S[n][w][z["y"] == 0] > t).mean())
                if n not in teach and (z["y"] == 1).sum() >= 10:
                    fd.setdefault(k, []).append((S[n][w][z["y"] == 1] > t).mean())
            des = np.mean([(S[n][w] > t).mean() for n in DESERT if n in S])
            row = " ".join(f"{np.mean(fa[k]) if k in fa else float('nan'):8.3f}" for k in head)
            found = "  ".join(f"{np.mean(fd[k]) if k in fd else float('nan'):5.2f}" for k in ("island", "space_dig"))
            print(f"  {w:22s} {ap:5.3f}  {row}  |        {found}  | {des:.3f}")


if __name__ == "__main__":
    main()
