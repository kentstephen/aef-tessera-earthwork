"""The three models taught on every teaching site, kept for scoring new ground, with each rule's cutoff.

Two recipes, as the held-out scores were made:
  us      the US digging sites and the US burns and dunes (aggregate.py's final models)
  abroad  those plus the island lidar and the satellite-laser labels (abroad_teach.py's final models)
Each recipe's cutoffs are set on its own held-out digging scores (data/scores, data/scores-abroad) so each rule
finds the same share of moved ground there. Fitted once and kept in data/models/<recipe>.pkl.

Use: from models import final; m = final("us"); m["models"]["both"], m["bounds"], m["cutoff"][0.6]["both_either"]
"""

import os
import pickle

import numpy as np

from aggregate import cutoff
from quick import RULES

HERE = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(HERE, "data", "models")
SCORE_DIRS = {"us": os.path.join(HERE, "data", "scores"), "abroad": os.path.join(HERE, "data", "scores-abroad")}
CATCH = (0.5, 0.6, 0.7)


def cutoffs(sd):
    """Each quick.py rule's cutoff at each catch rate, from the held-out digging scores in sd."""
    Z = [dict(np.load(f"{sd}/{f}", allow_pickle=True)) for f in sorted(os.listdir(sd)) if f.endswith(".npz")]
    Z = [z for z in Z if str(z["role"]) == "teach"]
    S = [{w: f(z, tuple(z["bounds"])) for w, f in RULES.items()} for z in Z]
    return {c: {w: float(cutoff([s[w][z["y"] == 1] for s, z in zip(S, Z)], c)) for w in RULES} for c in CATCH}


def final(recipe):
    path = f"{DIR}/{recipe}.pkl"
    if os.path.exists(path):
        out = pickle.load(open(path, "rb"))
        if set(out["cutoff"][CATCH[0]]) != set(RULES):  # a rule added since: cutoffs again
            out["cutoff"] = cutoffs(SCORE_DIRS[recipe])
            pickle.dump(out, open(path, "wb"))
        return out
    from abroad_teach import fit_rows, rows
    from aggregate import tes_gate_bounds
    T, W, D, teach, quiet, abroad = rows(("islands", "space") if recipe == "abroad" else ())
    D.clear()
    models = {w: fit_rows(T, W, list(T), w) for w in ("aef", "tes", "both")}
    us = teach + quiet
    bounds = tes_gate_bounds(T, {n: np.arange(len(T[n]["y"])) for n in us}, us)
    out = {"models": models, "bounds": bounds, "cutoff": cutoffs(SCORE_DIRS[recipe]), "taught": list(T)}
    os.makedirs(DIR, exist_ok=True)
    pickle.dump(out, open(path, "wb"))
    return out


if __name__ == "__main__":
    for r in ("us", "abroad"):
        m = final(r)
        print(r, "bounds", np.round(m["bounds"], 3), "cutoffs at 60%", {w: round(v, 3) for w, v in m["cutoff"][0.6].items()})
