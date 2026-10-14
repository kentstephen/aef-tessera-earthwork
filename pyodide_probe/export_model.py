"""The abroad recipe's three models, its TESSERA gate bounds and the cutoffs (60% of held-out digging found) of the
radar-safe rule and of each map layer, as plain arrays for the Pyodide probes and earthwork_map.py (models/):
uv run python pyodide_probe/export_model.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import final

m = final("abroad")
out = {f"{w}_{k}": v for w, lr in m["models"].items() for k, v in (("w", lr.coef_[0].astype(np.float32)), ("b", np.float32(lr.intercept_[0])))}
# each map layer's cutoff too (aggregate: either_radar, aef: aef_gated, tessera: tes), for earthwork_map.py's slider
cuts = {f"cut_{k}": np.float32(m["cutoff"][0.6][r]) for k, r in (("aggregate", "either_radar"), ("aef", "aef_gated"), ("tessera", "tes"))}
here = os.path.dirname(os.path.abspath(__file__))
for path in (os.path.join(here, "model_abroad.npz"), os.path.join(os.path.dirname(here), "models", "model_abroad.npz")):
    np.savez(path, **out, **cuts, bounds=np.array(m["bounds"], np.float32), cutoff=np.float32(m["cutoff"][0.6]["either_radar"]))
