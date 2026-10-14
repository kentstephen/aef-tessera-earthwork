"""The abroad recipe's three models, its TESSERA gate bounds and the radar-safe rule's cutoff (60% of held-out
digging found) as plain arrays for sql_probe.py: uv run python pyodide_probe/export_model.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import final

m = final("abroad")
out = {f"{w}_{k}": v for w, lr in m["models"].items() for k, v in (("w", lr.coef_[0].astype(np.float32)), ("b", np.float32(lr.intercept_[0])))}
np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_abroad.npz"), **out,
         bounds=np.array(m["bounds"], np.float32), cutoff=np.float32(m["cutoff"][0.6]["either_radar"]))
