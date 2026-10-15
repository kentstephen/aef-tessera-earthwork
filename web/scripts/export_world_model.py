"""The world recipe's three layers (aggregate, AlphaEarth, TESSERA), its TESSERA gate bounds and each layer's
cutoff (60% of held-out digging found), from models/model_world.npz to web/public/models (run from the repo root):

    uv run python web/scripts/export_world_model.py models/model_world.npz web/public/models/world.json
"""
import json
import sys

import numpy as np

src, dst = sys.argv[1], sys.argv[2]
m = np.load(src)
f = lambda v: [float(x) for x in np.asarray(v, np.float32).ravel()]
out = {
    "aef": {"w": f(m["aef_w"]), "b": float(m["aef_b"])},
    "tessera": {"w": f(m["tes_w"]), "b": float(m["tes_b"])},
    # the first 256 weights take AlphaEarth's [b, a, a*b, (a-b)^2], the last 512 TESSERA's
    "aggregate": {"w": f(m["both_w"]), "b": float(m["both_b"])},
    # TESSERA's change (1 - cos) at which the aggregate's gate opens and is fully open
    "bounds": f(m["bounds"]),
    "cut": {k: float(m[f"cut_{k}"]) for k in ("aggregate", "aef", "tessera")},
}
assert len(out["aef"]["w"]) == 256 and len(out["tessera"]["w"]) == 512 and len(out["aggregate"]["w"]) == 768
with open(dst, "w") as fh:
    json.dump(out, fh, separators=(",", ":"))
print(dst, {k: len(v["w"]) for k, v in out.items() if isinstance(v, dict) and "w" in v}, out["bounds"], out["cut"])
