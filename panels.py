"""Site panels: the lidar's truth, the Earthwork model in AlphaEarth, in TESSERA, and where they agree.

For each site, five maps on AlphaEarth's 10 m grid over a dark ground:
  lidar      moved more than 0.5 m (light) and unchanged (gray), from the two 3DEP flights
  AlphaEarth the Earthwork model's score with the map's change gate, the full cividis ramp
  TESSERA    the same model taught on TESSERA, the full viridis ramp
  agreement  who calls it moved: AlphaEarth only (blue), both (yellow), TESSERA only (green), set by the
             difference of the two scores; opacity is the stronger of the two, so ground neither calls
             moved fades out instead of reading as agreement. Agreement is the lightest stop on purpose.
  agree      the agreement score itself, the lower of the two (what the false alarm numbers measure),
             dark to the agreement panel's "both" yellow
A digging site is scored by models taught on the other sites; test-only ground (farms, marshes, the storm
coast, the islands and the other places abroad) by models taught on every digging and quiet site. A place with
no truth drops the lidar map.

Run: uv run python panels.py [site ...]   (writes results/panels/<site>.png)
"""

import json
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from compare import feats, fit_models, gate, load, teaching_names
from tessera_sites import OUT

HERE = os.path.dirname(os.path.abspath(__file__))
GROUND = "#0E0E10"
INK, MUTED = "#E8E8EA", "#9A9AA2"
AGREE = LinearSegmentedColormap.from_list("agree", ["#5B8BD6", "#FDE725", "#35B779"])
BOTH = LinearSegmentedColormap.from_list("both", ["#1C1C22", "#FDE725"])
SHOW = ["katy-tx", "huntsville-al", "las-vegas-se-nv", "mountain-pass-ca", "hamilton-farm-in", "fresno-farm-ca", "brazoria-tx"]


def score_all(models, d, chunk=150000):
    ok = np.flatnonzero(d["ok"])
    out = {w: np.full(len(d["ok"]), np.nan, np.float32) for w in ("aef", "tes")}
    for i in range(0, len(ok), chunk):
        sl = ok[i:i + chunk]
        out["aef"][sl] = gate(models["aef"].predict_proba(feats(d, sl, "aef"))[:, 1], d["aplain"][sl])
        out["tes"][sl] = models["tes"].predict_proba(feats(d, sl, "tes"))[:, 1]
    return out


def rgba(cmap, v, alpha):
    c = cmap(np.nan_to_num(v))
    c[..., 3] = np.where(np.isfinite(v), alpha, 0)
    return c


def panel(name, D, models):
    d = D[name]
    shape = json.loads(str(np.load(f"{OUT}/{name}.npz", allow_pickle=True)["meta"]))["shape"]
    s = score_all(models, d)
    A, T = (s[w].reshape(shape) for w in ("aef", "tes"))
    y = d["y"].reshape(shape)
    strength = np.fmax(A, T)
    diff = (T - A + 1) / 2  # 0 AlphaEarth only, .5 both, 1 TESSERA only

    truthless = d["kind"] == "notruth"
    fig, ax = plt.subplots(1, 4 if truthless else 5, figsize=(20 if truthless else 25, 5.6), facecolor=GROUND)
    truth = np.zeros((*shape, 4))
    truth[y == 0] = (0.32, 0.32, 0.35, 1)
    truth[y == 1] = (0.95, 0.95, 0.92, 1)
    layers = [
        ("lidar: moved > 0.5 m (light), unchanged (gray)", truth),
        ("AlphaEarth model (gated), cividis", rgba(plt.get_cmap("cividis"), A, 1.0)),
        ("TESSERA model, viridis", rgba(plt.get_cmap("viridis"), T, 1.0)),
        ("AlphaEarth only · both · TESSERA only", rgba(AGREE, diff, np.clip((strength - 0.2) / 0.5, 0, 1))),
        ("agree: the lower of the two scores", rgba(BOTH, np.fmin(A, T), 1.0)),
    ][truthless:]
    for a, (title, img) in zip(ax, layers):
        a.set_facecolor(GROUND)
        a.imshow(img, interpolation="nearest")
        a.set_title(title, color=INK, fontsize=11, loc="left")
        a.set_xticks([]), a.set_yticks([])
        for sp in a.spines.values():
            sp.set_visible(False)
    r = {}
    if os.path.exists(os.path.join(HERE, "results", "compare.json")):
        r = json.load(open(os.path.join(HERE, "results", "compare.json")))["sites"].get(name, {})
    note = f"{name} · {d['kind']} · AlphaEarth/TESSERA years {d['years'][0]} to {d['years'][1]}"
    if "ap" in r:
        note += f" · AP held out: AlphaEarth {r['ap']['aef_gated']:.2f}, TESSERA {r['ap']['tes']:.2f}, agree {r['ap']['agree']:.2f}"
    elif "false_alarm" in r:
        fa = r["false_alarm"]
        note += f" · unchanged ground scored > .5: AlphaEarth {fa['aef_gated']:.1%}, TESSERA {fa['tes']:.1%}, agree {fa['agree']:.1%}"
    fig.text(0.01, 0.02, note, color=MUTED, fontsize=10)
    # the agreement key: the ramp, and what fades
    k = fig.add_axes([0.52, 0.075, 0.2, 0.025] if truthless else [0.61, 0.075, 0.17, 0.025])
    k.imshow(np.linspace(0, 1, 256)[None], cmap=AGREE, aspect="auto")
    k.set_xticks([0, 127.5, 255], ["AlphaEarth only", "both", "TESSERA only"], color=MUTED, fontsize=9)
    k.set_yticks([])
    for sp in k.spines.values():
        sp.set_visible(False)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.12, wspace=0.03)
    os.makedirs(os.path.join(HERE, "results", "panels"), exist_ok=True)
    out = os.path.join(HERE, "results", "panels", f"{name}.png")
    fig.savefig(out, dpi=110, facecolor=GROUND)
    plt.close(fig)
    return out


def main():
    names = sys.argv[1:] or SHOW
    D = {n: load(n) for n in set(names) | set(teaching_names())}
    for n in names:
        print(panel(n, D, fit_models(D, held_out=n)))


if __name__ == "__main__":
    main()
