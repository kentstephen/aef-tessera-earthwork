/// <reference lib="webworker" />
// The view's analytics: Pyodide with numpy, DuckDB and h3 in this worker. The map hands over the
// AlphaEarth chunks it has already read (no second download); here each pixel is scored with the same
// model as the GPU's, labeled with its H3 res 9 cell, and DuckDB answers the card's questions.
//
// Messages in:  {type: "init", models}
//               {type: "chunk", key, r, c, w, h, first, last}   (Int8Array [64, h, w] each)
//               {type: "query", id, keys, bbox, model, divCells, divNames}
// Messages out: {type: "status", text} | {type: "ready"} | {type: "result", id, result} | {type: "error", id, error}

const PYODIDE = "https://cdn.jsdelivr.net/pyodide/v314.0.7/full/pyodide.mjs";

const PY = `
import json
import numpy as np
import duckdb
import pyarrow as pa
import h3

R = 8.983111749910169e-05
X0, Y0 = -180.0, 83.68570533713473
CHUNK = 256
RES = 9
GATE_LO, GATE_HI = 0.05, 0.15

models = {}
chunks = {}   # key -> (first [64, h, w] int8, last, r, c, w, h)
scored = {}   # (key, model) -> float32 [h * w], NaN where a year has no data
cells = {}    # key -> uint64 [h * w], each pixel's H3 res 9 cell
con = duckdb.connect()

def _py(x):
    """A JavaScript value as Python (typed arrays as memoryviews, arrays as lists)."""
    return x.to_py() if hasattr(x, "to_py") else x

def set_models(js):
    for k, m in json.loads(js).items():
        models[k] = (np.asarray(m["w"], np.float32), float(m["b"]))

def add_chunk(key, r, c, w, h, first, last):
    first = np.frombuffer(_py(first), np.int8).copy().reshape(64, h, w)
    last = np.frombuffer(_py(last), np.int8).copy().reshape(64, h, w)
    chunks[key] = (first, last, r, c, w, h)
    # a few dozen chunks at most, as the map keeps
    while len(chunks) > 64:
        old = next(iter(chunks))
        chunks.pop(old)
        cells.pop(old, None)
        for k in [k for k in scored if k[0] == old]:
            scored.pop(k)

def _deq(v):
    f = v.astype(np.float32) / 127.5
    return f * np.abs(f)

def score(key, model):
    k = (key, model)
    if k in scored:
        return scored[k]
    first, last, r, c, w, h = chunks[key]
    W, B = models[model]
    F = _deq(first).reshape(64, -1).T
    L = _deq(last).reshape(64, -1).T
    bad = (first == -128).any(0).ravel() | (last == -128).any(0).ravel()
    b = F / np.maximum(np.linalg.norm(F, axis=1), 1e-9)[:, None]
    a = L / np.maximum(np.linalg.norm(L, axis=1), 1e-9)[:, None]
    z = np.c_[b, a, a * b, (a - b) ** 2] @ W + B
    cos = (a * b).sum(1)
    gate = np.clip(((1 - cos) - GATE_LO) / (GATE_HI - GATE_LO), 0, 1)
    s = np.where(bad, np.nan, gate / (1 + np.exp(-z))).astype(np.float32)
    scored[k] = s
    return s

def lonlat(key):
    first, last, r, c, w, h = chunks[key]
    ys, xs = np.divmod(np.arange(w * h), w)
    lat = Y0 - (r * CHUNK + ys + 0.5) * R
    lon = X0 + (c * CHUNK + xs + 0.5) * R
    return lon, lat

def cells_of(key):
    if key not in cells:
        lon, lat = lonlat(key)
        cells[key] = np.fromiter((h3.str_to_int(h3.latlng_to_cell(la, lo, RES)) for la, lo in zip(lat, lon)), np.uint64, len(lat))
    return cells[key]

def query(keys, bbox, model, div_cells, div_names):
    keys, div_names = list(_py(keys)), list(_py(div_names))
    W_, S_, E_, N_ = list(_py(bbox))
    div_cells = np.frombuffer(_py(div_cells), np.uint64) if div_cells is not None else np.zeros(0, np.uint64)
    parts = {"cell": [], "score": [], "ha": []}
    for key in keys:
        if key not in chunks:
            continue
        lon, lat = lonlat(key)
        m = (lon >= W_) & (lon <= E_) & (lat >= S_) & (lat <= N_)
        if not m.any():
            continue
        parts["cell"].append(cells_of(key)[m])
        parts["score"].append(score(key, model)[m])
        # a pixel is R degrees square: its area shrinks with the cosine of its latitude
        parts["ha"].append((R * 111_320.0) ** 2 * np.cos(np.radians(lat[m])) / 10_000)
    if not parts["cell"]:
        return json.dumps({"pixels": 0})
    px = pa.table({
        "cell": pa.array(np.concatenate(parts["cell"])),
        "score": pa.array(np.concatenate(parts["score"])),
        "ha": pa.array(np.concatenate(parts["ha"]).astype(np.float32)),
    })
    div = pa.table({"cell": pa.array(div_cells), "name": pa.array(div_names, pa.string())})
    con.register("px", px)
    con.register("div", div)
    total = con.sql("""
        SELECT count(*) AS pixels,
               sum(ha) FILTER (WHERE NOT isnan(score)) AS read_ha,
               coalesce(sum(ha) FILTER (WHERE score > 0.5), 0) AS likely_ha
        FROM px""").fetchone()
    digs = con.sql("""
        SELECT p.cell, sum(p.ha) FILTER (WHERE p.score > 0.5) AS ha, max(p.score) AS peak, any_value(d.name) AS name
        FROM px p LEFT JOIN div d USING (cell)
        GROUP BY p.cell
        HAVING ha > 0
        ORDER BY ha DESC LIMIT 5""").fetchall()
    places = con.sql("""
        SELECT coalesce(d.name, '') AS name, sum(p.ha) FILTER (WHERE p.score > 0.5) AS ha, sum(p.ha) AS of_ha
        FROM px p LEFT JOIN div d USING (cell)
        GROUP BY 1
        HAVING ha > 0
        ORDER BY ha DESC LIMIT 6""").fetchall()
    con.unregister("px")
    con.unregister("div")
    return json.dumps({
        "pixels": int(total[0]), "read_ha": float(total[1] or 0), "likely_ha": float(total[2] or 0),
        "digs": [{"cell": h3.int_to_str(int(c)), "ha": float(a), "peak": float(p), "name": n or ""} for c, a, p, n in digs],
        "places": [{"name": n, "ha": float(a), "of_ha": float(o)} for n, a, o in places],
    })
`;

type Py = {
  runPython: (code: string) => unknown;
  globals: { get: (name: string) => (...args: unknown[]) => unknown };
  loadPackage: (pkgs: string[]) => Promise<void>;
  toPy: (v: unknown) => unknown;
};

let py: Py | null = null;
const queue: MessageEvent[] = [];
const say = (text: string) => postMessage({ type: "status", text });

async function boot(models: unknown) {
  say("Starting DuckDB in your browser…");
  const { loadPyodide } = await import(/* @vite-ignore */ PYODIDE);
  const p = (await loadPyodide()) as Py;
  await p.loadPackage(["numpy", "duckdb", "pyarrow", "h3"]);
  p.runPython(PY);
  p.globals.get("set_models")(JSON.stringify(models));
  py = p;
  postMessage({ type: "ready" });
  for (const e of queue.splice(0)) handle(e);
}

function handle(e: MessageEvent) {
  const d = e.data;
  if (!py) {
    queue.push(e);
    return;
  }
  if (d.type === "chunk") {
    py.globals.get("add_chunk")(d.key, d.r, d.c, d.w, d.h, d.first, d.last);
  } else if (d.type === "query") {
    try {
      const out = py.globals.get("query")(d.keys, d.bbox, d.model, d.divCells, d.divNames) as string;
      postMessage({ type: "result", id: d.id, result: JSON.parse(out) });
    } catch (err) {
      const msg = String((err as Error).message ?? err).trim().split("\n");
      postMessage({ type: "error", id: d.id, error: msg[msg.length - 1] });
    }
  }
}

onmessage = (e: MessageEvent) => {
  if (e.data.type === "init") {
    boot(e.data.models).catch((err) => postMessage({ type: "error", id: -1, error: String(err?.message ?? err) }));
    return;
  }
  handle(e);
};
