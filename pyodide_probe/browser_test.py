"""In real browsers (Playwright: Chromium, WebKit for Safari, Firefox): how much memory a page's WebAssembly and Pyodide
can take, and whether the crawl's peaks table survives a reload in the browser's own storage (IndexedDB, through
Pyodide's IDBFS mount; DuckDB writes and reads it as Parquet).

Run: uv run --no-project --with playwright python pyodide_probe/browser_test.py [chromium webkit firefox brave]
(playwright install chromium webkit firefox, once)
"""
import functools
import http.server
import json
import os
import sys
import tempfile
import threading
import time

from playwright.sync_api import sync_playwright

BRAVE = "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser"  # Chromium underneath, with its own shields
PYODIDE = "https://cdn.jsdelivr.net/pyodide/v314.0.7/full/pyodide.mjs"
TILES = int(os.environ.get("PEAK_TILES", "1000"))  # tiles of peaks saved: 1,000 tiles is about 10,500 square km

PAGE = f"""<!doctype html><meta charset="utf-8"><title>probe</title>
<script type="module">
import {{ loadPyodide }} from "{PYODIDE}";
window.loadPy = async (pk) => {{ const py = await loadPyodide(); await py.loadPackage(pk); window.py = py; return true; }};
</script>"""

RAW = """() => {  // grow one WebAssembly memory 64 MB at a time until the engine refuses
  const m = new WebAssembly.Memory({initial: 1, maximum: 65536});
  let pages = 1;
  try { while (pages < 65536) { m.grow(1024); pages += 1024; } } catch (e) {}
  return Math.round(pages * 65536 / 1e6);
}"""

PY_ALLOC = """async () => window.py.runPythonAsync(`
import numpy as np, pyodide_js
held, mb = [], 0
try:
    while True:
        held.append(np.ones(100_000_000, np.uint8)); mb += 100
except MemoryError:
    pass
heap = pyodide_js._module.HEAP8.length
del held
str(mb) + " " + str(round(heap / 1e6))
`)"""

SAVE = f"""async () => {{
  const py = window.py, t0 = performance.now();
  py.FS.mkdirTree("/persist"); py.FS.mount(py.FS.filesystems.IDBFS, {{}}, "/persist");
  await py.runPythonAsync(`
import duckdb, numpy as np
con = duckdb.connect()
n = {TILES} * 1056
con.sql(f"CREATE TABLE peaks AS SELECT (i // 33)::INTEGER AS brow, (i % 33)::INTEGER AS bcol, random()::FLOAT AS peak FROM range({{n}}) t(i)")
con.sql("COPY peaks TO '/persist/peaks.parquet' (FORMAT parquet)")
`);
  const t1 = performance.now();
  await new Promise((ok, no) => py.FS.syncfs(false, e => e ? no(e) : ok()));
  const est = await navigator.storage.estimate();
  return {{ write_ms: Math.round(t1 - t0), sync_ms: Math.round(performance.now() - t1),
           bytes: py.FS.stat("/persist/peaks.parquet").size, quota_mb: Math.round(est.quota / 1e6) }};
}}"""

LOAD = """async () => {
  const py = window.py, t0 = performance.now();
  py.FS.mkdirTree("/persist"); py.FS.mount(py.FS.filesystems.IDBFS, {}, "/persist");
  await new Promise((ok, no) => py.FS.syncfs(true, e => e ? no(e) : ok()));
  const t1 = performance.now();
  const rows = await py.runPythonAsync(`
import duckdb
duckdb.sql("SELECT count(*) FROM '/persist/peaks.parquet'").fetchone()[0]
`);
  return { restore_ms: Math.round(t1 - t0), read_ms: Math.round(performance.now() - t1), rows: Number(rows) };
}"""


def serve(root):
    h = functools.partial(http.server.SimpleHTTPRequestHandler, directory=root)
    h.log_message = lambda *a: None
    s = http.server.ThreadingHTTPServer(("127.0.0.1", 0), h)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    return s


def main():
    root = tempfile.mkdtemp()
    open(os.path.join(root, "index.html"), "w").write(PAGE)
    srv = serve(root)
    url = f"http://127.0.0.1:{srv.server_port}/index.html"
    out = {}
    with sync_playwright() as p:
        for name in sys.argv[1:] or ("chromium", "webkit", "firefox"):
            r = out[name] = {}
            try:
                # a fresh temporary profile, never the user's own
                b = p.chromium.launch(executable_path=BRAVE) if name == "brave" else getattr(p, name).launch()
                r["version"] = b.version
                ctx = b.new_context()
                pg = ctx.new_page()
                pg.set_default_timeout(600_000)
                pg.goto(url)
                r["raw_wasm_memory_mb"] = pg.evaluate(RAW)
                t = time.time()
                pg.evaluate("async () => window.loadPy(['numpy'])")
                r["pyodide_numpy_load_s"] = round(time.time() - t, 1)
                mb, heap = pg.evaluate(PY_ALLOC).split()
                r["numpy_allocated_mb"], r["pyodide_heap_at_limit_mb"] = int(mb), int(heap)
                pg.reload()  # a fresh page for the storage test
                pg.evaluate("async () => window.loadPy(['numpy', 'duckdb'])")
                r["save"] = pg.evaluate(SAVE)
                pg.reload()
                pg.evaluate("async () => window.loadPy(['numpy', 'duckdb'])")
                r["after_reload"] = pg.evaluate(LOAD)
                b.close()
            except Exception as e:
                r["error"] = str(e).splitlines()[0][:300]
            print(name, json.dumps(r), flush=True)
    srv.shutdown()
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "browser_test.json")
    old = json.load(open(path)) if os.path.exists(path) else {}
    json.dump({**old, **out}, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
