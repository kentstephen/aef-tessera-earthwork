// Runs one of the probes in Pyodide under Node: node run.mjs fetch_reader.py
// (npm install pyodide@314.0.7 first; the same Pyodide line marimo's WASM build bundles)
import { loadPyodide } from "pyodide";
import fs from "fs";
const t0 = Date.now();
const py = await loadPyodide();
await py.loadPackage(["numpy", "zarr", "numcodecs", "crc32c"]);
console.log("load s", (Date.now() - t0) / 1000);
await py.runPythonAsync(fs.readFileSync(process.argv[2] || "fetch_reader.py", "utf8"));
