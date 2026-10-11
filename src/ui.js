// The Earthwork map's interface, carried over from aef-earthwork's earthwork.py (its ChangeMap widget)
// with the kernel's parts done here: the earthwork is scored on the GPU per 10 m pixel (earthwork.ts)
// instead of folded to hexagons, and the Sentinel-2 footprints come from the STAC search in the page (s2.ts)
import * as maplibregl from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import maplibreWorkerUrl from "maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url";
import {MapLibreOverlay as MapboxOverlay} from "@deck.gl/maplibre";
import {BitmapLayer, PathLayer} from "@deck.gl/layers";
import {TileLayer} from "@deck.gl/geo-layers";
import {COGLayer, MosaicLayer} from "@developmentseed/deck.gl-geotiff";
import {CreateTexture} from "@developmentseed/deck.gl-raster/gpu-modules";
import {DecoderPool, GeoTIFF, PerOriginSemaphore} from "@developmentseed/geotiff";
import GeoTiffWorker from "@developmentseed/geotiff/pool/worker?worker";
import {latLngToCell, getResolution, cellToBoundary, cellToLatLng, isValidCell, polygonToCells} from "h3-js";
import {Protocol as PMProtocol} from "pmtiles";
maplibregl.setWorkerUrl(maplibreWorkerUrl);
maplibregl.addProtocol("pmtiles", new PMProtocol().tile);

// dark basemap, trying it; light was
// https://basemaps.cartocdn.com/gl/positron-gl-style/style.json
const STYLE = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json";
const FONTS = "https://fonts.googleapis.com/css2?family=Instrument+Sans:wdth,wght@75..100,400..700&display=swap";
const rgba = (c, a) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;
const fmt = (n) => Number(n).toLocaleString("en-US");
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
const INK = [230, 233, 236];

function bytesOf(v) {
  if (!v) return null;
  if (v instanceof DataView) return new Uint8Array(v.buffer, v.byteOffset, v.byteLength);
  if (v instanceof ArrayBuffer) return new Uint8Array(v);
  if (v.buffer) return new Uint8Array(v.buffer, v.byteOffset || 0, v.byteLength);
  return null;
}
const copyOf = (u8) => u8.buffer.slice(u8.byteOffset, u8.byteOffset + u8.byteLength);
const el_ = (tag, cls, html) => { const e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; };
const ICON = {
  search: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/></svg>',
  more: '<svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><circle cx="5" cy="12" r="1.8"/><circle cx="12" cy="12" r="1.8"/><circle cx="19" cy="12" r="1.8"/></svg>',
  expand: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5"/></svg>',
  chev: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><path d="M6 9l6 6 6-6"/></svg>',
  pair: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="4" width="18" height="16" rx="2"/><path d="M12 4v16"/></svg>',
  shrink: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 4v5H4M15 4v5h5M9 20v-5H4M15 20v-5h5"/></svg>',
};
// how much a hexagon moved, in words, from its 0..1 level in this view
const howMuch = (t) => t >= 0.75 ? "a lot" : t >= 0.4 ? "a fair amount" : t >= 0.15 ? "a little" : "barely";
const FAIR = 1 + Math.round(254 * 0.4);  // the level byte at "a fair amount"
const HB = 17;  // bytes per hexagon in hattrs (the 9th: 1 on built ground; 10 to 16 the model's; 17 earthwork, see _paint)
// kinds of change, largest first: Okabe-Ito, made to stay apart for
// red-weak and other color vision; quiet ground (0) faint gray
const KIND_RGB = [[230, 159, 0], [86, 180, 233], [0, 158, 115], [240, 228, 66], [0, 114, 178], [204, 121, 167]];
const kindCss = (k) => `rgb(${KIND_RGB[(k - 1) % KIND_RGB.length].join(",")})`;
const lcPair = (a, b) => a === b ? `stays ${a}` : `${a} → ${b}`;
// [[year, class], ..] in runs: "<b>cropland</b> 2021 to 2022, then <b>built-up</b> 2023 to 2025"
function readRuns(rs) {
  const out = [];
  for (const [y, nm] of rs) {
    const last = out[out.length - 1];
    if (last && last.nm === nm && last.y1 === y - 1) last.y1 = y;
    else out.push({nm, y0: y, y1: y});
  }
  return out.map((r) => `<b>${r.nm ? r.nm : "unread"}</b> ${r.y0 === r.y1 ? r.y0 : `${r.y0} to ${r.y1}`}`).join(", then ");
}
// the whole history, when read (zoomed in): codes from _trajectory
const HIST_WORD = {1: "one step, then held", 2: "came back", 3: "changes this much most years", 4: "kept changing after", 5: "too recent to tell"};
function histText(code, k, hy, ratio) {
  if (code === 1) return `One step into ${k}, then it held through ${hy[1]}.`;
  if (code === 2) return `It changed into ${k}, then came back: by ${hy[1]} it is closer to the ground before than to ${k}. Fields, water and seasons do this; buildings rarely do.`;
  if (code === 3) return `It changes about this much most years${ratio != null ? ` (this step is ${ratio.toFixed(1)} times its usual step, ${hy[0]} to ${hy[1]})` : ""}: ground that turns over, like a field.`;
  if (code === 4) return `It kept changing after ${k}, more than the usual step most years, the way a site still building out, a quarry or a mine does.`;
  if (code === 5) return `The change is into ${hy[1]}, the last year AlphaEarth has: too recent to tell whether it holds.`;
  return "";
}

export function render({model, el, earthwork, s2Items, credits, analytics}) {
  let cfg = {};
  try { cfg = JSON.parse(model.get("config") || "{}"); } catch (e) { cfg = {}; }
  if (!document.getElementById("at-fonts")) {
    const f = document.createElement("link"); f.id = "at-fonts"; f.rel = "stylesheet"; f.href = FONTS; document.head.appendChild(f);
  }

  const S2Y = cfg.s2_years || [2022, 2023, 2024, 2025];
  const VIR = (cfg.viridis || "440154fde725").match(/.{6}/g).map((h) => [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16)));
  const vir = (t) => { t = Math.max(0, Math.min(1, t)) * (VIR.length - 1); const i = Math.min(VIR.length - 2, Math.floor(t)), f = t - i; return VIR[i].map((v, j) => Math.round(v + (VIR[i + 1][j] - v) * f)); };
  const virCss = (n) => Array.from({length: n}, (_, i) => `rgb(${vir(i / (n - 1)).join(",")})`).join(",");
  const A_FILL = cfg.alpha_fill || 235, A_QUIET = cfg.alpha_quiet || 70, A_DIM = 45;
  const EWZ = cfg.ew_zoom || 12;
  const HEXZ = cfg.hex_zoom || 9, HOLD_MS = cfg.hold_ms || 200, SLOP = cfg.hold_slop || 5;
  const st = {
    gmode: "earth", want: "earth", noPick: false, hideKinds: new Set(), focus: "all", y0: cfg.aef_from || 2022, y1: cfg.aef_to || 2025,
    imgYear: cfg.s2_year || S2Y[S2Y.length - 1], labels: true, s2scale: Number(cfg.s2_scale) || 1, s2comp: cfg.s2_comp || "tci",
    fit: !!cfg.fit, holding: false,
    // the pair (P) and its left side (Sentinel-2)
    pair: false, left: "s2",
  };

  // ---- the frame ----------------------------------------------------
  const root = el_("div", "at");
  const pane = el_("div", "at-pane");
  const mapEl = el_("div", "at-map");
  const mapEl2 = el_("div", "at-map2");
  const bar = el_("div", "at-bar", "<i></i>");
  // an error stays until the next status; copy takes its full text, x closes it
  const msg = el_("div", "at-msg at-glass", '<span class="at-msg-t"></span><span class="at-msg-act"><button data-act="copy">copy</button><button data-act="x" title="Close">\u00d7</button></span>');
  const msgTx = msg.querySelector(".at-msg-t");
  msg.addEventListener("click", (e) => {
    const b = e.target && e.target.closest && e.target.closest("button[data-act]");
    if (!b) return;
    e.stopPropagation();
    if (b.getAttribute("data-act") === "x") { msg.style.display = "none"; return; }
    const t = msgTx.textContent;
    const done = () => { b.textContent = "copied"; setTimeout(() => { b.textContent = "copy"; }, 1200); };
    const fallback = () => { const ta = document.createElement("textarea"); ta.value = t; ta.style.position = "fixed"; ta.style.opacity = "0"; root.appendChild(ta); ta.select(); try { document.execCommand("copy"); done(); } catch (e2) {} ta.remove(); };
    if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(t).then(done, fallback); else fallback();
  });
  const creditsEl = el_("div", "at-credits", credits || "");
  pane.append(mapEl2, mapEl, bar, msg, creditsEl);
  root.appendChild(pane);
  el.appendChild(root);

  // top left: search, and the one control row for the hexagons
  const top = el_("div", "at-top");
  const search = el_("div", "at-search at-glass", ICON.search);
  const gc = el_("input"); gc.type = "search"; gc.placeholder = "Search";
  gc.title = "a place, or an H3 string (/)"; gc.autocomplete = "off"; gc.spellcheck = false;
  const hits = el_("div", "at-hits at-glass");
  const gcKey = el_("kbd", "", "/"); gcKey.title = "/ searches";
  search.append(gc, gcKey, hits);
  const panel = el_("div", "at-panel at-glass");
  // every panel folds; the fold is remembered in this browser
  const keep = (k, v) => { try { if (v === undefined) return localStorage.getItem("aef-lc-" + k) === "1"; localStorage.setItem("aef-lc-" + k, v ? "1" : "0"); } catch (e) {} return false; };
  const panelHd = el_("div", "at-hd", `<span class="t">Earthwork</span>`);
  const panelCb = el_("button", "at-cb", ICON.chev);
  const panelRamp = el_("i", "at-ramp ew-hramp");
  panelRamp.title = "the chance the ground was dug, filled or graded: unlikely to likely";
  const panelYears = el_("span", "ew-hy");
  panelHd.append(panelRamp, panelYears, panelCb);
  panel.appendChild(panelHd);
  const foldPanel = (on) => { panel.classList.toggle("collapsed", on); panelCb.title = on ? "open the Earthwork card" : "fold the Earthwork card"; keep("panel", on); };
  panelHd.onclick = (e) => { e.stopPropagation(); foldPanel(!panel.classList.contains("collapsed")); };
  foldPanel(keep("panel"));
  panelRamp.style.background = `linear-gradient(90deg,${virCss(8)})`;
  const rowOf = (label) => { const r = el_("div", "at-row"); if (label) r.appendChild(el_("span", "at-lab", label)); panel.appendChild(r); return r; };
  // Kinds of change is drawn only zoomed in with its land cover read (hmeta.kinds_ready), and only
  // once asked for: the map never switches to it by itself, its button flashes when it is ready
  //
  const kindsWait = () => st.want === "kinds" && !hmeta.kinds_ready;
  const drawnMode = () => kindsWait() ? "much" : st.want;
  const segOf = (row, items, isOn, onClick) => {
    const seg = el_("div", "seg-s col");
    const bs = items.map(([k, label, title, key]) => { const b = el_("button", "", label); b.title = title ? title + (key ? " (" + key + ")" : "") : ""; if (key) b.appendChild(el_("kbd", "", key)); b.onclick = () => onClick(k); seg.appendChild(b); return b; });
    row.appendChild(seg);
    return () => items.forEach(([k], i) => bs[i].classList.toggle("on", isOn(k)));
  };
  const rFill = rowOf("Color by");
  rFill.classList.add("top");
  // KINDS OF CHANGE COMMENTED OUT: AEF Change on built ground only
  const styleSeg = segOf(rFill, [["earth", "Earthwork", "the chance the ground itself moved (dug, filled, graded) between the first and last year read: a model taught by 3DEP repeat lidar", "E"],
                                  /* ["kinds", "Kinds of change", "the ground that moved most, grouped by the way it moved: the same color changed the same way. Click a kind in the key to hide or show it", "A"], */
                                  ],
                          (k) => k === st.gmode, (k) => { if (k === "kinds" && !hmeta.kinds_ready) return; st.want = k; st.gmode = drawnMode(); recolorHex(); styleRows(); renderYear(); update(); });
  // one layer (Earthwork): no choice to show
  rFill.style.display = "none";
  // the two models (EW_MODELS in the kernel), side by side under the title, the one drawn lit and in
  // words under them; M switches them
  const MODELS = cfg.ew_models || ["current"];
  const MODEL_INFO = {
    current: ["Current", "Taught on US lidar only. Stronger in the US."],
    abroad: ["Global", "Taught on the US lidar and on truth from outside the US. Stronger outside it."],
  };
  st.model = cfg.model || "current";
  const rModel = rowOf("Model");
  const styleModelSeg = segOf(rModel, MODELS.map((k) => [k, (MODEL_INFO[k] || [k])[0], (MODEL_INFO[k] || [k, ""])[1], ""]), (k) => k === st.model, (k) => setModel(k));
  rModel.querySelector(".seg-s").classList.remove("col");
  const modelKey = el_("kbd", "at-mkey", "M");
  modelKey.title = "M swaps the model";
  rModel.appendChild(modelKey);
  const modelWhy = el_("span", "at-mwhy");
  rModel.appendChild(modelWhy);
  const styleModel = () => { styleModelSeg(); modelWhy.textContent = (MODEL_INFO[st.model] || ["", ""])[1]; };
  styleModel();
  function setModel(k) { if (k === st.model) return; st.model = k; send("model", {model: k}); styleModel(); note(`Model: ${(MODEL_INFO[k] || [k])[0]}`, 2500); }
  if (MODELS.length < 2) rModel.style.display = "none";
  // while Kinds of change waits: its button grayed, and a line under it with the zoom it
  // appears at and the zoom now
  const rSoon = rowOf("");
  rSoon.classList.add("at-soon");
  const soonTxt = el_("span", "t"), soonZ = el_("span", "z");
  rSoon.append(soonTxt, soonZ);
  const styleSoon = () => {
    const z = map ? map.getZoom() : 0, KZ = cfg.kinds_zoom || 11.8;
    const show = st.gmode !== "earth" && z >= HEXZ && !hmeta.kinds_ready;
    rSoon.style.display = show ? "" : "none";
    if (!show) return;
    // what reads the ground now: the view's land cover reader
    soonTxt.textContent = z < KZ ? `Built ground read from ESA WorldCover 2021 until zoom ${KZ}` : "Built ground: reading the land cover";
    soonZ.textContent = `zoom ${z.toFixed(1)}`;
  };
  // the moment Kinds of change is ready, a soft ring on its button, twice; and a small dot on it while
  // it is ready but AEF Change is the one chosen
  let kindsWaited = false;
  const styleFill = () => {
    styleSeg();
    // the Kinds of change button's wait / ready cues, out with it
    // const b = rFill.querySelector("button"), ready = !!hmeta.kinds_ready;
    // b.classList.toggle("wait", !ready);
    // if (ready && kindsWaited) { b.classList.remove("ready"); void b.offsetWidth; b.classList.add("ready"); }
    // kindsWaited = !ready;
    // b.classList.toggle("fresh", ready && st.want !== "kinds");
    styleSoon();
  };
  const rKey = rowOf("");
  rKey.classList.add("keep");
  const keyEl = el_("span", "at-key");
  rKey.appendChild(keyEl);
  // the window: the years AlphaEarth is read over; drag either end, it rereads on release
  const rWin = rowOf("Years");
  const aefYears = cfg.aef_years || [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025];
  const win = el_("span", "at-win");
  const wTrk = el_("span", "trk"), wSpn = el_("span", "spn"), wTks = el_("span", "tks");
  for (const y of aefYears) wTks.appendChild(el_("span", "", `<i>’${String(y).slice(-2)}</i>`));
  const mkR = () => { const r = el_("input"); r.type = "range"; r.min = 0; r.max = aefYears.length - 1; r.step = 1; r.title = "the years AlphaEarth is read over: drag either end, it rereads when you let go"; return r; };
  const rFrom = mkR(), rTo = mkR();
  const wTxt = el_("span", "at-wtxt");
  win.append(wTrk, wSpn, wTks, rFrom, rTo);
  rWin.append(win, wTxt);
  function styleWin() {
    const i0 = Math.max(0, aefYears.indexOf(st.y0)), i1 = Math.max(0, aefYears.indexOf(st.y1)), n = Math.max(1, aefYears.length - 1);
    rFrom.value = i0; rTo.value = i1;
    rFrom.style.zIndex = i0 === n ? 3 : 2; rTo.style.zIndex = i1 === 0 ? 3 : 2;
    const usable = (win.clientWidth || 170) - 16;
    wSpn.style.left = (8 + usable * i0 / n) + "px"; wSpn.style.width = (usable * (i1 - i0) / n) + "px";
    wTxt.textContent = `${st.y0} to ${st.y1}`;
    panelYears.textContent = `${st.y0} to ${st.y1}`;
  }
  const onDrag = (which) => {
    let a = Number(rFrom.value), b = Number(rTo.value);
    if (a >= b) { if (which === "from") a = b - 1; else b = a + 1; }
    a = Math.max(0, a); b = Math.min(aefYears.length - 1, b);
    st.y0 = aefYears[a]; st.y1 = aefYears[b]; styleWin();
  };
  let winSent = [st.y0, st.y1];
  const winRelease = () => { if (st.y0 !== winSent[0] || st.y1 !== winSent[1]) { winSent = [st.y0, st.y1]; send("aef"); update(); } };
  rFrom.addEventListener("input", () => onDrag("from")); rTo.addEventListener("input", () => onDrag("to"));
  rFrom.addEventListener("change", winRelease); rTo.addEventListener("change", winRelease);
  try { new ResizeObserver(styleWin).observe(win); } catch (e) {}
  function styleKey() {
    const y0 = hmeta.y0 || st.y0, y1 = hmeta.y1 || st.y1;
    const out_ = map && map.getZoom() < EWZ;
    // the title is what is drawn, open or folded
    panelHd.querySelector(".t").textContent = {earth: "Earthwork", kinds: "Kinds of change", much: "AEF Change"}[st.gmode] || "AEF Change";
    styleSoon();
    if (out_) {
      keyEl.innerHTML = `<span class="why">Zoom in to ${EWZ} for earthwork. AlphaEarth is read at 10 m with no coarser levels, so a wider view would read too much.</span>`;
      return;
    }
    if (st.gmode === "kinds") {
      const ks = hmeta.kinds || [];
      const FOC = [["all", "All", "every hexagon in full", "Q"], ["built", "Built", "full only where the land cover changed and was built-up, road or construction in some year", "W"]];
      let h = `<div class="seg-f">` + FOC.map(([k, t, tip, key]) => `<button data-focus="${k}" class="${st.focus === k ? "on" : ""}" title="${tip} (${key})">${t} <kbd>${key}</kbd></button>`).join("") + `</div><div class="at-kinds">`;
      ks.forEach((q, j) => {
        const k = j + 1, off = st.hideKinds.has(k);
        h += `<button data-kind="${k}" class="${off ? "off" : ""}" title="${off ? "show" : "hide"} kind ${k}"><i style="background:${kindCss(k)}"></i><b>${k}</b><span>${fmt(q.n)}${q.year ? `, most ${q.year}` : ""}${q.from ? `, ${lcPair(q.from, q.to)}` : ""}</span></button>`;
      });
      h += `</div><span class="why">The ground that moved most from ${y0} to ${y1}, grouped by the way it moved: one color, one way of changing. Beside each: hexagons, the most common year, and what AlphaEarth reads most of it as in ${y0} and in ${y1}, taught in this view by ${esc(hmeta.lc_source || "land cover maps")}. Click one to hide it.</span>`;
      keyEl.innerHTML = h;
      keyEl.querySelectorAll("[data-focus]").forEach((b) => { b.onclick = (e) => { e.stopPropagation(); st.focus = b.dataset.focus; recolorHex(); styleKey(); update(); }; });
      keyEl.querySelectorAll("[data-kind]").forEach((b) => { b.onclick = (e) => { e.stopPropagation(); const k = +b.dataset.kind; st.hideKinds.has(k) ? st.hideKinds.delete(k) : st.hideKinds.add(k); recolorHex(); renderYear(); styleKey(); update(); }; });
      return;
    }
    if (st.gmode === "earth") { keyEl.innerHTML = `<span>unlikely</span><i class="at-ramp" style="background:linear-gradient(90deg,${virCss(8)})"></i><span>likely</span>`; keyEl.title = `The chance the ground itself was dug, filled or graded, ${y0} to ${y1}`; return; }
    keyEl.innerHTML = `Built ground (built-up, road or construction in ${y1}): barely <i class="at-ramp" style="background:linear-gradient(90deg,${virCss(8)})"></i> a lot, ${y0} to ${y1}`;
  }
  function styleRows() { styleFill(); styleWin(); styleKey(); }
  // the pair's imagery card (renderYear) waits in the top left column, shown only in the pair
  const s2c = el_("div", "at-yc at-s2c holding at-glass");
  top.append(search, s2c);
  pane.appendChild(top);
  // the pair's right column: the Earthwork panel and the hexagon card move in (setPair)
  const rcol = el_("div", "at-rcol");

  // top right: settings and fill the window, then the year card
  const tools = el_("div", "at-tools");
  const bMore = el_("button", "at-btn at-glass", ICON.more + "<kbd>O</kbd>"); bMore.title = "settings and about (O)";
  const bFit = el_("button", "at-btn at-glass", ICON.expand); bFit.title = "fill the window (X); full screen (F)";
  const bPair = el_("button", "at-btn at-glass", ICON.pair + "<kbd>P</kbd>"); bPair.title = "pair the map with Sentinel-2 (P)";
  bPair.onclick = () => setPair(!st.pair);
  const bHide = el_("button", "at-btn at-glass", `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M2 12s3.6-7 10-7 10 7 10 7-3.6 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/></svg><kbd>H</kbd>`);
  bHide.title = "hide or show the readout (H)";
  bHide.onclick = (e) => { e.stopPropagation(); setCards(!cardsHidden); };
  tools.append(bPair, bMore, bHide);
  let cardsHidden = keep("readout");
  function setCards(hidden) { cardsHidden = hidden; keep("readout", hidden); root.classList.toggle("ew-read-hidden", hidden); bHide.classList.toggle("on", hidden); }
  pane.appendChild(tools);
  const yc = el_("div", "at-yc at-glass");
  // the old floating hexagon card: kept hidden (the hexagon drops down in the top right card)
  const fc = el_("div", "at-yc at-fc at-glass");
  pane.appendChild(yc);
  pane.appendChild(fc);
  pane.appendChild(rcol);
  const ewCol = el_("div", "ew-col-r");
  ewCol.append(panel, yc);
  pane.appendChild(ewCol);
  const tip = el_("div", "at-tip");
  pane.appendChild(tip);
  const more = el_("div", "at-more at-glass");
  const item = (title, sub, ctl) => { const r = el_("div", "item"); r.append(el_("div", "", `${title}${sub ? `<small>${sub}</small>` : ""}`), ctl); more.appendChild(r); return r; };
  const sw = (get, set) => { const b = el_("button", "at-sw"); b.setAttribute("role", "switch"); const sty = () => { b.classList.toggle("on", !!get()); b.setAttribute("aria-checked", String(!!get())); }; b.onclick = () => { set(!get()); sty(); }; sty(); b.sty = sty; return b; };
  const gam = el_("input"); gam.type = "range"; gam.min = 0.3; gam.max = 2.5; gam.step = 0.1; gam.value = st.s2scale;
  gam.title = "imagery brightness (gamma); double-click for 1.0";
  let gamT = null;
  gam.oninput = () => { st.s2scale = Number(gam.value); clearTimeout(gamT); gamT = setTimeout(() => send("s2scale"), 250); };
  gam.ondblclick = () => { gam.value = 1; gam.oninput(); };
  item("Imagery brightness", "<kbd>;</kbd> <kbd>'</kbd> step it", gam);
  // the imagery colors (S2_COMPOSITES in the kernel): false colors from the raw bands,
  // stretched for the view, where the true color image clips bright ground
  const COMPS = [["tci", "True color", "Earth Genome's true color image"],
                 ["urban", "Urban", "B12 B11 B4 from the raw bands, each stretched to the view: built ground and bare soil apart by hue"],
                 ["blueyellow", "Blue-yellow", "B11 B11 B2 from the raw bands, each stretched to the view: sand yellow, concrete blue"],
                 // the indexes are drawn in the browser only (no kernel tiles for them)
                 ...(cfg.s2_gpu !== false ? [
                   ["ndbi", "NDBI", "(B11 - B8A) / (B11 + B8A), stretched to the view: built-up and bare ground orange, vegetation and water blue"],
                   ["mineral", "Mineral", "Bare ground by its minerals, stretched to the view: iron oxides (B4 / B2) yellow, clays (B11 / B12) blue, vegetation and water dark"]] : [])];
  // what the hold card calls each, always shown while the imagery is up
  const COMP_NAME = {tci: "True color (TCI)", urban: "False color: Urban (B12 B11 B4)", blueyellow: "False color: Blue-yellow (B11 B11 B2)",
                     ndbi: "NDBI: orange built-up or bare, blue vegetation or water", mineral: "Mineral: yellow iron oxides, blue clays, dark vegetation and water"};
  // true color only for now: the false colors' stretch for the view was the kernel's
  COMPS.splice(1);
  const compBox = el_("div");
  const styleComp = segOf(compBox, COMPS.map(([k, l, t]) => [k, l, t, ""]), (k) => k === st.s2comp, (k) => setComp(k));
  // new colors are new tiles: the year shown loads alone first again (s2Warm)
  function setComp(k) { st.s2comp = k; s2Warm = false; send("s2comp", {comp: k}); styleComp(); renderYear(); }
  item("Imagery colors", "C steps them", compBox).style.display = COMPS.length > 1 ? "" : "none";
  const swLab = sw(() => st.labels, (v) => { st.labels = v; labels(v); });
  item("Place names", "<kbd>L</kbd>", swLab);
  // picking: the tooltip and the click's card, off to look at the map (or shoot it) clean
  const rPick = rowOf("Picking");
  const swPick = sw(() => !st.noPick, (v) => { st.noPick = !v; tip.style.display = "none"; hover = null; update(); });
  swPick.title = "the tooltip and the card on a click (Q)";
  rPick.append(swPick, el_("span", "at-pick", "<span>Tooltip and click</span><kbd>Q</kbd>"));
  more.appendChild(el_("hr"));
  const bAbout = el_("button", "at-chip", "About this map <kbd>?</kbd>"); bAbout.title = "about this map (?)"; bAbout.style.margin = "4px 10px 6px";
  more.appendChild(bAbout);
  pane.appendChild(more);

  const about = el_("div", "at-about");
  about.innerHTML = `<div class="box at-glass">
    <h2>Earthwork</h2>
    <p><b>Earthwork</b> (E) is the chance the ground itself was dug, filled or graded between the first and last year read: a model on AlphaEarth taught where 3DEP lidar flew the same ground twice. Pair (P) with Sentinel-2 to see what it is.</p>
    <p>Earthwork is scored on your GPU for every 10 m pixel, from zoom ${EWZ}. <b>Click</b> a spot for its account.</p>
    <p><b>Hold space</b> to see the Sentinel-2 yearly imagery (Earth Genome, 2022 to 2025) instead of the earthwork; scroll while holding to step through the years.</p>
    <p><small>Keys: hold space for the imagery, scroll or [ and ] for its year, B its first year or its latest; P pairs the map with Sentinel-2; M swaps the model (Current, Global); Q the tooltip and picking off and on; H hides and shows the readout; the arrows fold the cards; O settings; ? this page; Shift + arrows turn and tilt the map; F full screen; ; and ' its brightness; L place names; / search (a place, or paste an H3 string); Esc close.</small></p>
    <p><small>AlphaEarth Foundations by Google and Google DeepMind (CC BY 4.0). ESA WorldCover 10 m 2021 v200, contains modified Copernicus Sentinel data processed by the ESA WorldCover consortium (CC BY 4.0). Impact Observatory, Microsoft and Esri 10 m annual land use and land cover v02, via Microsoft Planetary Computer (CC BY 4.0). Overture Maps transportation and land use, &copy;&nbsp;OpenStreetMap contributors (ODbL), from Overture's PMTiles. Sentinel-2 mosaics by Earth Genome (CC BY 4.0). Place names from Overture Maps divisions, &copy;&nbsp;OpenStreetMap contributors, Overture Maps Foundation (ODbL), with geoBoundaries, Esri Community Maps contributors and LINZ (CC BY 4.0): the PMTiles and, via Source Cooperative, fused/overture. Search by Photon over OpenStreetMap (ODbL). Basemap by Carto.</small></p>
    <div style="margin-top:12px"><button class="at-chip">Close</button></div></div>`;
  pane.appendChild(about);
  about.querySelector("button").onclick = () => { about.style.display = "none"; };
  about.onclick = (e) => { if (e.target === about) about.style.display = "none"; };
  bAbout.onclick = () => { more.style.display = "none"; about.style.display = "flex"; };

  // the controls' effects (the kernel's, in the notebook): the years and the model are the earthwork
  // layer's own props, the brightness the imagery shader's
  const send = (act) => {
    if (act === "aef" || act === "model") { if (cardData && cardData.kind === "spot") rescoreSpot(); av = null; styleKey(); renderYear(); viewSoon(); }
    update();
  };

  // ---- status ------------------------------------------------------------
  const ERR = /failed|error|no match|search:|timed? ?out|zoom in|^(deck|map|load|boot|\w+ tile):/i;
  let msgT = null;
  const note = (t, ms) => { msgTx.textContent = t; msg.style.display = t ? "block" : "none"; msg.classList.toggle("err", ERR.test(t)); clearTimeout(msgT); if (ms) msgT = setTimeout(() => { msg.style.display = "none"; }, ms); };
  const say = (t) => {
    t = (t || "").replace(/​/g, "");
    if (ERR.test(t)) { note(t); bar.classList.remove("busy"); return; }
    const busy = t.split(" | ").filter((p) => p.includes("…"));
    const names = [];
    // each dataset being read, by name, once
    for (const p of busy)
      for (const [nm, re] of [["AlphaEarth", /AlphaEarth/], ["WorldCover", /WorldCover/], ["Impact Observatory", /Impact Observatory/], ["the model", /^the model|fitting/]])
        if (re.test(p) && !names.includes(nm)) names.push(nm);
    bar.classList.toggle("busy", names.length > 0);
    note(names.length ? "Loading " + (names.length > 1 ? names.slice(0, -1).join(", ") + " and " + names[names.length - 1] : names[0]) + "…" : "");
  };

  // ---- tiles ----------------------------------------------------------------
  const pending = new Map();
  let tseq = 0;
  const tstat = {asked: 0, got: 0, empty: 0, err: 0, abort: 0};
  const tlog = [];  // per tile, for the tests: asked, arrived, kernel times, bytes
  const tlogOf = new Map();
  model.on("msg:custom", (m, buffers) => {
    if (!m || m.kind !== "tile") return;
    const p = pending.get(m.id);
    if (!p) return;
    pending.delete(m.id);
    const lg = tlogOf.get(m.id);
    if (lg) { lg.got = Date.now(); lg.kt = m.kt || null; lg.bytes = buffers && buffers.length ? (buffers[0].byteLength || 0) : 0; lg.err = m.err || null; tlogOf.delete(m.id); }
    if (m.err) { tstat.err++; p.reject(new Error(m.err)); return; }
    if (m.empty || !buffers || !buffers.length) { tstat.empty++; p.resolve(null); return; }
    tstat.got++;
    p.resolve(bytesOf(buffers[0]));
  });
  const ask = (src, year, index, signal) => new Promise((resolve, reject) => {
    const id = ++tseq; tstat.asked++;
    pending.set(id, {resolve, reject});
    const lg = {id, src, year, z: index.z, x: index.x, y: index.y, asked: Date.now()};
    tlog.push(lg); tlogOf.set(id, lg); if (tlog.length > 4000) tlog.splice(0, 1000);
    model.send({kind: "tile", id, src, year, x: index.x, y: index.y, z: index.z});
    if (signal) signal.addEventListener("abort", () => { if (pending.has(id)) lg.aborted = Date.now(); tlogOf.delete(id); pending.delete(id); tstat.abort++; const e = new Error("aborted"); e.name = "AbortError"; reject(e); });
  });
  const pngBitmap = (u8) => createImageBitmap(new Blob([u8], {type: "image/png"}));

  // ---- the hexagons -----------------------------------------------------------
  let hexes = [], N = 0, res = -1, hexIndex = new Map(), hattrs = null, hmeta = {}, hcol = null, hcol32 = null, hexSeq = 0, hover = null, picked = null, imgPick = null;
  function recolorHex() {
    if (!N || !hattrs || hattrs.length !== HB * N) { hcol = null; hcol32 = null; return; }
    hcol = new Uint8Array(4 * N);
    hcol32 = new Uint32Array(hcol.buffer);
    for (let i = 0; i < N; i++) {
      const a8 = HB * i, o = 4 * i, lv = hattrs[a8 + 1];
      let col, a;
      if (st.gmode === "earth") {
        // the chance the ground moved, on all ground: quiet ground faint, likely earthwork in full ink
        const v = hattrs[a8 + 16];
        if (!v) continue;
        const t = (v - 1) / 254;
        col = vir(t); a = Math.round(A_QUIET + (A_FILL - A_QUIET) * t);
      } else if (st.gmode === "kinds") {
        // its kind's color, fuller the more it moved; quiet ground faint gray
        if (!lv) continue;
        const kd = hattrs[a8 + 6], t = (lv - 1) / 254;
        if (!kd) { col = [150, 156, 162]; a = 40; }
        else if (st.hideKinds.has(kd)) continue;
        else {
          col = KIND_RGB[(kd - 1) % KIND_RGB.length]; a = Math.round(110 + (A_FILL - 110) * t);
          // the key's Built: all but built-up, road and construction
          // that changed made more see-through
          if (st.focus === "built" && hattrs[a8 + 7] !== 3) a = A_DIM;
        }
      } else {
        // built ground only (BUILT_SHARE in the kernel)
        if (!lv || !hattrs[a8 + 8]) continue;
        const t = (lv - 1) / 254;
        // quiet ground faint, change in full ink
        col = vir(t); a = Math.round(A_QUIET + (A_FILL - A_QUIET) * t);
      }
      hcol[o] = col[0]; hcol[o + 1] = col[1]; hcol[o + 2] = col[2]; hcol[o + 3] = a;
    }
    hexSeq++;
  }
  // Q: no tooltip, no hover outline and no picking, so the map can be looked at (or shot) clean
  const hexAt = (ll) => { if (st.noPick || res < 0 || !map || map.getZoom() < HEXZ) return -1; try { const h = latLngToCell(ll.lat, ll.lng, res); const i = hexIndex.get(h); return i == null || (st.gmode === "much" && hattrs && !hattrs[HB * i + 8]) ? -1 : i; } catch (e) { return -1; } };
  function hexWords(i) {
    const o = HB * i, yb = hattrs[o], lv = hattrs[o + 1];
    if (!lv) return "No AlphaEarth data here.";
    if (st.gmode === "earth") {
      const v = hattrs[o + 16];
      return v ? `<b>Earthwork ${Math.round(100 * (v - 1) / 254)}%</b>: the chance the ground itself was dug, filled or graded, ${hmeta.y0 || st.y0} to ${hmeta.y1 || st.y1}` : "No AlphaEarth data here.";
    }
    let s;
    if (st.gmode === "kinds") {
      const kd = hattrs[o + 6];
      s = kd ? `<b>Kind ${kd}</b>, changed ${howMuch((lv - 1) / 254)}${yb ? `, most in ${2000 + yb}` : ""}` : `<b>Not grouped</b>: changed ${howMuch((lv - 1) / 254)}, less than the ground that moved most`;
      if (hattrs[o + 4]) s += `<br>History: ${HIST_WORD[hattrs[o + 4]]}`;
    } else {
      s = `<b>Changed ${howMuch((lv - 1) / 254)}</b>${yb ? `, most in ${2000 + yb}` : ""}`;
      if (hattrs[o + 4]) s += `<br>History: ${HIST_WORD[hattrs[o + 4]]}`;
    }
    return s;
  }

  // ---- the year card ------------------------------------------------------------
  let cardData = null;
  // one bar per year you can scroll to, plus any other year the window
  // dates; an imagery year the window cannot date is an empty slot
  function viewCounts() {
    const out = {}, dated = {};
    const y0 = hmeta.y0 || st.y0, y1 = hmeta.y1 || st.y1;
    const ys = [...new Set([...S2Y, ...Array.from({length: Math.max(0, y1 - y0)}, (_, i) => y0 + 1 + i)])].sort((a, b) => a - b);
    for (const y of ys) { out[y] = 0; dated[y] = y > y0 && y <= y1; }
    let total = 0;
    if (hattrs) for (let i = 0; i < N; i++) {
      const o = HB * i, lv = hattrs[o + 1];
      if (lv && hattrs[o + 8]) total++;
      if (st.gmode === "kinds") { const yb = hattrs[o], kd = hattrs[o + 6]; if (yb && kd && !st.hideKinds.has(kd) && out[2000 + yb] != null) out[2000 + yb]++; }
      else { const yb = hattrs[o]; if (yb && lv >= FAIR && hattrs[o + 8] && out[2000 + yb] != null) out[2000 + yb]++; }
    }
    return {years: out, dated, total};
  }
  // bars per year: one series, ink on a baseline; the imagery year in
  // full ink with its count, the others lighter; a tooltip per bar
  function yearBars(c) {
    const ys = Object.keys(c.years).map(Number);
    if (!ys.length) return "";
    const W = 346, H = 78, base = 60, gap = 6, bw = Math.min(56, (W - gap * (ys.length - 1)) / ys.length);
    const x0 = (W - (bw * ys.length + gap * (ys.length - 1))) / 2;
    const max = Math.max(1, ...ys.map((y) => c.years[y]));
    let s = `<svg width="${W}" height="${H}" role="img" aria-label="hexagons in view that changed a fair amount or more, by the year of their biggest change">`;
    ys.forEach((y, i) => {
      const v = c.years[y], h = v ? Math.max(3, (base - 14) * v / max) : 0, x = x0 + i * (bw + gap), cur = y === st.imgYear;
      const r = Math.min(4, h / 2);
      if (h) s += `<path d="M${x},${base} v${-(h - r)} q0,${-r} ${r},${-r} h${bw - 2 * r} q${r},0 ${r},${r} v${h - r} z" fill="${rgba(INK, cur ? 0.9 : 0.28)}"/>`;
      const why = c.dated[y] ? `${fmt(v)} hexagon${v === 1 ? "" : "s"} changed most between the ${y - 1} and ${y} pictures` : `${y} is outside the years read: widen them to ${y - 1} to date changes into ${y}`;
      s += `<rect x="${x - gap / 2}" y="0" width="${bw + gap}" height="${H}" fill="transparent" data-tip="${why}"/>`;
      if (!c.dated[y]) s += `<line x1="${x}" x2="${x + bw}" y1="${base - 1}" y2="${base - 1}" stroke="rgba(230,233,236,.3)" stroke-dasharray="2 3"/>`;
      if (cur && v) s += `<text class="lbl" x="${x + bw / 2}" y="${base - h - 4}" text-anchor="middle">${fmt(v)}</text>`;
      s += `<text x="${x + bw / 2}" y="${H - 4}" text-anchor="middle"${cur ? ' class="lbl"' : ""}>${y}</text>`;
    });
    s += `<line x1="0" x2="${W}" y1="${base + 0.5}" y2="${base + 0.5}" stroke="rgba(230,233,236,.25)"/></svg>`;
    return s;
  }
  // the clicked hexagon's steps, each as a multiple of that year's
  // median step in view: the biggest in full ink, a dashed line at 1
  function stepBars(rel, steps, years, big) {
    if (!rel || !rel.length) return "";
    const W = 346, H = 74, base = 56, gap = 6, n = rel.length, bw = Math.min(46, (W - gap * (n - 1)) / n);
    const x0 = (W - (bw * n + gap * (n - 1))) / 2;
    const vmax = Math.max(2, ...rel.filter((v) => v != null));
    const yOf = (v) => base - (base - 10) * Math.min(1, v / vmax);
    let s = `<svg width="${W}" height="${H}" role="img" aria-label="AlphaEarth year-to-year change for this hexagon">`;
    rel.forEach((v, i) => {
      const x = x0 + i * (bw + gap), y = years[i];
      if (v != null) {
        const top_ = yOf(v), h = base - top_, r = Math.min(4, h / 2);
        if (h > 0.5) s += `<path d="M${x},${base} v${-(h - r)} q0,${-r} ${r},${-r} h${bw - 2 * r} q${r},0 ${r},${r} v${h - r} z" fill="${rgba(INK, y === big ? 0.9 : 0.28)}"/>`;
      }
      s += `<rect x="${x - gap / 2}" y="0" width="${bw + gap}" height="${H}" fill="transparent" data-tip="${y - 1} to ${y}: ${v == null ? "no data" : `${v.toFixed(1)} times the usual step here that year (${steps[i].toFixed(3)})`}"/>`;
      s += `<text x="${x + bw / 2}" y="${H - 4}" text-anchor="middle"${y === big ? ' class="lbl"' : ""}>${n > 5 ? `’${String(y).slice(-2)}` : `’${String(y - 1).slice(-2)} to ’${String(y).slice(-2)}`}</text>`;
    });
    const ty = yOf(1); s += `<line x1="0" x2="${W}" y1="${ty}" y2="${ty}" stroke="rgba(230,233,236,.55)" stroke-dasharray="3 3"/><text x="${W}" y="${ty - 3}" text-anchor="end">usual step here</text>`;
    s += `<line x1="0" x2="${W}" y1="${base + 0.5}" y2="${base + 0.5}" stroke="rgba(230,233,236,.25)"/></svg>`;
    return s;
  }
  function hexSection(c) {
    if (!c || !c.kind) return "";
    let h = `<div class="hex"><button class="x" title="close (Esc)" aria-label="close">×</button>`;
    if (c.place && c.place.length) h += `<div class="place">${c.place.map((q) => typeof q === "string" ? esc(q) : esc(q.name) + (q.tag ? ` <span>(${esc(q.tag)})</span>` : "")).join(", ")}</div>`;
    if (c.kind === "note") return h + `<h3>${esc(c.title || "")}</h3></div>`;
    if (c.kind === "spot") {
      const lat_lon = `${c.lat.toFixed(6)}, ${c.lon.toFixed(6)}`;
      h += `<h3>This spot</h3><div class="coords"><code title="lat, long">${lat_lon}</code><button data-copy="${lat_lon}">copy</button></div>`;
      h += `<h4>Earthwork</h4>`;
      h += c.score == null
        ? `<p>No score here: ${map && map.getZoom() < EWZ ? `zoom in to ${EWZ} so AlphaEarth is read here` : "AlphaEarth has no data for this pixel in one of the years, or it is still loading"}.</p>`
        : `<p><b>${Math.round(100 * c.score)}%</b>: the chance the ground itself was dug, filled or graded from ${c.y0} to ${c.y1}, at this 10 m pixel.</p>`;
      h += `<p class="sub">Pair with Sentinel-2 (<kbd>P</kbd>) or hold space to see what it is.</p>`;
      return h.replace('<button class="x" title="close (Esc)" aria-label="close">×</button>', "") + `</div>`;
    }
    h += `<h3>This hexagon</h3>`;
    // its H3 string and center, each copyable, however it was picked
    if (c.cell) {
      let ll = null; try { ll = cellToLatLng(c.cell); } catch (e) {}
      const lat_lon = ll ? `${ll[0].toFixed(6)}, ${ll[1].toFixed(6)}` : "";
      h += `<div class="coords"><code title="H3 string">${esc(c.cell)}</code><button data-copy="${esc(c.cell)}">copy</button>`;
      if (lat_lon) h += `<code title="lat, long of the cell's center">${lat_lon}</code><button data-copy="${lat_lon}">copy</button>`;
      h += `</div>`;
    }
    // ONE VOICE PER LAYER: the card speaks for the layer the map is colored by, nothing else
    const mode = st.gmode;
    if (c.level == null) h += `<p>No AlphaEarth data here.</p>`;
    else if (mode === "earth") {
      const i = c.cell ? hexIndex.get(c.cell) : null, v = i != null && hattrs ? hattrs[HB * i + 16] : 0;
      h += `<h4>Earthwork</h4>`;
      h += v ? `<p><b>${Math.round(100 * (v - 1) / 254)}%</b>: the chance the ground itself was dug, filled or graded from ${c.y0} to ${c.y1} (its highest-scoring patch).</p>` : `<p>No score here.</p>`;
      h += `<p class="sub">Pair with Sentinel-2 (<kbd>P</kbd>) or hold space to see what it is.</p>`;
    } else {
      h += `<h4>AEF Change</h4>`;
      h += `<p>The ground changed <b>${howMuch(c.level)}</b> from ${c.y0} to ${c.y1}, compared with the rest of the view. Its year-to-year change stood out most between the <b>${c.big - 1} and ${c.big}</b> pictures${c.stand != null ? `, ${c.stand.toFixed(1)} times the usual step in view that year` : ""}.</p>`;
      if (c.hist && c.ccode) h += `<p class="sub">${histText(c.ccode, c.big, c.hist, c.cratio)}${c.hbig > 0 && (c.hbig <= c.y0 || c.hbig > c.y1) ? ` Its biggest step from ${c.hist[0]} to ${c.hist[1]} was into ${c.hbig}, outside the years read.` : ""}</p>`;
      h += stepBars(c.rel, c.steps, c.step_years, c.big);
      if (S2Y.includes(c.big) && S2Y.includes(c.big - 1)) h += `<p class="sub">Hold space with the pointer near it and scroll between ${c.big - 1} and ${c.big} to see what happened.</p>`;
      else if (c.big) h += `<p class="sub">The imagery starts in ${S2Y[0]}, so there is no picture from before ${c.big} to compare.</p>`;
    }
    if (c.km2) h += `<p class="sub">${c.km2 < 0.1 ? `${Math.round(c.km2 * 1e6).toLocaleString("en-US")} m²` : `${c.km2.toFixed(2)} km²`} hexagon.</p>`;
    return h + `</div>`;
  }
  let ycFolded = false, s2Folded = keep("s2card"), ycOpenedFor = null;
  // the imagery's year, its colors and keys: the card's head while held, the pair's left card
  const cbOf = (folded) => `<button class="at-cb" title="${folded ? "open the card" : "fold the card"}">${ICON.chev}</button>`;
  const imgYr = (cbH, help) => `<div class="yr img"><b>${st.imgYear}</b><span class="comp">${COMP_NAME[st.s2comp] || ""}</span>${cbH}<span class="comps">${COMPS.map(([k, l, t]) => `<button data-comp="${k}" class="${k === st.s2comp ? "on" : ""}" title="${esc(t)}">${l}</button>`).join("")}</span><span class="help">${help}</span></div>`;
  // ---- the view's analytics: DuckDB in a worker, over the pixels on screen ----------------------------
  let av = null, avStatus = "", avT = null, avSeq = 0;
  const haFmt = (v) => v >= 100 ? `${Math.round(v).toLocaleString("en-US")} ha` : v >= 10 ? `${v.toFixed(1)} ha` : `${v.toFixed(2)} ha`;
  const divName = (v) => { if (!v) return null; if (typeof v !== "string") return v.primary || null; try { const o = JSON.parse(v); return o.primary || v; } catch (e) { return v; } };
  // the view's towns and counties as H3 res 9 cells (a town wins over its county), from the divisions
  // the map has loaded
  function divisionCells() {
    const byCell = new Map();
    if (map && map.getSource("ov-div")) for (const sub of ["county", "locality"]) {
      let fs = [];
      try { fs = map.querySourceFeatures("ov-div", {sourceLayer: "division_area", filter: ["all", ["==", ["get", "subtype"], sub], ["==", ["get", "class"], "land"]]}); } catch (e) {}
      for (const f of fs) {
        const p = f.properties || {}, name = divName(p["@name"] || p.names);
        if (!name) continue;
        const g = f.geometry, polys = g.type === "Polygon" ? [g.coordinates] : g.type === "MultiPolygon" ? g.coordinates : [];
        for (const poly of polys) { let cs = []; try { cs = polygonToCells(poly, 9, true); } catch (e) {} for (const c of cs) byCell.set(c, name); }
      }
    }
    const cells = new BigUint64Array(byCell.size), names = [];
    let i = 0;
    for (const [c, n] of byCell) { cells[i++] = BigInt("0x" + c); names.push(n); }
    return {cells, names};
  }
  function viewSoon() { clearTimeout(avT); avT = setTimeout(runView, 700); }
  async function runView() {
    if (!analytics || !map || st.holding || map.getZoom() < EWZ) return;
    if (!analytics.isReady()) return;
    const b = map.getBounds(), bbox = [b.getWest(), b.getSouth(), b.getEast(), b.getNorth()];
    const {chunks, wanted} = earthwork.chunksIn(bbox, st.y0, st.y1);
    if (!chunks.length) return;
    const s = ++avSeq, d = divisionCells();
    try {
      const r = await analytics.query({chunks, bbox, model: st.model, divCells: d.cells, divNames: d.names});
      if (s !== avSeq) return;
      av = r; avStatus = chunks.length < wanted ? "Still loading part of the view" : "";
    } catch (e) {
      if (s !== avSeq) return;
      avStatus = "DuckDB: " + e.message;
    }
    renderYear();
  }
  function viewSection() {
    if (!av && !avStatus) return `<p class="ew-r1">Reading this view…</p>`;
    let h = "";
    if (av && av.pixels) {
      h += `<p class="ew-r1"><b>${haFmt(av.likely_ha)}</b> likely moved in view</p>`;
      if (av.places && av.places.length) {
        const max = Math.max(...av.places.map((q) => q.ha));
        h += `<div class="at-lc">` + av.places.slice(0, 4).map((q) => `<span>${esc(q.name || "Elsewhere")}</span><i style="width:${Math.max(2, 100 * q.ha / max)}%"></i><span>${haFmt(q.ha)}</span>`).join("") + `</div>`;
      }
      if (av.digs && av.digs.length) {
        h += `<p class="ew-sub">Most moved, about 10 ha each</p><div class="ew-digs">`;
        h += av.digs.slice(0, 3).map((q) => `<button data-dig="${q.cell}"><span>${esc(q.name || "Elsewhere")}</span><b>${haFmt(q.ha)}</b></button>`).join("");
        h += `</div>`;
      }
    } else if (av) h += `<p class="ew-r1">No AlphaEarth read in view yet</p>`;
    if (avStatus) h += `<p class="ew-sub">${esc(avStatus)}</p>`;
    return h;
  }
  function renderYear() {
    // the imagery's card: its year, while the imagery shows (held, or the pair's left)
    const showImg = st.holding || (st.pair && st.left === "s2");
    s2c.innerHTML = `<div class="at-hd ew-s2hd"><span class="t">Sentinel-2</span><b class="ew-s2y">${st.imgYear}</b>${cbOf(s2Folded)}</div>`
      + `<div class="ew-keys">${showImg
        ? `<kbd>[</kbd><kbd>]</kbd> year, <kbd>B</kbd> ${S2Y[0]} or ${S2Y[S2Y.length - 1]}${st.holding ? ", or scroll" : ""}`
        : `Hold <kbd>Space</kbd> to see it, or <kbd>P</kbd> beside the map`}</div>`;
    s2c.className = "at-yc at-s2c at-glass" + (s2Folded ? " ew-folded" : "") + (showImg ? " ew-showing" : "");
    s2c.querySelector(".ew-s2hd").onclick = (e) => { e.stopPropagation(); s2Folded = !s2Folded; keep("s2card", s2Folded); renderYear(); };
    // the readout: the view's numbers and the clicked spot
    const pc = cardData && cardData.kind === "spot" ? cardData : null;
    let h = map && map.getZoom() < EWZ ? `<p class="ew-r1">Zoom in to ${EWZ} for earthwork</p>` : viewSection();
    if (pc) {
      const where = (pc.place && pc.place[0]) || "This spot", lat_lon = `${pc.lat.toFixed(5)}, ${pc.lon.toFixed(5)}`;
      h += `<div class="ew-spot"><div class="ew-spot-l"><span><b>${esc(where)}</b> ${pc.score == null ? "no score here" : `${Math.round(100 * pc.score)}%`}</span>`
        + `<button class="ew-x" title="close (Esc)" aria-label="close">\u00d7</button></div>`
        + `<button class="ew-ll" data-copy="${lat_lon}" title="copy">${lat_lon}</button></div>`;
    }
    yc.innerHTML = h;
    fc.style.display = "none";
    yc.className = "at-yc at-glass ew-read" + (h ? "" : " ew-off");
    const bx = yc.querySelector(".ew-x");
    if (bx) bx.onclick = (e) => { e.stopPropagation(); closeCard(); };
  }
  // the card never scrolls: when its content is taller
  // than the pane below it, it is scaled down from its top right
  // corner to fit. On a narrow screen it keeps its own scroll.
  function fitCard() {
    return;
    yc.style.transform = "";
    if (window.matchMedia("(max-width:760px)").matches) return;
    const avail = pane.getBoundingClientRect().bottom - yc.getBoundingClientRect().top - 12, need = yc.offsetHeight;
    if (avail > 0 && need > avail) yc.style.transform = `scale(${avail / need})`;
  }
  // the hexagon's card lives in the top right card (renderYear); fc stays hidden
  let fcMore = false, fcFor = null;
  for (const el of [yc, fc]) el.addEventListener("pointermove", (e) => {
    const t = e.target && e.target.getAttribute && e.target.getAttribute("data-tip");
    if (!t) { tip.style.display = "none"; return; }
    const p = pane.getBoundingClientRect();
    tip.textContent = t;
    tip.style.display = "block";
    tip.style.left = Math.max(8, e.clientX - p.left - tip.offsetWidth - 14) + "px";
    tip.style.top = (e.clientY - p.top + 12) + "px";
  });
  for (const el of [yc, fc]) el.addEventListener("pointerleave", () => { tip.style.display = "none"; });
  // the imagery colors, from the hold card's own buttons and the pair's in the panel
  for (const el of [yc, s2c]) el.addEventListener("click", (e) => {
    const b = e.target && e.target.closest && e.target.closest("[data-comp]");
    if (!b) return;
    e.stopPropagation();
    setComp(b.getAttribute("data-comp"));
  });
  yc.addEventListener("click", (e) => {
    const b = e.target && e.target.closest && e.target.closest("[data-dig]");
    if (!b || !map) return;
    e.stopPropagation();
    const cell = b.getAttribute("data-dig"), [lat, lon] = cellToLatLng(cell);
    searched = cell; update();
    map.flyTo({center: [lon, lat], zoom: Math.max(map.getZoom(), 15), duration: 1600, essential: true});
  });
  for (const el of [yc, fc]) el.addEventListener("click", (e) => {
    const b = e.target && e.target.closest && e.target.closest("[data-copy]");
    if (!b) return;
    e.stopPropagation();
    const t = b.getAttribute("data-copy");
    const done = () => { b.textContent = "copied"; setTimeout(() => { b.textContent = "copy"; }, 1200); };
    // the notebook may sit in an iframe without clipboard permission (molab): fall back to a selection copy
    const fallback = () => { const ta = document.createElement("textarea"); ta.value = t; ta.style.position = "fixed"; ta.style.opacity = "0"; root.appendChild(ta); ta.select(); try { document.execCommand("copy"); done(); } catch (e2) {} ta.remove(); };
    if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(t).then(done, fallback); else fallback();
  });
  function renderCard() {
    picked = cardData && cardData.cell ? cardData.cell : null;
    // a new click opens a folded card once; folding it again stays folded
    if (cardData && cardData.n != null && cardData.n !== ycOpenedFor) { ycOpenedFor = cardData.n; if (ycFolded) { ycFolded = false; keep("card", false); } }
    renderYear(); update();
  }
  function closeCard() { cardData = null; picked = null; imgPick = null; spot = null; renderYear(); update(); }
  function rescoreSpot() {
    if (!cardData || cardData.kind !== "spot") return;
    cardData.score = earthwork.scoreAt(cardData.lon, cardData.lat, st.y0, st.y1, st.model);
    cardData.y0 = st.y0; cardData.y1 = st.y1;
  }
  // a pick: the same cell again clears it
  function pickCell(cell, ll, pt, onImagery) {
    // a searched cell goes with the next click, inside it or anywhere
    // else, and the map stays where it is; a click inside
    // it only clears it
    if (searched) {
      let inside = false; try { inside = latLngToCell(ll.lat, ll.lng, getResolution(searched)) === searched; } catch (e) {}
      searched = null; update();
      if (inside) return;
    }
    if (cell && cell === picked) { closeCard(); return; }
    imgPick = null;
    const ad = adminAt(pt), nm = (v) => { if (!v) return null; if (typeof v !== "string") return v.primary || null; try { const o = JSON.parse(v); return o.primary || v; } catch (e) { return v; } };
    spot = {lon: ll.lng, lat: ll.lat};
    cardData = {kind: "spot", n: ++seq, lon: ll.lng, lat: ll.lat, place: [nm(ad.locality), nm(ad.county), nm(ad.region)].filter(Boolean)};
    rescoreSpot();
    renderCard();
  }

  // ---- the layers --------------------------------------------------------------
  let map = null, ov = null, map2 = null, ov2 = null;
  const slot = (m = map) => { const want = cfg.labels_slot || "watername_ocean"; const s = m && m.getStyle && m.getStyle(); if (!s || !s.layers || s.layers.some((x) => x.id === want)) return want; const l = s.layers.find((x) => x.type === "symbol"); return (l && l.id) || want; };
  // ---- Sentinel-2 true color, read and drawn here by deck.gl-raster (Development Seed's
  // MosaicLayer + COGLayer, as iceye-view.py and segments-map's ICEYE): the kernel only names
  // the footprints under the view (its STAC search, s2i); the browser opens each one's TCI COG
  // on data.source.coop and decodes its tiles in workers, so no tile waits on the kernel. The
  // year shown loads first, the other years once it is in (a scroll while holding is then
  // instant). Yearly footprints paint over the fill ones. The false colors too, from each
  // footprint's raw band COGs; the kernel only takes their stretch for the view (cfg.s2_stretch)
  const S2GPU = cfg.s2_gpu !== false;
  const S2_WORKER = "https://esm.sh/@developmentseed/geotiff@0.8.0/es2022/dist/pool/worker.mjs";
  let s2Pool = null, s2Warm = false;
  const s2Tiffs = new Map();  // url -> Promise<GeoTIFF>: the headers, kept for the session
  // year -> {ids, batches}: a batch is one answer of the kernel (a tile's footprints, each
  // footprint in the first batch that named it) and gets its own MosaicLayer: a MosaicLayer
  // looks for its sources only when the camera moves, so footprints added to one after the
  // camera stopped would wait for the next pan; a new layer looks at once
  const s2Src = new Map();
  // the requests to data.source.coop, 16 at a time (HTTP/2: one connection; the default is 6),
  // the rest queued in order: the year shown first, yearly footprints before the fill ones,
  // then the nearest the center of the view
  const s2Limiter = new PerOriginSemaphore({maxRequests: 16});
  const s2Asked = new Set();  // "year/z/x/y" asked of the kernel
  const s2Stat = {tiles: 0, pending: 0, failed: 0, last: 0, years: {}};  // years: per year ("2025", "2025 fill") {tiles, pending, last}, for the tests
  const s2Tiff = (url, opts, year, fill) => {
    let p = s2Tiffs.get(url);
    if (!p) {
      const near = opts && opts.getPriority;
      const getPriority = () => [year === st.imgYear ? 0 : 1, fill ? 1 : 0, near ? near() : 0];
      p = GeoTIFF.fromUrl(url, {concurrencyLimiter: s2Limiter, getPriority});
      s2Tiffs.set(url, p);
      p.catch(() => { if (s2Tiffs.get(url) === p) s2Tiffs.delete(url); });  // a failed read is asked again next time
    }
    return p;
  };
  // the view and half of it again on every side: where footprints are looked for and mounted
  function s2View() {
    const bs = [map, st.pair ? map2 : null].filter(Boolean).map((m) => m.getBounds());
    const W_ = Math.min(...bs.map((b) => b.getWest())), E_ = Math.max(...bs.map((b) => b.getEast()));
    const S_ = Math.min(...bs.map((b) => b.getSouth())), N_ = Math.max(...bs.map((b) => b.getNorth()));
    const dx = (E_ - W_) / 2, dy = (N_ - S_) / 2;
    return [W_ - dx, Math.max(-85, S_ - dy), E_ + dx, Math.min(85, N_ + dy)];
  }
  function s2Discover() {
    if (!map || !S2GPU || !(st.holding || st.pair || map.getZoom() >= 9)) return;
    const z = Math.max(cfg.s2_min_z || 7, Math.min(9, Math.floor(map.getZoom()))), n = 2 ** z;
    const tx = (lon) => Math.min(n - 1, Math.max(0, Math.floor((lon + 180) / 360 * n)));
    const ty = (lat) => { const v = Math.sin(lat * Math.PI / 180); return Math.min(n - 1, Math.max(0, Math.floor((0.5 - Math.log((1 + v) / (1 - v)) / (4 * Math.PI)) * n))); };
    const [W_, S_, E_, N_] = s2View(), x0 = tx(W_), x1 = tx(E_), y0 = ty(N_), y1 = ty(S_);
    if ((x1 - x0 + 1) * (y1 - y0 + 1) > 64) return;
    for (const year of [st.imgYear, ...S2Y.filter((y) => y !== st.imgYear)])
      for (let x = x0; x <= x1; x++) for (let y = y0; y <= y1; y++) {
        const k = `${year}/${z}/${x}/${y}`;
        if (s2Asked.has(k)) continue;
        s2Asked.add(k);
        s2Items(year, {z, x, y}).then((items) => {
          let e = s2Src.get(year);
          if (!e) { e = {ids: new Set(), batches: []}; s2Src.set(year, e); }
          const bt = {key: k.replace(/\//g, "-"), yearly: [], fill: [], bbox: [180, 90, -180, -90]};
          for (const it of items) {
            if (!it.bbox || e.ids.has(it.id)) continue;
            e.ids.add(it.id);
            bt[it.fill ? "fill" : "yearly"].push({id: it.id, url: it.url, bbox: it.bbox, bands: it.bands || {}});
            bt.bbox = [Math.min(bt.bbox[0], it.bbox[0]), Math.min(bt.bbox[1], it.bbox[1]), Math.max(bt.bbox[2], it.bbox[2]), Math.max(bt.bbox[3], it.bbox[3])];
          }
          if (bt.yearly.length || bt.fill.length) { e.batches.push(bt); update(); }
        }, () => { s2Asked.delete(k); });
      }
  }
  async function s2TileData(image, {device, x, y, signal, pool}, key) {
    const sy = s2Stat.years[key] || (s2Stat.years[key] = {tiles: 0, pending: 0, last: 0});
    s2Stat.pending++; sy.pending++;
    try {
      const {array} = await image.fetchTile(x, y, {boundless: false, pool, signal});
      const {width, height, data} = array, px = width * height;
      // TCI's black (0, 0, 0) is nodata: alpha 0 there, so the linear filter's blend of an edge
      // pixel with its nodata neighbor carries a falling alpha the shader can undo (a black
      // test alone let the half-dark blend through: a dark line along every footprint edge)
      let rgba = data;
      if (data.length === 3 * px) {
        rgba = new Uint8Array(4 * px);
        for (let i = 0; i < px; i++) {
          const r = data[3 * i], g = data[3 * i + 1], b = data[3 * i + 2];
          rgba[4 * i] = r; rgba[4 * i + 1] = g; rgba[4 * i + 2] = b; rgba[4 * i + 3] = r || g || b ? 255 : 0;
        }
      }
      s2Stat.tiles++; sy.tiles++;
      return {texture: device.createTexture({data: rgba, format: "rgba8unorm", width, height, sampler: {magFilter: "linear", minFilter: "linear"}}), width, height};
    } catch (e) {
      if (!(signal && signal.aborted)) s2Stat.failed++;
      throw e;
    } finally {
      s2Stat.pending--; sy.pending--;
      s2Stat.last = sy.last = Date.now();
      // the year shown is in: mount the others
      if (!s2Warm && s2Stat.pending === 0) { s2Warm = true; setTimeout(update, 250); }
    }
  }
  // nodata (alpha 0) is dropped; at a footprint edge the filtered color is the edge pixel's
  // color times the filtered alpha (nodata is black), so dividing by alpha gives the pixel back
  // with no dark fringe, and alpha under a half ends the footprint midway through the texel.
  // Then the strip's gamma, v -> v ** (1 / gamma), as the kernel's tiles had
  const S2Look = {
    name: "s2Look",
    fs: `uniform s2LookUniforms {
  float gamma;
} s2Look;
`,
    inject: {"fs:DECKGL_FILTER_COLOR": `
  if (color.a < 0.5) discard;
  color = vec4(pow(color.rgb / color.a, vec3(1.0 / s2Look.gamma)), 1.0);
`},
    uniformTypes: {gamma: "f32"},
    getUniforms: (q) => ({gamma: q.gamma}),
  };
  // ---- the false colors (S2_COMPOSITES in the kernel): R, G, B from the raw bands (uint16
  // reflectance x 10,000). The layer walks the first 20 m band's tiles; each other band is read
  // at the level of the same pixel size and placed by its own geotransform (the yearly
  // footprints share one grid for every band, the fill ones put the 10 m bands on another
  // origin and other blocks), from the blocks under the tile. Packed as sqrt(reflectance) in
  // 8 bits (fine steps in the dark ground, where most of it lies); nodata (a band at 0) gets
  // alpha 0 as in TCI
  // rgb: three bands, each over its stretch (S2Stretch). bands + q: quantities computed here per
  // pixel (the kernel's S2_INDEXES computes the same ones for the stretch), kept as half floats
  // and colored by S2Index (kind 0: one index on a ramp, 1: mineral)
  const s2Clamp = (x) => Math.min(1, Math.max(0, x));
  const s2Bare = (b3, b4, b11, b8a) => s2Clamp((0.45 - (b8a - b4) / (b8a + b4)) / 0.25) * s2Clamp((0.05 - (b3 - b11) / (b3 + b11)) / 0.15);
  const S2_FALSE = {
    urban: {rgb: ["B12", "B11", "B04"]},
    blueyellow: {rgb: ["B11", "B11", "B02"]},
    ndbi: {bands: ["B11", "B8A"], kind: 0, q: (v, o) => { o[0] = (v[0] - v[1]) / (v[0] + v[1]); o[1] = o[2] = 0; }},
    mineral: {bands: ["B02", "B03", "B04", "B11", "B12", "B8A"], kind: 1,
              q: (v, o) => { o[0] = v[2] / v[0]; o[1] = v[3] / v[4]; o[2] = s2Bare(v[1], v[2], v[3], v[5]); }},
  };
  const S2_20M = new Set(["B11", "B12", "B8A"]);
  const s2BandsOf = (comp) => S2_FALSE[comp].rgb || S2_FALSE[comp].bands;
  const s2Prim = (comp) => s2BandsOf(comp).find((b) => S2_20M.has(b));
  // float -> half float bits (rgba16float takes them as a Uint16Array)
  const s2F32 = new Float32Array(1), s2U32 = new Uint32Array(s2F32.buffer);
  function s2Half(v) {
    s2F32[0] = v;
    const x = s2U32[0], sg = (x >>> 16) & 0x8000, e = ((x >>> 23) & 0xff) - 112;
    if (e <= 0 || v !== v) return sg;
    if (e >= 31) return sg | 0x7bff;
    return sg | (e << 10) | ((x & 0x7fffff) >>> 13);
  }
  const S2_HALF_ONE = s2Half(1);
  const S2_SQRT = new Uint8Array(65536);
  for (let v = 0; v < 65536; v++) S2_SQRT[v] = Math.round(255 * Math.sqrt(Math.min(v, 10000) / 10000));
  // a block on another grid lies under up to four tiles: read once for all of them (no
  // tile's abort signal, so one tile let go does not fail its neighbors)
  const s2Blocks = new Map();
  function s2Block(url, im, bx, by, pool) {
    const k = `${url}|${im.width}|${bx}|${by}`;
    let p = s2Blocks.get(k);
    if (!p) {
      p = im.fetchTile(bx, by, {boundless: false, pool}).then(({array}) => ({d: array.data || array.bands[0], w: array.width, h: array.height}), () => null);
      s2Blocks.set(k, p);
      if (s2Blocks.size > 256) s2Blocks.delete(s2Blocks.keys().next().value);
    }
    return p;
  }
  async function s2BandTileData(image, {device, x, y, signal, pool}, key, geotiff, src, comp, year, fill) {
    const sy = s2Stat.years[key] || (s2Stat.years[key] = {tiles: 0, pending: 0, last: 0});
    s2Stat.pending++; sy.pending++;
    try {
      const spec = S2_FALSE[comp], bands = s2BandsOf(comp), prim = s2Prim(comp);
      const tp = image.transform, tw = image.tileWidth, th = image.tileHeight;
      const X0 = tp[2] + x * tw * tp[0], Y0 = tp[5] + y * th * tp[4];  // the tile's corner, in meters
      const block = async (im, bx, by) => {
        try {
          const {array} = await im.fetchTile(bx, by, {boundless: false, pool, signal});
          return {d: array.data || array.bands[0], w: array.width, h: array.height};
        } catch (e) { if (signal && signal.aborted) throw e; return null; }  // no block there
      };
      const one = async (b) => {
        if (b === prim) return block(image, x, y);
        const g = src.bands[b] ? await s2Tiff(src.bands[b], null, year, fill) : null;
        const im = g && [g, ...g.overviews].find((q) => Math.abs(q.transform[0] / tp[0] - 1) < 0.01);
        if (!im) return null;
        const q = im.transform, c0 = Math.round((X0 - q[2]) / q[0]), r0 = Math.round((Y0 - q[5]) / q[4]);
        const bw = im.tileWidth, bh = im.tileHeight;
        if (bw === tw && bh === th && c0 === x * tw && r0 === y * th) return block(im, x, y);
        const out = new Uint16Array(tw * th), jobs = [];
        for (let by = Math.max(0, Math.floor(r0 / bh)); by <= Math.min(Math.ceil(im.height / bh) - 1, Math.floor((r0 + th - 1) / bh)); by++)
          for (let bx = Math.max(0, Math.floor(c0 / bw)); bx <= Math.min(Math.ceil(im.width / bw) - 1, Math.floor((c0 + tw - 1) / bw)); bx++)
            jobs.push(s2Block(src.bands[b], im, bx, by, pool).then((t) => {
              if (!t) return;
              // the block's pixels in the tile's frame
              const ox = bx * bw - c0, oy = by * bh - r0;
              for (let r = Math.max(0, -oy); r < t.h && r + oy < th; r++) {
                const c = Math.max(0, -ox), n = Math.min(t.w, tw - ox) - c;
                if (n > 0) out.set(t.d.subarray(r * t.w + c, r * t.w + c + n), (r + oy) * tw + c + ox);
              }
            }));
        await Promise.all(jobs);
        return {d: out, w: tw, h: th};
      };
      const ub = [...new Set(bands)], got = new Map(await Promise.all(ub.map(async (b) => [b, await one(b)])));
      const p = got.get(prim);
      if (!p) throw new Error("no " + prim + " tile");
      const {w: width, h: height} = p, chans = bands.map((b) => got.get(b));
      if (spec.q) {
        // the quantities as half floats, times alpha (0 at nodata, so the filter's blend at an
        // edge divides back out as in TCI)
        const px = new Uint16Array(4 * width * height), v = new Float32Array(bands.length), q = new Float32Array(3);
        if (chans.every(Boolean)) {
          for (let r = 0; r < height; r++) for (let c = 0; c < width; c++) {
            let ok = true;
            for (let k = 0; k < bands.length; k++) {
              const a = chans[k], x = r < a.h && c < a.w ? a.d[r * a.w + c] : 0;
              if (!x) { ok = false; break; }
              v[k] = x;
            }
            if (!ok) continue;
            spec.q(v, q);
            const o = 4 * (r * width + c);
            px[o] = s2Half(q[0]); px[o + 1] = s2Half(q[1]); px[o + 2] = s2Half(q[2]); px[o + 3] = S2_HALF_ONE;
          }
        }
        s2Stat.tiles++; sy.tiles++;
        return {texture: device.createTexture({data: px, format: "rgba16float", width, height, sampler: {magFilter: "linear", minFilter: "linear"}}), width, height};
      }
      const rgba = new Uint8Array(4 * width * height);
      if (chans.every(Boolean)) {
        for (let r = 0; r < height; r++) for (let c = 0; c < width; c++) {
          const o = 4 * (r * width + c);
          let ok = true;
          for (let k = 0; k < 3; k++) {
            const a = chans[k], v = r < a.h && c < a.w ? a.d[r * a.w + c] : 0;
            if (!v) { ok = false; break; }
            rgba[o + k] = S2_SQRT[v];
          }
          if (ok) rgba[o + 3] = 255; else rgba[o] = rgba[o + 1] = rgba[o + 2] = 0;
        }
      }
      s2Stat.tiles++; sy.tiles++;
      return {texture: device.createTexture({data: rgba, format: "rgba8unorm", width, height, sampler: {magFilter: "linear", minFilter: "linear"}}), width, height};
    } catch (e) {
      if (!(signal && signal.aborted)) s2Stat.failed++;
      throw e;
    } finally {
      s2Stat.pending--; sy.pending--;
      s2Stat.last = sy.last = Date.now();
      if (!s2Warm && s2Stat.pending === 0) { s2Warm = true; setTimeout(update, 250); }
    }
  }
  // as S2Look, then back from sqrt to reflectance and each band over its stretch (the
  // kernel's 98th percentile for the view, as a fraction), then the gamma
  const S2Stretch = {
    name: "s2Stretch",
    fs: `uniform s2StretchUniforms {
  float gamma;
  vec3 scale;
} s2Stretch;
`,
    inject: {"fs:DECKGL_FILTER_COLOR": `
  if (color.a < 0.5) discard;
  vec3 v = color.rgb / color.a;
  v = clamp(v * v / s2Stretch.scale, 0.0, 1.0);
  color = vec4(pow(v, vec3(1.0 / s2Stretch.gamma)), 1.0);
`},
    uniformTypes: {gamma: "f32", scale: "vec3<f32>"},
    getUniforms: (q) => ({gamma: q.gamma, scale: q.scale}),
  };
  // the indexes: each quantity over its stretch (the kernel's 2nd to 98th percentile for the view).
  // NDBI on blue (low: vegetation, water) to a light middle to orange (high: built-up, bare).
  // Mineral: iron oxides yellow, clays blue, both near white, on bare ground; vegetation dark.
  // Blue against yellow and orange, never red against green
  const S2Index = {
    name: "s2Index",
    fs: `uniform s2IndexUniforms {
  vec3 lo;
  vec3 hi;
  float kind;
} s2Index;
`,
    inject: {"fs:DECKGL_FILTER_COLOR": `
  if (color.a < 0.5) discard;
  vec3 q = color.rgb / color.a;
  vec3 t = clamp((q - s2Index.lo) / max(s2Index.hi - s2Index.lo, vec3(1e-6)), 0.0, 1.0);
  vec3 c;
  if (s2Index.kind < 0.5) {
    vec3 lo = vec3(0.13, 0.33, 0.62), mid = vec3(0.95, 0.93, 0.88), hi = vec3(0.89, 0.47, 0.08);
    c = t.r < 0.5 ? mix(lo, mid, t.r * 2.0) : mix(mid, hi, t.r * 2.0 - 1.0);
  } else {
    c = vec3(0.05, 0.06, 0.07) + t.b * (vec3(0.10) + t.r * vec3(0.95, 0.76, 0.18) + t.g * vec3(0.22, 0.50, 0.92));
  }
  color = vec4(clamp(c, 0.0, 1.0), 1.0);
`},
    uniformTypes: {lo: "vec3<f32>", hi: "vec3<f32>", kind: "f32"},
    getUniforms: (q) => ({lo: q.lo, hi: q.hi, kind: q.kind}),
  };
  const s2CogLayers = (year, left = false) => {
    if (!s2Pool) s2Pool = new DecoderPool({size: Math.min(6, navigator.hardwareConcurrency || 4),
      createWorker: () => new GeoTiffWorker()});
    const e = s2Src.get(year);
    if (!e || !map) return [];
    // shown or hidden by visible, not opacity: the tile layers inside MosaicLayer and COGLayer
    // keep each tile's raster layer as it was when the tile came in (an opacity change never
    // reaches them, so a tile loaded while holding stayed after the hold), while deck.gl checks
    // every parent's visible on each draw. Hidden years still load their tiles
    const visible = (left ? st.pair && st.left === "s2" : st.holding) && year === st.imgYear;
    const gamma = Number(st.s2scale) || 1, comp = S2_FALSE[st.s2comp] ? st.s2comp : "tci";
    const pre = (left ? "s2gl-" : "s2g-") + (comp === "tci" ? "" : comp + "-");
    // the false colors wait, still loading, for the stretch of their view
    const sx = cfg.s2_stretch, stretched = comp === "tci" || (sx && sx.comp === comp);
    const spec = S2_FALSE[comp];
    const scale = !spec || !spec.rgb ? null : spec.rgb.map((b) => ((sx && sx[b]) || 3000) / 10000);
    const lo = (sx && sx.lo) || [0, 0, 0], hi = (sx && sx.hi) || [1, 1, 1];
    const look = !spec ? null : spec.rgb ? {module: S2Stretch, props: {gamma, scale}} : {module: S2Index, props: {lo, hi, kind: spec.kind}};
    const [W_, S_, E_, N_] = s2View();
    const live = e.batches.filter((bt) => bt.bbox[0] < E_ && bt.bbox[2] > W_ && bt.bbox[1] < N_ && bt.bbox[3] > S_);
    // every batch's fill under every batch's yearly footprints
    const parts = [...live.filter((bt) => bt.fill.length).map((bt) => [bt, "fill"]), ...live.filter((bt) => bt.yearly.length).map((bt) => [bt, "yearly"])];
    return parts.map(([bt, part]) => new MosaicLayer({
      id: pre + bt.key + "-" + part, sources: bt[part], maxCacheSize: 0, minZoom: cfg.s2_min_z || 7, visible: visible && stretched, beforeId: slot(left ? map2 : map),
      // a footprint that cannot be read (the STAC lists it but the bucket lacks it, or its header
      // failed) is nothing there: a rejection here leaves MosaicLayer a tile with no content,
      // which it then cannot draw ("Cannot destructure property 'source' ... null")
      getSource: (src, o) => s2Tiff(comp === "tci" ? src.url : src.bands[s2Prim(comp)], o, year, part === "fill").catch(() => null),
      renderSource: (src, {data, signal}) => !data ? null : comp === "tci" ? new COGLayer({
        id: pre + year + "-" + src.id, geotiff: data, getTileData: (img, o) => s2TileData(img, o, year + (part === "fill" ? " fill" : "")), pool: s2Pool, signal,
        refinementStrategy: "best-available", maxRequests: 16,
        renderTile: (d) => ({renderPipeline: [{module: CreateTexture, props: {textureName: d.texture}}, {module: S2Look, props: {gamma}}]}),
        updateTriggers: {renderTile: [gamma]},
      }) : new COGLayer({
        id: pre + year + "-" + src.id, geotiff: data, pool: s2Pool, signal, refinementStrategy: "best-available", maxRequests: 16,
        getTileData: (img, o) => s2BandTileData(img, o, comp + " " + year + (part === "fill" ? " fill" : ""), data, src, comp, year, part === "fill"),
        renderTile: (d) => ({renderPipeline: [{module: CreateTexture, props: {textureName: d.texture}}, look]}),
        updateTriggers: {renderTile: [gamma, ...(scale || []), ...lo, ...hi]},
      }),
    }));
  };
  // the years mounted: the one shown, the rest once it has loaded
  const s2Years = () => (s2Warm ? [st.imgYear, ...S2Y.filter((y) => y !== st.imgYear)] : [st.imgYear]);
  // every imagery year stays mounted once the view is close enough, the
  // ones not shown at opacity 0, so their tiles load ahead and a scroll
  // while holding is instant
  const s2Layer = (year, left = false) => new TileLayer({
    id: (left ? "s2l-" : "s2-") + year + "-g" + (cfg.s2_gen || 0),
    getTileData: async ({index, signal}) => { const u8 = await ask("s2", year, index, signal); return u8 ? pngBitmap(u8) : null; },
    onTileError: (e) => { if (!e || e.name !== "AbortError") say("s2 tile: " + ((e && e.message) || e)); },
    tileSize: cfg.tile || 256, minZoom: cfg.s2_min_z || 7, maxZoom: 14, refinementStrategy: "best-available", debounceTime: 120, beforeId: slot(left ? map2 : map),
    opacity: (left ? st.pair && st.left === "s2" : st.holding) && year === st.imgYear ? 1 : 0,
    renderSubLayers: (p) => { if (!p.data) return null; const {west, south, east, north} = p.tile.bbox; return new BitmapLayer(p, {data: null, image: p.data, bounds: [west, south, east, north]}); },
  });
  // the hexagons: the frame's cells drawn as geometry, colored from hcol. A zoom or pan only
  // rescales what is on screen (nothing to fetch per tile), and a new frame replaces the old
  // one whole the moment its cells arrive. hexData is made once per frame, so a hover or a
  // mode change does not rebuild the shapes
  let hexData = null;
  const hexLayer = (visible) => new H3HexagonLayer({
    id: "hexes", visible,
    data: hexData,
    getHexagon: (_, {index}) => hexes[index],
    getFillColor: (_, {index, target}) => { const o = 4 * index; target[0] = hcol[o]; target[1] = hcol[o + 1]; target[2] = hcol[o + 2]; target[3] = hcol[o + 3]; return target; },
    updateTriggers: {getFillColor: [hexSeq]},
    filled: true, stroked: false, extruded: false, highPrecision: true, pickable: false,
    beforeId: slot(),
  });
  const ring = (h) => { try { return cellToBoundary(h, true); } catch (e) { return null; } };
  const outline = (id, h, color, width, m = map) => { const r = h ? ring(h) : null; return r ? new PathLayer({id, data: [r], getPath: (d) => d, getColor: color, widthUnits: "pixels", getWidth: width, beforeId: slot(m)}) : null; };
  // the picked spot: its 10 m AlphaEarth pixel (the mosaic's grid)
  const AEF_R = 8.983111749910169e-05, AEF_Y0 = 83.68570533713473;
  let spot = null;
  function spotLayer(id, m = map) {
    if (!spot) return null;
    const x0 = -180 + Math.floor((spot.lon + 180) / AEF_R) * AEF_R, y1 = AEF_Y0 - Math.floor((AEF_Y0 - spot.lat) / AEF_R) * AEF_R;
    const ring = [[x0, y1], [x0 + AEF_R, y1], [x0 + AEF_R, y1 - AEF_R], [x0, y1 - AEF_R], [x0, y1]];
    return new PathLayer({id, data: [ring], getPath: (d) => d, getColor: [255, 200, 40, 255], widthUnits: "pixels", getWidth: 3, widthMinPixels: 3, beforeId: slot(m)});
  }
  // a chunk read: the year card's line and the tooltip under a still pointer
  let tileT = null;
  function tileIn() { viewSoon(); clearTimeout(tileT); tileT = setTimeout(() => { if (cardData && cardData.kind === "spot" && cardData.score == null) { rescoreSpot(); } renderYear(); if (lastPt && !st.holding) showHexTip(lastPt); }, 200); }
  function layers() {
    const out = [];
    const z = map ? map.getZoom() : 0;
    // preloaded from zoom 9 only: below it the tiles are decimated from L5 (slow). Every year
    // stays mounted (the one shown loads first, a scroll is instant); the kernel's tiles (no
    // GPU path) keep only the year shown for false colors, which read three bands. Paired,
    // the imagery is the left map's, so none here
    if (!st.pair && (st.holding || z >= 9)) {
      if (S2GPU) { s2Discover(); for (const y of s2Years()) out.push(...s2CogLayers(y)); }
      else for (const y of S2Y) if (st.s2comp === "tci" || y === st.imgYear) out.push(s2Layer(y));
    }
    // while holding: the imagery, and over it only the two outlines
    //
    // kept in the stack while hidden (holding, zoomed out) so its shapes stay built
    out.push(earthwork.layer({visible: !st.holding && z >= EWZ, first: st.y0, last: st.y1, model: st.model, minZoom: EWZ, onTile: tileIn}));
    const hv = hover != null && hover >= 0 ? outline("hover", hexes[hover], [255, 255, 255, 235], 2) : null;
    if (hv) out.push(hv);
    // gold on the dark basemap (was near-black on the light one)
    const sc = searched ? outline("searched", searched, [255, 200, 40, 255], 3) : null;
    if (sc) out.push(sc);
    const pk = picked ? outline("picked", picked, [255, 200, 40, 255], 3) : null;
    if (pk) out.push(pk);
    const sp = spotLayer("spot");
    if (sp) out.push(sp);
    return out;
  }
  // the pair's left side: the imagery years, mounted as on the map so a year step is instant,
  // and over the imagery the same outlines as on the right: the hexagon under
  // the pointer (on either side), the searched one and the picked one
  function layers2() {
    if (!st.pair || !map2) return [];
    let out;
    if (S2GPU) { s2Discover(); out = s2Years().flatMap((y) => s2CogLayers(y, true)); }
    else out = S2Y.filter((y) => st.s2comp === "tci" || y === st.imgYear).map((y) => s2Layer(y, true));
    const hv = hover != null && hover >= 0 ? outline("hover-l", hexes[hover], [255, 255, 255, 235], 2, map2) : null;
    if (hv) out.push(hv);
    const sc = searched ? outline("searched-l", searched, [255, 200, 40, 255], 3, map2) : null;
    if (sc) out.push(sc);
    const pk = picked ? outline("picked-l", picked, [255, 200, 40, 255], 3, map2) : null;
    if (pk) out.push(pk);
    const sp = spotLayer("spot-l", map2);
    if (sp) out.push(sp);
    return out;
  }
  function update() {
    if (ov) ov.setProps({layers: layers()});
    if (ov2) ov2.setProps({layers: layers2()});
  }
  function labels(on) {
    for (const m of [map, map2]) {
      if (!m || !m.isStyleLoaded()) continue;
      (m.getStyle().layers || []).forEach((l) => { if (l.layout && l.layout["text-field"] !== undefined) m.setLayoutProperty(l.id, "visibility", on ? "visible" : "none"); });
    }
  }

  // ---- the pair (P) ---------------------------------------------------------------
  // a second map on the left, made the first time the pair opens; either map moves the
  // other. The right is the map as it was (its layer, card and clicks); the left shows
  // Sentinel-2 (the year as the hold leaves it)
  let syncing = false;
  function boot2() {
    if (map2) return;
    map2 = new maplibregl.Map({container: mapEl2, style: STYLE, center: map.getCenter(), zoom: map.getZoom(), attributionControl: false});
    map2.keyboard.disable();
    if (root._otf) { root._otf.map2 = map2; root._otf.layers2 = () => layers2().map((l) => l.id); }
    ov2 = new MapboxOverlay({interleaved: true, layers: [], onError: (e) => say("deck, left: " + (e && e.message ? e.message : e))});
    map2.addControl(ov2);
    map2.on("load", () => { labels(st.labels); update(); });
    const follow = (a, b) => a.on("move", () => {
      if (syncing || !st.pair) return;
      syncing = true;
      b.jumpTo({center: a.getCenter(), zoom: a.getZoom(), bearing: a.getBearing(), pitch: a.getPitch()});
      syncing = false;
    });
    follow(map, map2); follow(map2, map);
    // the left side hovers and picks as the right does (no tooltip, as over the imagery held):
    // the two sides are the same size under one camera, so a point is the same place on both
    map2.on("mousemove", (e) => { const i = hexAt(e.lngLat); if (i !== hover) { hover = i; update(); } });
    map2.on("mouseout", () => { if (hover != null && hover >= 0) { hover = null; update(); } });
    map2.on("click", (e) => { if (st.noPick) return; const i = hexAt(e.lngLat); pickCell(i >= 0 ? hexes[i] : null, e.lngLat, e.point, st.left === "s2"); });
    new ResizeObserver(() => { try { map2.resize(); } catch (e) {} }).observe(mapEl2);
  }
  function setPair(on) {
    if (on === st.pair || !map) return;
    if (on && st.holding) endHold(null);
    st.pair = on;
    root.classList.toggle("pair", on);
    // paired, the Earthwork panel and the hexagon card go over the hexagons on the right

    bPair.classList.toggle("on", on);
    bPair.title = on ? "back to one map (P)" : "pair the map with Sentinel-2 (P)";
    // the same zoom on both sides (each shows half the ground of one map), so the model and the
    // hexagons work in the pair exactly as on one map
    if (on) {
      boot2();
      map2.jumpTo({center: map.getCenter(), zoom: map.getZoom(), bearing: map.getBearing(), pitch: map.getPitch()});
      if (st.left === "s2" && st.s2comp !== "tci") send("s2comp", {comp: st.s2comp});
    }
    renderYear(); update();
  }

  // ---- hold: the imagery ------------------------------------------------------------
  // hold SPACE, only (the mouse stays free, so it can rest off the
  // building you are looking at, where the pointer would cover a small
  // one; the press and hold on the map is gone). The
  // imagery opens on the year the last
  // hold left off at (2022 the first time); while holding, the wheel anywhere over the map or the card
  // steps the imagery year instead of zooming; letting go of whichever
  // started it ends it
  let holdT = null, holdAt = null, holdBy = null, wheelAcc = 0, lastStep = 0, suppressClick = false, lastPt = null;
  const stepImg = (d) => { const i = S2Y.indexOf(st.imgYear); const n = S2Y[Math.max(0, Math.min(S2Y.length - 1, (i < 0 ? S2Y.length - 1 : i) + d))]; if (n !== st.imgYear) { st.imgYear = n; renderYear(); update(); } };
  function beginHold(x, y, by) {
    holdT = null;
    if (!map || st.holding) return;
    st.holding = true;
    holdBy = by;
    if (by === "mouse") suppressClick = true;
    mapEl.classList.add("holding");
    mapEl.classList.toggle("key", by === "key");
    // the wheel is the year while holding (the capture listener on root
    // keeps it from the map, so scrollZoom stays on: disabling it
    // mid-zoom left maplibre's zoom marked active, and scroll froze
    // after the hold); a space hold leaves the map free to drag ("i'd
    // like to be able to move the map when space is pressed"), a mouse
    // hold cannot (the press is the hold)
    if (by === "mouse") map.dragPan.disable();
    tip.style.display = "none";
    // false colors are stretched for the view: taken again if this hold is somewhere else
    if (st.s2comp !== "tci") send("s2comp", {comp: st.s2comp});
    renderYear(); update();
  }
  function endHold(by) {
    if (by !== "key") { clearTimeout(holdT); holdT = null; holdAt = null; }
    if (!st.holding || (by && by !== holdBy)) return;
    st.holding = false; holdBy = null;
    mapEl.classList.remove("holding");
    if (map) map.dragPan.enable();
    wheelAcc = 0;
    renderYear(); update();
    if (lastPt && map) showHexTip(lastPt);
  }
  mapEl.addEventListener("pointermove", (e) => {
    lastPt = {x: e.clientX, y: e.clientY};
    if (holdT && holdAt && Math.hypot(e.clientX - holdAt.x, e.clientY - holdAt.y) > SLOP) { clearTimeout(holdT); holdT = null; holdAt = null; }
  }, true);
  const endMouse = () => endHold("mouse"), endAny = () => endHold(null);
  window.addEventListener("pointerup", endMouse, true);
  window.addEventListener("pointercancel", endMouse, true);
  window.addEventListener("blur", endAny);
  // the space bar: down starts a hold at the pointer (or the map's
  // center before the pointer has been over it), up ends it
  function spaceDown() {
    // paired, the imagery is already on the left
    if (st.holding || !map || st.pair) return;
    const r = mapEl.getBoundingClientRect();
    const inMap = lastPt && lastPt.x >= r.left && lastPt.x <= r.right && lastPt.y >= r.top && lastPt.y <= r.bottom;
    const pt = inMap ? lastPt : {x: r.left + r.width / 2, y: r.top + r.height / 2};
    beginHold(pt.x, pt.y, "key");
  }
  const onKeyUp = (e) => {
    if (e.key === " ") endHold("key");
  };
  window.addEventListener("keyup", onKeyUp);
  root.addEventListener("wheel", (e) => {
    if (!st.holding) return;
    e.preventDefault(); e.stopPropagation();
    wheelAcc += e.deltaY;
    const now = performance.now();
    if (Math.abs(wheelAcc) >= 40 && now - lastStep > 140) { stepImg(wheelAcc > 0 ? 1 : -1); wheelAcc = 0; lastStep = now; }
  }, {capture: true, passive: false});
  root.addEventListener("pointermove", (e) => { lastPt = {x: e.clientX, y: e.clientY}; }, true);
  function showHexTip(pt) {
    if (st.noPick || !map || map.getZoom() < EWZ) { tip.style.display = "none"; return; }
    const r = mapEl.getBoundingClientRect();
    if (pt.x < r.left || pt.x > r.right || pt.y < r.top || pt.y > r.bottom) { tip.style.display = "none"; return; }
    // only over the map itself, never over a card
    const under = document.elementFromPoint(pt.x, pt.y);
    if (!under || !mapEl.contains(under)) { tip.style.display = "none"; return; }
    const ll = map.unproject([pt.x - r.left, pt.y - r.top]);
    const v = earthwork.scoreAt(ll.lng, ll.lat, st.y0, st.y1, st.model);
    if (v == null) { tip.style.display = "none"; return; }
    const p = pane.getBoundingClientRect();
    tip.innerHTML = `<b>${Math.round(100 * v)}%</b> earthwork, ${st.y0} to ${st.y1}`;
    tip.style.display = "block";
    tip.style.left = (pt.x - p.left + 14) + "px";
    tip.style.top = (pt.y - p.top + 14) + "px";
  }


  // ---- search ------------------------------------------------------------------------
  const PHOTON = "https://photon.komoot.io/api/";
  let gcHits = [], gcSel = -1, gcTimer = null, gcSeq = 0, searched = null;
  // drop the searched cell's outline, staying where the map is
  function unsearch() {
    searched = null; update();
  }
  // an H3 string in the box is a cell, not a place
  const h3Of = (q) => { const h = q.trim().toLowerCase(); try { return /^[0-9a-f]{15}$/.test(h) && isValidCell(h) ? h : null; } catch (e) { return null; } };
  const hitName = (f) => { if (f.h3) return "H3 " + f.h3; const p = f.properties || {}; return [p.name, p.street && !p.name ? p.street : null, p.city && p.city !== p.name ? p.city : null, p.state, p.country].filter(Boolean).join(", "); };
  const hitKind = (f) => { if (f.h3) { const [la, lo] = cellToLatLng(f.h3); return `res ${getResolution(f.h3)}, ${la.toFixed(5)}, ${lo.toFixed(5)}`; } const p = f.properties || {}; return [p.osm_value, p.type].filter((x) => x && x !== "yes").join(", "); };
  const gcHide = () => { hits.style.display = "none"; hits.replaceChildren(); gcSel = -1; };
  const gcShow = () => {
    hits.replaceChildren();
    if (!gcHits.length) { gcHide(); return; }
    gcHits.forEach((f, i) => { const r = el_("div", "at-hit" + (i === gcSel ? " sel" : "")); r.textContent = hitName(f); const k = el_("small"); k.textContent = hitKind(f); r.appendChild(k); r.onmousedown = (e) => { e.preventDefault(); gcFly(f); }; r.onmouseenter = () => { gcSel = i; gcShow(); }; hits.appendChild(r); });
    hits.style.display = "block";
  };
  const gcAsk = async () => {
    const q = gc.value.trim();
    if (!q && searched) { searched = null; update(); }
    if (q.length < 2) { gcHits = []; gcHide(); return; }
    const s = ++gcSeq;
    const h3 = h3Of(q);
    if (h3) { gcHits = [{h3}]; gcSel = 0; gcShow(); return; }
    const params = new URLSearchParams({q, limit: "6", lang: "en"});
    if (map) { const c = map.getCenter(); params.set("lon", c.lng.toFixed(4)); params.set("lat", c.lat.toFixed(4)); }
    try { const r = await fetch(PHOTON + "?" + params.toString()); const d = await r.json(); if (s !== gcSeq) return; gcHits = (d.features || []).filter((f) => f.geometry && f.geometry.coordinates); gcSel = gcHits.length ? 0 : -1; gcShow(); }
    catch (e) { if (s === gcSeq) note("search: " + e.message, 4000); }
  };
  const gcFly = (f) => {
    if (f.h3) {
      // to the zoom that draws hexagons of the cell's own res, the middle
      // of its zooms; finer than the finest drawn,
      // about 80 px across; never out past the hexagons
      const [lat, lon] = cellToLatLng(f.h3), r = getResolution(f.h3), L = cfg.res_ladder;
      const zoom = Math.max(HEXZ, Math.min(17, L && r <= L[3] ? L[0] + (r - L[2] + 0.5) * L[1]
        : Math.log2(78271.5 * Math.cos(lat * Math.PI / 180) * 80 / (2 * 1281256 / Math.pow(Math.sqrt(7), r)))));
      searched = f.h3; gcHits = []; gcHide(); gc.blur(); update();
      if (map) map.flyTo({center: [lon, lat], zoom, duration: 2200, essential: true});
      return;
    }
    searched = null;
    const [lon, lat] = f.geometry.coordinates;
    const ext = (f.properties || {}).extent;
    let zoom = 12;
    if (ext && ext.length === 4) { const span = Math.max(Math.abs(ext[2] - ext[0]), Math.abs(ext[1] - ext[3]) * 2, 0.01); zoom = Math.log2(360 * ((mapEl.clientWidth || 1200) / 512) / span) - 0.3; }
    zoom = Math.max(HEXZ, Math.min(16, zoom));
    gc.value = hitName(f); gcHits = []; gcHide(); gc.blur();
    if (map) map.flyTo({center: [lon, lat], zoom, duration: 2200, essential: true});
  };
  gc.addEventListener("input", () => { clearTimeout(gcTimer); gcTimer = setTimeout(gcAsk, 250); });
  gc.addEventListener("focus", () => { if (gcHits.length) gcShow(); });
  gc.addEventListener("blur", () => setTimeout(gcHide, 120));
  gc.addEventListener("keydown", (e) => {
    e.stopPropagation();
    if (e.key === "ArrowDown" && gcHits.length) { gcSel = (gcSel + 1) % gcHits.length; gcShow(); e.preventDefault(); }
    else if (e.key === "ArrowUp" && gcHits.length) { gcSel = (gcSel - 1 + gcHits.length) % gcHits.length; gcShow(); e.preventDefault(); }
    else if (e.key === "Enter") { e.preventDefault(); if (gcHits.length) gcFly(gcHits[Math.max(0, gcSel)]); else { clearTimeout(gcTimer); gcAsk().then(() => { if (gcHits.length) gcFly(gcHits[0]); else note("no match: " + gc.value.trim(), 4000); }); } }
    else if (e.key === "Escape") { gcHide(); gc.blur(); }
  });

  // ---- fill the window --------------------------------------------------------------
  const FIT_CLS = "at-fit-on";
  if (!document.getElementById("at-fit-style")) {
    const s = document.createElement("style"); s.id = "at-fit-style";
    s.textContent = ["notebook-actions-dropdown", "cell-actions-button", "drag-button", "expand-output-button", "fullscreen-output-button", "chrome-sidebar", "chrome-footer", "chrome-controls-top-right", "chrome-controls-bottom-right"].map((t) => "html." + FIT_CLS + " [data-testid='" + t + "']").join(",") + ",html." + FIT_CLS + " div[class*='top-[25vh]']{display:none!important}html." + FIT_CLS + "{overflow:hidden}";
    document.head.appendChild(s);
  }
  function sizes() {
    st.fit = true;
    root.classList.toggle("fit", st.fit);
    document.documentElement.classList.toggle(FIT_CLS, st.fit);
    pane.style.height = st.fit ? "100vh" : (cfg.height || 780) + "px";
    bFit.innerHTML = st.fit ? ICON.shrink : ICON.expand; bFit.title = st.fit ? "back to the notebook (X or Esc); full screen (F)" : "fill the window (X); full screen (F)";
    setTimeout(() => { try { map && map.resize(); } catch (e) {} }, 30);
  }
  bFit.onclick = () => { st.fit = !st.fit; sizes(); };
  bMore.onclick = (e) => { e.stopPropagation(); const open = more.style.display !== "block"; more.style.display = open ? "block" : "none"; yc.style.visibility = open ? "hidden" : ""; bMore.classList.toggle("on", open); };
  more.addEventListener("click", (e) => e.stopPropagation());
  const closeMore = () => { more.style.display = "none"; yc.style.visibility = ""; bMore.classList.remove("on"); };
  root.addEventListener("click", () => { if (more.style.display === "block") closeMore(); });
  window.addEventListener("resize", () => { if (st.fit) sizes(); });

  // ---- keys ---------------------------------------------------------------------------
  root.tabIndex = 0;
  const onKey = (e) => {
    const path = e.composedPath ? e.composedPath() : [];
    if (!st.fit && !path.includes(root)) return;
    const tgt = path[0] || e.target;
    if (tgt && /^(INPUT|SELECT|TEXTAREA)$/.test(tgt.tagName)) return;
    const k = e.key, lo = st.y0, hi = st.y1;
    if (k === " ") { if (!e.repeat) spaceDown(); }
    // Shift + arrows: turn the map (left, right: 15 degrees) and tilt it (up, down: 10), as MapLibre's own
    // keys do; held, it keeps going
    else if (e.shiftKey && /^Arrow(Left|Right|Up|Down)$/.test(k)) {
      if (!map) return;
      const db = k === "ArrowLeft" ? -15 : k === "ArrowRight" ? 15 : 0, dp = k === "ArrowUp" ? 10 : k === "ArrowDown" ? -10 : 0;
      map.easeTo({bearing: map.getBearing() + db, pitch: Math.max(0, Math.min(map.getMaxPitch(), map.getPitch() + dp)), duration: e.repeat ? 120 : 250});
    }
    // Color by: E Earthwork
    else if (/^[eE]$/.test(k)) { st.want = "earth"; const m = drawnMode(); if (m !== st.gmode) { st.gmode = m; recolorHex(); renderYear(); update(); } styleRows(); }
    // Q: the tooltip and picking off and back on
    else if (k === "q" || k === "Q") { st.noPick = !st.noPick; swPick.sty(); tip.style.display = "none"; hover = null; update(); note(st.noPick ? "No tooltip or picking (Q to turn them back on)" : "Tooltip and picking on", 2500); }
    // the kinds key's Built (W)
    else if (k === "w" || k === "W") { if (st.gmode !== "kinds") return; st.focus = "built"; recolorHex(); styleKey(); update(); }
    else if (k === "p" || k === "P") setPair(!st.pair);
    else if (k === "[" || k === "]") stepImg(k === "]" ? 1 : -1);
    // B: the imagery's first year and its latest, back and forth (from any other year, the
    // latest), while it shows: holding space, or on the pair's left side
    else if (k === "b" || k === "B") {
      if (!(st.holding || (st.pair && st.left === "s2"))) return;
      const a = S2Y[0], b = S2Y[S2Y.length - 1];
      st.imgYear = st.imgYear === b ? a : b; renderYear(); update();
    }
    else if (k === ";" || k === "'") { st.s2scale = Math.round(10 * Math.max(0.3, Math.min(2.5, st.s2scale + (k === "'" ? 0.1 : -0.1)))) / 10; gam.value = st.s2scale; clearTimeout(gamT); gamT = setTimeout(() => send("s2scale"), 250); }
    else if (k === "l" || k === "L") { st.labels = !st.labels; labels(st.labels); swLab.sty(); }
    else if (k === "x" || k === "X") { st.fit = !st.fit; sizes(); }
    else if (k === "c" || k === "C") { const i = COMPS.findIndex(([c]) => c === st.s2comp); setComp(COMPS[(i + 1) % COMPS.length][0]); }
    else if ((k === "m" || k === "M") && MODELS.length > 1) setModel(MODELS[(MODELS.indexOf(st.model) + 1) % MODELS.length]);
    // full screen (the browser's own; Esc or F again leaves it), filling the window inside it
    else if (k === "f" || k === "F") {
      if (document.fullscreenElement) document.exitFullscreen().catch(() => {});
      else { if (!st.fit) { st.fit = true; sizes(); } (root.requestFullscreen ? root.requestFullscreen() : Promise.reject()).catch(() => note("full screen is not allowed here", 2500)); }
    }
    else if (k === "/") gc.focus();
    // H: every card folded, or all open again
    else if (k === "h" || k === "H") setCards(!cardsHidden);
    else if (k === "o" || k === "O") bMore.onclick(e);
    else if (k === "?") { closeMore(); about.style.display = about.style.display === "flex" ? "none" : "flex"; }
    else if (k === "Escape") { if (searched) unsearch(); else if (about.style.display === "flex") about.style.display = "none"; else if (more.style.display === "block") closeMore(); else if (cardData) closeCard(); else if (st.fit) { st.fit = false; sizes(); } }
    else return;
    e.preventDefault();
  };
  window.addEventListener("keydown", onKey);
  // the earthwork model on the GPU is ready: draw with it
  window.addEventListener("earthwork-ready", () => update());
  window.addEventListener("analytics-status", (e) => { avStatus = e.detail; renderYear(); });
  window.addEventListener("analytics-ready", () => viewSoon());

  // ---- camera and click ----------------------------------------------------------------
  let seq = 0, lastView = "";
  function sendView() {
    if (!map) return;
    const c = map.getCenter();
    const v = {longitude: c.lng, latitude: c.lat, zoom: map.getZoom(), w: mapEl.clientWidth, h: mapEl.clientHeight};
    const key = JSON.stringify(v);
    if (key === lastView) return;
    lastView = key; v.n = ++seq;
    model.set("view", JSON.stringify(v)); model.save_changes();
  }
  const adminAt = (pt) => {
    const out = {};
    const one = (k) => { const id = "ov-div-" + k; if (!map.getLayer(id)) return null; const fs = map.queryRenderedFeatures(pt, {layers: [id]}); return fs && fs.length ? fs[0].properties : null; };
    try { const r = one("region"); if (r) out.region = r["@name"] || r.names || null; const c = one("county"); if (c) out.county = c["@name"] || c.names || null; const l = one("locality"); if (l) out.locality = l["@name"] || l.names || null; } catch (e) {}
    return out;
  };
  function boot() {
    const home = cfg.home || {longitude: 3.6, latitude: 6.46, zoom: 11};
    map = new maplibregl.Map({container: mapEl, style: STYLE, center: [home.longitude, home.latitude], zoom: home.zoom, attributionControl: false, hash: true});
    map.keyboard.disable();
    map.doubleClickZoom.enable();
    root._otf = {map, st, cfg: () => cfg, hmeta: () => hmeta, hexShown: () => (hexData && hcol ? hmeta.seq : 0), tiles: () => tlog, s2: () => ({...s2Stat, warm: s2Warm, sources: Object.fromEntries([...s2Src].map(([y, e]) => [y, e.ids.size])), batches: Object.fromEntries([...s2Src].map(([y, e]) => [y, e.batches.length]))})};  // for headless tests
    map.addControl(new maplibregl.NavigationControl({showCompass: false}), "bottom-left");
    ov = new MapboxOverlay({interleaved: true, layers: [], onError: (e) => say("deck: " + (e && e.message ? e.message : e))});
    map.addControl(ov);
    map.on("load", () => {
      labels(st.labels);
      if (cfg.div_pm && !map.getSource("ov-div")) {
        try {
          map.addSource("ov-div", {type: "vector", url: "pmtiles://" + cfg.div_pm});
          for (const k of ["region", "county", "locality"]) map.addLayer({id: "ov-div-" + k, type: "fill", source: "ov-div", "source-layer": "division_area", filter: ["all", ["==", ["get", "subtype"], k], ["==", ["get", "class"], "land"]], paint: {"fill-opacity": 0}}, slot());
        } catch (e) { console.error("divisions", e); }
      }
      update(); sendView(); renderYear();
    });
    map.on("moveend", sendView);
    // a new view: the footprints under it asked for
    // (and the batches near it mounted, the far ones let go)
    map.on("moveend", () => { if (S2GPU) { s2Discover(); update(); } });
    map.on("zoomend", () => { update(); renderYear(); styleKey(); });
    map.on("moveend", viewSoon);
    // a zoom in crossing 8.3 (the kernel's _AHEAD_ZOOM) tells the kernel
    // at once, so it starts reading before the hexagons' zoom
    let aheadSent = false;
    map.on("zoom", () => { styleSoon(); const z = map.getZoom(); if (z < 8.3) aheadSent = false; else if (!aheadSent && z < HEXZ) { aheadSent = true; sendView(); } });
    map.on("mousemove", (e) => {
      if (holdT) return;
      // over the imagery: the white outline follows the pointer, no tooltip
      if (st.holding) { const i = hexAt(e.lngLat); if (i !== hover) { hover = i; update(); } return; }
      showHexTip({x: e.originalEvent.clientX, y: e.originalEvent.clientY});
    });
    map.on("mouseout", () => { tip.style.display = "none"; if (hover != null && hover >= 0) { hover = null; update(); } });
    map.on("click", (e) => {
      if (suppressClick) { suppressClick = false; return; }
      if (st.noPick) return;
      const i = hexAt(e.lngLat);
      pickCell(i >= 0 ? hexes[i] : null, e.lngLat, e.point, st.holding);
    });
    map.on("error", (ev) => { if (ev && ev.error && ev.error.message && !/tile|404/i.test(ev.error.message)) say("map: " + ev.error.message); });
    new ResizeObserver(() => { try { map.resize(); } catch (e) {} fitCard(); }).observe(mapEl);
    window.__cmMaps = () => [map];
    window.__cmState = () => ({st: Object.assign({}, st), hex: N, res, hmeta, tiles: tstat, card: cardData, status: model.get("status"),
      lc: hattrs ? Array.from({length: N}, (_, i) => hattrs[HB * i + 7]).reduce((c, v) => (c[v]++, c), [0, 0, 0, 0]) : null});
    window.__cmTiles = () => ({log: tlog, frames: flog});
    // for tests: the center of the first hexagon whose biggest step is year y and that moved a fair amount
    window.__cmHexAt = (y) => { const b = map.getBounds(); for (let i = 0; i < N; i++) if (hattrs && hattrs[HB * i] === y - 2000 && hattrs[HB * i + 1] >= FAIR) { const r = cellToBoundary(hexes[i], true); const c = r.slice(0, -1).reduce((a, p) => [a[0] + p[0] / (r.length - 1), a[1] + p[1] / (r.length - 1)], [0, 0]); if (b.contains(c) && map.project(c).x < mapEl.clientWidth - 420 && map.project(c).y > 200) return c; } return null; };
  }

  // ---- the kernel's data -----------------------------------------------------------------
  const flog = [];  // per frame received, for the tests
  const loadHex = () => {
    const tl = performance.now();
    const cb = bytesOf(model.get("cells")), ab = bytesOf(model.get("hattrs"));
    const seq0 = hmeta.seq;
    try { hmeta = JSON.parse(model.get("hmeta") || "{}"); } catch (e) { hmeta = {}; }
    // the kinds are grouped again for every new frame: what was hidden no longer means the same
    if (hmeta.seq !== seq0) st.hideKinds.clear();
    // zoomed out of Kinds of change: back to AEF Change, and it stays there when zooming in again
    if (kindsWait()) st.want = "much";
    st.gmode = drawnMode();
    styleFill();
    if (!cb || !cb.length) { hexes = []; N = 0; hexIndex = new Map(); res = -1; hattrs = null; hcol = null; hcol32 = null; hexData = null; renderYear(); styleKey(); update(); return; }
    const ids = new BigUint64Array(copyOf(cb));
    N = ids.length; hexes = new Array(N); hexIndex = new Map();
    for (let i = 0; i < N; i++) { const h = ids[i].toString(16); hexes[i] = h; hexIndex.set(h, i); }
    try { res = getResolution(hexes[0]); } catch (e) { res = -1; }
    hattrs = ab && ab.length === HB * N ? new Uint8Array(copyOf(ab)) : null;
    hexData = {length: N};
    hover = null;
    recolorHex(); renderYear(); styleKey(); update();
    flog.push({seq: hmeta.seq, n: N, t: Date.now(), ms: performance.now() - tl});
  };
  let pendHex = null;
  const hexSoon = () => { clearTimeout(pendHex); pendHex = setTimeout(loadHex, 0); };
  model.on("change:cells", hexSoon);
  model.on("change:hattrs", hexSoon);
  model.on("change:hmeta", hexSoon);
  model.on("change:card", renderCard);
  model.on("change:status", () => say(model.get("status")));
  model.on("change:config", () => { try { cfg = JSON.parse(model.get("config") || "{}"); } catch (e) { cfg = {}; } update(); });
  try {
    sizes(); boot(); styleRows(); loadHex(); renderCard(); setCards(cardsHidden); say(model.get("status"));
  } catch (e) { say("boot: " + e.message); console.error(e); }
  return () => { window.removeEventListener("keydown", onKey); window.removeEventListener("keyup", onKeyUp); window.removeEventListener("pointerup", endMouse, true); window.removeEventListener("pointercancel", endMouse, true); window.removeEventListener("blur", endAny); document.documentElement.classList.remove(FIT_CLS); try { map && map.remove(); } catch (e) {} };
}

