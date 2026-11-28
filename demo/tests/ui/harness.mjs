// Test harness for the search UI (src/vocalocate_demo/ui/app.js).
//
// app.js is a classic browser script with no build step and no exports, so the tests load the
// real index.html into jsdom and run the unmodified app.js inside the page's VM context under its
// real file name. That way c8 (V8 coverage) attributes every executed line to app.js, and the
// tests can reach the app's top-level state and functions (S, render, doSearch, ...) through
// `app.ev("...")`, because top-level declarations of scripts in one context share a scope.
//
// Browser APIs jsdom lacks (fetch, canvas, Web Audio, media playback, matchMedia, the mic,
// the clipboard) are replaced with small fakes that the tests can steer.
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
import vm from "node:vm";
import { JSDOM, VirtualConsole } from "jsdom";
import { afterEach } from "node:test";

// Close every page a test opened, even when an assertion failed before its own close().
const open = new Set();
afterEach(async () => {
  for (const app of open) app.close();
  await flush(); // let late fetches and decodes settle before the windows go away
  for (const app of open) app.w.close();
  open.clear();
});

const UI = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../src/vocalocate_demo/ui");
const APP_JS = path.join(UI, "app.js");
const HTML = readFileSync(path.join(UI, "index.html"), "utf8").replace(/<script[^>]*><\/script>/, "");
const SCRIPT = new vm.Script(readFileSync(APP_JS, "utf8"), { filename: APP_JS });

export const READY = () => ({
  libraries: [{ name: "SFX", state: "ready", roots: ["D:\\SFX"], files: 3, models_ready: ["m1", "m2"] }],
  models: [{ name: "m1", label: "M1: the default model" }, { name: "m2", label: "M2" }],
  examples: [{ id: 7, name: "dog.wav", category: "animal", hint: "bark" }],
  hints: ["metallic", "short"],
  text_state: "ready",
  boot: { state: "ready", message: "" },
  needs: [],
  ready: true,
  settings: { model: "m1", autostart: false, autostart_supported: true, gpu: { available: true, name: "RTX 5080" }, device: "auto", device_active: "auto", restart_supported: true },
  home: "C:\\VL", lake: "C:\\VL\\lake", source: "D:\\models", clap: "clap-v1",
});

export const RESULTS = () => ({
  library: "SFX", searched: 3, ms: 0.4, model: "m1", model_label: "M1: the default model", text: "metal", folders: ["a", "b"],
  results: [
    { id: 1, rank: 1, name: "clang.wav", path: "D:\\SFX\\proj\\clang.wav", project: "proj", score: 0.9, voice_score: 0.8, text_score: 0.7 },
    { id: 2, rank: 2, name: "a:b.wav", path: "D:\\SFX\\misc\\a:b.wav", score: 0.5 },
    { id: 3, rank: 3, name: "c.wav", path: "/x/SFX/c.wav", score: 0.2 },
  ],
});

// An AudioBuffer-shaped object.
export function audioBuffer(channels = 2, length = 16000, sampleRate = 16000) {
  const data = Array.from({ length: channels }, (_, c) => Float32Array.from({ length }, (_, i) => Math.sin(i / 7 + c) * 0.5));
  return { numberOfChannels: channels, length, sampleRate, duration: length / sampleRate, getChannelData: (c) => data[c] };
}

export const flush = async (n = 20) => { for (let i = 0; i < n; i++) await new Promise((r) => setImmediate(r)); };

/**
 * Boot the app in a fresh page.
 * routes: { "GET /api/status": value | (body, init) => value, ... }. A value is
 *   { status, body } (body may be undefined for a non-JSON reply), or the string "offline"
 *   to make fetch reject. A missing route answers 404 "not found".
 */
export async function boot({ routes = {}, status = READY(), storage = {}, options = {} } = {}) {
  const calls = [];
  const vc = new VirtualConsole(); // swallow jsdom's "not implemented" noise
  const dom = new JSDOM(HTML, { url: "http://localhost/", runScripts: "outside-only", pretendToBeVisual: true, virtualConsole: vc });
  const w = dom.window;
  for (const [k, v] of Object.entries(storage)) w.localStorage.setItem(k, v);

  const table = { "GET /api/status": { status: 200, body: status }, "GET /api/health": { status: 200, body: {} }, ...routes };
  const reply = (status, body) => ({
    ok: status >= 200 && status < 300, status,
    json: async () => { if (body === undefined) throw new Error("not json"); return body; },
    arrayBuffer: async () => new ArrayBuffer(8),
  });
  w.fetch = async (url, init = {}) => {
    const method = (init && init.method) || "GET";
    const body = init && init.body ? JSON.parse(init.body) : undefined;
    calls.push({ method, url, body });
    let r = table[`${method} ${url}`];
    if (typeof r === "function") r = await r(body, init);
    if (r === "offline") throw new TypeError("Failed to fetch");
    if (!r) return reply(404, { error: "not found" });
    return reply(r.status ?? 200, r.body);
  };
  const app = {
    dom, w, calls, routes: table, closed: false,
    $: (s) => w.document.querySelector(s),
    ev: (code) => vm.runInContext(code, dom.getInternalVMContext()),
    set(key, value) { table[key] = value; },
    close() {
      if (app.closed) return;
      app.closed = true;
      table["GET /api/health"] = { status: 200, body: {} };
      table["GET /api/status"] = { status: 200, body: READY() };
      try { vm.runInContext("S.online = true; clearTimeout(S.pollT)", dom.getInternalVMContext()); } catch (e) { /* closed */ }
      w.setTimeout = () => 0; // the window itself is closed after the test, in afterEach
    },
  };

  // Rendering + media fakes.
  const ctx2d = { setTransform() {}, clearRect() {}, fillRect() { app.rects = (app.rects || 0) + 1; }, fillStyle: "" };
  w.HTMLCanvasElement.prototype.getContext = () => ctx2d;
  w.HTMLElement.prototype.scrollIntoView = function () {};
  w.matchMedia = (q) => ({ matches: !!options.reducedMotion && q.includes("reduced-motion"), addEventListener(_, f) { app.onScheme = f; } });
  w.CSS = { escape: (s) => String(s).replace(/["\\]/g, "\\$&") };
  w.URL.createObjectURL = () => "blob:fake";
  w.URL.revokeObjectURL = () => {};
  w.document.execCommand = () => { if (options.execThrows) throw new Error("no"); return true; };
  w.HTMLMediaElement.prototype.pause = function () {};
  w.HTMLMediaElement.prototype.load = function () {};
  w.HTMLMediaElement.prototype.play = function () { return options.playFails ? Promise.reject(new Error("blocked")) : Promise.resolve(); };
  Object.defineProperty(w.navigator, "clipboard", {
    configurable: true,
    value: { writeText: async (t) => { if (options.clipboardFails) throw new Error("denied"); app.copied = t; } },
  });

  // Web Audio + microphone fakes.
  const node = () => ({ connect() {}, disconnect() { if (options.disconnectThrows) throw new Error("gone"); }, port: {} });
  class FakeCtx {
    constructor() { this.sampleRate = 16000; this.state = "suspended"; this.destination = {}; app.audioCtx = this; }
    resume() { this.state = "running"; return Promise.resolve(); }
    close() { this.closed = true; }
    createMediaStreamSource() { return node(); }
    createGain() { return { gain: { value: 1 }, connect() {} }; }
    createScriptProcessor() { const n = node(); app.processor = n; return n; }
    decodeAudioData(buf, ok, err) {
      if (options.decodeFails) { err(new Error("bad audio")); return undefined; }
      const ab = options.decoded || audioBuffer();
      if (options.decodePromise) return Promise.resolve(ab);
      ok(ab);
      return undefined;
    }
    get audioWorklet() { return options.worklet ? { addModule: async () => {} } : undefined; }
  }
  w.AudioContext = FakeCtx;
  w.AudioWorkletNode = class { constructor() { const n = node(); Object.assign(this, n); app.worklet = this; } };
  if (options.noMic) Object.defineProperty(w.navigator, "mediaDevices", { configurable: true, value: undefined });
  else Object.defineProperty(w.navigator, "mediaDevices", {
    configurable: true,
    value: { getUserMedia: async () => { if (options.micDenied) throw new Error("denied"); return { getTracks: () => [{ stop() { app.trackStopped = true; } }] }; } },
  });

  open.add(app);
  SCRIPT.runInContext(dom.getInternalVMContext());
  await flush();
  return app;
}

// Feed recorded samples into a running recording (worklet or ScriptProcessor path).
export function feed(app, samples) {
  if (app.worklet && app.worklet.port.onmessage) app.worklet.port.onmessage({ data: samples });
  else app.processor.onaudioprocess({ inputBuffer: { getChannelData: () => samples } });
}

export function key(app, init, target = app.w.document.body) {
  const e = new app.w.KeyboardEvent("keydown", { bubbles: true, cancelable: true, ...init });
  target.dispatchEvent(e);
  return e;
}
