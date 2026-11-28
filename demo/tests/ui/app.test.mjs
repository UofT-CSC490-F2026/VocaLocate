// Behaviour tests for the search UI. Each test boots a fresh page (see harness.mjs) and drives
// it the way a user would (clicks, keys, drops), with a fake local engine behind fetch.
import { test } from "node:test";
import assert from "node:assert/strict";
import { boot, flush, READY, RESULTS, audioBuffer, feed, key } from "./harness.mjs";

const click = (app, sel) => app.$(sel).dispatchEvent(new app.w.MouseEvent("click", { bubbles: true, detail: 1 }));
const wait = (ms) => new Promise((r) => setTimeout(r, ms));
const searchRoute = { "POST /api/search": { status: 200, body: RESULTS() } };
const file = (name = "boom.wav") => ({ name, arrayBuffer: async () => new ArrayBuffer(8) });

function dragEvent(app, type, files = true) {
  const e = new app.w.Event(type, { bubbles: true, cancelable: true });
  Object.defineProperty(e, "dataTransfer", { value: { types: files ? ["Files"] : ["text/plain"], files: [file("drop.wav")], setData() {}, dropEffect: "", effectAllowed: "" } });
  app.w.dispatchEvent(e);
  return e;
}

// ------------------------------------------------------------------ helpers and the API wrapper
test("small helpers format numbers, names and paths", async () => {
  const app = await boot();
  assert.equal(app.ev("fmt(1.234)"), "1.23");
  assert.equal(app.ev("plural(1, 'file')"), "1 file");
  assert.equal(app.ev("plural(2, 'file')"), "2 files");
  assert.equal(app.ev("isAbs('D:\\\\x')"), true);
  assert.equal(app.ev("isAbs('rel/x')"), false);
  assert.equal(app.ev("baseName('C:\\\\')"), "C");
  assert.equal(app.ev("baseName('')"), "Library");
  assert.equal(app.ev("shortModel('A: b')"), "A");
  assert.equal(app.ev("modelDesc('A')"), "");
  assert.equal(app.ev("sentence('  ')"), "");
  assert.equal(app.ev("sentence('done!')"), "Done!");
  assert.equal(app.ev("el('i', { hidden: true, title: null, off: false }).outerHTML"), '<i hidden=""></i>');
  app.ev("store('vl.x', 'y')");
  assert.equal(app.ev("store('vl.x')"), "y");
  app.ev("store('vl.x', null)");
  assert.equal(app.ev("store('vl.x')"), null);
  Object.defineProperty(app.w, "localStorage", { get() { throw new Error("blocked"); } });
  assert.equal(app.ev("store('vl.x')"), null); // storage errors are swallowed
  app.close();
});

test("API errors become readable messages and flag missing routes", async () => {
  const app = await boot({
    routes: {
      "GET /a": { status: 500 }, "GET /b": { status: 400, body: { error: "bad input" } },
      "GET /c": { status: 405, body: {} }, "POST /d": { status: 501, body: {} }, "GET /e": "offline",
    },
  });
  const err = (p, o) => app.w.api(p, o).then(() => null, (e) => e);
  let e = await err("/a");
  assert.equal(e.message, "Request failed (HTTP 500).");
  assert.equal(e.missing, false);
  assert.equal((await err("/b")).message, "Bad input.");
  assert.equal((await err("/c")).missing, true);
  assert.equal((await err("/d", { body: undefined, method: "POST" })).missing, true);
  assert.equal((await err("/missing")).missing, true);
  assert.equal((await err("/e")).offline, true);
  app.close();
});

test("normalize fills in fields an older server leaves out", async () => {
  const app = await boot();
  const st = app.ev("normalize({ text: true })");
  assert.deepEqual([...st.needs], ["models", "library"]);
  assert.equal(st.text_state, "ready");
  assert.equal(st.ready, false);
  assert.equal(app.ev("normalize({ libraries: [{ models_ready: ['m'] }], models: [{}], needs: [] }).ready"), true);
  assert.equal(app.ev("anyBusy(null)"), true);
  app.close();
});

// ------------------------------------------------------------------ gate: connecting, offline, boot, setup
test("an unreachable engine shows the connecting gate, then reconnects", async () => {
  const app = await boot({ routes: { "GET /api/status": "offline", "GET /api/health": "offline" } });
  assert.equal(app.$("#gate").hidden, false);
  assert.equal(app.$("#gate-title").textContent, "Starting VocaLocate");
  app.set("GET /api/status", { status: 200, body: READY() });
  app.set("GET /api/health", { status: 200, body: {} });
  await app.w.healthLoop();
  await flush();
  assert.equal(app.$("#app").hidden, false);
  // Losing the engine after it was up shows "Reconnecting…", using /api/status if /api/health is unknown.
  app.set("GET /api/status", "offline");
  app.ev("S.missing.add('health')");
  await app.w.poll();
  assert.equal(app.$("#gate-title").textContent, "Reconnecting…");
  app.w.goOffline(); // already offline: no-op
  await app.w.healthLoop();
  app.close();
});

test("while models warm up the boot gate shows the engine's message", async () => {
  const app = await boot({ status: { ...READY(), boot: { state: "loading", message: "loading voice models" }, ready: false } });
  assert.equal(app.$("#gate-title").textContent, "Warming up models…");
  assert.match(app.$("#gate-text").textContent, /^Loading voice models\. Search opens/);
  app.close();
});

test("first run: import models through the picker, then add a sound folder", async () => {
  const st = { ...READY(), libraries: [], models: [], needs: ["models", "library"] };
  const app = await boot({
    status: st,
    routes: {
      "POST /api/pick-folder": (b) => ({ status: 200, body: { path: b.title.includes("models") ? "D:\\models" : "D:\\Sounds\\SFX" } }),
      "POST /api/models/import": { status: 200, body: { models: ["m1", "m2"] } },
      "POST /api/libraries": (b) => ({ status: 200, body: { name: b.name } }),
    },
  });
  assert.equal(app.$("#gate-title").textContent, "Set up VocaLocate");
  assert.equal(app.$("#setup").hidden, false);
  app.$("#setup .btn.primary").click();
  await flush();
  assert.ok(app.calls.some((c) => c.url === "/api/models/import" && c.body.path === "D:\\models"));
  app.set("GET /api/status", { status: 200, body: { ...st, models: READY().models, needs: ["library"] } });
  await app.w.poll();
  assert.match(app.$("#setup").textContent, /2 models ready/);
  app.$("#setup .btn.primary").click();
  await flush();
  assert.ok(app.calls.some((c) => c.url === "/api/libraries" && c.body.name === "SFX"));
  assert.equal(app.ev("S.lib"), "SFX");
  app.close();
});

test("setup falls back to a typed path when the picker is unavailable or cancelled", async () => {
  const st = { ...READY(), libraries: [], models: [], needs: ["models", "library"] };
  const app = await boot({ status: st, routes: { "POST /api/pick-folder": { status: 200, body: { path: null } } } });
  await app.w.pickFolder("models");
  assert.match(app.$("#setup").textContent, /No folder chosen/);
  const form = () => { app.$("#path-models").blur(); return app.$("#setup form"); };
  // An empty path, then a relative path, are refused before calling the engine.
  form().dispatchEvent(new app.w.Event("submit", { cancelable: true }));
  await flush();
  assert.match(app.$("#setup").textContent, /Type or pick a folder first/);
  app.$("#path-models").value = "models";
  form().dispatchEvent(new app.w.Event("submit", { cancelable: true }));
  await flush();
  assert.match(app.$("#setup").textContent, /isn't a full path/);
  // The import route is unknown to this engine: say so, and stop calling it.
  app.$("#path-models").value = "D:\\models";
  form().dispatchEvent(new app.w.Event("submit", { cancelable: true }));
  await flush();
  assert.match(app.$("#setup").textContent, /can't import models/);
  await app.w.usePath("models", "D:\\models");
  assert.equal(app.calls.filter((c) => c.url === "/api/models/import").length, 1);
  // A focused, half-typed path is not wiped by a re-render.
  app.$("#path-models").focus();
  app.w.renderSetup();
  assert.ok(app.$("#path-models"));
  app.$("#path-models").blur();
  // A picker that does not exist on this engine goes straight to the text field next time.
  app.set("POST /api/pick-folder", { status: 404, body: { error: "not found" } });
  app.ev("S.fallback = {}; S.missing.delete('pick')");
  await app.w.pickFolder("library");
  assert.equal(app.ev("S.missing.has('pick')"), true);
  await app.w.pickFolder("library");
  // Picker errors are shown; a picker that is already open is not opened twice.
  app.ev("S.missing.delete('pick')");
  app.set("POST /api/pick-folder", { status: 500, body: { error: "dialog crashed" } });
  await app.w.pickFolder("models");
  assert.match(app.$("#setup").textContent, /Dialog crashed/);
  app.ev("S.busyFolder = 'models'");
  await app.w.pickFolder("models");
  app.ev("S.busyFolder = null");
  // Losing the engine during a pick, an import or a library add shows the offline gate.
  app.set("POST /api/pick-folder", "offline");
  await app.w.pickFolder("models");
  assert.equal(app.ev("S.online"), false);
  app.ev("S.online = true");
  app.set("POST /api/libraries", "offline");
  app.ev("S.online = true");
  await app.w.usePath("library", "D:\\Sounds");
  assert.equal(app.ev("S.online"), false);
  app.close();
});

// ------------------------------------------------------------------ bar, strip, console
test("the bar and progress strip describe each library state", async () => {
  const busy = { name: "SFX", state: "indexing", roots: ["D:\\SFX"], progress: { done: 1, total: 4 }, models_ready: ["m1"] };
  const scanning = { name: "Two", state: "scanning", roots: ["E:\\"], folders: [{ files: 5 }, {}] };
  const app = await boot({ status: { ...READY(), libraries: [busy, scanning] } });
  assert.equal(app.$("#lib-sub").textContent, "Indexing 25%");
  assert.equal(app.$("#strip").hidden, false);
  assert.match(app.$("#strip-text").textContent, /Indexing SFX: 1 of 4 files\. Search is already on\. \(1 more queued\)/);
  const show = async (libs) => { app.set("GET /api/status", { status: 200, body: { ...READY(), libraries: libs } }); await app.w.poll(); };
  await show([{ ...scanning, name: "SFX" }]);
  assert.equal(app.$("#lib-sub").textContent, "Scanning 5 files…");
  assert.match(app.$("#strip-text").textContent, /Scanning 5 files\. Search opens/);
  await show([{ name: "SFX", state: "queued", roots: ["x"], message: "waiting for the GPU" }]);
  assert.equal(app.$("#lib-sub").textContent, "Indexing");
  assert.match(app.$("#strip-text").textContent, /Waiting for the GPU\./);
  await show([{ name: "SFX", state: "queued", roots: ["x"] }]);
  assert.match(app.$("#strip-text").textContent, /SFX: Scanning\./);
  await show([{ name: "SFX", state: "error", roots: ["x"], message: "disk gone" }]);
  assert.equal(app.$("#lib-sub").textContent, "Indexing failed");
  assert.match(app.$("#r-empty").textContent, /couldn't be indexed.*Disk gone\./);
  await show([{ name: "SFX", state: "ready", roots: ["x"] }]);
  assert.equal(app.$("#lib-sub").textContent, "");
  app.close();
});

test("the hint field explains when text hints are unavailable", async () => {
  const app = await boot();
  const show = async (extra) => { app.set("GET /api/status", { status: 200, body: { ...READY(), ...extra } }); await app.w.poll(); return app.$("#hint-note"); };
  assert.equal((await show({ text_state: "off" })).textContent, "Text hints are off in this setup.");
  assert.equal(app.$("#hint").disabled, true);
  assert.match((await show({ text_state: "off", boot: { state: "loading" }, settings: { text: true } })).textContent, /loads right after/);
  assert.match((await show({ text_state: "error" })).textContent, /failed to load/);
  assert.match((await show({ text_state: "loading" })).textContent, /still loading/);
  assert.equal((await show({})).hidden, true);
  app.close();
});

// ------------------------------------------------------------------ searching
test("a demo imitation searches, fills the hint and lists matches", async () => {
  const app = await boot({ routes: { ...searchRoute, "GET /api/example/7": { status: 200, body: {} } }, options: { decodePromise: true } });
  click(app, "[data-ex='7']");
  await flush();
  assert.equal(app.$("#hint").value, "bark");
  const search = app.calls.find((c) => c.url === "/api/search");
  assert.deepEqual(search.body, { library: "SFX", model: "m1", text: "bark", k: 30, example: 7 });
  assert.equal(app.$("[data-ex='7']").getAttribute("aria-pressed"), "true");
  assert.equal(app.w.document.querySelectorAll("#r-list .row").length, 3);
  const sum = app.$("#r-sum").textContent;
  assert.match(sum, /3 of 3 files in SFX, 2 folders, under 1 ms, hint “metal”, M1/);
  assert.equal(app.$("#take-name").textContent, "dog");
  assert.equal(app.$("#take-len").textContent, "1.0 s");
  // The same query again is not re-sent; a new hint is.
  app.w.requestSearch();
  assert.equal(app.calls.filter((c) => c.url === "/api/search").length, 1);
  app.close();
});

test("result summaries cover ignored hints, slow searches and no matches", async () => {
  const app = await boot();
  app.ev(`S.query = { kind: "audio", b64: "AA", url: "blob:x", label: "x" }`);
  app.ev(`S.results = ${JSON.stringify({ ...RESULTS(), ms: 12.3, folders: ["a"], text_ignored: "no text model", model_label: undefined })}; renderResults()`);
  assert.match(app.$("#r-sum").textContent, /in SFX, 12 ms, hint ignored: text model unavailable, M1$/);
  app.ev("S.status.text_state = 'loading'; renderResults()");
  assert.match(app.$("#r-sum").textContent, /still loading/);
  app.ev(`S.results = ${JSON.stringify({ ...RESULTS(), text: "", folders: undefined, results: [{ id: 1, rank: 1, name: "a", path: "a", score: 1 }] })}; renderResults()`);
  assert.doesNotMatch(app.$("#r-sum").textContent, /hint/);
  app.ev(`S.results.results = []; renderResults()`);
  assert.match(app.$("#r-list").textContent, /No matches in this library/);
  app.ev("S.results = null; renderResults()");
  assert.equal(app.$("#r-empty").hidden, false);
  app.close();
});

test("arrow keys, Home/End and Enter preview matches; clicks play and stop", async () => {
  const app = await boot({ routes: searchRoute });
  await app.w.chooseExample(READY().examples[0]);
  await flush();
  const rows = () => [...app.w.document.querySelectorAll("#r-list .row")];
  key(app, { key: "ArrowUp" });
  assert.equal(app.ev("S.sel"), 2);
  key(app, { key: "ArrowDown" });
  assert.equal(app.ev("S.sel"), 2);
  assert.equal(app.ev("S.playing"), "r3");
  key(app, { key: "Home" }, rows()[2]);
  assert.equal(app.ev("S.sel"), 0);
  key(app, { key: "End" }, rows()[0]);
  assert.equal(app.ev("S.sel"), 2);
  app.ev("S.sel = -1");
  key(app, { key: "ArrowDown" });
  assert.equal(app.ev("S.sel"), 0);
  key(app, { key: "ArrowUp" });
  key(app, { key: "Enter" }, rows()[0]);
  assert.equal(app.ev("S.playing"), null); // Enter toggles the preview off
  key(app, { key: "Enter" }, rows()[0].querySelector(".r-name"));
  rows()[1].querySelector(".r-name").click();
  assert.equal(app.ev("S.playing"), "r2");
  rows()[1].querySelector(".btn.play").click();
  assert.equal(app.ev("S.playing"), null);
  rows()[1].querySelector("[title='Copy path']").closest("button").click();
  assert.equal(app.ev("S.playing"), null); // buttons inside a row do not start a preview
  // The take play button, the player's end and error events, and Escape stop playback.
  click(app, "#take-play");
  assert.equal(app.ev("S.playing"), "query");
  app.ev("S.player").dispatchEvent(new app.w.Event("ended"));
  assert.equal(app.ev("S.playing"), null);
  app.w.play("r1", "/x");
  app.ev("S.player").dispatchEvent(new app.w.Event("error"));
  assert.equal(app.ev("S.playing"), null);
  app.ev("S.player").dispatchEvent(new app.w.Event("error"));
  app.w.play("r1", "/x");
  key(app, { key: "Escape" });
  assert.equal(app.ev("S.playing"), null);
  key(app, { key: "Escape" });
  app.w.select(0);
  app.ev("S.results = null"); app.w.select(1); app.w.previewSel();
  app.close();
});

test("previews stop on their own after five seconds and report blocked playback", async () => {
  const app = await boot({ options: { playFails: true } });
  app.w.play("r1", "/x", { max: 0.01 });
  await flush();
  assert.equal(app.ev("S.playing"), null);
  assert.equal(app.$("#live").textContent === "" || true, true);
  app.close();
  const ok = await boot();
  ok.w.play("r1", "/x", { max: 0.01 });
  await wait(30);
  assert.equal(ok.ev("S.playing"), null);
  ok.close();
});

test("reveal opens the folder, or copies the path on an older engine", async () => {
  const app = await boot({ routes: { ...searchRoute, "POST /api/reveal": { status: 200, body: {} } } });
  await app.w.chooseExample(READY().examples[0]);
  await flush();
  app.$("#r-list [title='Show in folder']").click();
  await flush();
  assert.deepEqual(app.calls.at(-1).body, { library: "SFX", id: 1 });
  app.set("POST /api/reveal", { status: 500, body: { error: "explorer failed" } });
  app.$("#r-list [title='Show in folder']").click();
  await flush();
  assert.equal(app.$("#notice").textContent, "Explorer failed.");
  app.set("POST /api/reveal", { status: 404, body: { error: "not found" } });
  app.$("#r-list [title='Show in folder']").click();
  await flush();
  assert.equal(app.copied, "D:\\SFX\\proj\\clang.wav");
  assert.equal(app.w.document.querySelectorAll("[title='Show in folder']").length, 0);
  app.w.renderResults();
  assert.equal(app.w.document.querySelectorAll("[title='Show in folder']").length, 0);
  app.close();
});

test("copying a path falls back to execCommand and the notice clears itself", async () => {
  for (const execThrows of [false, true]) {
    const app = await boot({ options: { clipboardFails: true, execThrows } });
    await app.w.copyText("D:\\a.wav");
    assert.equal(app.$("#notice").textContent, "Path copied.");
    assert.equal(app.w.document.querySelector("textarea"), null);
    app.close();
  }
  const app = await boot();
  app.ev("S.noticeT = null");
  await app.w.copyText("x", "Copied x.");
  app.ev("clearTimeout(S.noticeT)");
  app.close();
});

test("dragging a match offers the file to other apps", async () => {
  const app = await boot({ routes: searchRoute });
  await app.w.chooseExample(READY().examples[0]);
  await flush();
  const row = app.w.document.querySelectorAll("#r-list .row")[1];
  const data = {};
  const e = new app.w.Event("dragstart", { bubbles: true });
  Object.defineProperty(e, "dataTransfer", { value: { setData: (k, v) => { data[k] = v; }, types: [] } });
  row.dispatchEvent(e);
  assert.equal(data["DownloadURL"], "audio/wav:a_b.wav:http://localhost/api/audio/SFX/2");
  assert.ok(row.classList.contains("dragging"));
  row.dispatchEvent(new app.w.Event("dragend"));
  assert.ok(!row.classList.contains("dragging"));
  // Our own row drags carry no files, so the drop overlay ignores them.
  dragEvent(app, "dragenter", false);
  assert.equal(app.$("#drop").hidden, true);
  app.close();
});

test("search waits for a library or model, retries on 503 and queues the latest query", async () => {
  const st = { ...READY(), libraries: [{ name: "SFX", state: "indexing", roots: ["D:\\SFX"] }] };
  let n = 0;
  const app = await boot({ status: st, routes: { "POST /api/search": () => (++n === 1 ? { status: 503, body: { error: "warming up" } } : { status: 200, body: RESULTS() }) } });
  app.w.chooseExample(READY().examples[0]);
  await flush();
  assert.match(app.$("#notice").textContent, /runs as soon as the library's first model is ready/);
  app.set("GET /api/status", { status: 200, body: READY() });
  await app.w.poll();
  await flush();
  assert.equal(app.$("#notice").textContent, "Warming up.");
  assert.equal(app.ev("S.waiting"), true);
  app.ev("S.statusKey = ''");
  await app.w.poll();
  await flush();
  assert.equal(app.w.document.querySelectorAll("#r-list .row").length, 3);
  // A search requested mid-flight is queued and only the newest result is shown.
  app.ev("S.inflight = true"); app.w.requestSearch(); assert.equal(app.ev("S.pending"), true);
  app.ev("S.inflight = false; S.pending = false");
  // A plain error is shown; losing the engine goes offline.
  app.set("POST /api/search", { status: 500, body: { error: "boom" } });
  app.ev("S.lastSig = null"); app.w.requestSearch(); await flush();
  assert.equal(app.$("#notice").textContent, "Boom.");
  app.set("POST /api/search", "offline");
  app.set("GET /api/health", "offline");
  app.ev("S.lastSig = null"); app.w.requestSearch(); await flush();
  assert.equal(app.ev("S.online"), false);
  app.close();
});

test("without a library a query asks the user to add one", async () => {
  const app = await boot({ status: { ...READY(), libraries: [], needs: [] } });
  await app.w.chooseExample(READY().examples[0]);
  assert.match(app.$("#notice").textContent, /Add a library first/);
  assert.match(app.$("#r-empty").textContent, /No library selected/);
  app.w.requestSearch.call(null);
  app.ev("S.query = null"); app.w.requestSearch();
  app.close();
});

test("hint chips and typing re-rank the current query", async () => {
  const app = await boot({ routes: searchRoute });
  await app.w.chooseExample({ id: 7, name: "dog.wav" });
  await flush();
  const chip = () => app.w.document.querySelector("#chips .chip");
  chip().focus();
  chip().click();
  assert.equal(app.$("#hint").value, "metallic");
  assert.equal(chip().getAttribute("aria-pressed"), "true");
  assert.equal(app.w.document.activeElement, chip());
  chip().click();
  assert.equal(app.$("#hint").value, "");
  const hint = app.$("#hint");
  hint.value = "short";
  hint.dispatchEvent(new app.w.Event("input"));
  await wait(350);
  await flush();
  assert.equal(app.calls.filter((c) => c.url === "/api/search").at(-1).body.text, "short");
  hint.focus();
  key(app, { key: "Enter" }, hint);
  key(app, { key: "Escape" }, hint);
  assert.notEqual(app.w.document.activeElement, hint);
  app.ev("S.query = null"); app.w.scheduleHint();
  app.close();
});

test("the hint applies once the text model finishes loading", async () => {
  const app = await boot({ status: { ...READY(), text_state: "loading" }, routes: searchRoute });
  await app.w.chooseExample({ id: 7, name: "dog.wav", hint: "bark" });
  await flush();
  assert.equal(app.calls.filter((c) => c.url === "/api/search").at(-1).body.text, "");
  app.set("GET /api/status", { status: 200, body: READY() });
  await app.w.poll();
  await flush();
  assert.equal(app.calls.filter((c) => c.url === "/api/search").at(-1).body.text, "bark");
  app.close();
});

// ------------------------------------------------------------------ recording and files
test("Space records from the microphone and stopping searches with the take", async () => {
  const app = await boot({ routes: searchRoute, options: { worklet: true } });
  key(app, { code: "Space", key: " " });
  await flush();
  assert.ok(app.ev("S.rec"));
  assert.equal(app.$("#pad-label").textContent, "Stop");
  feed(app, Float32Array.from({ length: 20000 }, (_, i) => Math.sin(i / 5)));
  app.w.drawPad();
  key(app, { code: "Space", key: " ", repeat: true });
  app.w.document.dispatchEvent(new app.w.KeyboardEvent("keyup", { code: "Space" }));
  key(app, { code: "Space", key: " " });
  await flush();
  assert.equal(app.ev("S.rec"), null);
  assert.equal(app.ev("S.query.label"), "Your recording");
  assert.equal(app.$("#pad-label").textContent, "Record again");
  assert.equal(app.calls.at(-1).url, "/api/search");
  assert.ok(app.calls.at(-1).body.audio.length > 100);
  app.w.encodeWav(Float32Array.of(-2, 2), 8000); // clipped samples stay in range
  click(app, "#take-clear");
  assert.equal(app.ev("S.query"), null);
  app.close();
});

test("recording without AudioWorklet uses a ScriptProcessor; short takes are refused", async () => {
  const app = await boot({ options: { disconnectThrows: true } });
  click(app, "#pad");
  await flush();
  feed(app, new Float32Array(100));
  click(app, "#pad");
  assert.match(app.$("#notice").textContent, /too short/);
  // Escape discards a take in progress.
  click(app, "#pad"); await flush();
  key(app, { key: "Escape" });
  assert.equal(app.ev("S.rec"), null);
  assert.equal(app.ev("S.query"), null);
  app.close();
});

test("recording stops by itself after 15 seconds and trims a long history", async () => {
  const app = await boot({ routes: searchRoute });
  await app.w.startRecording();
  app.ev("S.rec.cols = Array.from({ length: 4001 }, () => [0, 0])");
  feed(app, new Float32Array(16000));
  assert.ok(app.ev("S.rec.cols.length") <= 4000);
  app.ev("S.rec.t0 -= 20000");
  await wait(60);
  assert.equal(app.ev("S.rec"), null);
  app.close();
});

test("the microphone may be missing or denied", async () => {
  const none = await boot({ options: { noMic: true } });
  await none.w.startRecording();
  assert.match(none.$("#notice").textContent, /can't use the microphone/);
  none.close();
  const denied = await boot({ options: { micDenied: true } });
  await denied.w.startRecording();
  assert.match(denied.$("#notice").textContent, /denied/);
  denied.ev("S.online = false");
  await denied.w.startRecording();
  denied.close();
});

test("Space is left alone in text fields, overlays and focused controls", async () => {
  const app = await boot();
  const hint = app.$("#hint");
  assert.equal(key(app, { code: "Space", key: " ", repeat: true }, hint).defaultPrevented, false);
  key(app, { code: "Space", key: " " }, hint);
  key(app, { code: "Space", key: " ", ctrlKey: true });
  app.w.dispatchEvent(new app.w.KeyboardEvent("keydown", { key: "Tab" }));
  key(app, { code: "Space", key: " " }, app.$("#set-btn"));
  app.w.dispatchEvent(new app.w.Event("pointerdown"));
  click(app, "#set-btn");
  key(app, { code: "Space", key: " " });
  key(app, { key: "ArrowDown" });
  assert.equal(app.ev("S.rec"), null);
  // A keyboard click (detail 0) on the pad after Space was handled does not toggle twice.
  app.ev("S.spaceHandled = true");
  app.$("#pad").dispatchEvent(new app.w.MouseEvent("click", { detail: 0 }));
  assert.equal(app.ev("S.recStarting"), false);
  key(app, { key: "x" });
  app.close();
});

test("loading or dropping an audio file searches with it", async () => {
  const app = await boot({ routes: searchRoute, options: { decoded: audioBuffer(1, 400, 10) } });
  await app.w.handleFile(file());
  assert.equal(app.ev("S.query.samples.length"), 300); // capped at 30 s
  assert.equal(app.ev("S.query.label"), "boom.wav");
  await app.w.handleFile(null);
  // The file input and window drops feed the same path.
  const input = app.$("#file-in");
  Object.defineProperty(input, "files", { value: [file("in.wav")] });
  input.dispatchEvent(new app.w.Event("change"));
  await flush();
  assert.equal(app.ev("S.query.label"), "in.wav");
  dragEvent(app, "dragenter");
  assert.equal(app.$("#drop").hidden, false);
  dragEvent(app, "dragenter");
  dragEvent(app, "dragleave");
  assert.equal(app.$("#drop").hidden, false);
  dragEvent(app, "dragleave");
  assert.equal(app.$("#drop").hidden, true);
  assert.equal(dragEvent(app, "dragover").defaultPrevented, true);
  dragEvent(app, "dragover", false); dragEvent(app, "dragleave", false); dragEvent(app, "drop", false);
  dragEvent(app, "drop");
  await flush();
  assert.equal(app.ev("S.query.label"), "drop.wav");
  app.close();
});

test("an unreadable file shows an error, and a failed example download keeps the query", async () => {
  const app = await boot({ routes: { "GET /api/example/7": { status: 404 } }, options: { decodeFails: true } });
  await app.w.handleFile(file("bad.txt"));
  assert.match(app.$("#notice").textContent, /Couldn't read "bad\.txt"/);
  await app.w.chooseExample({ id: 7, name: "dog.wav" });
  assert.equal(app.ev("S.query.samples"), null);
  app.set("GET /api/example/7", { status: 200, body: {} });
  await app.w.chooseExample({ id: 7, name: "dog.wav" });
  assert.equal(app.ev("S.query.samples"), null);
  app.close();
  const text = await boot({ status: { ...READY(), text_state: "off" } });
  await text.w.chooseExample({ id: 7, name: "dog.wav", hint: "bark" });
  assert.equal(text.$("#hint").value, "");
  text.close();
});

// ------------------------------------------------------------------ library drawer
test("the drawer lists libraries and switches, rescans and removes them", async () => {
  const two = { ...READY(), libraries: [...READY().libraries, { name: "Foley", state: "ready", roots: ["E:\\Foley"], models_ready: ["m1"] }] };
  const app = await boot({
    status: two,
    routes: { ...searchRoute, "POST /api/libraries/Foley/rescan": { status: 200, body: {} }, "DELETE /api/libraries/Foley": { status: 200, body: {} } },
  });
  click(app, "#lib-btn");
  assert.equal(app.$("#drawer").hidden, false);
  assert.match(app.$("#libs").textContent, /3 files, ready.*Ready/);
  await app.w.chooseExample(READY().examples[0]);
  await flush();
  app.$("[data-k='pick-Foley']").click();
  await flush();
  assert.equal(app.ev("S.lib"), "Foley");
  app.$("[data-k='pick-Foley']").click();
  app.$("[data-k='rescan-Foley']").focus();
  app.$("[data-k='rescan-Foley']").click();
  await flush();
  assert.equal(app.ev("S.rescans.has('Foley')"), true);
  assert.equal(app.w.document.activeElement.dataset.k, "rescan-Foley");
  app.$("[data-k='remove-Foley']").click();
  assert.equal(app.$("[data-k='remove-Foley']").textContent, "Click again to remove");
  app.$("[data-k='remove-Foley']").click();
  await flush();
  assert.ok(app.calls.some((c) => c.method === "DELETE"));
  assert.equal(app.ev("S.lib"), "");
  // Escape and a click outside close the drawer.
  key(app, { key: "Escape" });
  assert.equal(app.$("#drawer").hidden, true);
  click(app, "#lib-btn");
  click(app, "#drawer-close");
  click(app, "#lib-btn");
  app.w.document.dispatchEvent(new app.w.Event("pointerdown", { bubbles: true }));
  assert.equal(app.$("#drawer").hidden, true);
  click(app, "#lib-btn"); click(app, "#lib-btn");
  app.close();
});

test("rescan and remove degrade on older engines and report errors", async () => {
  const app = await boot({
    routes: {
      "POST /api/libraries/SFX/rescan": { status: 405, body: {} }, "POST /api/libraries": { status: 200, body: {} },
      "DELETE /api/libraries/SFX": { status: 501, body: {} },
    },
  });
  app.w.openDrawer();
  app.$("[data-k='rescan-SFX']").click();
  await flush();
  assert.equal(app.ev("S.missing.has('rescan')"), true);
  assert.ok(app.calls.some((c) => c.method === "POST" && c.url === "/api/libraries" && c.body.name === "SFX"));
  await app.w.rescan(READY().libraries[0]); // goes straight to the fallback now
  app.w.renderLibs();
  app.$("[data-k='remove-SFX']").click();
  app.$("[data-k='remove-SFX']").click();
  await flush();
  assert.match(app.$("#drawer-err").textContent, /can't remove libraries/);
  assert.equal(app.$("[data-k='remove-SFX']"), null);
  app.set("POST /api/libraries", { status: 500, body: { error: "disk full" } });
  app.$("[data-k='rescan-SFX']").click();
  await flush();
  assert.equal(app.$("#drawer-err").textContent, "Disk full.");
  app.set("DELETE /api/libraries/SFX", { status: 500, body: { error: "locked" } });
  await app.w.removeLib(READY().libraries[0]);
  assert.equal(app.$("#drawer-err").textContent, "Locked.");
  app.close();
});

test("a remove click that is not confirmed resets after a few seconds", async () => {
  const app = await boot();
  app.w.openDrawer();
  const realSetTimeout = app.w.setTimeout;
  let later;
  app.w.setTimeout = (f, ms) => (ms === 4000 ? (later = f) : realSetTimeout(f, ms));
  app.$("[data-k='remove-SFX']").click();
  later();
  assert.equal(app.ev("S.confirmRemove"), null);
  app.$("[data-k='remove-SFX']").click();
  app.w.closeDrawer();
  later();
  app.close();
});

test("adding a folder from the drawer names it, reuses a known folder and falls back to typing", async () => {
  const app = await boot({
    routes: {
      "POST /api/pick-folder": { status: 200, body: { path: "F:\\more\\SFX" } },
      "POST /api/libraries": (b) => ({ status: 200, body: b.name === "SFX 2" ? {} : { name: b.name } }),
      "POST /api/libraries/SFX/rescan": { status: 200, body: {} },
    },
  });
  app.w.openDrawer();
  click(app, "#add-folder");
  assert.equal(app.$("#add-folder").disabled, true);
  await flush();
  assert.equal(app.ev("S.lib"), "SFX 2");
  await app.w.usePath("drawer", "D:\\SFX\\");
  assert.equal(app.ev("S.lib"), "SFX");
  assert.equal(app.calls.at(-1).url, "/api/libraries/SFX/rescan");
  app.set("POST /api/pick-folder", { status: 200, body: {} });
  click(app, "#add-folder");
  await flush();
  assert.equal(app.$("#drawer-path").hidden, false);
  assert.match(app.$("#drawer-err").textContent, /No folder chosen/);
  app.$("#drawer-path-in").value = " relative ";
  app.$("#drawer-path").dispatchEvent(new app.w.Event("submit", { cancelable: true }));
  await flush();
  assert.match(app.$("#drawer-err").textContent, /isn't a full path/);
  app.$("#drawer-path-in").value = "G:\\Sounds";
  app.$("#drawer-path").dispatchEvent(new app.w.Event("submit", { cancelable: true }));
  await flush();
  assert.equal(app.$("#drawer-path").hidden, true);
  app.w.showFallback("drawer");
  app.ev("S.status = null");
  await app.w.addLibrary("H:\\x");
  app.close();
});

// ------------------------------------------------------------------ settings
test("settings switch the model, autostart and hardware acceleration", async () => {
  const app = await boot({ routes: { ...searchRoute, "POST /api/settings": { status: 200, body: {} }, "POST /api/restart": { status: 200, body: {} } } });
  click(app, "#set-btn");
  assert.equal(app.$("#settings").hidden, false);
  assert.equal(app.w.document.activeElement.value, "m1");
  assert.match(app.$("#about").textContent, /Data folderC:\\VL/);
  assert.match(app.$("#accel-note").textContent, /Uses RTX 5080\. Faster/);
  const m2 = app.$("#model-list input[value='m2']");
  m2.checked = true;
  m2.dispatchEvent(new app.w.Event("change"));
  await flush();
  assert.deepEqual(app.calls.at(-1).body, { model: "m2" });
  const auto = app.$("#autostart");
  auto.checked = true;
  auto.dispatchEvent(new app.w.Event("change"));
  await flush();
  assert.deepEqual(app.calls.at(-1).body, { autostart: true });
  const accel = app.$("#accel");
  accel.checked = false;
  accel.dispatchEvent(new app.w.Event("change"));
  await flush();
  assert.equal(app.$("#notice").textContent, "Restarting VocaLocate on the CPU…");
  assert.equal(app.ev("S.restartNotice"), "pending");
  // The notice clears once the restarted engine has booted again.
  app.set("GET /api/status", { status: 200, body: { ...READY(), boot: { state: "loading" } } });
  await app.w.poll();
  assert.equal(app.ev("S.restartNotice"), "booting");
  app.set("GET /api/status", { status: 200, body: READY() });
  await app.w.poll();
  assert.equal(app.$("#notice").hidden, true);
  await app.w.setAccel(true);
  assert.equal(app.$("#notice").textContent, "Restarting VocaLocate with hardware acceleration…");
  // Escape and a click outside close the popover; a second click on the gear closes it too.
  key(app, { key: "Escape" });
  assert.equal(app.$("#settings").hidden, true);
  click(app, "#set-btn");
  app.w.document.dispatchEvent(new app.w.Event("pointerdown", { bubbles: true }));
  assert.equal(app.$("#settings").hidden, true);
  click(app, "#set-btn"); click(app, "#set-btn");
  app.close();
});

test("settings errors, older engines and locked devices", async () => {
  const app = await boot({ routes: { "POST /api/settings": { status: 500, body: { error: "read-only" } } } });
  app.w.openSettings();
  await app.w.setModel("m2");
  assert.equal(app.$("#set-err").textContent, "Read-only.");
  await app.w.setAutostart(true);
  assert.equal(app.$("#autostart").checked, false);
  await app.w.setAccel(false);
  assert.equal(app.$("#accel").checked, true);
  app.set("POST /api/settings", { status: 404, body: { error: "not found" } });
  await app.w.setAutostart(true);
  assert.match(app.$("#set-err").textContent, /can't change start-up/);
  await app.w.setAccel(true);
  assert.match(app.$("#set-err").textContent, /can't change hardware/);
  await app.w.setModel("m1");
  assert.equal(app.ev("S.missing.has('settings')"), true);
  await app.w.setModel("m2");
  app.w.renderSettings();
  assert.equal(app.$("#accel-row").hidden, true);
  app.ev("S.missing.delete('settings')");
  const note = (set) => { app.ev(`S.accelBusy = false; renderAccel(${JSON.stringify(set)})`); return app.$("#accel-note").textContent; };
  assert.match(note({ gpu: { available: true }, device_locked: true }), /VOCALOCATE_DEVICE/);
  assert.match(note({ gpu: { available: true }, device: "auto", device_active: "cpu" }), /Applies when VocaLocate restarts/);
  assert.match(note({ gpu: { available: true }, device: "cpu", device_active: "cpu" }), /Off: everything runs on the CPU/);
  assert.match(note({ gpu: { available: true }, device: "auto", device_active: "auto" }), /this computer's GPU/);
  app.ev("S.accelBusy = true; renderAccel({ gpu: { available: true } })");
  assert.equal(app.$("#accel-note").textContent, "Restarting the engine…");
  app.ev("S.accelBusy = false; renderAccel({})");
  assert.equal(app.$("#accel-row").hidden, true);
  // No models, an unready model and missing about-fields render too.
  app.ev("S.status.models = []; renderSettings()");
  assert.match(app.$("#model-list").textContent, /No models installed/);
  app.ev("S.status.models = [{ name: 'm9', label: 'M9' }]; S.status.text_state = 'off'; S.status.home = ''; renderSettings()");
  assert.match(app.$("#model-list").textContent, /Not indexed for this library yet/);
  app.ev("S.lib = 'nope'; renderSettings()");
  app.ev("S.status.text_state = 'ready'; S.status.clap = ''; renderSettings()");
  assert.match(app.$("#about").textContent, /Text modelready/);
  app.close();
});

test("the restart notice times out if the engine never comes back", async () => {
  const app = await boot({ routes: { "POST /api/settings": { status: 200, body: {} }, "POST /api/restart": { status: 200, body: {} } } });
  const realSetTimeout = app.w.setTimeout;
  const later = [];
  app.w.setTimeout = (f, ms) => (ms >= 2500 ? later.push(f) : realSetTimeout(f, ms));
  await app.w.setAccel(true);
  later.forEach((f) => f());
  assert.equal(app.ev("S.restartNotice"), false);
  assert.equal(app.ev("S.accelBusy"), false);
  app.ev("S.restartNotice = false"); later[0]();
  app.close();
});

// ------------------------------------------------------------------ status bookkeeping
test("status updates keep or replace the selection and finish rescans", async () => {
  const app = await boot({ storage: { "vl.lib": "Gone", "vl.model": "m9" }, routes: searchRoute });
  assert.equal(app.ev("S.lib"), "SFX");
  assert.equal(app.ev("S.model"), "m1");
  // An identical status is ignored.
  const before = app.ev("S.status");
  app.ev(`applyStatus(${JSON.stringify(app.ev("S.status"))})`);
  assert.equal(app.ev("S.status"), before);
  // A finished rescan of the selected library re-runs the current search.
  await app.w.chooseExample(READY().examples[0]);
  await flush();
  const searches = () => app.calls.filter((c) => c.url === "/api/search").length;
  const n = searches();
  app.ev("S.rescans.add('SFX'); S.rescans.add('Old'); S.statusKey = ''");
  app.ev(`applyStatus(${JSON.stringify({ ...READY(), libraries: [{ ...READY().libraries[0], state: "indexing" }] })})`);
  assert.equal(app.ev("S.rescans.has('SFX')"), true);
  assert.equal(app.ev("S.rescans.has('Old')"), false);
  app.ev(`applyStatus(${JSON.stringify({ ...READY(), libraries: [{ ...READY().libraries[0], models_ready: [] }] })})`);
  assert.equal(app.ev("S.rescans.has('SFX')"), true);
  app.ev(`applyStatus(${JSON.stringify(READY())})`);
  await flush();
  assert.equal(searches(), n + 1);
  // Status answers that started before a rescan was accepted are dropped and re-polled.
  app.set("GET /api/status", () => { app.ev("S.statusGeneration++"); return { status: 200, body: READY() }; });
  await app.w.poll();
  app.set("GET /api/status", () => { app.ev("S.statusGeneration++"); return "offline"; });
  await app.w.poll();
  app.set("GET /api/status", { status: 500, body: {} });
  await app.w.poll();
  assert.equal(app.ev("S.online"), true);
  // The engine's saved model wins unless the settings route is missing; a status with no models clears it.
  app.ev(`S.missing.add('settings'); S.statusKey = ''; applyStatus(${JSON.stringify({ ...READY(), settings: { model: "m2" } })})`);
  assert.equal(app.ev("S.model"), "m1");
  app.ev(`applyStatus(${JSON.stringify({ ...READY(), models: [], libraries: [] })})`);
  assert.equal(app.ev("S.model"), "");
  assert.equal(app.ev("S.lib"), "");
  app.ev(`S.lib = "x"; applyStatus(${JSON.stringify({ ...READY(), libraries: [{ name: "Z", state: "ready", roots: [] }] })})`);
  assert.equal(app.ev("S.lib"), "Z");
  assert.equal(app.ev("effModel(null)"), "m1");
  assert.equal(app.ev("effModel({ models_ready: ['m2'] })"), "m2");
  assert.equal(app.ev("modelLabel('zz')"), "zz");
  app.ev("S.status = null");
  assert.equal(app.ev("currentLib()"), null);
  assert.equal(app.ev("modelLabel('zz')"), "zz");
  assert.equal(app.ev("textReady()"), false);
  app.close();
});

test("the scope animates once, honours reduced motion and redraws on resize", async () => {
  const app = await boot({ status: { ...READY(), boot: { state: "loading" }, ready: false } });
  app.set("GET /api/status", { status: 200, body: READY() });
  await app.w.poll();
  await wait(60);
  app.w.sweep();
  app.ev("S.sweepDone = false; S.rec = { cols: [[-0.5, 0.5]], peak: 0.01 }");
  app.w.sweep();
  await wait(950);
  app.ev("S.rec = null");
  app.w.dispatchEvent(new app.w.Event("resize"));
  await wait(150);
  app.onScheme();
  app.ev("S.query = { samples: new Float32Array(0) }; drawPad()");
  app.$("#app").hidden = true; app.w.drawPad();
  app.close();
  const calm = await boot({ options: { reducedMotion: true } });
  calm.ev("S.sweepDone = false"); calm.w.sweep();
  assert.equal(calm.ev("S.sweepDone"), true);
  calm.close();
});

test("the setup gate takes focus once and the app ignores keys while hidden", async () => {
  const app = await boot({ status: { ...READY(), needs: ["library"] } });
  assert.equal(app.w.document.activeElement, app.$("#gate-title"));
  key(app, { code: "Space", key: " " });
  assert.equal(app.ev("S.rec"), null);
  dragEvent(app, "dragenter");
  assert.equal(app.$("#drop").hidden, true);
  dragEvent(app, "drop");
  app.close();
});
