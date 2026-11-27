// Exercise the actual UI rescan/status/search coordination without a browser or server.
const assert = require("assert");
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const source = fs.readFileSync(path.join(__dirname, "../src/vocalocate_demo/ui/app.js"), "utf8");
const context = vm.createContext({ console, setTimeout, clearTimeout,
  document: { querySelector: () => ({ hidden: false }) } });
vm.runInContext(source.replace(/\nboot\(\);\s*$/, "\n"), context);
vm.runInContext(`
  render = () => {};
  announce = () => {};
  showNotice = () => {};
  pollSoon = () => {};
  api = async () => ({});
  globalThis.calls = 0;
  globalThis.realDoSearch = doSearch;
  doSearch = () => { calls++; };
  globalThis.state = S;
  globalThis.rescanLibrary = rescan;
  globalThis.removeLibrary = removeLib;
  globalThis.receiveStatus = applyStatus;
  globalThis.pollNow = poll;
  globalThis.sig = searchSig;
`, context);

const s = context.state;
const library = (name, state = "ready") => ({ name, state, roots: ["/sfx"], models_ready: ["shared"] });
const status = (libs) => ({ libraries: libs, models: [{ name: "shared" }], settings: { model: "shared" },
  text_state: "ready", boot: { state: "ready" } });

async function main() {
  s.lib = "sounds";
  s.model = "shared";
  s.hint = "metallic";
  s.query = { kind: "audio", b64: "audio", url: "blob:query" };
  s.status = status([library("sounds")]);
  s.results = { results: [] };
  s.lastSig = context.sig();
  s.statusKey = JSON.stringify(s.status);

  // A fast rescan may finish before any busy state is polled, even with identical status.
  await context.rescanLibrary(library("sounds"));
  context.receiveStatus(status([library("sounds")]));
  assert.equal(context.calls, 1, "the existing query should run again after a fast rescan");
  context.receiveStatus(status([library("sounds")]));
  assert.equal(context.calls, 1, "the query should not rerun repeatedly on later polls");

  // A slower rescan should not rerun until the replacement index is ready.
  await context.rescanLibrary(library("sounds"));
  context.receiveStatus(status([library("sounds", "scanning")]));
  assert.equal(context.calls, 1);
  context.receiveStatus(status([library("sounds")]));
  assert.equal(context.calls, 2);

  // An in-flight search must queue another search, not lose the reindex refresh.
  s.inflight = true;
  s.pending = false;
  await context.rescanLibrary(library("sounds"));
  context.receiveStatus(status([library("sounds")]));
  assert.equal(s.pending, true);
  assert.equal(context.calls, 2);
  s.inflight = false;
  s.pending = false;

  // No search on failed indexing, an unrelated library, or when there is no query.
  await context.rescanLibrary(library("sounds"));
  context.receiveStatus(status([library("sounds", "error")]));
  assert.equal(context.calls, 2);
  await context.rescanLibrary(library("other"));
  context.receiveStatus(status([library("sounds"), library("other")]));
  assert.equal(context.calls, 2);
  s.query = null;
  await context.rescanLibrary(library("sounds"));
  context.receiveStatus(status([library("sounds")]));
  assert.equal(context.calls, 2);

  // If a model isn't available at completion, wait for it instead of dropping the refresh.
  s.query = { kind: "audio", b64: "audio", url: "blob:query" };
  await context.rescanLibrary(library("sounds"));
  const noModels = library("sounds");
  noModels.models_ready = [];
  context.receiveStatus(status([noModels]));
  assert.equal(context.calls, 2);
  assert.equal(s.rescans.has("sounds"), true);
  context.receiveStatus(status([library("sounds")]));
  assert.equal(context.calls, 3);

  // A removed library must not leave a pending refresh for a later library with the same name.
  await context.rescanLibrary(library("other"));
  await context.removeLibrary(library("other"));
  assert.equal(s.rescans.has("other"), false);

  // A poll started before rescan must not mistake the old ready state for its completion.
  vm.runInContext(`
    globalThis.resolveOldPoll = null;
    api = (path) => path === '/api/status'
      ? new Promise((resolve) => { resolveOldPoll = resolve; })
      : Promise.resolve({});
  `, context);
  const oldPoll = context.pollNow();
  await context.rescanLibrary(library("sounds"));
  context.resolveOldPoll(status([library("sounds")]));
  await oldPoll;
  assert.equal(context.calls, 3);
  assert.equal(s.rescans.has("sounds"), true);
  context.receiveStatus(status([library("sounds")]));
  assert.equal(context.calls, 4);

  // Exercise the real request function and verify that the returned results replace the old ones.
  vm.runInContext(`
    doSearch = realDoSearch;
    renderResults = () => {};
    api = async (path) => path === '/api/search'
      ? { results: [{ name: 'new.wav' }], library: 'sounds' }
      : {};
  `, context);
  s.results = { results: [{ name: "old.wav" }] };
  s.lastSig = context.sig();
  await context.rescanLibrary(library("sounds"));
  context.receiveStatus(status([library("sounds")]));
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(s.results.results[0].name, "new.wav", "the UI should display the refreshed search response");

  // Do not show the in-flight old result while a fresh post-rescan search is queued.
  vm.runInContext(`
    globalThis.searchResolvers = [];
    globalThis.renderedNames = [];
    renderResults = () => { renderedNames.push(S.results.results[0].name); };
    api = (path) => path === '/api/search'
      ? new Promise((resolve) => { searchResolvers.push(resolve); })
      : Promise.resolve({});
  `, context);
  const firstSearch = context.realDoSearch();
  assert.equal(context.searchResolvers.length, 1);
  await context.rescanLibrary(library("sounds"));
  context.receiveStatus(status([library("sounds")]));
  assert.equal(s.pending, true);
  context.searchResolvers[0]({ results: [{ name: "stale.wav" }], library: "sounds" });
  await firstSearch;
  assert.equal(context.renderedNames.length, 0, "the superseded result should not be rendered");
  assert.equal(context.searchResolvers.length, 2);
  context.searchResolvers[1]({ results: [{ name: "fresh.wav" }], library: "sounds" });
  await new Promise((resolve) => setImmediate(resolve));
  assert.deepEqual(Array.from(context.renderedNames), ["fresh.wav"]);
}

main().catch((err) => { console.error(err); process.exitCode = 1; });
