import { test } from "node:test";
import assert from "node:assert/strict";
import { boot } from "./harness.mjs";

test("boots into the search screen when a library is ready", async () => {
  const app = await boot();
  assert.equal(app.$("#app").hidden, false);
  assert.equal(app.$("#lib-name").textContent, "SFX");
  app.close();
});
