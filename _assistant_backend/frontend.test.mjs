import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";
import { test } from "node:test";
const js = fs.readFileSync(new URL("../assets/assistant.js", import.meta.url), "utf8");
function fixture(endpoint = "https://mock.example/ask") {
  class Control {
    events = {}; dataset = {}; hidden = true; attributes = {}; value = ""; checked = false;
    textContent = ""; disabled = false; open = false;
    addEventListener(ev, fn) { this.events[ev] = fn; }
    setAttribute(key, value) { this.attributes[key] = value; }
    focus() { this.focused = true; }
    emit(event, more = {}) { return this.events[event]?.({ preventDefault() {}, ...more }); }
    show() { this.open = true; }
    close() { this.open = false; this.emit("close"); }
  }
  const root = new Control(), checkbox = new Control();
  const selectors = ["dialog", "form", ".rb-assistant-launch", ".rb-assistant-close", "#rb-assistant-code",
    "#rb-assistant-question", ".rb-assistant-context", ".rb-assistant-status", ".rb-assistant-answer", ".rb-assistant-send"];
  const map = Object.fromEntries(selectors.map((key) => [key, new Control()]));
  map[".rb-assistant-context"].querySelector = () => checkbox;
  root.querySelector = (key) => map[key];
  root.dataset.rbEndpoint = endpoint;
  const document = { querySelector: (key) => ({
    "[data-rb-endpoint]": root, "article .post-body": { innerText: "公开文章" },
    "article h1": { textContent: "GPU 架构" },
  })[key] };
  const requests = [];
  const env = { document, window: { clearTimeout, setTimeout }, URL, AbortController,
    fetch: async (destination, options) => {
      requests.push({ destination: String(destination), options });
      return Response.json({ answer: "<b>纯文本，不执行 HTML</b>" });
    },
  };
  vm.runInNewContext(js, env);
  map["#rb-assistant-code"].value = "demo_test_value_0123456789abcdefghijk";
  map["#rb-assistant-question"].value = "什么是 L1？";
  return { map, requests, checkbox };
}
test("widget opens and closes with focus; context requires opt-in; answer stays text", async () => {
  const { map, requests, checkbox } = fixture();
  const launch = map[".rb-assistant-launch"], dialog = map.dialog;
  assert.equal(launch.hidden, false);
  launch.emit("click");
  assert.equal(dialog.open, true);
  assert.equal(launch.attributes["aria-expanded"], "true");
  await map.form.emit("submit");
  assert.equal(requests[0].destination, "https://mock.example/ask");
  assert.equal(JSON.parse(requests[0].options.body).context, "");
  checkbox.checked = true;
  await map.form.emit("submit");
  assert.equal(JSON.parse(requests[1].options.body).context, "GPU 架构\n公开文章");
  assert.equal(map[".rb-assistant-answer"].textContent, "<b>纯文本，不执行 HTML</b>");
  dialog.emit("keydown", { key: "Escape" });
  assert.equal(dialog.open, false);
  assert.equal(launch.attributes["aria-expanded"], "false");
  assert.ok(launch.focused);
});
test("unsafe or misconfigured endpoint leaves entry hidden", () => {
  for (const endpoint of ["http://mock.example/ask", "https://mock.example/not-ask", "https://name@mock.example/ask"]) {
    const { map, requests } = fixture(endpoint);
    assert.equal(map[".rb-assistant-launch"].hidden, true);
    assert.equal(requests.length, 0);
  }
});
