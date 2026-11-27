import assert from "node:assert/strict";
import { DatabaseSync } from "node:sqlite";
import { test } from "node:test";
import { DailyQuota, handle } from "./worker.mjs";

const origin = "https://blueyo0.github.io";
const code = "test_only_access_0123456789abcdefghij";
const url = "https://worker.example/ask";

function fixture() {
  const db = new DatabaseSync(":memory:");
  const sql = {
    exec(query, ...args) {
      const statement = db.prepare(query);
      if (query.includes("RETURNING")) return { toArray: () => statement.all(...args) };
      statement.run(...args);
    },
  };
  const q = new DailyQuota({ storage: { sql } }, { DAILY_LIMIT: "2" });
  const env = {
    DEEPSEEK_API_KEY: "fake-test-key",
    ASSISTANT_ACCESS_CODE: code,
    QUOTA: { idFromName: (n) => n, get: () => ({ fetch: (url, init) => q.fetch(new Request(url, init)) }) },
  };
  let calls = 0;
  const upstream = async (endpoint, init) => {
    calls++;
    assert.equal(endpoint, "https://api.deepseek.com/chat/completions");
    assert.equal(init.headers.Authorization, "Bearer fake-test-key");
    assert.equal(init.redirect, "error");
    const body = JSON.parse(init.body);
    assert.equal(body.model, "deepseek-flash");
    assert.equal(body.max_tokens, 700);
    assert.equal(body.thinking.type, "disabled");
    assert.deepEqual(JSON.parse(body.messages[1].content), { question: "什么是 L1？", article_excerpt: "公开文章" });
    assert.ok(!init.body.includes(code));
    return Response.json({ choices: [{ message: { content: "L1 是 SM 本地缓存。" } }] });
  };
  return { env, upstream, count: () => calls };
}

function req({ source = origin, access = code, body = JSON.stringify({ question: "什么是 L1？", context: "公开文章" }), method = "POST" } = {}) {
  return new Request(url, {
    method, headers: { Origin: source, "Content-Type": "application/json", "X-Assistant-Code": access },
    ...(method === "POST" ? { body } : {}),
  });
}

test("only configured origin; preflight never calls the model", async () => {
  const f = fixture();
  let r = await handle(req({ source: "https://other.example" }), f.env, f.upstream);
  assert.equal(r.status, 403);
  assert.equal(r.headers.get("Access-Control-Allow-Origin"), null);
  r = await handle(req({ method: "OPTIONS" }), f.env, f.upstream);
  assert.equal(r.status, 204);
  assert.equal(r.headers.get("Access-Control-Allow-Origin"), origin);
  assert.equal(f.count(), 0);
});

test("wrong code and malformed or oversized input fail before model and quota", async () => {
  const f = fixture();
  const rejected = [
    [req({ access: code.replace("test", "xxxx") }), 401],
    [req({ body: "{oops" }), 400],
    [req({ body: "x".repeat(25_000) }), 413],
    [req({ body: JSON.stringify({ question: " ", context: "" }) }), 400],
  ];
  for (const [request, status] of rejected) assert.equal((await handle(request, f.env, f.upstream)).status, status);
  assert.equal(f.count(), 0);
  assert.equal((await handle(req(), f.env, f.upstream)).status, 200);
});

test("concurrent accepted requests cannot overrun daily quota", async () => {
  const f = fixture();
  const result = await Promise.all(Array.from({ length: 5 }, () => handle(req(), f.env, f.upstream)));
  assert.deepEqual(result.map((r) => r.status).sort(), [200, 200, 429, 429, 429]);
  const r = result.find((r) => r.status === 200);
  assert.equal(r.headers.get("Cache-Control"), "no-store");
  assert.deepEqual(await r.json(), { answer: "L1 是 SM 本地缓存。" });
  assert.equal(f.count(), 2);
});

test("missing secrets and failed quota fail closed; upstream details are not returned", async () => {
  const f = fixture();
  assert.equal((await handle(req(), { ...f.env, DEEPSEEK_API_KEY: "" }, f.upstream)).status, 503);
  const broken = { idFromName: () => "x", get: () => ({ fetch: () => { throw Error("offline"); } }) };
  assert.equal((await handle(req(), { ...f.env, QUOTA: broken }, f.upstream)).status, 503);
  assert.equal(f.count(), 0);
  const failed = await handle(req(), f.env, async () => new Response("sensitive upstream diagnostics", { status: 401 }));
  assert.equal(failed.status, 502);
  assert.deepEqual(await failed.json(), { error: "upstream" });
});

test("a truncated visible answer is marked, and reasoning alone is never returned", async () => {
  const f = fixture();
  const r = await handle(req(), f.env, async () => Response.json({
    choices: [{ finish_reason: "length", message: { content: "可见的部分", reasoning_content: "do not relay" } }],
  }));
  assert.equal(r.status, 200);
  assert.deepEqual(await r.json(), { answer: "可见的部分", truncated: true });
  const empty = await handle(req(), f.env, async () => Response.json({
    choices: [{ finish_reason: "length", message: { content: "  ", reasoning_content: "do not relay" } }],
  }));
  assert.equal(empty.status, 502);
  assert.deepEqual(await empty.json(), { error: "upstream" });
});
