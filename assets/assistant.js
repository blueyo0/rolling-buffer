(() => {
  "use strict";
  const root = document.querySelector("[data-rb-endpoint]");
  if (!root) return;
  let endpoint;
  try { endpoint = new URL(root.dataset.rbEndpoint); } catch { return; }
  if (endpoint.protocol !== "https:" || endpoint.username || endpoint.password || endpoint.search || endpoint.hash || endpoint.pathname !== "/ask") return;
  const open = root.querySelector(".rb-assistant-launch");
  const panel = root.querySelector("dialog");
  const form = root.querySelector("form");
  const close = root.querySelector(".rb-assistant-close");
  const code = root.querySelector("#rb-assistant-code");
  const question = root.querySelector("#rb-assistant-question");
  const context = root.querySelector(".rb-assistant-context");
  const checkbox = context.querySelector("input");
  const status = root.querySelector(".rb-assistant-status");
  const answer = root.querySelector(".rb-assistant-answer");
  const send = root.querySelector(".rb-assistant-send");
  const article = document.querySelector("article .post-body");
  if (typeof panel.show !== "function") return;
  context.hidden = !article;
  open.hidden = false;
  open.addEventListener("click", () => {
    if (panel.open) { panel.close(); return; }
    panel.show();
    open.setAttribute("aria-expanded", "true");
    (code.value ? question : code).focus();
  });
  close.addEventListener("click", () => panel.close());
  panel.addEventListener("keydown", (event) => {
    if (event.key === "Escape") { event.preventDefault(); panel.close(); }
  });
  panel.addEventListener("close", () => {
    open.setAttribute("aria-expanded", "false");
    open.focus();
  });
  const errors = {
    401: "访问码不正确。",
    429: "今天的试用额度已用完。",
    503: "问答服务暂不可用。",
  };
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const prompt = question.value.trim();
    if (send.disabled || !prompt) return;
    const pass = code.value.trim();
    if (!/^[A-Za-z0-9_-]{32,128}$/.test(pass)) {
      status.textContent = "请输入完整的试用访问码，不是 DeepSeek API 密钥。";
      return;
    }
    send.disabled = true;
    status.textContent = "正在等 DeepSeek 回复…";
    answer.hidden = true;
    answer.textContent = "";
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), 32_000);
    try {
      const excerpt = article && checkbox.checked
        ? `${document.querySelector("article h1")?.textContent || ""}\n${article.innerText}`.trim().slice(0, 4500)
        : "";
      const response = await fetch(endpoint, {
        method: "POST", mode: "cors", credentials: "omit", redirect: "error", signal: controller.signal,
        headers: { "Content-Type": "application/json", "X-Assistant-Code": pass },
        body: JSON.stringify({ question: prompt, context: excerpt }),
      });
      if (!response.ok) {
        status.textContent = errors[response.status] || "这一条没有答出来，请稍后再试。";
        return;
      }
      const result = await response.json();
      if (typeof result.answer !== "string" || !result.answer) throw new Error("invalid answer");
      answer.textContent = result.answer;
      answer.hidden = false;
      status.textContent = result.truncated === true ? "回复已截断，可以把问题拆小再问。" : "";
    } catch {
      status.textContent = "暂时连不上问答服务，请稍后再试。";
    } finally {
      window.clearTimeout(timeout);
      send.disabled = false;
    }
  });
})();
