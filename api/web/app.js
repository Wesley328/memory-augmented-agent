const chatForm = document.querySelector("#chat-form");
const queryInput = document.querySelector("#query-input");
const chatLog = document.querySelector("#chat-log");
const sendButton = document.querySelector("#send-button");
const formHint = document.querySelector("#form-hint");
const clearChatButton = document.querySelector("#clear-chat");

const CHAT_EMPTY_STATE_HTML = `
  <section id="chat-empty-state" class="chat-empty-state">
    <p class="chat-empty-label">Ready</p>
    <h3>开始一轮新的对话</h3>
    <p class="chat-empty-copy">
      输入问题后，系统会返回回答，并在后台完成规划、检索与记忆更新。
    </p>
  </section>
`;

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
    ...options,
  });

  const text = await response.text();
  const payload = text ? JSON.parse(text) : null;
  if (!response.ok) {
    const message = payload?.detail || `Request failed with status ${response.status}`;
    throw new Error(message);
  }
  return payload;
}

function appendMessage(role, html, extraClass = "") {
  const emptyState = document.querySelector("#chat-empty-state");
  if (emptyState) {
    emptyState.remove();
  }
  const article = document.createElement("article");
  article.className = `message message-${role} ${extraClass}`.trim();
  article.innerHTML = `
    <div class="message-role">${escapeHtml(role)}</div>
    <div class="message-body">${html}</div>
  `;
  chatLog.appendChild(article);
  chatLog.scrollTop = chatLog.scrollHeight;
  return article;
}

function buildRetrievedMemoriesHtml(items) {
  if (!items?.length) {
    return `<p class="meta-line">本轮没有召回 memory。</p>`;
  }

  return `
    <div class="retrieved-list">
      ${items
        .slice(0, 3)
        .map(
          (item) => `
            <article class="retrieved-item">
              <div class="chip-row">
                <span class="chip">${escapeHtml(item.memory.type)}</span>
                <span class="chip">score ${Number(item.score).toFixed(2)}</span>
                <span class="chip">confidence ${Number(item.confidence).toFixed(2)}</span>
              </div>
              <p>${escapeHtml(item.memory.content)}</p>
            </article>
          `,
        )
        .join("")}
    </div>
  `;
}

function appendAssistantTurn(result) {
  const planner = result.turn_plan || {};
  const selfCheck = result.self_check;
  const lifecycle = result.memory_lifecycle;

  const html = `
    <div>${escapeHtml(result.answer || "")}</div>
    <details class="message-details">
      <summary>查看本轮分析与记忆信息</summary>
      <div class="message-meta">
        <section class="meta-group">
          <h3>Planner</h3>
          <p class="meta-line">route=${escapeHtml(result.query_route || "general")} · action=${escapeHtml(planner.action || "n/a")}</p>
          <p class="meta-line">${escapeHtml(planner.reason || "No planner reason available.")}</p>
        </section>
        <section class="meta-group">
          <h3>Self-check</h3>
          <p class="meta-line">${escapeHtml(selfCheck?.summary || "No self-check result.")}</p>
        </section>
        <section class="meta-group">
          <h3>Memory lifecycle</h3>
          <p class="meta-line">${
            lifecycle
              ? `extracted=${lifecycle.extracted} · added=${lifecycle.added} · updated=${lifecycle.updated} · versioned=${lifecycle.versioned} · removed=${lifecycle.removed}`
              : "This turn did not update memory."
          }</p>
        </section>
        <section class="meta-group">
          <h3>Retrieved memories</h3>
          ${buildRetrievedMemoriesHtml(result.retrieved_memories)}
        </section>
      </div>
    </details>
  `;

  appendMessage("assistant", html);
}

function appendErrorMessage(error) {
  appendMessage(
    "system",
    `<span class="warning-text">${escapeHtml(error.message || "请求失败，请检查后端日志。")}</span>`,
  );
}

async function handleSubmit(event) {
  event.preventDefault();
  const query = queryInput.value.trim();
  if (!query) {
    return;
  }

  appendMessage("user", escapeHtml(query));
  queryInput.value = "";
  sendButton.disabled = true;
  formHint.textContent = "正在请求 /chat ...";
  const loadingMessage = appendMessage("system", "正在思考并更新记忆...", "message-loading");

  try {
    const result = await fetchJson("/chat", {
      method: "POST",
      body: JSON.stringify({ query }),
    });
    loadingMessage.remove();
    appendAssistantTurn(result);
  } catch (error) {
    loadingMessage.remove();
    appendErrorMessage(error);
  } finally {
    sendButton.disabled = false;
    formHint.textContent = "按 Enter 发送，Shift + Enter 换行";
    queryInput.focus();
  }
}

chatForm.addEventListener("submit", handleSubmit);

queryInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    chatForm.requestSubmit();
  }
});

clearChatButton.addEventListener("click", () => {
  chatLog.innerHTML = CHAT_EMPTY_STATE_HTML;
});
