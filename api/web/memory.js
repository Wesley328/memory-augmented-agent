const memorySummary = document.querySelector("#memory-summary");
const memoryList = document.querySelector("#memory-list");
const lineageSummary = document.querySelector("#lineage-summary");
const lineageList = document.querySelector("#lineage-list");
const sourceSummary = document.querySelector("#source-summary");
const sourceList = document.querySelector("#source-list");
const detailSummary = document.querySelector("#detail-summary");
const detailBody = document.querySelector("#detail-body");
const traceSummary = document.querySelector("#trace-summary");
const traceList = document.querySelector("#trace-list");
const traceDetailSummary = document.querySelector("#trace-detail-summary");
const traceDetailBody = document.querySelector("#trace-detail-body");
const statTotal = document.querySelector("#stat-total");
const statActive = document.querySelector("#stat-active");
const statSummary = document.querySelector("#stat-summary");
const statSuperseded = document.querySelector("#stat-superseded");
const refreshMemoriesButton = document.querySelector("#refresh-memories");
const refreshTracesButton = document.querySelector("#refresh-traces");
const clearLineageButton = document.querySelector("#clear-lineage");
const clearSourcesButton = document.querySelector("#clear-sources");
const clearDetailButton = document.querySelector("#clear-detail");
const clearTraceDetailButton = document.querySelector("#clear-trace-detail");
const memoryStatusFilter = document.querySelector("#memory-status-filter");
const memoryTypeFilter = document.querySelector("#memory-type-filter");
const memorySummaryOnlyToggle = document.querySelector("#memory-summary-only");
const resetMemoryFiltersButton = document.querySelector("#reset-memory-filters");
const traceRouteFilter = document.querySelector("#trace-route-filter");
const traceActionFilter = document.querySelector("#trace-action-filter");
const resetTraceFiltersButton = document.querySelector("#reset-trace-filters");

const DETAIL_PRIORITY_KEYS = [
  "memory_id",
  "lineage_id",
  "status",
  "version",
  "topic_key",
  "summary_scope",
  "summary_kind",
  "summary_source_count",
  "derived_from",
  "query_route_at_extraction",
  "source_turn_id",
  "last_updated_at",
  "superseded_at",
  "version_reason",
  "supersedes",
  "superseded_by",
];

const DETAIL_EXCLUDED_KEYS = new Set([
  "source_user_message",
  "source_assistant_message",
]);

const memoryViewState = {
  status: "",
  memoryType: "",
  summaryOnly: false,
  limit: 24,
  selectedLineageId: null,
  selectedMemoryId: null,
  selectedSummaryMemoryId: null,
  selectedSourceMemoryIds: [],
  selectedTraceTurnId: null,
  loadedMemories: [],
  loadedLineageItems: [],
  loadedSourceItems: [],
  loadedTraces: [],
  selectedMemoryRecord: null,
  selectedTraceRecord: null,
  traceRoute: "",
  traceAction: "",
  traceLimit: 12,
};

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

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function formatTime(value) {
  if (!value) {
    return "n/a";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return new Intl.DateTimeFormat("zh-CN", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}

function shortId(value) {
  if (!value) {
    return "n/a";
  }
  return String(value).slice(0, 8);
}

function escapeSelectorValue(value) {
  if (globalThis.CSS && typeof globalThis.CSS.escape === "function") {
    return globalThis.CSS.escape(value);
  }
  return String(value).replaceAll('"', '\\"');
}

function formatMetadataValue(value) {
  if (value === null || value === undefined || value === "") {
    return "n/a";
  }
  if (Array.isArray(value)) {
    if (!value.length) {
      return "[]";
    }
    return value.map((item) => shortId(item)).join(", ");
  }
  if (typeof value === "object") {
    return JSON.stringify(value);
  }
  return String(value);
}

function getMemoryFilterLabels() {
  const labels = [];
  if (memoryViewState.status) {
    labels.push(`status=${memoryViewState.status}`);
  }
  if (memoryViewState.summaryOnly) {
    labels.push("summary-only=true");
  } else if (memoryViewState.memoryType) {
    labels.push(`type=${memoryViewState.memoryType}`);
  }
  if (!labels.length) {
    return "当前展示默认最近 memories。";
  }
  return `当前筛选：${labels.join(" · ")}`;
}

function getMemoryById(memoryId) {
  if (!memoryId) {
    return null;
  }

  const collections = [
    memoryViewState.loadedMemories,
    memoryViewState.loadedLineageItems,
    memoryViewState.loadedSourceItems,
  ];

  for (const items of collections) {
    const found = items.find((item) => item?.metadata?.memory_id === memoryId);
    if (found) {
      return found;
    }
  }
  return null;
}

function getTraceByTurnId(turnId) {
  if (turnId === null || turnId === undefined || turnId === "") {
    return null;
  }

  const normalizedTurnId = Number(turnId);
  return (
    memoryViewState.loadedTraces.find((item) => Number(item.turn_id) === normalizedTurnId) || null
  );
}

function getMemoriesForTurn(turnId) {
  if (!turnId) {
    return [];
  }
  return memoryViewState.loadedMemories.filter(
    (item) => Number(item.metadata?.source_turn_id) === Number(turnId),
  );
}

function buildMetadataRows(metadata) {
  const seenKeys = new Set();
  const orderedKeys = [];

  for (const key of DETAIL_PRIORITY_KEYS) {
    if (key in metadata && !DETAIL_EXCLUDED_KEYS.has(key)) {
      orderedKeys.push(key);
      seenKeys.add(key);
    }
  }

  for (const key of Object.keys(metadata).sort()) {
    if (seenKeys.has(key) || DETAIL_EXCLUDED_KEYS.has(key)) {
      continue;
    }
    orderedKeys.push(key);
  }

  return orderedKeys
    .map(
      (key) => `
        <div class="detail-row">
          <dt>${escapeHtml(key)}</dt>
          <dd>${escapeHtml(formatMetadataValue(metadata[key]))}</dd>
        </div>
      `,
    )
    .join("");
}

function renderDetailPanel() {
  const memory = memoryViewState.selectedMemoryRecord;
  if (!memory) {
    detailSummary.innerHTML = `<p class="muted">点击任意 memory 的 “View detail” 按钮，查看它的 metadata 细节。</p>`;
    detailBody.innerHTML = "";
    return;
  }

  const metadata = memory.metadata || {};
  const isSummary = Boolean(metadata.is_summary) || memory.type === "summary";
  const sourceMessage = metadata.source_user_message || metadata.source_assistant_message;

  detailSummary.innerHTML = `
    <p class="muted">
      当前查看 memory=<strong>${escapeHtml(shortId(metadata.memory_id))}</strong>
      · lineage=<strong>${escapeHtml(shortId(metadata.lineage_id))}</strong>
      · status=<strong>${escapeHtml(metadata.status || "active")}</strong>
    </p>
  `;

  detailBody.innerHTML = `
    <article class="detail-card">
      <div class="chip-row">
        <span class="chip">${escapeHtml(memory.type)}</span>
        <span class="chip">confidence ${Number(memory.confidence).toFixed(2)}</span>
        <span class="chip">importance ${Number(memory.importance).toFixed(2)}</span>
        ${isSummary ? `<span class="chip chip-neutral">summary</span>` : ""}
      </div>
      <p class="detail-content">${escapeHtml(memory.content)}</p>
      <div class="detail-grid">
        <div class="detail-metric">
          <span>timestamp</span>
          <strong>${escapeHtml(formatTime(memory.timestamp))}</strong>
        </div>
        <div class="detail-metric">
          <span>embedding_dim</span>
          <strong>${escapeHtml(memory.embedding_dim ?? "n/a")}</strong>
        </div>
      </div>
      <section class="detail-section">
        <h4>Metadata Fields</h4>
        <dl class="detail-list">
          ${buildMetadataRows(metadata)}
        </dl>
      </section>
      ${
        metadata.source_turn_id
          ? `
      <section class="detail-section">
        <h4>Trace Link</h4>
        <p class="detail-trace">
          source_turn_id=<strong>${escapeHtml(metadata.source_turn_id)}</strong>
        </p>
        <div class="memory-actions">
          <button
            type="button"
            class="inline-button"
            data-action="view-turn-trace"
            data-turn-id="${escapeHtml(metadata.source_turn_id)}"
          >
            View turn trace
          </button>
        </div>
      </section>`
          : ""
      }
      ${
        sourceMessage
          ? `
      <section class="detail-section">
        <h4>Source Trace</h4>
        ${
          metadata.source_user_message
            ? `<p class="detail-trace"><strong>User:</strong> ${escapeHtml(metadata.source_user_message)}</p>`
            : ""
        }
        ${
          metadata.source_assistant_message
            ? `<p class="detail-trace"><strong>Assistant:</strong> ${escapeHtml(metadata.source_assistant_message)}</p>`
            : ""
        }
      </section>`
          : ""
      }
    </article>
  `;
}

function setSelectedMemory(memoryId, record = null) {
  memoryViewState.selectedMemoryId = memoryId || null;
  memoryViewState.selectedMemoryRecord = record || getMemoryById(memoryId);
  applyMemoryHighlights();
  renderDetailPanel();
}

function setSelectedTrace(turnId, record = null) {
  memoryViewState.selectedTraceTurnId = turnId === null || turnId === undefined ? null : Number(turnId);
  memoryViewState.selectedTraceRecord = record || getTraceByTurnId(turnId);
  applyMemoryHighlights();
  renderTraceInspector();
}

function applyMemoryHighlights() {
  const selectedSourceIds = new Set(memoryViewState.selectedSourceMemoryIds);
  const cards = memoryList.querySelectorAll(".memory-item");

  cards.forEach((card) => {
    const memoryId = card.dataset.memoryId || "";
    const lineageId = card.dataset.lineageId || "";
    const sourceTurnId = card.dataset.sourceTurnId || "";

    card.classList.toggle("memory-item-selected", memoryId === memoryViewState.selectedMemoryId);
    card.classList.toggle(
      "memory-item-lineage",
      Boolean(memoryViewState.selectedLineageId) && lineageId === memoryViewState.selectedLineageId,
    );
    card.classList.toggle("memory-item-source", selectedSourceIds.has(memoryId));
    card.classList.toggle("memory-item-summary-focus", memoryId === memoryViewState.selectedSummaryMemoryId);
    card.classList.toggle(
      "memory-item-trace",
      Boolean(memoryViewState.selectedTraceTurnId) &&
        Number(sourceTurnId) === Number(memoryViewState.selectedTraceTurnId),
    );
  });

  const inspectorItems = document.querySelectorAll(".lineage-item");
  inspectorItems.forEach((item) => {
    const memoryId = item.dataset.memoryId || "";
    item.classList.toggle("lineage-item-selected", memoryId === memoryViewState.selectedMemoryId);
  });

  const traceItems = traceList.querySelectorAll(".trace-item");
  traceItems.forEach((item) => {
    const turnId = item.dataset.turnId || "";
    item.classList.toggle(
      "trace-item-selected",
      Boolean(memoryViewState.selectedTraceTurnId) &&
        Number(turnId) === Number(memoryViewState.selectedTraceTurnId),
    );
  });
}

function scrollToMemoryCard(memoryId) {
  if (!memoryId) {
    return;
  }

  const selectorMemoryId = escapeSelectorValue(memoryId);
  const card = memoryList.querySelector(`[data-memory-id="${selectorMemoryId}"]`);
  if (!card) {
    return;
  }

  card.scrollIntoView({ behavior: "smooth", block: "nearest" });
  card.classList.remove("memory-item-bump");
  void card.offsetWidth;
  card.classList.add("memory-item-bump");
}

function renderMemories(payload) {
  const items = payload?.items || [];
  memoryViewState.loadedMemories = items;

  memorySummary.innerHTML = `
    <p class="muted">
      ${escapeHtml(getMemoryFilterLabels())}
      共展示 <strong>${items.length}</strong> 条 memory。这里最适合观察 type、confidence、version、summary、source 和 lineage 演化。
    </p>
  `;

  if (!items.length) {
    memoryList.innerHTML = `<p class="muted">当前筛选条件下没有 memory，试试切换 status/type 或关闭 summary-only。</p>`;
    return;
  }

  memoryList.innerHTML = items
    .map((item) => {
      const metadata = item.metadata || {};
      const status = metadata.status || "active";
      const version = metadata.version ?? 1;
      const topicKey = metadata.topic_key || "n/a";
      const lineageId = metadata.lineage_id || "";
      const memoryId = metadata.memory_id || "";
      const sourceTurnId = metadata.source_turn_id ?? "";
      const summaryScope = metadata.summary_scope;
      const summarySourceIds = Array.isArray(metadata.summary_source_ids)
        ? metadata.summary_source_ids.filter(Boolean)
        : [];
      const isSummary = Boolean(metadata.is_summary) || item.type === "summary";

      return `
        <article
          class="memory-item"
          data-memory-id="${escapeHtml(memoryId)}"
          data-lineage-id="${escapeHtml(lineageId)}"
          data-source-turn-id="${escapeHtml(sourceTurnId)}"
        >
          <div class="chip-row">
            <span class="chip">${escapeHtml(item.type)}</span>
            <span class="chip">confidence ${Number(item.confidence).toFixed(2)}</span>
            <span class="chip">importance ${Number(item.importance).toFixed(2)}</span>
            ${isSummary ? `<span class="chip chip-neutral">summary</span>` : ""}
          </div>
          <p class="memory-content">${escapeHtml(item.content)}</p>
          <p class="memory-meta">
            status=${escapeHtml(status)} · version=v${escapeHtml(version)} · topic=${escapeHtml(topicKey)}
          </p>
          <p class="memory-meta">
            memory_id=${escapeHtml(shortId(memoryId))} · lineage=${escapeHtml(shortId(lineageId))}
            ${summaryScope ? ` · scope=${escapeHtml(summaryScope)}` : ""}
          </p>
          <p class="memory-meta">
            time=${escapeHtml(formatTime(item.timestamp))}
            ${sourceTurnId ? ` · source_turn=${escapeHtml(sourceTurnId)}` : ""}
          </p>
          <div class="memory-actions">
            <button
              type="button"
              class="inline-button"
              data-action="view-detail"
              data-memory-id="${escapeHtml(memoryId)}"
            >
              View detail
            </button>
            <button
              type="button"
              class="inline-button"
              data-action="view-lineage"
              data-memory-id="${escapeHtml(memoryId)}"
              data-lineage-id="${escapeHtml(lineageId)}"
              ${lineageId ? "" : "disabled"}
            >
              View lineage
            </button>
            ${
              summarySourceIds.length
                ? `
            <button
              type="button"
              class="inline-button"
              data-action="view-sources"
              data-memory-id="${escapeHtml(memoryId)}"
              data-summary-scope="${escapeHtml(summaryScope || "")}"
              data-source-ids="${escapeHtml(summarySourceIds.join(","))}"
            >
              View sources
            </button>`
                : ""
            }
          </div>
        </article>
      `;
    })
    .join("");

  applyMemoryHighlights();
}

function buildTraceFilterLabels() {
  const labels = [];
  if (memoryViewState.traceRoute) {
    labels.push(`route=${memoryViewState.traceRoute}`);
  }
  if (memoryViewState.traceAction) {
    labels.push(`action=${memoryViewState.traceAction}`);
  }
  if (!labels.length) {
    return "当前展示最近 turn traces。";
  }
  return `当前筛选：${labels.join(" · ")}`;
}

function renderTraceList(payload) {
  const items = payload?.items || [];
  memoryViewState.loadedTraces = items;

  traceSummary.innerHTML = `
    <p class="muted">
      ${escapeHtml(buildTraceFilterLabels())}
      共展示 <strong>${items.length}</strong> 条 trace。这里最适合观察 route、planner、self-check 与 memory lifecycle 的整轮行为。
    </p>
  `;

  if (!items.length) {
    traceList.innerHTML = `<p class="muted trace-empty">当前筛选条件下没有 trace，先去聊天页发起几轮对话。</p>`;
    return;
  }

  traceList.innerHTML = items
    .map((item) => {
      const turnPlan = item.turn_plan || {};
      const selfCheck = item.self_check || null;
      const lifecycle = item.memory_lifecycle || null;
      const relatedCount = getMemoriesForTurn(item.turn_id).length;

      return `
        <article class="trace-item" data-turn-id="${escapeHtml(item.turn_id)}">
          <div class="chip-row">
            <span class="chip">turn ${escapeHtml(item.turn_id)}</span>
            <span class="chip">${escapeHtml(item.query_route || "general")}</span>
            <span class="chip chip-neutral">${escapeHtml(turnPlan.action || "n/a")}</span>
          </div>
          <p class="trace-query">${escapeHtml(item.query || "")}</p>
          <p class="memory-meta">
            time=${escapeHtml(formatTime(item.created_at))} · retrieved=${escapeHtml(
              item.retrieved_memories?.length ?? 0,
            )} · related_memories=${escapeHtml(relatedCount)}
          </p>
          <p class="memory-meta">
            self_check=${escapeHtml(selfCheck?.summary || "n/a")}
            ${
              lifecycle
                ? ` · lifecycle=${escapeHtml(
                    `+${lifecycle.added} ~${lifecycle.updated} v${lifecycle.versioned} -${lifecycle.removed}`,
                  )}`
                : ""
            }
          </p>
          <div class="memory-actions">
            <button
              type="button"
              class="inline-button"
              data-action="view-trace"
              data-turn-id="${escapeHtml(item.turn_id)}"
            >
              View trace
            </button>
            <button
              type="button"
              class="inline-button"
              data-action="focus-trace-memories"
              data-turn-id="${escapeHtml(item.turn_id)}"
            >
              Focus memories
            </button>
          </div>
        </article>
      `;
    })
    .join("");

  applyMemoryHighlights();
}

function renderLineage(items, lineageId) {
  memoryViewState.selectedLineageId = lineageId;
  memoryViewState.loadedLineageItems = items;

  if (!items?.length) {
    lineageSummary.innerHTML = `<p class="muted">没有查到 lineage=${escapeHtml(shortId(lineageId))} 的版本链。</p>`;
    lineageList.innerHTML = "";
    applyMemoryHighlights();
    return;
  }

  const latest = items[items.length - 1];
  lineageSummary.innerHTML = `
    <p class="muted">
      当前查看 lineage=<strong>${escapeHtml(shortId(lineageId))}</strong>，共 <strong>${items.length}</strong> 个版本。
      最新状态是 <strong>${escapeHtml(latest.metadata?.status || "active")}</strong>。
    </p>
  `;

  lineageList.innerHTML = items
    .map((item) => {
      const metadata = item.metadata || {};
      const version = metadata.version ?? 1;
      const status = metadata.status || "active";
      const memoryId = metadata.memory_id || "";
      const supersedes = metadata.supersedes ? `supersedes=${shortId(metadata.supersedes)}` : "";
      const supersededBy = metadata.superseded_by ? `superseded_by=${shortId(metadata.superseded_by)}` : "";
      const relationBits = [supersedes, supersededBy].filter(Boolean);

      return `
        <article
          class="lineage-item"
          data-memory-id="${escapeHtml(memoryId)}"
          data-lineage-id="${escapeHtml(lineageId)}"
        >
          <div class="chip-row">
            <span class="chip">v${escapeHtml(version)}</span>
            <span class="chip chip-neutral">${escapeHtml(status)}</span>
            <span class="chip">${escapeHtml(item.type)}</span>
          </div>
          <p class="memory-content">${escapeHtml(item.content)}</p>
          <p class="memory-meta">time=${escapeHtml(formatTime(item.timestamp))}</p>
          <p class="memory-meta">
            memory_id=${escapeHtml(shortId(memoryId))}
            ${relationBits.length ? ` · ${escapeHtml(relationBits.join(" · "))}` : ""}
          </p>
          <div class="memory-actions">
            <button
              type="button"
              class="inline-button"
              data-action="view-detail"
              data-memory-id="${escapeHtml(memoryId)}"
            >
              View detail
            </button>
            <button
              type="button"
              class="inline-button"
              data-action="focus-memory"
              data-memory-id="${escapeHtml(memoryId)}"
              data-lineage-id="${escapeHtml(lineageId)}"
            >
              Focus in list
            </button>
          </div>
        </article>
      `;
    })
    .join("");

  applyMemoryHighlights();
}

function renderSourceInspector(summaryMemoryId, summaryScope, sourceIds, items) {
  memoryViewState.selectedSummaryMemoryId = summaryMemoryId;
  memoryViewState.selectedSourceMemoryIds = sourceIds;
  memoryViewState.loadedSourceItems = items;

  if (!items.length) {
    sourceSummary.innerHTML = `
      <p class="muted">
        当前查看 summary=<strong>${escapeHtml(shortId(summaryMemoryId))}</strong>，
        但没有查到可用的 source memories。
      </p>
    `;
    sourceList.innerHTML = "";
    applyMemoryHighlights();
    return;
  }

  sourceSummary.innerHTML = `
    <p class="muted">
      当前查看 summary=<strong>${escapeHtml(shortId(summaryMemoryId))}</strong>
      ${summaryScope ? ` · scope=<strong>${escapeHtml(summaryScope)}</strong>` : ""}
      ，共关联 <strong>${sourceIds.length}</strong> 条 source memory。
    </p>
  `;

  sourceList.innerHTML = items
    .map((item) => {
      const metadata = item.metadata || {};
      const memoryId = metadata.memory_id || "";
      const lineageId = metadata.lineage_id || "";

      return `
        <article
          class="lineage-item"
          data-memory-id="${escapeHtml(memoryId)}"
          data-lineage-id="${escapeHtml(lineageId)}"
        >
          <div class="chip-row">
            <span class="chip">${escapeHtml(item.type)}</span>
            <span class="chip">confidence ${Number(item.confidence).toFixed(2)}</span>
            <span class="chip">importance ${Number(item.importance).toFixed(2)}</span>
          </div>
          <p class="memory-content">${escapeHtml(item.content)}</p>
          <p class="memory-meta">
            memory_id=${escapeHtml(shortId(memoryId))} · lineage=${escapeHtml(shortId(lineageId))}
          </p>
          <p class="memory-meta">time=${escapeHtml(formatTime(item.timestamp))}</p>
          <div class="memory-actions">
            <button
              type="button"
              class="inline-button"
              data-action="view-detail"
              data-memory-id="${escapeHtml(memoryId)}"
            >
              View detail
            </button>
            <button
              type="button"
              class="inline-button"
              data-action="focus-memory"
              data-memory-id="${escapeHtml(memoryId)}"
              data-lineage-id="${escapeHtml(lineageId)}"
            >
              Focus in list
            </button>
          </div>
        </article>
      `;
    })
    .join("");

  applyMemoryHighlights();
}

function clearLineageInspector() {
  memoryViewState.selectedLineageId = null;
  memoryViewState.loadedLineageItems = [];
  if (!memoryViewState.selectedSummaryMemoryId) {
    setSelectedMemory(null, null);
  } else {
    applyMemoryHighlights();
  }
  lineageSummary.innerHTML = `<p class="muted">点击某条 memory 上的 “View lineage” 按钮查看版本链。</p>`;
  lineageList.innerHTML = "";
  applyMemoryHighlights();
}

function clearSourceInspector() {
  memoryViewState.selectedSummaryMemoryId = null;
  memoryViewState.selectedSourceMemoryIds = [];
  memoryViewState.loadedSourceItems = [];
  if (!memoryViewState.selectedLineageId) {
    setSelectedMemory(null, null);
  } else {
    applyMemoryHighlights();
  }
  sourceSummary.innerHTML = `<p class="muted">点击 summary memory 上的 “View sources” 按钮查看它由哪些原子记忆派生而来。</p>`;
  sourceList.innerHTML = "";
  applyMemoryHighlights();
}

function clearDetailInspector() {
  memoryViewState.selectedMemoryId = null;
  memoryViewState.selectedMemoryRecord = null;
  renderDetailPanel();
  applyMemoryHighlights();
}

function clearTraceInspector() {
  memoryViewState.selectedTraceTurnId = null;
  memoryViewState.selectedTraceRecord = null;
  renderTraceInspector();
  applyMemoryHighlights();
}

function buildMemoriesUrl() {
  const params = new URLSearchParams();
  params.set("limit", String(memoryViewState.limit));

  if (memoryViewState.status) {
    params.set("status", memoryViewState.status);
  }
  if (memoryViewState.summaryOnly) {
    params.set("summary_only", "true");
  } else if (memoryViewState.memoryType) {
    params.set("memory_type", memoryViewState.memoryType);
  }

  return `/memories?${params.toString()}`;
}

function buildTracesUrl() {
  const params = new URLSearchParams();
  params.set("limit", String(memoryViewState.traceLimit));
  if (memoryViewState.traceRoute) {
    params.set("query_route", memoryViewState.traceRoute);
  }
  if (memoryViewState.traceAction) {
    params.set("planner_action", memoryViewState.traceAction);
  }
  return `/traces?${params.toString()}`;
}

function renderStats(stats) {
  statTotal.textContent = String(stats?.total ?? "--");
  statActive.textContent = String(stats?.active ?? "--");
  statSummary.textContent = String(stats?.summary ?? "--");
  statSuperseded.textContent = String(stats?.superseded ?? "--");
}

async function loadStats() {
  try {
    const stats = await fetchJson("/memory-stats");
    renderStats(stats);
  } catch (error) {
    statTotal.textContent = "!";
    statActive.textContent = "!";
    statSummary.textContent = "!";
    statSuperseded.textContent = "!";
  }
}

async function loadMemories() {
  try {
    const payload = await fetchJson(buildMemoriesUrl());
    renderMemories(payload);
    if (memoryViewState.selectedMemoryId) {
      setSelectedMemory(memoryViewState.selectedMemoryId);
    }
    if (memoryViewState.selectedTraceRecord) {
      renderTraceInspector();
      applyMemoryHighlights();
    }
  } catch (error) {
    memorySummary.innerHTML = `<p class="warning-text">${escapeHtml(error.message)}</p>`;
    memoryList.innerHTML = "";
  }
}

async function loadTraces() {
  try {
    const payload = await fetchJson(buildTracesUrl());
    renderTraceList(payload);
    if (memoryViewState.selectedTraceTurnId) {
      setSelectedTrace(memoryViewState.selectedTraceTurnId);
    }
  } catch (error) {
    traceSummary.innerHTML = `<p class="warning-text">${escapeHtml(error.message)}</p>`;
    traceList.innerHTML = "";
  }
}

async function loadLineage(lineageId, memoryId) {
  if (!lineageId) {
    clearLineageInspector();
    return;
  }

  setSelectedMemory(memoryId, getMemoryById(memoryId));
  memoryViewState.selectedLineageId = lineageId;
  lineageSummary.innerHTML = `<p class="muted">正在加载 lineage=${escapeHtml(shortId(lineageId))} ...</p>`;
  lineageList.innerHTML = "";

  try {
    const payload = await fetchJson(`/memories/${encodeURIComponent(lineageId)}/lineage`);
    renderLineage(payload.items || [], lineageId);
    if (memoryId) {
      scrollToMemoryCard(memoryId);
    }
  } catch (error) {
    lineageSummary.innerHTML = `<p class="warning-text">${escapeHtml(error.message)}</p>`;
    lineageList.innerHTML = "";
  }
}

async function loadSummarySources(summaryMemoryId, summaryScope, sourceIds) {
  if (!summaryMemoryId || !sourceIds.length) {
    clearSourceInspector();
    return;
  }

  setSelectedMemory(summaryMemoryId, getMemoryById(summaryMemoryId));
  memoryViewState.selectedSummaryMemoryId = summaryMemoryId;
  memoryViewState.selectedSourceMemoryIds = sourceIds;
  sourceSummary.innerHTML = `
    <p class="muted">
      正在加载 summary=<strong>${escapeHtml(shortId(summaryMemoryId))}</strong> 的 source memories...
    </p>
  `;
  sourceList.innerHTML = "";
  applyMemoryHighlights();

  try {
    const params = new URLSearchParams();
    params.set("memory_ids", sourceIds.join(","));
    params.set("limit", String(sourceIds.length));
    const payload = await fetchJson(`/memories?${params.toString()}`);
    const sourceMap = new Map((payload.items || []).map((item) => [item.metadata?.memory_id, item]));
    const orderedItems = sourceIds.map((sourceId) => sourceMap.get(sourceId)).filter(Boolean);
    renderSourceInspector(summaryMemoryId, summaryScope, sourceIds, orderedItems);
    scrollToMemoryCard(summaryMemoryId);
  } catch (error) {
    sourceSummary.innerHTML = `<p class="warning-text">${escapeHtml(error.message)}</p>`;
    sourceList.innerHTML = "";
  }
}

function syncMemoryFiltersFromDom() {
  memoryViewState.status = memoryStatusFilter.value;
  memoryViewState.memoryType = memoryTypeFilter.value;
  memoryViewState.summaryOnly = memorySummaryOnlyToggle.checked;

  if (memoryViewState.summaryOnly) {
    memoryTypeFilter.disabled = true;
  } else {
    memoryTypeFilter.disabled = false;
  }
}

function syncTraceFiltersFromDom() {
  memoryViewState.traceRoute = traceRouteFilter.value;
  memoryViewState.traceAction = traceActionFilter.value;
}

function focusMemoriesForTurn(turnId) {
  const related = getMemoriesForTurn(turnId);
  if (!related.length) {
    return;
  }
  const first = related[0];
  const memoryId = first.metadata?.memory_id || "";
  if (!memoryId) {
    return;
  }
  setSelectedMemory(memoryId, first);
  scrollToMemoryCard(memoryId);
}

function renderTraceInspector() {
  const trace = memoryViewState.selectedTraceRecord;
  if (!trace) {
    traceDetailSummary.innerHTML = `<p class="muted">在左侧 Trace 面板中选择任意轮次，查看 planner、self-check、retrieval 与 memory lifecycle 细节。</p>`;
    traceDetailBody.innerHTML = "";
    return;
  }

  const turnPlan = trace.turn_plan || {};
  const selfCheck = trace.self_check || null;
  const lifecycle = trace.memory_lifecycle || null;
  const relatedMemories = getMemoriesForTurn(trace.turn_id);

  traceDetailSummary.innerHTML = `
    <p class="muted">
      当前查看 turn=<strong>${escapeHtml(trace.turn_id)}</strong>
      · route=<strong>${escapeHtml(trace.query_route || "general")}</strong>
      · action=<strong>${escapeHtml(turnPlan.action || "n/a")}</strong>
      · related_memories=<strong>${escapeHtml(relatedMemories.length)}</strong>
    </p>
  `;

  traceDetailBody.innerHTML = `
    <article class="detail-card">
      <div class="chip-row">
        <span class="chip">turn ${escapeHtml(trace.turn_id)}</span>
        <span class="chip">${escapeHtml(trace.query_route || "general")}</span>
        <span class="chip chip-neutral">${escapeHtml(turnPlan.action || "n/a")}</span>
      </div>
      <div class="detail-grid">
        <div class="detail-metric">
          <span>created_at</span>
          <strong>${escapeHtml(formatTime(trace.created_at))}</strong>
        </div>
        <div class="detail-metric">
          <span>retrieved</span>
          <strong>${escapeHtml(trace.retrieved_memories?.length ?? 0)}</strong>
        </div>
      </div>
      <section class="detail-section">
        <h4>User Query</h4>
        <p class="detail-trace">${escapeHtml(trace.query || "")}</p>
      </section>
      <section class="detail-section">
        <h4>Assistant Answer</h4>
        <p class="detail-trace">${escapeHtml(trace.answer || "")}</p>
      </section>
      <section class="detail-section">
        <h4>Planner</h4>
        <dl class="detail-list">
          <div class="detail-row">
            <dt>reason</dt>
            <dd>${escapeHtml(turnPlan.reason || "n/a")}</dd>
          </div>
          <div class="detail-row">
            <dt>response_language</dt>
            <dd>${escapeHtml(turnPlan.response_language || "n/a")}</dd>
          </div>
          ${
            turnPlan.tool_name
              ? `
          <div class="detail-row">
            <dt>tool_name</dt>
            <dd>${escapeHtml(turnPlan.tool_name)}</dd>
          </div>`
              : ""
          }
          ${
            turnPlan.tool_query
              ? `
          <div class="detail-row">
            <dt>tool_query</dt>
            <dd>${escapeHtml(turnPlan.tool_query)}</dd>
          </div>`
              : ""
          }
        </dl>
      </section>
      <section class="detail-section">
        <h4>Self-check</h4>
        <p class="detail-trace">${escapeHtml(selfCheck?.summary || "No self-check result.")}</p>
      </section>
      <section class="detail-section">
        <h4>Memory Lifecycle</h4>
        <p class="detail-trace">${
          lifecycle
            ? escapeHtml(
                `extracted=${lifecycle.extracted} · added=${lifecycle.added} · updated=${lifecycle.updated} · versioned=${lifecycle.versioned} · removed=${lifecycle.removed}`,
              )
            : "This turn did not update memory."
        }</p>
      </section>
      <section class="detail-section">
        <h4>Retrieved Memories</h4>
        ${
          trace.retrieved_memories?.length
            ? `
        <div class="retrieved-list">
          ${trace.retrieved_memories
            .slice(0, 4)
            .map(
              (item) => `
                <article class="retrieved-item">
                  <div class="chip-row">
                    <span class="chip">${escapeHtml(item.memory?.type || "memory")}</span>
                    <span class="chip">score ${escapeHtml(Number(item.score || 0).toFixed(2))}</span>
                  </div>
                  <p>${escapeHtml(item.memory?.content || "")}</p>
                </article>
              `,
            )
            .join("")}
        </div>`
            : `<p class="muted">这一轮没有召回 memory。</p>`
        }
      </section>
      <section class="detail-section">
        <h4>Memory Link</h4>
        <p class="detail-trace">该轮写入的 memory 会按 <code>source_turn_id=${escapeHtml(trace.turn_id)}</code> 在左侧主列表中高亮。</p>
        <div class="memory-actions">
          <button
            type="button"
            class="inline-button"
            data-action="focus-trace-memories"
            data-turn-id="${escapeHtml(trace.turn_id)}"
          >
            Focus memories from this turn
          </button>
        </div>
      </section>
    </article>
  `;
}

refreshMemoriesButton.addEventListener("click", async () => {
  await Promise.all([loadStats(), loadMemories(), loadTraces()]);
});
refreshTracesButton.addEventListener("click", loadTraces);
clearLineageButton.addEventListener("click", clearLineageInspector);
clearSourcesButton.addEventListener("click", clearSourceInspector);
clearDetailButton.addEventListener("click", clearDetailInspector);
clearTraceDetailButton.addEventListener("click", clearTraceInspector);

[memoryStatusFilter, memoryTypeFilter, memorySummaryOnlyToggle].forEach((element) => {
  element.addEventListener("change", async () => {
    syncMemoryFiltersFromDom();
    await loadMemories();
  });
});

resetMemoryFiltersButton.addEventListener("click", async () => {
  memoryStatusFilter.value = "";
  memoryTypeFilter.value = "";
  memorySummaryOnlyToggle.checked = false;
  syncMemoryFiltersFromDom();
  await loadMemories();
});

[traceRouteFilter, traceActionFilter].forEach((element) => {
  element.addEventListener("change", async () => {
    syncTraceFiltersFromDom();
    await loadTraces();
  });
});

resetTraceFiltersButton.addEventListener("click", async () => {
  traceRouteFilter.value = "";
  traceActionFilter.value = "";
  syncTraceFiltersFromDom();
  await loadTraces();
});

memoryList.addEventListener("click", async (event) => {
  const target = event.target;
  if (!(target instanceof HTMLElement)) {
    return;
  }

  const action = target.dataset.action;
  if (action === "view-detail") {
    const memoryId = target.dataset.memoryId || "";
    if (memoryId) {
      setSelectedMemory(memoryId, getMemoryById(memoryId));
      scrollToMemoryCard(memoryId);
    }
    return;
  }

  if (action === "view-lineage") {
    await loadLineage(target.dataset.lineageId || "", target.dataset.memoryId || "");
    return;
  }

  if (action === "view-sources") {
    const sourceIds = (target.dataset.sourceIds || "")
      .split(",")
      .map((item) => item.trim())
      .filter(Boolean);
    await loadSummarySources(
      target.dataset.memoryId || "",
      target.dataset.summaryScope || "",
      sourceIds,
    );
  }
});

detailBody.addEventListener("click", async (event) => {
  const target = event.target;
  if (!(target instanceof HTMLElement)) {
    return;
  }

  const action = target.dataset.action;
  if (action === "view-turn-trace") {
    const turnId = target.dataset.turnId || "";
    if (!turnId) {
      return;
    }
    const existing = getTraceByTurnId(turnId);
    if (existing) {
      setSelectedTrace(turnId, existing);
      focusMemoriesForTurn(turnId);
      return;
    }
    try {
      const payload = await fetchJson(`/traces/${encodeURIComponent(turnId)}`);
      setSelectedTrace(turnId, payload);
      focusMemoriesForTurn(turnId);
    } catch (error) {
      traceDetailSummary.innerHTML = `<p class="warning-text">${escapeHtml(error.message)}</p>`;
      traceDetailBody.innerHTML = "";
    }
  }
});

traceList.addEventListener("click", async (event) => {
  const target = event.target;
  if (!(target instanceof HTMLElement)) {
    return;
  }

  const action = target.dataset.action;
  if (action === "view-trace") {
    const turnId = target.dataset.turnId || "";
    if (!turnId) {
      return;
    }
    const existing = getTraceByTurnId(turnId);
    if (existing) {
      setSelectedTrace(turnId, existing);
      return;
    }
    try {
      const payload = await fetchJson(`/traces/${encodeURIComponent(turnId)}`);
      setSelectedTrace(turnId, payload);
    } catch (error) {
      traceDetailSummary.innerHTML = `<p class="warning-text">${escapeHtml(error.message)}</p>`;
      traceDetailBody.innerHTML = "";
    }
    return;
  }

  if (action === "focus-trace-memories") {
    const turnId = target.dataset.turnId || "";
    const existing = getTraceByTurnId(turnId);
    if (existing) {
      setSelectedTrace(turnId, existing);
    }
    focusMemoriesForTurn(turnId);
  }
});

traceDetailBody.addEventListener("click", (event) => {
  const target = event.target;
  if (!(target instanceof HTMLElement)) {
    return;
  }

  const action = target.dataset.action;
  if (action !== "focus-trace-memories") {
    return;
  }

  const turnId = target.dataset.turnId || "";
  const existing = getTraceByTurnId(turnId);
  if (existing) {
    setSelectedTrace(turnId, existing);
  }
  focusMemoriesForTurn(turnId);
});

function bindInspectorFocus(container) {
  container.addEventListener("click", (event) => {
    const target = event.target;
    if (!(target instanceof HTMLElement)) {
      return;
    }

    const action = target.dataset.action;
    if (action === "view-detail") {
      const memoryId = target.dataset.memoryId || "";
      if (memoryId) {
        setSelectedMemory(memoryId, getMemoryById(memoryId));
      }
      return;
    }

    if (action !== "focus-memory") {
      return;
    }

    const memoryId = target.dataset.memoryId || "";
    const lineageId = target.dataset.lineageId || "";
    if (memoryId) {
      setSelectedMemory(memoryId, getMemoryById(memoryId));
    }
    if (lineageId) {
      memoryViewState.selectedLineageId = lineageId;
    }
    applyMemoryHighlights();
    scrollToMemoryCard(memoryId);
  });
}

bindInspectorFocus(lineageList);
bindInspectorFocus(sourceList);
bindInspectorFocus(traceDetailBody);

syncMemoryFiltersFromDom();
syncTraceFiltersFromDom();
renderDetailPanel();
renderTraceInspector();
loadStats();
loadMemories();
loadTraces();
