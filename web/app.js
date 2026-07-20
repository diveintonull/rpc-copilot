"use strict";

const EVENT_TYPES = new Set([
  "status",
  "text",
  "reference",
  "recommendation",
  "trace",
  "done",
  "error",
]);
const TERMINAL_EVENT_TYPES = new Set(["done", "error"]);
const TRACE_FIELDS = [
  "node",
  "tool",
  "action",
  "validation_action",
  "failure_count",
  "failure_codes",
  "duration_ms",
  "status",
];
const ANSWER_RENDER_INTERVAL_MS = 40;

const MODE_CONFIG = {
  regulation_qa: {
    description:
      "提出一个法规问题，回答中的事实应绑定右侧证据。",
    placeholder: "例如：管理员身份鉴别有哪些要求？",
    examples: [
      "管理员身份鉴别有哪些要求？",
      "身份鉴别为什么要组合两种技术？",
    ],
  },
  clause_comparison: {
    description:
      "指定两个明确条款，分别保留左右证据，再说明相同点与差异。",
    placeholder:
      "例如：比较 GB/T 22239-2019 8.1.4.1 与 ISO 27001:2022 A.5.17",
    examples: [
      "比较 GB/T 22239 与 GB/T 35273 的身份相关安全要求",
      "比较《数据安全法》与《网络安全法》的安全事件处置要求",
    ],
  },
  gap_analysis: {
    description:
      "把企业当前控制事实与法规证据对照；结论需要人工复核。",
    placeholder: "例如：检查当前管理员登录控制有哪些差距",
    examples: [
      "检查管理员身份鉴别控制差距",
      "评估当前管理员身份鉴别措施",
    ],
  },
};

class StreamProtocolError extends Error {
  constructor(message) {
    super(message);
    this.name = "StreamProtocolError";
  }
}

class StreamInterruptedError extends Error {
  constructor(message = "stream closed before a terminal event") {
    super(message);
    this.name = "StreamInterruptedError";
  }
}

class SSEFrameParser {
  constructor(onEvent) {
    if (typeof onEvent !== "function") {
      throw new TypeError("onEvent must be a function");
    }
    this.onEvent = onEvent;
    this.buffer = "";
  }

  push(chunk) {
    if (typeof chunk !== "string") {
      throw new TypeError("SSE chunk must be a string");
    }
    this.buffer += chunk;

    let boundary = this.findBoundary();
    while (boundary !== null) {
      const frame = this.buffer.slice(0, boundary.index);
      this.buffer = this.buffer.slice(boundary.index + boundary.length);
      this.parseFrame(frame);
      boundary = this.findBoundary();
    }
  }

  findBoundary() {
    const match = /\r?\n\r?\n/.exec(this.buffer);
    if (match === null) {
      return null;
    }
    return { index: match.index, length: match[0].length };
  }

  parseFrame(frame) {
    if (!frame.trim()) {
      return;
    }

    const dataLines = [];
    for (const line of frame.replaceAll("\r\n", "\n").split("\n")) {
      if (line.startsWith(":")) {
        continue;
      }
      if (line === "data") {
        dataLines.push("");
      } else if (line.startsWith("data:")) {
        const value = line.slice(5);
        dataLines.push(value.startsWith(" ") ? value.slice(1) : value);
      }
    }

    if (dataLines.length === 0) {
      return;
    }

    let payload;
    try {
      payload = JSON.parse(dataLines.join("\n"));
    } catch (_error) {
      throw new StreamProtocolError("SSE data is not valid JSON");
    }
    this.onEvent(payload);
  }

  finish() {
    if (this.buffer.trim()) {
      throw new StreamProtocolError("stream ended inside an SSE frame");
    }
    this.buffer = "";
  }
}

class StreamSession {
  constructor(requestId, onEvent = () => {}) {
    if (typeof requestId !== "string" || !requestId.trim()) {
      throw new TypeError("requestId must not be blank");
    }
    if (typeof onEvent !== "function") {
      throw new TypeError("onEvent must be a function");
    }
    this.requestId = requestId;
    this.onEvent = onEvent;
    this.terminal = false;
    this.terminalType = null;
  }

  accept(event) {
    if (event === null || typeof event !== "object" || Array.isArray(event)) {
      throw new StreamProtocolError("SSE event must be an object");
    }
    if (!EVENT_TYPES.has(event.type)) {
      throw new StreamProtocolError(`unknown SSE event type: ${event.type}`);
    }
    if (typeof event.request_id !== "string" || !event.request_id.trim()) {
      throw new StreamProtocolError("SSE request_id must not be blank");
    }
    if (
      event.data === null ||
      typeof event.data !== "object" ||
      Array.isArray(event.data)
    ) {
      throw new StreamProtocolError("SSE event data must be an object");
    }
    if (event.request_id !== this.requestId) {
      return { accepted: false, reason: "request_mismatch" };
    }
    if (this.terminal) {
      return { accepted: false, reason: "after_terminal" };
    }

    this.onEvent(event);
    if (TERMINAL_EVENT_TYPES.has(event.type)) {
      this.terminal = true;
      this.terminalType = event.type;
    }
    return { accepted: true, reason: null };
  }

  finish() {
    if (!this.terminal) {
      throw new StreamInterruptedError();
    }
    return this.terminalType;
  }
}

function assertContract(condition, message) {
  if (!condition) {
    throw new Error(`frontend contract failed: ${message}`);
  }
}

function eventFrame(type, requestId, data, newline = "\r\n") {
  const payload = JSON.stringify({ type, request_id: requestId, data });
  return `event: ${type}${newline}data: ${payload}${newline}${newline}`;
}

function runContractSelfTests() {
  const observed = [];
  const session = new StreamSession("selftest-order", (event) => {
    observed.push(event.type);
  });
  const parser = new SSEFrameParser((event) => session.accept(event));
  const stream = [
    eventFrame("reference", "selftest-order", { parent_id: "law#1" }),
    eventFrame("text", "selftest-order", { delta: "answer [1]" }),
    eventFrame("status", "selftest-order", { status: "running" }),
    eventFrame("done", "selftest-order", { status: "completed" }),
  ].join("");

  let offset = 0;
  for (const size of [1, 2, 5, 3, 11, 7, 19, 23, 29]) {
    parser.push(stream.slice(offset, offset + size));
    offset += size;
  }
  if (offset < stream.length) {
    parser.push(stream.slice(offset));
  }
  parser.finish();
  assertContract(
    observed.join(",") === "reference,text,status,done",
    "out-of-order events must retain arrival order",
  );
  assertContract(session.finish() === "done", "done must close the session");

  const late = session.accept({
    type: "text",
    request_id: "selftest-order",
    data: { delta: "must be ignored" },
  });
  assertContract(
    !late.accepted && late.reason === "after_terminal",
    "events after done must be ignored",
  );

  const disconnected = new StreamSession("selftest-disconnect");
  disconnected.accept({
    type: "status",
    request_id: "selftest-disconnect",
    data: { status: "running" },
  });
  let interruptionDetected = false;
  try {
    disconnected.finish();
  } catch (error) {
    interruptionDetected = error instanceof StreamInterruptedError;
  }
  assertContract(
    interruptionDetected,
    "close without done or error must be an interruption",
  );

  const mismatch = new StreamSession("selftest-current");
  const mismatched = mismatch.accept({
    type: "done",
    request_id: "selftest-old",
    data: { status: "completed" },
  });
  assertContract(
    !mismatched.accepted && mismatched.reason === "request_mismatch",
    "events from an old request must be ignored",
  );

  return {
    passed: true,
    checks: [
      "fragmented CRLF frames",
      "out-of-order nonterminal events",
      "terminal boundary",
      "unexpected disconnect",
      "request isolation",
    ],
  };
}

function makeRequestId() {
  if (globalThis.crypto && typeof globalThis.crypto.randomUUID === "function") {
    return globalThis.crypto.randomUUID();
  }
  return `web-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function isNonEmptyString(value) {
  return typeof value === "string" && value.trim().length > 0;
}

function safeRecommendationText(data) {
  for (const key of ["text", "recommendation", "action", "message"]) {
    if (isNonEmptyString(data[key])) {
      return data[key].trim();
    }
  }
  return "收到一条未包含可展示文本的建议。";
}

function appendInlineMarkdown(parent, text, documentRoot, createCitation) {
  const pattern =
    /(\*\*([^*\n]+)\*\*|__([^_\n]+)__|\x60([^\x60\n]+)\x60|\*([^*\n]+)\*|_([^_\n]+)_|\[(\d+)\])/g;
  let cursor = 0;
  let match = pattern.exec(text);
  while (match !== null) {
    parent.append(documentRoot.createTextNode(text.slice(cursor, match.index)));
    let node;
    if (match[2] !== undefined || match[3] !== undefined) {
      node = documentRoot.createElement("strong");
      node.textContent = match[2] ?? match[3];
    } else if (match[4] !== undefined) {
      node = documentRoot.createElement("code");
      node.textContent = match[4];
    } else if (match[5] !== undefined || match[6] !== undefined) {
      node = documentRoot.createElement("em");
      node.textContent = match[5] ?? match[6];
    } else {
      node = createCitation(Number.parseInt(match[7], 10));
    }
    parent.append(node);
    cursor = pattern.lastIndex;
    match = pattern.exec(text);
  }
  parent.append(documentRoot.createTextNode(text.slice(cursor)));
}

function splitTableRow(line) {
  return line.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map(
    (cell) => cell.trim(),
  );
}

function isTableSeparator(line) {
  const cells = splitTableRow(line);
  return (
    cells.length > 0 &&
    cells.every((cell) => /^:?-{3,}:?$/.test(cell))
  );
}

function startsMarkdownBlock(lines, index) {
  const line = lines[index] ?? "";
  const next = lines[index + 1] ?? "";
  const trimmed = line.trim();
  const fence = "\x60\x60\x60";
  return (
    !trimmed ||
    trimmed.startsWith(fence) ||
    /^#{1,6}\s+\S/.test(line) ||
    /^\s*([-*+]\s+\S|\d+[.)]\s+\S|>\s?)/.test(line) ||
    /^\s*((-{3,})|(\*{3,})|(_{3,}))\s*$/.test(line) ||
    (line.includes("|") && isTableSeparator(next))
  );
}

function renderMarkdownInto(
  container,
  markdown,
  { documentRoot, createCitation },
) {
  container.replaceChildren();
  const lines = markdown.replaceAll("\r\n", "\n").split("\n");
  const appendInline = (parent, text) => {
    appendInlineMarkdown(parent, text, documentRoot, createCitation);
  };

  let index = 0;
  while (index < lines.length) {
    const line = lines[index];
    if (!line.trim()) {
      index += 1;
      continue;
    }

    const fence = "\x60\x60\x60";
    if (line.trim().startsWith(fence)) {
      const language = line.trim().slice(fence.length).trim();
      const codeLines = [];
      index += 1;
      while (
        index < lines.length &&
        !lines[index].trim().startsWith(fence)
      ) {
        codeLines.push(lines[index]);
        index += 1;
      }
      if (index < lines.length) {
        index += 1;
      }
      const pre = documentRoot.createElement("pre");
      const code = documentRoot.createElement("code");
      if (language) {
        code.dataset.language = language;
      }
      code.textContent = codeLines.join("\n");
      pre.append(code);
      container.append(pre);
      continue;
    }

    if (line.includes("|") && isTableSeparator(lines[index + 1] ?? "")) {
      const table = documentRoot.createElement("table");
      const head = documentRoot.createElement("thead");
      const headRow = documentRoot.createElement("tr");
      for (const cellText of splitTableRow(line)) {
        const cell = documentRoot.createElement("th");
        appendInline(cell, cellText);
        headRow.append(cell);
      }
      head.append(headRow);
      table.append(head);
      index += 2;

      const body = documentRoot.createElement("tbody");
      while (
        index < lines.length &&
        lines[index].trim() &&
        lines[index].includes("|")
      ) {
        const row = documentRoot.createElement("tr");
        for (const cellText of splitTableRow(lines[index])) {
          const cell = documentRoot.createElement("td");
          appendInline(cell, cellText);
          row.append(cell);
        }
        body.append(row);
        index += 1;
      }
      if (body.childNodes.length > 0) {
        table.append(body);
      }
      const wrapper = documentRoot.createElement("div");
      wrapper.className = "markdown-table-wrap";
      wrapper.append(table);
      container.append(wrapper);
      continue;
    }

    const headingMatch = /^(#{1,6})\s+(.+)$/.exec(line);
    if (headingMatch) {
      const heading = documentRoot.createElement(
        "h" + String(headingMatch[1].length),
      );
      appendInline(heading, headingMatch[2]);
      container.append(heading);
      index += 1;
      continue;
    }

    if (/^\s*((-{3,})|(\*{3,})|(_{3,}))\s*$/.test(line)) {
      container.append(documentRoot.createElement("hr"));
      index += 1;
      continue;
    }

    if (/^\s*>\s?/.test(line)) {
      const quote = documentRoot.createElement("blockquote");
      while (index < lines.length && /^\s*>\s?/.test(lines[index])) {
        if (quote.childNodes.length > 0) {
          quote.append(documentRoot.createElement("br"));
        }
        appendInline(quote, lines[index].replace(/^\s*>\s?/, ""));
        index += 1;
      }
      container.append(quote);
      continue;
    }

    const unordered = /^\s*[-*+]\s+\S/.test(line);
    const ordered = /^\s*\d+[.)]\s+\S/.test(line);
    if (unordered || ordered) {
      const list = documentRoot.createElement(ordered ? "ol" : "ul");
      const itemPattern = ordered
        ? /^\s*\d+[.)]\s+(.+)$/
        : /^\s*[-*+]\s+(.+)$/;
      let itemMatch = itemPattern.exec(lines[index]);
      while (itemMatch) {
        const item = documentRoot.createElement("li");
        appendInline(item, itemMatch[1]);
        list.append(item);
        index += 1;
        itemMatch =
          index < lines.length ? itemPattern.exec(lines[index]) : null;
      }
      container.append(list);
      continue;
    }

    const paragraph = documentRoot.createElement("p");
    let firstLine = true;
    while (index < lines.length && !startsMarkdownBlock(lines, index)) {
      if (!firstLine) {
        paragraph.append(documentRoot.createElement("br"));
      }
      appendInline(paragraph, lines[index]);
      firstLine = false;
      index += 1;
    }
    if (firstLine) {
      // Streaming can temporarily leave an incomplete Markdown marker such as
      // "1.  ", "- ", or "# ". Always consume one line so rendering can
      // never spin forever while the rest of the token is still in flight.
      appendInline(paragraph, lines[index]);
      index += 1;
    }
    container.append(paragraph);
  }
}

class GRCApplication {
  constructor(documentRoot) {
    this.document = documentRoot;
    this.mode = "regulation_qa";
    this.running = false;
    this.requestId = null;
    this.answerText = "";
    this.references = [];
    this.referenceIds = new Set();
    this.referenceGeneration = 0;
    this.traceCount = 0;
    this.answerContent = null;
    this.answerStatus = null;
    this.answerRenderTimer = null;
    this.answerRenderFrame = null;

    this.elements = this.readElements();
    this.bindInteractions();
    this.selectMode(this.mode);
    this.checkHealth();
    this.showSelfTestResult();
  }

  readElements() {
    const byId = (id) => this.document.getElementById(id);
    return {
      healthDot: byId("health-dot"),
      healthLabel: byId("health-label"),
      requestLabel: byId("request-label"),
      modeDescription: byId("mode-description"),
      connectionPill: byId("connection-pill"),
      connectionLabel: byId("connection-label"),
      selftestBanner: byId("selftest-banner"),
      timeline: byId("chat-timeline"),
      recommendationSection: byId("recommendation-section"),
      recommendationList: byId("recommendation-list"),
      exampleList: byId("example-list"),
      form: byId("chat-composer"),
      queryInput: byId("query-input"),
      controlGroup: byId("control-input-group"),
      controlInput: byId("control-input"),
      stopButton: byId("stop-button"),
      sendButton: byId("send-button"),
      evidenceCount: byId("evidence-count"),
      evidenceEmpty: byId("evidence-empty"),
      evidenceList: byId("evidence-list"),
      traceCount: byId("trace-count"),
      traceEmpty: byId("trace-empty"),
      traceList: byId("trace-list"),
      evidencePanel: byId("evidence-panel"),
      tracePanel: byId("trace-panel"),
      announcer: byId("live-announcer"),
    };
  }

  bindInteractions() {
    for (const button of this.document.querySelectorAll("[data-mode]")) {
      button.addEventListener("click", () => this.selectMode(button.dataset.mode));
    }
    for (const tab of this.document.querySelectorAll("[data-panel]")) {
      tab.addEventListener("click", () => this.selectPanel(tab.dataset.panel));
    }
    this.elements.form.addEventListener("submit", (event) => {
      event.preventDefault();
      this.startRequest();
    });
    this.elements.queryInput.addEventListener("keydown", (event) => {
      if (event.key !== "Enter" || event.isComposing || event.shiftKey) {
        return;
      }
      event.preventDefault();
      this.elements.form.requestSubmit();
    });
    this.elements.stopButton.addEventListener("click", () => this.stopRequest());
  }

  selectMode(mode) {
    if (this.running || !Object.hasOwn(MODE_CONFIG, mode)) {
      return;
    }
    this.mode = mode;
    const config = MODE_CONFIG[mode];
    for (const button of this.document.querySelectorAll("[data-mode]")) {
      const active = button.dataset.mode === mode;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-pressed", String(active));
    }
    this.elements.modeDescription.textContent = config.description;
    this.elements.queryInput.placeholder = config.placeholder;
    this.elements.controlGroup.hidden = mode !== "gap_analysis";
    this.renderExamples(config.examples);
  }

  renderExamples(examples) {
    this.elements.exampleList.replaceChildren();
    for (const example of examples) {
      const button = this.document.createElement("button");
      button.type = "button";
      button.className = "example-button";
      button.textContent = example;
      button.addEventListener("click", () => {
        this.elements.queryInput.value = example;
        this.elements.queryInput.focus();
      });
      this.elements.exampleList.append(button);
    }
  }

  selectPanel(panel) {
    const evidenceActive = panel === "evidence";
    this.elements.evidencePanel.hidden = !evidenceActive;
    this.elements.tracePanel.hidden = evidenceActive;
    for (const tab of this.document.querySelectorAll("[data-panel]")) {
      const active = tab.dataset.panel === panel;
      tab.classList.toggle("is-active", active);
      tab.setAttribute("aria-selected", String(active));
      tab.tabIndex = active ? 0 : -1;
    }
  }

  async checkHealth() {
    try {
      const response = await fetch("/ready", {
        headers: { Accept: "application/json" },
      });
      if (!response.ok) {
        throw new Error("health endpoint failed");
      }
      const data = await response.json();
      if (!new Set(["ok", "ready"]).has(data.status)) {
        throw new Error("service is not ready");
      }
      this.elements.healthDot.classList.add("is-healthy");
      this.elements.healthDot.classList.remove("is-unhealthy");
      this.elements.healthLabel.textContent = "服务正常";
    } catch (_error) {
      this.elements.healthDot.classList.remove("is-healthy");
      this.elements.healthDot.classList.add("is-unhealthy");
      this.elements.healthLabel.textContent = "服务不可用";
    }
  }

  showSelfTestResult() {
    const params = new URLSearchParams(globalThis.location.search);
    if (params.get("selftest") !== "1") {
      return;
    }
    this.elements.selftestBanner.hidden = false;
    try {
      const result = runContractSelfTests();
      this.elements.selftestBanner.textContent =
        `前端契约自检通过：${result.checks.join("、")}`;
    } catch (error) {
      this.elements.selftestBanner.classList.add("is-failed");
      this.elements.selftestBanner.textContent = `前端契约自检失败：${error.message}`;
    }
  }

  resetResultPanels() {
    this.cancelScheduledAnswerRender();
    this.answerText = "";
    this.references = [];
    this.referenceIds.clear();
    this.referenceGeneration = 0;
    this.traceCount = 0;
    this.answerContent = null;
    this.answerStatus = null;
    this.elements.evidenceList.replaceChildren();
    this.elements.traceList.replaceChildren();
    this.elements.recommendationList.replaceChildren();
    this.elements.evidenceEmpty.hidden = false;
    this.elements.traceEmpty.hidden = false;
    this.elements.recommendationSection.hidden = true;
    this.elements.evidenceCount.textContent = "0";
    this.elements.traceCount.textContent = "0";
  }

  addUserMessage(query, controlText) {
    const article = this.document.createElement("article");
    article.className = "message message-user";

    const avatar = this.document.createElement("div");
    avatar.className = "message-avatar";
    avatar.setAttribute("aria-hidden", "true");
    avatar.textContent = "你";

    const body = this.document.createElement("div");
    body.className = "message-body";
    const role = this.document.createElement("p");
    role.className = "message-role";
    role.textContent = MODE_CONFIG[this.mode].description;
    const content = this.document.createElement("div");
    content.className = "message-content";
    const queryParagraph = this.document.createElement("p");
    queryParagraph.textContent = query;
    content.append(queryParagraph);
    if (controlText) {
      const controlParagraph = this.document.createElement("p");
      controlParagraph.textContent = `\n当前控制：${controlText}`;
      content.append(controlParagraph);
    }
    body.append(role, content);
    article.append(avatar, body);
    this.elements.timeline.append(article);
  }

  addAssistantMessage() {
    const article = this.document.createElement("article");
    article.className = "message message-assistant";

    const avatar = this.document.createElement("div");
    avatar.className = "message-avatar";
    avatar.setAttribute("aria-hidden", "true");
    avatar.textContent = "AI";

    const body = this.document.createElement("div");
    body.className = "message-body";
    const role = this.document.createElement("p");
    role.className = "message-role";
    role.textContent = "GRC Copilot";
    const content = this.document.createElement("div");
    content.className = "message-content";
    content.textContent = "正在建立流式连接…";
    const status = this.document.createElement("div");
    status.className = "message-status";
    const statusDot = this.document.createElement("span");
    statusDot.className = "message-status-dot";
    statusDot.setAttribute("aria-hidden", "true");
    const statusText = this.document.createElement("span");
    statusText.textContent = "等待 Agent 事件";
    status.append(statusDot, statusText);
    body.append(role, content, status);
    article.append(avatar, body);
    this.elements.timeline.append(article);

    this.answerContent = content;
    this.answerStatus = statusText;
    this.answerArticle = article;
    this.scrollTimeline();
  }

  renderAnswer() {
    if (!this.answerContent) {
      return;
    }
    if (!this.answerText) {
      this.answerContent.replaceChildren();
      this.answerContent.textContent = "正在等待回答内容…";
      return;
    }
    renderMarkdownInto(this.answerContent, this.answerText, {
      documentRoot: this.document,
      createCitation: (number) => this.makeCitationButton(number),
    });
  }

  cancelScheduledAnswerRender() {
    if (this.answerRenderTimer !== null) {
      globalThis.clearTimeout(this.answerRenderTimer);
      this.answerRenderTimer = null;
    }
    if (this.answerRenderFrame !== null) {
      globalThis.cancelAnimationFrame(this.answerRenderFrame);
      this.answerRenderFrame = null;
    }
  }

  scheduleAnswerRender() {
    if (
      this.answerRenderTimer !== null ||
      this.answerRenderFrame !== null
    ) {
      return;
    }
    this.answerRenderTimer = globalThis.setTimeout(() => {
      this.answerRenderTimer = null;
      this.answerRenderFrame = globalThis.requestAnimationFrame(() => {
        this.answerRenderFrame = null;
        this.renderAnswer();
        this.scrollTimeline();
      });
    }, ANSWER_RENDER_INTERVAL_MS);
  }

  flushAnswerRender() {
    this.cancelScheduledAnswerRender();
    this.renderAnswer();
    this.scrollTimeline();
  }

  makeCitationButton(number) {
    const button = this.document.createElement("button");
    button.type = "button";
    button.className = "citation-button";
    button.textContent = "[" + String(number) + "]";
    const available = number >= 1 && number <= this.references.length;
    button.disabled = !available;
    button.title = available
      ? "定位证据 " + String(number)
      : "证据 " + String(number) + " 尚未到达";
    if (available) {
      button.addEventListener("click", () => this.focusEvidence(number));
    }
    return button;
  }

  addReference(data) {
    const generation = Number.isInteger(data.generation)
      ? data.generation
      : this.referenceGeneration;
    if (generation < this.referenceGeneration) {
      return;
    }
    this.referenceGeneration = generation;
    if (!isNonEmptyString(data.parent_id) || this.referenceIds.has(data.parent_id)) {
      return;
    }
    this.referenceIds.add(data.parent_id);
    this.references.push(data);
    const number = this.references.length;

    const card = this.document.createElement("article");
    card.className = "evidence-card";
    card.id = `evidence-${number}`;
    card.tabIndex = -1;
    card.dataset.parentId = data.parent_id;

    const header = this.document.createElement("div");
    header.className = "evidence-card-header";
    const badge = this.document.createElement("span");
    badge.className = "evidence-number";
    badge.textContent = String(number);
    const score = this.document.createElement("span");
    score.className = "evidence-score";
    score.textContent =
      typeof data.score === "number"
        ? `RELEVANCE ${(data.score * 100).toFixed(0)}%`
        : "EXACT CLAUSE";
    header.append(badge, score);

    const title = this.document.createElement("h3");
    title.className = "evidence-title";
    const source = isNonEmptyString(data.source_id) ? data.source_id : "未知来源";
    const section = isNonEmptyString(data.section_number)
      ? ` · ${data.section_number}`
      : "";
    title.textContent = `${source}${section}`;

    const text = this.document.createElement("p");
    text.className = "evidence-text";
    text.textContent = isNonEmptyString(data.text)
      ? data.text
      : "该引用未包含可展示的条款正文。";

    let visual = null;
    if (data.modality === "image" && isNonEmptyString(data.image_url)) {
      visual = this.document.createElement("img");
      visual.className = "evidence-page-image";
      visual.src = data.image_url;
      visual.alt = Number.isInteger(data.page_number)
        ? `${source} page ${data.page_number}`
        : `${source} rendered page`;
      visual.loading = "lazy";
      visual.decoding = "async";
    }

    const meta = this.document.createElement("div");
    meta.className = "evidence-meta";
    const pageLabel = Number.isInteger(data.page_number)
      ? `PAGE ${data.page_number}`
      : null;
    for (const value of [data.modality, pageLabel, data.version, data.parent_id]) {
      if (isNonEmptyString(value)) {
        const item = this.document.createElement("span");
        item.textContent = value;
        meta.append(item);
      }
    }
    card.append(header, title);
    if (visual) {
      card.append(visual);
    }
    card.append(text, meta);
    this.elements.evidenceList.append(card);
    this.elements.evidenceEmpty.hidden = true;
    this.elements.evidenceCount.textContent = String(number);
    this.renderAnswer();
  }

  resetReferences(data) {
    const generation = Number.isInteger(data.generation)
      ? data.generation
      : this.referenceGeneration + 1;
    if (generation < this.referenceGeneration) {
      return;
    }
    this.referenceGeneration = generation;
    this.references = [];
    this.referenceIds.clear();
    this.elements.evidenceList.replaceChildren();
    this.elements.evidenceEmpty.hidden = false;
    this.elements.evidenceCount.textContent = "0";
    this.renderAnswer();
  }

  focusEvidence(number) {
    this.selectPanel("evidence");
    const card = this.document.getElementById(`evidence-${number}`);
    if (!card) {
      return;
    }
    for (const item of this.elements.evidenceList.children) {
      item.classList.remove("is-highlighted");
    }
    card.classList.add("is-highlighted");
    card.scrollIntoView({ behavior: "smooth", block: "center" });
    card.focus({ preventScroll: true });
    globalThis.setTimeout(() => card.classList.remove("is-highlighted"), 1900);
    this.announce(`已定位证据 ${number}`);
  }

  addTrace(data) {
    const safe = {};
    for (const field of TRACE_FIELDS) {
      const value = data[field];
      if (field === "duration_ms" || field === "failure_count") {
        if (typeof value === "number" && Number.isFinite(value)) {
          safe[field] = value;
        }
      } else if (field === "failure_codes") {
        if (Array.isArray(value)) {
          safe[field] = value.filter(isNonEmptyString).map((item) => item.trim());
        }
      } else if (isNonEmptyString(value)) {
        safe[field] = value.trim();
      }
    }
    if (Object.keys(safe).length === 0) {
      return;
    }

    this.traceCount += 1;
    const item = this.document.createElement("li");
    item.className = "trace-item";
    item.dataset.status = safe.status || "observed";
    const marker = this.document.createElement("span");
    marker.className = "trace-marker";
    marker.setAttribute("aria-hidden", "true");
    marker.textContent = String(this.traceCount).padStart(2, "0");

    const body = this.document.createElement("div");
    body.className = "trace-body";
    const title = this.document.createElement("h3");
    title.className = "trace-title";
    title.textContent = safe.node || "Agent event";
    const meta = this.document.createElement("div");
    meta.className = "trace-meta";
    if (safe.tool) {
      const tool = this.document.createElement("span");
      tool.textContent = `工具 ${safe.tool}`;
      meta.append(tool);
    }
    if (safe.action) {
      const action = this.document.createElement("span");
      action.textContent = `动作 ${safe.action}`;
      meta.append(action);
    }
    if (safe.validation_action) {
      const validation = this.document.createElement("span");
      validation.textContent = `校验 ${safe.validation_action}`;
      meta.append(validation);
    }
    if (safe.failure_codes?.length) {
      const failures = this.document.createElement("span");
      const count = safe.failure_count ?? safe.failure_codes.length;
      failures.textContent = `失败 ${safe.failure_codes.join(", ")} × ${count}`;
      meta.append(failures);
    }
    if (safe.duration_ms !== undefined) {
      const duration = this.document.createElement("span");
      duration.textContent = `${safe.duration_ms} ms`;
      meta.append(duration);
    }
    if (safe.status) {
      const status = this.document.createElement("span");
      status.className = "trace-status";
      status.textContent = safe.status;
      meta.append(status);
    }
    body.append(title, meta);
    item.append(marker, body);
    this.elements.traceList.append(item);
    this.elements.traceEmpty.hidden = true;
    this.elements.traceCount.textContent = String(this.traceCount);
  }

  addRecommendation(data) {
    const item = this.document.createElement("div");
    item.className = "recommendation-item";
    item.textContent = safeRecommendationText(data);
    this.elements.recommendationList.append(item);
    this.elements.recommendationSection.hidden = false;
  }

  handleEvent(event) {
    const data = event.data;
    if (event.type === "status") {
      this.setConnection("running", "Agent 正在执行");
      if (this.answerStatus) {
        this.answerStatus.textContent = "已连接，正在接收事件";
      }
      if (this.answerContent && !this.answerText) {
        this.answerContent.textContent = "已连接，正在检索证据并生成回答…";
      }
    } else if (event.type === "text") {
      if (typeof data.delta === "string") {
        if (data.reset === true) {
          this.answerText = "";
        }
        this.answerText += data.delta;
        this.scheduleAnswerRender();
      }
    } else if (event.type === "reference") {
      if (data.reset === true) {
        this.resetReferences(data);
      } else {
        this.addReference(data);
      }
    } else if (event.type === "recommendation") {
      this.addRecommendation(data);
    } else if (event.type === "trace") {
      this.addTrace(data);
    } else if (event.type === "done") {
      this.flushAnswerRender();
      const status = isNonEmptyString(data.status) ? data.status : "completed";
      this.setConnection("completed", `请求结束 · ${status}`);
      if (this.answerStatus) {
        this.answerStatus.textContent = `终态：${status}`;
      }
      this.announce(`请求已结束，状态 ${status}`);
    } else if (event.type === "error") {
      this.flushAnswerRender();
      const status = isNonEmptyString(data.status) ? data.status : "failed";
      const message = isNonEmptyString(data.message)
        ? data.message
        : "Agent 执行未完成";
      this.setConnection("failed", `请求结束 · ${status}`);
      if (!this.answerText) {
        this.answerText = message;
        this.renderAnswer();
        this.answerArticle.classList.add("is-error");
      }
      if (this.answerStatus) {
        this.answerStatus.textContent = `终态：${status} · ${message}`;
      }
      this.announce(`请求未完成：${message}`);
    }
  }

  setConnection(state, label) {
    this.elements.connectionPill.dataset.state = state;
    this.elements.connectionLabel.textContent = label;
  }

  setRunning(running) {
    this.running = running;
    this.elements.sendButton.disabled = running;
    this.elements.stopButton.disabled = !running;
    this.elements.queryInput.disabled = running;
    this.elements.controlInput.disabled = running;
    for (const button of this.document.querySelectorAll("[data-mode]")) {
      button.disabled = running;
    }
  }

  async startRequest() {
    if (this.running) {
      return;
    }
    const query = this.elements.queryInput.value.trim();
    const controlText = this.elements.controlInput.value.trim();
    if (!query) {
      this.elements.queryInput.focus();
      return;
    }

    this.resetResultPanels();
    this.addUserMessage(query, this.mode === "gap_analysis" ? controlText : "");
    this.elements.queryInput.value = "";
    this.elements.controlInput.value = "";
    this.addAssistantMessage();
    this.requestId = makeRequestId();
    const currentRequestId = this.requestId;
    this.elements.requestLabel.textContent = `请求 ${currentRequestId.slice(0, 8)}`;
    this.setConnection("running", "正在连接事件流");
    this.setRunning(true);

    const session = new StreamSession(currentRequestId, (event) => {
      this.handleEvent(event);
    });
    const parser = new SSEFrameParser((event) => session.accept(event));

    try {
      const response = await fetch("/chat", {
        method: "POST",
        headers: {
          Accept: "text/event-stream",
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          request_id: currentRequestId,
          mode: this.mode,
          query,
          control_text: this.mode === "gap_analysis" ? controlText : "",
        }),
      });
      if (!response.ok) {
        throw new Error(`请求被服务器拒绝（HTTP ${response.status}）`);
      }
      if (!response.body) {
        throw new StreamInterruptedError("response has no readable body");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          break;
        }
        parser.push(decoder.decode(value, { stream: true }));
      }
      parser.push(decoder.decode());
      parser.finish();
      session.finish();
    } catch (error) {
      if (this.requestId !== currentRequestId) {
        return;
      }
      const interrupted = error instanceof StreamInterruptedError;
      const label = interrupted ? "连接中断，未收到终态" : "请求处理失败";
      this.setConnection(interrupted ? "interrupted" : "failed", label);
      if (!this.answerText) {
        this.answerText = interrupted
          ? "事件流意外断开。当前结果不完整，请重新运行。"
          : `无法完成请求：${error.message}`;
        this.renderAnswer();
        this.answerArticle.classList.add("is-error");
      }
      if (this.answerStatus) {
        this.answerStatus.textContent = label;
      }
      this.announce(label);
    } finally {
      if (this.requestId === currentRequestId) {
        this.setRunning(false);
      }
    }
  }

  async stopRequest() {
    if (!this.running || !this.requestId) {
      return;
    }
    const requestId = this.requestId;
    this.elements.stopButton.disabled = true;
    this.elements.stopButton.textContent = "正在停止…";
    this.setConnection("running", "正在请求停止");
    try {
      const response = await fetch(`/tasks/${encodeURIComponent(requestId)}/stop`, {
        method: "POST",
        headers: { Accept: "application/json" },
      });
      if (!response.ok && response.status !== 404) {
        throw new Error(`HTTP ${response.status}`);
      }
      this.announce("已发送停止请求，等待终态事件");
    } catch (_error) {
      this.announce("停止请求发送失败，事件流仍在继续");
      if (this.running) {
        this.elements.stopButton.disabled = false;
      }
    } finally {
      this.elements.stopButton.textContent = "停止";
    }
  }

  scrollTimeline() {
    this.elements.timeline.scrollTop = this.elements.timeline.scrollHeight;
  }

  announce(message) {
    this.elements.announcer.textContent = "";
    globalThis.setTimeout(() => {
      this.elements.announcer.textContent = message;
    }, 20);
  }
}

globalThis.GRCFrontend = Object.freeze({
  SSEFrameParser,
  StreamSession,
  StreamProtocolError,
  StreamInterruptedError,
  renderMarkdownInto,
  runContractSelfTests,
});

if (typeof document !== "undefined") {
  globalThis.addEventListener("DOMContentLoaded", () => {
    new GRCApplication(document);
  });
}
