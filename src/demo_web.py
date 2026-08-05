"""Browser interface for the curated static verification demo."""


DEMO_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta
    name="viewport"
    content="width=device-width, initial-scale=1.0"
  >
  <meta
    name="description"
    content="Static browser demo for a multimodal evidence verification agent."
  >
  <title>Multimodal Evidence Verification Agent</title>

  <style>
    :root {
      color-scheme: light;
      --background: #f4f6f8;
      --surface: #ffffff;
      --surface-soft: #f8fafc;
      --border: #d9e0e7;
      --text: #172033;
      --muted: #667085;
      --primary: #2457d6;
      --primary-soft: #e8efff;
      --supported: #18794e;
      --supported-soft: #e7f6ee;
      --refuted: #b42318;
      --refuted-soft: #feeceb;
      --insufficient: #9a6700;
      --insufficient-soft: #fff4ce;
      --shadow: 0 12px 30px rgba(15, 23, 42, 0.08);
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      min-height: 100vh;
      background: var(--background);
      color: var(--text);
      font-family:
        Inter,
        ui-sans-serif,
        system-ui,
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;
    }

    button,
    input,
    select {
      font: inherit;
    }

    .page-header {
      padding: 34px 32px 28px;
      color: white;
      background:
        radial-gradient(
          circle at top right,
          rgba(255, 255, 255, 0.18),
          transparent 36%
        ),
        linear-gradient(
          120deg,
          #173a8f,
          #2457d6 55%,
          #517ce8
        );
    }

    .header-content {
      width: min(1440px, 100%);
      margin: 0 auto;
    }

    .eyebrow {
      margin: 0 0 10px;
      font-size: 0.78rem;
      font-weight: 700;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      opacity: 0.82;
    }

    h1 {
      max-width: 900px;
      margin: 0;
      font-size: clamp(2rem, 4vw, 3.25rem);
      line-height: 1.08;
    }

    .subtitle {
      max-width: 820px;
      margin: 15px 0 0;
      font-size: 1rem;
      line-height: 1.7;
      opacity: 0.9;
    }

    .header-badges {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 20px;
    }

    .header-badge {
      padding: 7px 11px;
      border: 1px solid rgba(255, 255, 255, 0.26);
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.12);
      font-size: 0.82rem;
      font-weight: 650;
    }

    .layout {
      display: grid;
      grid-template-columns: minmax(250px, 330px) minmax(0, 1fr);
      gap: 22px;
      width: min(1440px, calc(100% - 40px));
      margin: 24px auto 48px;
    }

    .panel {
      border: 1px solid var(--border);
      border-radius: 18px;
      background: var(--surface);
      box-shadow: var(--shadow);
    }

    .sidebar {
      align-self: start;
      position: sticky;
      top: 20px;
      overflow: hidden;
    }

    .sidebar-header {
      padding: 20px;
      border-bottom: 1px solid var(--border);
    }

    .sidebar-header h2 {
      margin: 0;
      font-size: 1.05rem;
    }

    .sidebar-header p {
      margin: 7px 0 0;
      color: var(--muted);
      font-size: 0.86rem;
      line-height: 1.5;
    }

    .example-list {
      display: grid;
      gap: 8px;
      max-height: calc(100vh - 250px);
      padding: 12px;
      overflow-y: auto;
    }

    .example-button {
      width: 100%;
      padding: 13px;
      border: 1px solid transparent;
      border-radius: 12px;
      background: transparent;
      color: inherit;
      text-align: left;
      cursor: pointer;
      transition:
        background 140ms ease,
        border-color 140ms ease,
        transform 140ms ease;
    }

    .example-button:hover {
      border-color: var(--border);
      background: var(--surface-soft);
      transform: translateY(-1px);
    }

    .example-button.active {
      border-color: #a9bff5;
      background: var(--primary-soft);
    }

    .example-id {
      display: block;
      margin-bottom: 5px;
      color: var(--primary);
      font-size: 0.76rem;
      font-weight: 750;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }

    .example-claim {
      display: block;
      font-size: 0.88rem;
      font-weight: 620;
      line-height: 1.45;
    }

    .example-meta {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 8px;
    }

    .mini-badge {
      padding: 4px 7px;
      border-radius: 999px;
      background: #eef2f6;
      color: #475467;
      font-size: 0.7rem;
      font-weight: 700;
    }

    .content {
      min-width: 0;
      overflow: hidden;
    }

    .content-header {
      padding: 22px 24px;
      border-bottom: 1px solid var(--border);
    }

    .content-header-top {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 18px;
    }

    .claim-label {
      margin: 0 0 7px;
      color: var(--muted);
      font-size: 0.75rem;
      font-weight: 750;
      letter-spacing: 0.09em;
      text-transform: uppercase;
    }

    .claim {
      margin: 0;
      font-size: clamp(1.25rem, 2.5vw, 1.8rem);
      line-height: 1.35;
    }

    .label-badge {
      flex: 0 0 auto;
      padding: 8px 12px;
      border-radius: 999px;
      font-size: 0.78rem;
      font-weight: 800;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }

    .label-supported {
      background: var(--supported-soft);
      color: var(--supported);
    }

    .label-refuted {
      background: var(--refuted-soft);
      color: var(--refuted);
    }

    .label-insufficient {
      background: var(--insufficient-soft);
      color: var(--insufficient);
    }

    .metrics {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 18px;
    }

    .metric {
      min-width: 130px;
      padding: 10px 12px;
      border: 1px solid var(--border);
      border-radius: 11px;
      background: var(--surface-soft);
    }

    .metric-name {
      display: block;
      color: var(--muted);
      font-size: 0.7rem;
      font-weight: 700;
      letter-spacing: 0.05em;
      text-transform: uppercase;
    }

    .metric-value {
      display: block;
      margin-top: 4px;
      font-size: 0.92rem;
      font-weight: 750;
    }

    .main-grid {
      display: grid;
      grid-template-columns: minmax(280px, 0.9fr) minmax(320px, 1.1fr);
      gap: 20px;
      padding: 22px;
      align-items: start;
    }

    .image-frame {
      display: flex;
      align-items: center;
      justify-content: center;
      align-self: start;
      position: sticky;
      top: 20px;
      overflow: hidden;
      border: 1px solid var(--border);
      border-radius: 15px;
      background:
        linear-gradient(45deg, #eef1f5 25%, transparent 25%),
        linear-gradient(-45deg, #eef1f5 25%, transparent 25%),
        linear-gradient(45deg, transparent 75%, #eef1f5 75%),
        linear-gradient(-45deg, transparent 75%, #eef1f5 75%);
      background-position:
        0 0,
        0 10px,
        10px -10px,
        -10px 0;
      background-size: 20px 20px;
    }

    .demo-image {
      display: block;
      width: 100%;
      max-height: 590px;
      object-fit: contain;
    }

    .section-stack {
      display: grid;
      align-content: start;
      gap: 18px;
    }

    .section {
      padding: 18px;
      border: 1px solid var(--border);
      border-radius: 14px;
      background: var(--surface-soft);
    }

    .section h3 {
      margin: 0 0 11px;
      font-size: 1rem;
    }

    .section p {
      margin: 0;
      color: #344054;
      font-size: 0.92rem;
      line-height: 1.65;
    }

    .evidence-list,
    .trace-list {
      display: grid;
      gap: 11px;
      margin: 0;
      padding: 0;
      list-style: none;
    }

    .evidence-item {
      padding: 12px;
      border-left: 4px solid #8aa8ef;
      border-radius: 8px;
      background: white;
    }

    .evidence-top {
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      gap: 8px;
      margin-bottom: 6px;
    }

    .evidence-source {
      color: var(--primary);
      font-size: 0.72rem;
      font-weight: 800;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }

    .evidence-relevance {
      color: var(--muted);
      font-size: 0.72rem;
      font-weight: 700;
    }

    .evidence-content {
      color: #344054;
      font-size: 0.87rem;
      line-height: 1.55;
    }

    .trace-item {
      position: relative;
      padding: 13px 13px 13px 42px;
      border: 1px solid var(--border);
      border-radius: 11px;
      background: white;
    }

    .trace-number {
      position: absolute;
      top: 12px;
      left: 12px;
      display: grid;
      place-items: center;
      width: 22px;
      height: 22px;
      border-radius: 50%;
      background: var(--primary);
      color: white;
      font-size: 0.72rem;
      font-weight: 800;
    }

    .trace-name {
      margin: 0;
      color: var(--primary);
      font-size: 0.86rem;
      font-weight: 800;
    }

    .trace-summary {
      margin-top: 5px;
      color: #475467;
      font-size: 0.83rem;
      line-height: 1.5;
    }

    details {
      margin-top: 9px;
    }

    summary {
      color: var(--muted);
      font-size: 0.76rem;
      font-weight: 700;
      cursor: pointer;
    }

    pre {
      max-width: 100%;
      margin: 9px 0 0;
      padding: 10px;
      overflow-x: auto;
      border-radius: 8px;
      background: #111827;
      color: #dbeafe;
      font-size: 0.72rem;
      line-height: 1.45;
      white-space: pre-wrap;
      word-break: break-word;
    }

    .status {
      padding: 32px;
      color: var(--muted);
      text-align: center;
    }

    .error {
      color: var(--refuted);
    }

    .page-footer {
      width: min(1440px, calc(100% - 40px));
      margin: -22px auto 36px;
      color: var(--muted);
      font-size: 0.78rem;
      line-height: 1.6;
      text-align: center;
    }

    @media (max-width: 980px) {
      .layout {
        grid-template-columns: 1fr;
      }

      .sidebar {
        position: static;
      }

      .example-list {
        grid-template-columns: repeat(2, minmax(0, 1fr));
        max-height: none;
      }

      .main-grid {
        grid-template-columns: 1fr;
      }

      .image-frame {
        position: static;
      }
    }

    @media (max-width: 640px) {
      .page-header {
        padding: 28px 20px 24px;
      }

      .layout {
        width: min(100% - 24px, 1440px);
        margin-top: 14px;
      }

      .example-list {
        grid-template-columns: 1fr;
      }

      .content-header,
      .main-grid {
        padding: 17px;
      }

      .content-header-top {
        display: grid;
      }

      .label-badge {
        justify-self: start;
      }
    }
  </style>
</head>

<body>
  <header class="page-header">
    <div class="header-content">
      <p class="eyebrow">Static portfolio demo</p>

      <h1>
        Multimodal Evidence Verification Agent
      </h1>

      <p class="subtitle">
        A tool-using agent that verifies textual claims against
        image evidence through dynamic routing, visual inspection,
        optional OCR, structured evidence selection, and a
        verification reasoner.
      </p>

      <div class="header-badges">
        <span class="header-badge">6 curated examples</span>
        <span class="header-badge">3 verification labels</span>
        <span class="header-badge">Visual + OCR routes</span>
        <span class="header-badge">No live model calls</span>
      </div>
    </div>
  </header>

  <main class="layout">
    <aside class="panel sidebar">
      <div class="sidebar-header">
        <h2>Verification examples</h2>

        <p>
          Select an example to inspect its saved result,
          evidence, and tool-use trace.
        </p>
      </div>

      <div
        id="example-list"
        class="example-list"
      >
        <div class="status">
          Loading examples…
        </div>
      </div>
    </aside>

    <section class="panel content">
      <div id="content-status" class="status">
        Loading verification result…
      </div>

      <div id="result-content" hidden>
        <div class="content-header">
          <div class="content-header-top">
            <div>
              <p class="claim-label">Claim</p>
              <h2 id="claim" class="claim"></h2>
            </div>

            <span
              id="label-badge"
              class="label-badge"
            ></span>
          </div>

          <div class="metrics">
            <div class="metric">
              <span class="metric-name">Confidence</span>
              <span
                id="confidence"
                class="metric-value"
              ></span>
            </div>

            <div class="metric">
              <span class="metric-name">Tool route</span>
              <span
                id="route"
                class="metric-value"
              ></span>
            </div>

            <div class="metric">
              <span class="metric-name">Category</span>
              <span
                id="category"
                class="metric-value"
              ></span>
            </div>
          </div>
        </div>

        <div class="main-grid">
          <div class="image-frame">
            <img
              id="demo-image"
              class="demo-image"
              alt=""
            >
          </div>

          <div class="section-stack">
            <section class="section">
              <h3>Rationale</h3>
              <p id="rationale"></p>
            </section>

            <section class="section">
              <h3>Selected evidence</h3>
              <ul
                id="evidence-list"
                class="evidence-list"
              ></ul>
            </section>

            <section class="section">
              <h3>Tool trace</h3>
              <ol
                id="tool-trace"
                class="trace-list"
              ></ol>
            </section>
          </div>
        </div>
      </div>
    </section>
  </main>

  <footer class="page-footer">
    Results on this page are precomputed from a small curated
    functional dataset. They demonstrate system behavior and
    should not be interpreted as a generalization benchmark.
  </footer>

  <script>
    const exampleList = document.getElementById(
      "example-list"
    );

    const contentStatus = document.getElementById(
      "content-status"
    );

    const resultContent = document.getElementById(
      "result-content"
    );

    async function requestJson(url) {
      const response = await fetch(url);

      let payload = null;

      try {
        payload = await response.json();
      } catch (error) {
        throw new Error(
          `Invalid server response (${response.status}).`
        );
      }

      if (!response.ok) {
        const message = payload.detail
          || `Request failed (${response.status}).`;

        throw new Error(message);
      }

      return payload;
    }

    function buildImageUrl(imagePath) {
      const normalizedPath = String(
        imagePath || ""
      ).replaceAll("\\", "/");

      const prefix = "data/images/";

      if (!normalizedPath.startsWith(prefix)) {
        return "";
      }

      const relativePath = normalizedPath.slice(
        prefix.length
      );

      return "/demo-images/"
        + relativePath
          .split("/")
          .map(encodeURIComponent)
          .join("/");
    }

    function formatCategory(category) {
      return String(category || "")
        .replaceAll("_", " ")
        .replace(/\b\w/g, character => (
          character.toUpperCase()
        ));
    }

    function setActiveExample(exampleId) {
      document
        .querySelectorAll(".example-button")
        .forEach(button => {
          button.classList.toggle(
            "active",
            button.dataset.exampleId === exampleId
          );
        });
    }

    function renderEvidence(evidenceItems) {
      const evidenceList = document.getElementById(
        "evidence-list"
      );

      evidenceList.replaceChildren();

      if (!Array.isArray(evidenceItems)
          || evidenceItems.length === 0) {
        const emptyItem = document.createElement("li");
        emptyItem.className = "evidence-item";
        emptyItem.textContent = "No evidence was selected.";
        evidenceList.appendChild(emptyItem);
        return;
      }

      evidenceItems.forEach(item => {
        const listItem = document.createElement("li");
        listItem.className = "evidence-item";

        const top = document.createElement("div");
        top.className = "evidence-top";

        const source = document.createElement("span");
        source.className = "evidence-source";
        source.textContent = [
          item.modality,
          item.source,
        ].filter(Boolean).join(" · ");

        const relevance = document.createElement("span");
        relevance.className = "evidence-relevance";

        const relevanceValue = Number(
          item.relevance
        );

        relevance.textContent = Number.isFinite(
          relevanceValue
        )
          ? `Relevance ${Math.round(
              relevanceValue * 100
            )}%`
          : "";

        const content = document.createElement("div");
        content.className = "evidence-content";
        content.textContent = item.content || "";

        top.append(source, relevance);
        listItem.append(top, content);
        evidenceList.appendChild(listItem);
      });
    }

    function renderToolTrace(toolTrace) {
      const traceList = document.getElementById(
        "tool-trace"
      );

      traceList.replaceChildren();

      if (!Array.isArray(toolTrace)
          || toolTrace.length === 0) {
        const emptyItem = document.createElement("li");
        emptyItem.className = "trace-item";
        emptyItem.textContent = "No tool trace is available.";
        traceList.appendChild(emptyItem);
        return;
      }

      toolTrace.forEach((toolCall, index) => {
        const listItem = document.createElement("li");
        listItem.className = "trace-item";

        const number = document.createElement("span");
        number.className = "trace-number";
        number.textContent = String(index + 1);

        const name = document.createElement("p");
        name.className = "trace-name";
        name.textContent = toolCall.tool_name || "tool";

        const summary = document.createElement("div");
        summary.className = "trace-summary";
        summary.textContent = (
          toolCall.tool_output_summary
          || "No output summary."
        );

        const details = document.createElement("details");
        const detailsSummary = document.createElement(
          "summary"
        );

        detailsSummary.textContent = "View tool input";

        const pre = document.createElement("pre");
        pre.textContent = JSON.stringify(
          toolCall.tool_input || {},
          null,
          2
        );

        details.append(detailsSummary, pre);

        listItem.append(
          number,
          name,
          summary,
          details
        );

        traceList.appendChild(listItem);
      });
    }

    function renderResult(result) {
      const claim = document.getElementById("claim");
      const labelBadge = document.getElementById(
        "label-badge"
      );

      const confidence = document.getElementById(
        "confidence"
      );

      const route = document.getElementById("route");
      const category = document.getElementById(
        "category"
      );

      const rationale = document.getElementById(
        "rationale"
      );

      const demoImage = document.getElementById(
        "demo-image"
      );

      const label = String(
        result.predicted_label || "insufficient"
      ).toLowerCase();

      claim.textContent = result.claim || "";

      labelBadge.textContent = label;
      labelBadge.className = (
        `label-badge label-${label}`
      );

      confidence.textContent = (
        `${Math.round(
          Number(result.confidence || 0) * 100
        )}%`
      );

      route.textContent = result.predicted_use_ocr
        ? "Visual + OCR"
        : "Visual only";

      category.textContent = formatCategory(
        result.category
      );

      rationale.textContent = result.rationale || "";

      demoImage.src = buildImageUrl(
        result.image_path
      );

      demoImage.alt = (
        `Evidence image for claim: ${result.claim || ""}`
      );

      renderEvidence(
        result.evidence
      );

      renderToolTrace(
        result.tool_trace
      );

      contentStatus.hidden = true;
      resultContent.hidden = false;
    }

    async function loadExample(exampleId) {
      setActiveExample(exampleId);

      contentStatus.hidden = false;
      contentStatus.classList.remove("error");
      contentStatus.textContent = (
        "Loading verification result…"
      );

      resultContent.hidden = true;

      try {
        const result = await requestJson(
          `/demo/examples/${encodeURIComponent(
            exampleId
          )}`
        );

        renderResult(result);
      } catch (error) {
        contentStatus.hidden = false;
        contentStatus.classList.add("error");
        contentStatus.textContent = error.message;
      }
    }

    function createExampleButton(summary) {
      const button = document.createElement("button");

      button.type = "button";
      button.className = "example-button";
      button.dataset.exampleId = summary.example_id;

      const exampleId = document.createElement("span");
      exampleId.className = "example-id";
      exampleId.textContent = summary.example_id;

      const claim = document.createElement("span");
      claim.className = "example-claim";
      claim.textContent = summary.claim;

      const meta = document.createElement("span");
      meta.className = "example-meta";

      const label = document.createElement("span");
      label.className = "mini-badge";
      label.textContent = summary.predicted_label;

      const route = document.createElement("span");
      route.className = "mini-badge";
      route.textContent = summary.predicted_use_ocr
        ? "OCR"
        : "Visual";

      meta.append(label, route);

      button.append(
        exampleId,
        claim,
        meta
      );

      button.addEventListener(
        "click",
        () => loadExample(
          summary.example_id
        )
      );

      return button;
    }

    async function initializeDemo() {
      try {
        const payload = await requestJson(
          "/demo/examples"
        );

        exampleList.replaceChildren();

        payload.examples.forEach(summary => {
          exampleList.appendChild(
            createExampleButton(summary)
          );
        });

        if (payload.examples.length === 0) {
          exampleList.innerHTML = (
            '<div class="status">'
            + "No demo examples are available."
            + "</div>"
          );

          contentStatus.textContent = (
            "No verification result is available."
          );

          return;
        }

        await loadExample(
          payload.examples[0].example_id
        );
      } catch (error) {
        exampleList.innerHTML = (
          '<div class="status error"></div>'
        );

        exampleList.firstElementChild.textContent = (
          error.message
        );

        contentStatus.classList.add("error");
        contentStatus.textContent = error.message;
      }
    }

    initializeDemo();
  </script>
</body>
</html>
"""
