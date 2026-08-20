(() => {
  "use strict";

  const $ = (sel) => document.querySelector(sel);

  const els = {
    prompt: $("#prompt"),
    promptCount: $("#prompt-count"),
    temperature: $("#temperature"),
    temperatureOut: $("#temperature-out"),
    topK: $("#top_k"),
    topKOut: $("#top_k-out"),
    tokens: $("#max_new_tokens"),
    tokensOut: $("#max_new_tokens-out"),
    tempDial: $("#temp-dial"),
    topkDial: $("#topk-dial"),
    tokensDial: $("#tokens-dial"),
    generateBtn: $("#generate-btn"),
    generateLabel: $("#generate-label"),
    errorMsg: $("#error-msg"),
    output: $("#output"),
    outputMeta: $("#output-meta"),
    metaStep: $("#meta-step"),
    metaLoss: $("#meta-loss"),
    statusDot: $("#status-dot"),
    statusText: $("#status-text"),
    specs: $("#specs"),
  };

  const MIN_ANGLE = -120;
  const MAX_ANGLE = 120;

  function angleFor(input) {
    const min = parseFloat(input.min);
    const max = parseFloat(input.max);
    const val = parseFloat(input.value);
    const t = (val - min) / (max - min);
    return MIN_ANGLE + t * (MAX_ANGLE - MIN_ANGLE);
  }

  function wireDial(input, dialEl, outEl, formatter) {
    const update = () => {
      dialEl.style.setProperty("--angle", `${angleFor(input)}deg`);
      outEl.textContent = formatter(parseFloat(input.value));
    };
    input.addEventListener("input", update);
    update();
  }

  wireDial(els.temperature, els.tempDial, els.temperatureOut, (v) => v.toFixed(2));
  wireDial(els.topK, els.topkDial, els.topKOut, (v) => (v === 0 ? "off" : String(v)));
  wireDial(els.tokens, els.tokensDial, els.tokensOut, (v) => String(v));

  els.prompt.addEventListener("input", () => {
    els.promptCount.textContent = `${els.prompt.value.length} / 300`;
  });

  function setStatus(state, text) {
    els.statusDot.classList.remove("ok", "err");
    if (state) els.statusDot.classList.add(state);
    els.statusText.textContent = text;
  }

  function showError(message) {
    els.errorMsg.textContent = message;
    els.errorMsg.hidden = false;
  }

  function clearError() {
    els.errorMsg.hidden = true;
    els.errorMsg.textContent = "";
  }

  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }

  async function checkHealth() {
    try {
      const res = await fetch("/health");
      if (!res.ok) throw new Error(`status ${res.status}`);
      const data = await res.json();
      if (data.model_loaded) {
        setStatus("ok", "oven preheated — model loaded");
      } else {
        setStatus(null, "warming up the oven\u2026");
      }
    } catch (e) {
      setStatus("err", "kitchen's unreachable right now");
    }
  }

  async function loadInfo() {
    try {
      const res = await fetch("/api/info");
      if (!res.ok) throw new Error(`status ${res.status}`);
      const data = await res.json();
      const cfg = data.config || {};
      const rows = els.specs.querySelectorAll("dd");
      const params = data.total_params
        ? (data.total_params / 1e6).toFixed(2) + "M"
        : "\u2014";
      if (rows[0]) rows[0].textContent = params;
      if (rows[1]) rows[1].textContent = cfg.n_layer != null ? `${cfg.n_layer} (× ${cfg.n_head} heads)` : "\u2014";
      if (rows[2]) rows[2].textContent = cfg.block_size != null ? `${cfg.block_size} tokens` : "\u2014";
      if (rows[3]) rows[3].textContent = cfg.vocab_size != null ? cfg.vocab_size.toLocaleString() : "\u2014";
    } catch (e) {
      // Non-critical: the about section just keeps its loading placeholders.
    }
  }

  async function generate() {
    clearError();
    els.generateBtn.disabled = true;
    els.generateBtn.classList.add("working");
    els.generateLabel.textContent = "working\u2026";

    const payload = {
      prompt: els.prompt.value,
      temperature: parseFloat(els.temperature.value),
      top_k: parseInt(els.topK.value, 10),
      max_new_tokens: parseInt(els.tokens.value, 10),
    };

    try {
      const res = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const data = await res.json();

      if (!res.ok) {
        const detail = Array.isArray(data.detail)
          ? data.detail.map((d) => d.msg).join("; ")
          : data.detail || `Request failed (${res.status})`;
        throw new Error(detail);
      }

      const promptText = data.prompt || "";
      const fullText = data.generated_text || "";
      const rest = fullText.startsWith(promptText) ? fullText.slice(promptText.length) : fullText;

      els.output.innerHTML =
        `<span class="prompt-echo">${escapeHtml(promptText)}</span>${escapeHtml(rest)}`;

      if (data.meta && (data.meta.step != null || data.meta.val_loss != null)) {
        els.metaStep.textContent = data.meta.step != null ? `checkpoint step ${data.meta.step}` : "";
        els.metaLoss.textContent =
          data.meta.val_loss != null ? `val loss ${Number(data.meta.val_loss).toFixed(3)}` : "";
        els.outputMeta.hidden = false;
      }

      setStatus("ok", "oven preheated — model loaded");
    } catch (e) {
      showError(e.message || "Something boiled over. Try again.");
    } finally {
      els.generateBtn.disabled = false;
      els.generateBtn.classList.remove("working");
      els.generateLabel.textContent = "activate";
    }
  }

  els.generateBtn.addEventListener("click", generate);
  els.prompt.addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") generate();
  });

  checkHealth();
  loadInfo();
})();
