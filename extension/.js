(() => {
  // content.js
  var ortReady = false;
  var inferenceRunning = false;
  var inferenceQueue = [];
  function waitForORT() {
    return new Promise((resolve, reject) => {
      const start = Date.now();
      const check = () => {
        if (typeof globalThis.ort !== "undefined" && globalThis.ort.InferenceSession) {
          globalThis.ort.env.wasm.wasmPaths = chrome.runtime.getURL("wasm/");
          globalThis.ort.env.wasm.numThreads = 1;
          globalThis.ort.env.wasm.proxy = false;
          console.log("AISeeYou: ORT found on globalThis");
          resolve();
          return;
        }
        if (Date.now() - start > 15e3) {
          reject(new Error("ORT not found after 15s \u2014 is ort.min.js loaded before content.bundle.js in manifest.json?"));
          return;
        }
        setTimeout(check, 100);
      };
      check();
    });
  }
  var TOKENIZER = {
    vocab: null,
    maxLen: 128,
    CLS: 101,
    SEP: 102,
    PAD: 0,
    UNK: 100,
    async load() {
      const url = chrome.runtime.getURL("onnx_model/tokenizer.json");
      const res = await fetch(url);
      if (!res.ok) throw new Error("tokenizer.json fetch failed: " + res.status);
      const tj = await res.json();
      this.vocab = tj.model.vocab;
      console.log("AISeeYou: BERT tokenizer loaded, vocab:", Object.keys(this.vocab).length);
    },
    _isWhitespace(cp) {
      return cp === 32 || cp === 9 || cp === 10 || cp === 13;
    },
    _isControl(cp) {
      if (cp === 9 || cp === 10 || cp === 13) return false;
      return cp >= 0 && cp <= 31 || cp >= 127 && cp <= 159;
    },
    _isChineseCJK(cp) {
      return cp >= 19968 && cp <= 40959 || cp >= 13312 && cp <= 19903 || cp >= 63744 && cp <= 64255;
    },
    _isPunct(cp) {
      if (cp >= 33 && cp <= 47 || cp >= 58 && cp <= 64 || cp >= 91 && cp <= 96 || cp >= 123 && cp <= 126) return true;
      return /\p{P}|\p{S}/u.test(String.fromCodePoint(cp));
    },
    _normalize(text) {
      text = text.toLowerCase();
      let out = "";
      for (const ch of text) {
        const cp = ch.codePointAt(0);
        if (cp === 0 || cp === 65533) continue;
        if (this._isControl(cp)) continue;
        if (this._isWhitespace(cp)) {
          out += " ";
          continue;
        }
        if (this._isChineseCJK(cp)) {
          out += " " + ch + " ";
          continue;
        }
        out += ch;
      }
      return out;
    },
    _preTokenize(text) {
      const tokens = [];
      for (const word of text.split(/\s+/).filter(Boolean)) {
        let cur = "";
        for (const ch of word) {
          if (this._isPunct(ch.codePointAt(0))) {
            if (cur) {
              tokens.push(cur);
              cur = "";
            }
            tokens.push(ch);
          } else {
            cur += ch;
          }
        }
        if (cur) tokens.push(cur);
      }
      return tokens;
    },
    _wordpiece(word) {
      if (this.vocab[word] !== void 0) return [word];
      const chars = [...word];
      const out = [];
      let start = 0;
      while (start < chars.length) {
        let end = chars.length, found = null;
        while (start < end) {
          const sub = (start === 0 ? "" : "##") + chars.slice(start, end).join("");
          if (this.vocab[sub] !== void 0) {
            found = sub;
            break;
          }
          end--;
        }
        if (!found) return ["[UNK]"];
        out.push(found);
        start = end;
      }
      return out;
    },
    preprocess(text) {
      if (!this.vocab) throw new Error("Tokenizer not loaded");
      const words = this._preTokenize(this._normalize(text));
      const contentIds = [];
      for (const word of words) {
        for (const sw of this._wordpiece(word)) {
          contentIds.push(this.vocab[sw] ?? this.UNK);
          if (contentIds.length >= this.maxLen - 2) break;
        }
        if (contentIds.length >= this.maxLen - 2) break;
      }
      const ids = [this.CLS, ...contentIds, this.SEP];
      const seqLen = ids.length;
      while (ids.length < this.maxLen) ids.push(this.PAD);
      const attn = ids.map((_, i) => i < seqLen ? 1 : 0);
      const typeIds = new Array(this.maxLen).fill(0);
      console.log("AISeeYou: tokenized", seqLen, "tokens, padded to", this.maxLen);
      return {
        input_ids: new globalThis.ort.Tensor("int64", BigInt64Array.from(ids.map(BigInt)), [1, this.maxLen]),
        attention_mask: new globalThis.ort.Tensor("int64", BigInt64Array.from(attn.map(BigInt)), [1, this.maxLen]),
        token_type_ids: new globalThis.ort.Tensor("int64", BigInt64Array.from(typeIds.map(BigInt)), [1, this.maxLen])
      };
    }
  };
  var session = null;
  var logs = [];
  var MAX_WORDS = 800;
  var settings = {
    minWords: 30,
    threshold: 0.8,
    mode: "highlight"
  };
  var scannedHashes = /* @__PURE__ */ new Map();
  var PLATFORMS = [
    {
      name: "X",
      container: "article[data-testId='tweet']",
      textNode: "[data-testid='tweetText']"
    },
    {
      name: "Facebook",
      container: "[data-ad-preview='message']",
      textNode: null
    },
    {
      name: "Facebook2",
      container: "div[data-testid='post_message']",
      textNode: null
    },
    {
      name: "Threads",
      container: "article[role='article']",
      textNode: "div[dir='auto']"
    },
    {
      name: "LinkedIn",
      container: ".feed-shared-update-v2",
      textNode: ".feed-shared-update-v2__description"
    },
    {
      name: "Reddit",
      container: "shreddit-post",
      textNode: "[slot='text-body']"
    },
    {
      name: "Generic-article",
      container: "article",
      textNode: null
    },
    {
      name: "Generic-blockquote",
      container: "blockquote",
      textNode: null
    }
  ];
  function simpleHash(text) {
    let h = 0;
    for (let i = 0; i < text.length; i++) {
      h = (h << 5) - h + text.charCodeAt(i);
      h |= 0;
    }
    return h.toString();
  }
  function isVisible(el) {
    if (!el) return false;
    const rect = el.getBoundingClientRect();
    return rect.width > 0 && rect.height > 0;
  }
  function softmax(arr) {
    const max = Math.max(...arr);
    const exps = arr.map((x) => Math.exp(x - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map((x) => x / sum);
  }
  function resolveBestTextEl(containerEl, textSelector) {
    if (!textSelector) return containerEl;
    const candidates = [...containerEl.querySelectorAll(textSelector)];
    if (candidates.length === 0) return null;
    return candidates.reduce(
      (best, el) => (el.innerText?.length ?? 0) > (best.innerText?.length ?? 0) ? el : best
    );
  }
  async function loadModel() {
    const modelUrl = chrome.runtime.getURL("onnx_model/model.onnx");
    console.log("AISeeYou: loading model from", modelUrl);
    const res = await fetch(modelUrl);
    if (!res.ok) throw new Error("Model fetch failed: " + res.status);
    const modelBuffer = await res.arrayBuffer();
    console.log("AISeeYou: model buffer size:", modelBuffer.byteLength);
    session = await globalThis.ort.InferenceSession.create(modelBuffer, {
      executionProviders: ["wasm"]
    });
    console.log("AISeeYou: model loaded, inputs:", session.inputNames, "outputs:", session.outputNames);
  }
  function runNextInQueue() {
    if (inferenceRunning || inferenceQueue.length === 0) return;
    inferenceRunning = true;
    const { text, resolve, reject } = inferenceQueue.shift();
    const inputs = TOKENIZER.preprocess(text);
    session.run(inputs).then((output) => {
      const key = Object.keys(output)[0];
      const raw = Array.from(output[key].data);
      console.log("AISeeYou: raw output (" + raw.length + " values):", raw);
      let score;
      if (raw.length === 1) {
        score = raw[0];
      } else {
        const probs = softmax(raw);
        console.log("AISeeYou: probs[0] (human?):", probs[0], "probs[1] (AI?):", probs[1]);
        score = probs[1];
      }
      console.log("AISeeYou: final score:", score);
      inferenceRunning = false;
      resolve(score);
      runNextInQueue();
    }).catch((err) => {
      console.error("AISeeYou: session.run failed:", err);
      inferenceRunning = false;
      reject(err);
      runNextInQueue();
    });
  }
  function predict(text) {
    return new Promise((resolve, reject) => {
      inferenceQueue.push({ text, resolve, reject });
      runNextInQueue();
    });
  }
  function highlightElement(containerEl, textEl, score) {
    if (!containerEl.isConnected) return;
    if (containerEl.dataset.aiDone === "true") return;
    containerEl.dataset.aiDone = "true";
    const mid = settings.threshold - 0.15;
    const color = score > settings.threshold ? "rgba(255,0,0,0.35)" : score > mid ? "rgba(255,165,0,0.30)" : "rgba(0,200,0,0.20)";
    const badgeColor = score > settings.threshold ? "red" : score > mid ? "orange" : "green";
    containerEl.style.setProperty("outline", "2px solid " + badgeColor, "important");
    containerEl.style.setProperty("background", color, "important");
    if (containerEl.querySelector(".ai-detector-badge")) return;
    const badge = document.createElement("div");
    badge.className = "ai-detector-badge";
    badge.textContent = "AI " + (score * 100).toFixed(1) + "%";
    badge.setAttribute(
      "style",
      "display:block !important;width:fit-content !important;margin:4px 0 0 0 !important;padding:2px 8px !important;border-radius:4px !important;font-size:11px !important;font-weight:bold !important;font-family:sans-serif !important;color:white !important;background:" + badgeColor + " !important;line-height:1.6 !important;pointer-events:none !important;z-index:9999 !important;position:relative !important;"
    );
    containerEl.appendChild(badge);
  }
  function blockElement(containerEl) {
    if (!containerEl.isConnected) return;
    if (containerEl.dataset.aiBlocked === "true") return;
    containerEl.dataset.aiBlocked = "true";
    if (getComputedStyle(containerEl).position === "static") {
      containerEl.style.setProperty("position", "relative", "important");
    }
    const overlay = document.createElement("div");
    overlay.setAttribute(
      "style",
      "position:absolute !important;inset:0 !important;background:rgba(10,10,10,0.88) !important;display:flex !important;align-items:center !important;justify-content:center !important;z-index:9999 !important;border-radius:4px !important;"
    );
    const warning = document.createElement("div");
    warning.textContent = "This content has been blocked \u2014 high probability of being AI generated.";
    warning.setAttribute(
      "style",
      "padding:12px 16px !important;background:rgba(255,0,0,0.12) !important;border:1px solid red !important;border-radius:5px !important;font-size:12px !important;color:red !important;text-align:center !important;max-width:320px !important;"
    );
    overlay.appendChild(warning);
    containerEl.appendChild(overlay);
  }
  async function scanContainer(containerEl, textSelector, platformName) {
    if (!containerEl) return;
    if (!isVisible(containerEl)) return;
    if (containerEl.dataset.aiBlocked === "true") return;
    if (containerEl.dataset.aiQueued === "true") return;
    containerEl.dataset.aiQueued = "true";
    const textEl = resolveBestTextEl(containerEl, textSelector);
    if (!textEl) {
      console.warn("AISeeYou:", platformName, "\u2014 no text element, selector:", textSelector);
      return;
    }
    const text = textEl.innerText?.trim();
    if (!text) return;
    const words = text.split(/\s+/).length;
    console.log("AISeeYou:", platformName, "\u2014", words, "words");
    if (words < settings.minWords || words > MAX_WORDS) return;
    const hash = simpleHash(text);
    const now = Date.now();
    const prev = scannedHashes.get(hash);
    if (prev && now - prev < 15e3) return;
    scannedHashes.set(hash, now);
    try {
      const score = await predict(text);
      console.log("AISeeYou:", platformName, "\u2014 score:", score);
      logs.push({ platform: platformName, text: text.slice(0, 200), words, score, time: Date.now() });
      if (settings.mode === "block" && score > settings.threshold) {
        blockElement(containerEl);
      } else {
        highlightElement(containerEl, textEl, score);
      }
    } catch (err) {
      console.error("AISeeYou: predict failed:", err);
    }
  }
  function scanPage() {
    if (!ortReady) return;
    PLATFORMS.forEach(({ container, textNode, name }) => {
      const matches = document.querySelectorAll(container);
      if (matches.length > 0) {
        console.log("AISeeYou:", name, "\u2014 matched", matches.length);
      }
      matches.forEach((el) => scanContainer(el, textNode, name));
    });
  }
  function loadSettings(cb) {
    if (!chrome?.storage?.sync) {
      if (cb) cb();
      return;
    }
    chrome.storage.sync.get(
      ["minWords", "mode", "threshold"],
      (data) => {
        settings.minWords = data?.minWords ?? 30;
        settings.mode = data?.mode ?? "highlight";
        settings.threshold = data?.threshold ?? 0.8;
        if (cb) cb();
      }
    );
  }
  async function bootstrap() {
    try {
      console.log("AISeeYou: waiting for ORT...");
      await waitForORT();
      console.log("AISeeYou: ORT ready");
      await TOKENIZER.load();
      await loadModel();
      await new Promise((resolve) => loadSettings(resolve));
      ortReady = true;
      console.log("AISeeYou: bootstrap complete");
      scanPage();
    } catch (err) {
      console.error("AISeeYou: bootstrap failed:", err);
    }
  }
  bootstrap();
  var running = false;
  var observer = new MutationObserver(() => {
    if (running) return;
    running = true;
    setTimeout(() => {
      scanPage();
      running = false;
    }, 1200);
  });
  observer.observe(document.body, { childList: true, subtree: true });
  chrome.runtime.onMessage.addListener((msg) => {
    if (msg.action === "exportLogs") {
      const blob = new Blob(
        [JSON.stringify(logs, null, 2)],
        { type: "application/json" }
      );
      const url = URL.createObjectURL(blob);
      chrome.runtime.sendMessage({ action: "downloadLogs", url });
    }
    if (msg.action === "settingsChanged") {
      loadSettings();
    }
  });
})();
