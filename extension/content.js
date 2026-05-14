// ORT SETUP

// ort is imported at the top of this file via:
//   import * as ort from "onnxruntime-web";
// esbuild bundles it. WASM files must be in extension/wasm/ and listed
// in manifest.json web_accessible_resources.
// ort is NOT imported here — esbuild breaks ORT's internal WASM loading
// when it bundles it. Instead, ort.min.js is listed before content.bundle.js
// in manifest.json and exposes `ort` on globalThis automatically.
// WASM path is configured once ORT is confirmed ready in waitForORT().

let ortReady = false;
let inferenceRunning = false;
const inferenceQueue = [];

const platformCooldown = new Map();

function canScan(name, delay = 3000) {
    const now = Date.now();
    const last = platformCooldown.get(name) || 0;
    if (now - last < delay) return false;
    platformCooldown.set(name, now);
    return true;
}

function waitForORT() {
    return new Promise((resolve, reject) => {
        const start = Date.now();
        const check = () => {
            if (typeof globalThis.ort !== "undefined" && globalThis.ort.InferenceSession) {
                // Configure WASM paths now that we know ort is present
                globalThis.ort.env.wasm.wasmPaths = chrome.runtime.getURL("wasm/");
                globalThis.ort.env.wasm.numThreads = 1;
                globalThis.ort.env.wasm.proxy = false;
                console.log("AISeeYou: ORT found on globalThis");
                resolve();
                return;
            }
            if (Date.now() - start > 15000) {
                reject(new Error("ORT not found after 15s — is ort.min.js loaded before content.bundle.js in manifest.json?"));
                return;
            }
            setTimeout(check, 100);
        };
        check();
    });
}

// TOKENIZER
// BERT WordPiece — inlined so no external library or load-order dependency.
// Matches tokenizer.json exactly: BertNormalizer + BertPreTokenizer +
// WordPiece, sequence [CLS] tokens [SEP] padded to 128, outputs int64.

const TOKENIZER = {

    vocab: null,
    maxLen: 128,
    CLS: 101, SEP: 102, PAD: 0, UNK: 100,

    async load() {
        const url = chrome.runtime.getURL("onnx_model/tokenizer.json");
        const res = await fetch(url);
        if (!res.ok) throw new Error("tokenizer.json fetch failed: " + res.status);
        const tj = await res.json();
        this.vocab = tj.model.vocab;
        console.log("AISeeYou: BERT tokenizer loaded, vocab:", Object.keys(this.vocab).length);
    },

    _isWhitespace(cp) {
        return cp === 0x20 || cp === 0x09 || cp === 0x0a || cp === 0x0d;
    },

    _isControl(cp) {
        if (cp === 0x09 || cp === 0x0a || cp === 0x0d) return false;
        return (cp >= 0x00 && cp <= 0x1f) || (cp >= 0x7f && cp <= 0x9f);
    },

    _isChineseCJK(cp) {
        return (cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0x3400 && cp <= 0x4DBF) ||
            (cp >= 0xF900 && cp <= 0xFAFF);
    },

    _isPunct(cp) {
        if ((cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) ||
            (cp >= 91 && cp <= 96) || (cp >= 123 && cp <= 126)) return true;
        return /\p{P}|\p{S}/u.test(String.fromCodePoint(cp));
    },

    _normalize(text) {
        text = text.toLowerCase();
        let out = "";
        for (const ch of text) {
            const cp = ch.codePointAt(0);
            if (cp === 0 || cp === 0xfffd) continue;
            if (this._isControl(cp)) continue;
            if (this._isWhitespace(cp)) { out += " "; continue; }
            if (this._isChineseCJK(cp)) { out += " " + ch + " "; continue; }
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
                    if (cur) { tokens.push(cur); cur = ""; }
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
        if (this.vocab[word] !== undefined) return [word];
        const chars = [...word];
        const out = [];
        let start = 0;
        while (start < chars.length) {
            let end = chars.length, found = null;
            while (start < end) {
                const sub = (start === 0 ? "" : "##") + chars.slice(start, end).join("");
                if (this.vocab[sub] !== undefined) { found = sub; break; }
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
            attention_mask: new globalThis.ort.Tensor("int64", BigInt64Array.from(attn.map(BigInt)), [1, this.maxLen])//,
            // token_type_ids: new globalThis.ort.Tensor("int64", BigInt64Array.from(typeIds.map(BigInt)), [1, this.maxLen])
        };
    }
};


// STATE

let session = null;
const logs = [];
const MAX_WORDS = 800;

const settings = {
    minWords: 30,
    threshold: 0.80,
    mode: "highlight",
    showHuman: true
};

const scannedHashes = new Map();

// PLATFORMS

const PLATFORMS = [
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
    }
];

// UTILS

function simpleHash(text) {
    let h = 0;
    for (let i = 0; i < text.length; i++) {
        h = ((h << 5) - h) + text.charCodeAt(i);
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
    const exps = arr.map(x => Math.exp(x - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map(x => x / sum);
}

function resolveBestTextEl(containerEl, textSelector) {

    if (!textSelector) return containerEl;

    const candidates = [...containerEl.querySelectorAll(textSelector)];

    if (candidates.length === 0) return null;

    return candidates.reduce((best, el) =>
        (el.innerText?.length ?? 0) > (best.innerText?.length ?? 0) ? el : best
    );
}

// MODEL LOAD

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

// INFERENCE QUEUE
// ORT wasm only runs one session.run() at a time. A proper explicit queue
// (not a promise chain) is used so resolved promises are released from
// memory and don't accumulate across the page session.

function runNextInQueue() {

    if (inferenceRunning || inferenceQueue.length === 0) return;

    inferenceRunning = true;

    const { text, resolve, reject } = inferenceQueue.shift();

    const inputs = TOKENIZER.preprocess(text);

    session.run(inputs)
        .then(output => {

            const key = Object.keys(output)[0];
            const raw = Array.from(output[key].data);

            console.log("AISeeYou: raw output (" + raw.length + " values):", raw);

            let score;

            if (raw.length === 1) {
                // Single sigmoid output — value IS the AI probability
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
        })
        .catch(err => {
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

// HIGHLIGHT
// Background is applied to the containerEl (which always has block layout
// and real dimensions) rather than textEl, which on X/Threads can be an
// inline span whose layout collapses if display is changed.
// Badge is appended to containerEl so it always has a visible anchor.

function highlightElement(containerEl, textEl, score) {

    if (!containerEl.isConnected) return;

    // Remove old badge if re-applying
    const existing = containerEl.querySelector(".ai-detector-badge");
    if (existing) existing.remove();
    containerEl.style.removeProperty("background");
    containerEl.style.removeProperty("outline");

    const mid = settings.threshold - 0.15;

    if (!settings.showHuman && score <= settings.threshold) return;

    containerEl.dataset.aiDone = "true";

    const color =
        score > settings.threshold ? "rgba(255,0,0,0.35)"
        : score > mid ? "rgba(255,165,0,0.30)"
        : "rgba(0,200,0,0.20)";

    const badgeColor =
        score > settings.threshold ? "red"
        : score > mid ? "orange"
        : "green";

    containerEl.style.setProperty("outline", "2px solid " + badgeColor, "important");
    containerEl.style.setProperty("background", color, "important");

    const badge = document.createElement("div");
    badge.className = "ai-detector-badge";
    badge.textContent = "AI " + (score * 100).toFixed(1) + "%";
    badge.setAttribute("style",
        "display:block !important;" +
        "width:fit-content !important;" +
        "margin:4px 0 0 0 !important;" +
        "padding:2px 8px !important;" +
        "border-radius:4px !important;" +
        "font-size:11px !important;" +
        "font-weight:bold !important;" +
        "font-family:sans-serif !important;" +
        "color:white !important;" +
        "background:" + badgeColor + " !important;" +
        "line-height:1.6 !important;" +
        "pointer-events:none !important;" +
        "z-index:9999 !important;" +
        "position:relative !important;"
    );
    containerEl.appendChild(badge);
}
// BLOCK OVERLAY

function blockElement(containerEl) {

    if (!containerEl.isConnected) return;
    if (containerEl.dataset.aiBlocked === "true") return;

    containerEl.dataset.aiBlocked = "true";

    if (getComputedStyle(containerEl).position === "static") {
        containerEl.style.setProperty("position", "relative", "important");
    }

    const overlay = document.createElement("div");
    overlay.setAttribute("style",
        "position:absolute !important;" +
        "inset:0 !important;" +
        "background:rgba(10,10,10,0.88) !important;" +
        "display:flex !important;" +
        "align-items:center !important;" +
        "justify-content:center !important;" +
        "z-index:9999 !important;" +
        "border-radius:4px !important;"
    );

    const warning = document.createElement("div");
    warning.textContent = "This content has been blocked — high probability of being AI generated.";
    warning.setAttribute("style",
        "padding:12px 16px !important;" +
        "background:rgba(255,0,0,0.12) !important;" +
        "border:1px solid red !important;" +
        "border-radius:5px !important;" +
        "font-size:12px !important;" +
        "color:red !important;" +
        "text-align:center !important;" +
        "max-width:320px !important;"
    );

    overlay.appendChild(warning);
    containerEl.appendChild(overlay);
}

// SCAN ONE CONTAINER
async function scanContainer(containerEl, textSelector, platformName) {
    if (!containerEl) return;
    if (!isVisible(containerEl)) return;
    if (containerEl.dataset.aiBlocked === "true") return;
    if (containerEl.dataset.aiQueued === "true") return;

    containerEl.dataset.aiQueued = "true";

    // Expand "See more" before resolving text
    const seeMore = [...containerEl.querySelectorAll("div[role='button'], span[role='button']")]
        .find(el => /^\s*see more\s*$/i.test(el.innerText));

    if (seeMore) {
        seeMore.click();
        await new Promise(r => setTimeout(r, 500));
    }

    // Resolve AFTER expansion so innerText is full
    const textEl = resolveBestTextEl(containerEl, textSelector);
    if (!textEl) return;

    const text = textEl.innerText?.trim();
    if (!text) return;

    const words = text.split(/\s+/).length;
    if (words < settings.minWords || words > MAX_WORDS) return;

    const hash = simpleHash(text);
    const now = Date.now();
    const prev = scannedHashes.get(hash);
    if (prev && now - prev < 15000) return;

    scannedHashes.set(hash, now);

    debugTokens(text);

    try {
        const score = await predict(text);
        containerEl.dataset.aiScore = score;
        logs.push({ platform: platformName, text: text.slice(0, 200), words, score, time: Date.now() });
        applyResult(containerEl, textEl, score, platformName);
    } catch (err) {
        console.error("AISeeYou: predict failed:", err);
    }
}

function applyResult(containerEl, textEl, score, platformName) {

    if (!containerEl.isConnected) return;

    containerEl.dataset.aiDone = "false";
    containerEl.dataset.aiBlocked = "false";
    const badge = containerEl.querySelector(".ai-detector-badge");
    if (badge) badge.remove();
    containerEl.style.removeProperty("background");
    containerEl.style.removeProperty("outline");

    const mid = settings.threshold - 0.15;

    if (!settings.showHuman && score <= settings.threshold) return; // hide non-AI

    if (settings.mode === "block" && score > settings.threshold) {
        blockElement(containerEl);
    } else {
        highlightElement(containerEl, textEl, score);
    }
}

// SCAN PAGE

function scanPage() {

    if (!ortReady) return;

    PLATFORMS.forEach(({ container, textNode, name }) => {

        if (!canScan(name)) return;

        const matches = document.querySelectorAll(container);

        if (matches.length > 0) {
            console.log("AISeeYou:", name, "— matched", matches.length);
        }

        matches.forEach(el => scanContainer(el, textNode, name));
    });
}

// SETTINGS

function loadSettings(cb) {

    if (!chrome?.storage?.sync) {
        if (cb) cb();
        return;
    }

    chrome.storage.sync.get(
        ["minWords", "mode", "threshold", "showHuman"],
        (data) => {
            settings.minWords = data?.minWords ?? 30;
            settings.mode = data?.mode ?? "highlight";
            settings.threshold = data?.threshold ?? 0.80;
            settings.showHuman = data?.showHuman ?? true;
            if (cb) cb();
        }
    );
}

// BOOTSTRAP

async function bootstrap() {

    try {

        console.log("AISeeYou: waiting for ORT...");
        await waitForORT();
        console.log("AISeeYou: ORT ready");

        await TOKENIZER.load();
        await loadModel();

        await new Promise(resolve => loadSettings(resolve));

        ortReady = true;

        console.log("AISeeYou: bootstrap complete");

        scanPage();

    } catch (err) {
        console.error("AISeeYou: bootstrap failed:", err);
    }
}

bootstrap();

// MUTATION OBSERVER

let running = false;

const observer = new MutationObserver(() => {

    if (running) return;

    running = true;

    setTimeout(() => {
        scanPage();
        running = false;
    }, 1200);
});

observer.observe(document.body, { childList: true, subtree: true });

// MESSAGE LISTENER

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
    loadSettings(() => {
        document.querySelectorAll("[data-ai-score]").forEach(el => {
            const score = parseFloat(el.dataset.aiScore);
            el.dataset.aiDone = "false";
            const textEl = resolveBestTextEl(el, null);
            highlightElement(el, textEl, score);
        });
    });
}
});


function debugTokens(text) {
    const normalized = TOKENIZER._normalize(text);
    const pre = TOKENIZER._preTokenize(normalized);

    const pieces = [];
    const ids = [];

    for (const word of pre) {
        const wp = TOKENIZER._wordpiece(word);
        for (const p of wp) {
            pieces.push(p);
            ids.push(TOKENIZER.vocab[p] ?? TOKENIZER.UNK);
        }
    }

    const finalIds = [TOKENIZER.CLS, ...ids.slice(0, TOKENIZER.maxLen - 2), TOKENIZER.SEP];
    while (finalIds.length < TOKENIZER.maxLen) finalIds.push(TOKENIZER.PAD);

    console.log("normalized", normalized);
    console.log("preTokens", pre);
    console.log("wordpieces", pieces);
    console.log("tokenIds", finalIds);
    console.log("attentionMask", finalIds.map((_, i) => i < (ids.length + 2) ? 1 : 0));
}