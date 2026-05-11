const API_URL = "http://127.0.0.1:8000/predict";

let settings = {
    minWords: 30,
    debugMode: false,
    mode: "highlight",
    threshold: 0.80
};

const logs = [];

// CHANGED: Map instead of Set
const scannedHashes = new Map();

const SKIP_HIGHLIGHT_TAGS = new Set([
    "SCRIPT",
    "STYLE",
    "BUTTON",
    "A",
    "INPUT",
    "TEXTAREA",
    "SELECT",
    "LABEL"
]);

const PLATFORMS = [
    {
        container: "article[data-testid='tweet']",
        textNode: "[data-testid='tweetText']",
        name: "X"
    },
    {
        container: "[data-ad-preview='message']",
        textNode: null,
        name: "Facebook"
    },
    {
        container: "div[data-testid='post_message']",
        textNode: null,
        name: "Facebook2"
    },
    {
        container: "article[role='article']",
        textNode: "div[dir='auto']",
        name: "Threads"
    },
    {
        container: ".feed-shared-update-v2",
        textNode: ".feed-shared-update-v2__description",
        name: "LinkedIn"
    },
    {
        container: "shreddit-post",
        textNode: "[slot='text-body']",
        name: "Reddit"
    },
    {
        container: "article",
        textNode: null,
        name: "Generic-article"
    },
    {
        container: "blockquote",
        textNode: null,
        name: "Generic-blockquote"
    }
];

console.log("AISeeYou LOADED");
console.log("chrome.runtime exists:", !!chrome?.runtime);
console.log("chrome.storage exists:", !!chrome?.storage);

// ----------------------
// SETTINGS
// ----------------------
function loadSettings(cb) {

    if (!chrome || !chrome.storage || !chrome.storage.sync) {

        console.warn("AISeeYou: chrome.storage.sync not available");

        if (cb) cb();

        return;
    }

    chrome.storage.sync.get(
        ["minWords", "debugMode", "mode", "threshold"],
        (data) => {

            settings.minWords = data?.minWords ?? 30;
            settings.debugMode = data?.debugMode ?? false;
            settings.mode = data?.mode ?? "highlight";
            settings.threshold = data?.threshold ?? 0.80;

            if (cb) cb();
        }
    );
}

function dbg(...args) {

    if (settings.debugMode) {
        console.log("AISeeYou:", ...args);
    }
}

// ----------------------
// API CALL
// ----------------------
async function checkText(text) {

    try {

        const res = await fetch(API_URL, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ text })
        });

        if (!res.ok) return null;

        return await res.json();

    } catch {
        return null;
    }
}

// ----------------------
// HASH
// ----------------------
function simpleHash(text) {

    let hash = 0;

    for (let i = 0; i < text.length; i++) {

        hash = ((hash << 5) - hash) + text.charCodeAt(i);
        hash |= 0;
    }

    return hash.toString();
}

// ----------------------
// LOGGING
// ----------------------
function log(text, words, score) {

    logs.push({
        text,
        words,
        score,
        time: Date.now()
    });
}

// ----------------------
// VISIBILITY CHECK
// ----------------------
function isVisible(el) {

    const rect = el.getBoundingClientRect();

    return rect.width > 0 && rect.height > 0;
}

// ----------------------
// HIGHLIGHT
// ----------------------
function highlightElement(textEl, score) {

    if (!textEl.isConnected) return;

    if (textEl.dataset.aiScanned === "true") return;

    textEl.dataset.aiScanned = "true";

    const midThreshold = settings.threshold - 0.15;

    const color =
        score > settings.threshold
            ? "rgba(255,0,0,0.18)"
            : score > midThreshold
            ? "rgba(255,165,0,0.16)"
            : "rgba(0,200,0,0.10)";

    const badgeColor =
        score > settings.threshold
            ? "red"
            : score > midThreshold
            ? "orange"
            : "green";

    textEl.style.setProperty(
        "background",
        color,
        "important"
    );

    textEl.style.setProperty(
        "border-radius",
        "6px",
        "important"
    );

    textEl.style.setProperty(
        "padding",
        "2px 4px",
        "important"
    );

    if (textEl.querySelector(".ai-detector-badge")) return;

    const badge = document.createElement("span");

    badge.className = "ai-detector-badge";

    badge.textContent =
        " AI " + (score * 100).toFixed(1) + "%";

    badge.setAttribute(
        "style",
        `
        display:inline-block !important;
        margin-left:6px !important;
        margin-top:4px !important;
        padding:2px 6px !important;
        border-radius:4px !important;
        font-size:10px !important;
        font-weight:bold !important;
        color:white !important;
        background:${badgeColor} !important;
        vertical-align:middle !important;
        `
    );

    textEl.appendChild(badge);
}

// ----------------------
// BLOCK OVERLAY
// ----------------------
function blockElement(containerEl) {

    if (!containerEl.isConnected) return;

    if (containerEl.dataset.aiBlocked === "true") return;

    containerEl.dataset.aiBlocked = "true";

    if (getComputedStyle(containerEl).position === "static") {

        containerEl.style.setProperty(
            "position",
            "relative",
            "important"
        );
    }

    const overlay = document.createElement("div");

    overlay.className = "ai-detector-block-overlay";

    overlay.setAttribute(
        "style",
        `
        position:absolute !important;
        inset:0 !important;
        background:rgba(10,10,10,0.88) !important;
        display:flex !important;
        align-items:center !important;
        justify-content:center !important;
        z-index:9999 !important;
        border-radius:4px !important;
        `
    );

    const warning = document.createElement("div");

    warning.setAttribute(
        "style",
        `
        padding:12px 16px !important;
        background:rgba(255,0,0,0.12) !important;
        border:1px solid red !important;
        border-radius:5px !important;
        font-size:12px !important;
        color:red !important;
        text-align:center !important;
        max-width:320px !important;
        `
    );

    warning.textContent =
        "This content has been blocked — high probability of being AI generated.";

    overlay.appendChild(warning);

    containerEl.appendChild(overlay);
}

// ----------------------
// SCAN CONTAINER
// ----------------------
async function scanContainer(containerEl, textSelector, platformName) {

    if (!containerEl) return;

    if (containerEl.dataset.aiBlocked === "true") return;

    const textEl = textSelector
        ? containerEl.querySelector(textSelector)
        : containerEl;

    if (!textEl) {

        dbg("no textEl found for", platformName);

        return;
    }

    if (textEl.dataset.aiScanned === "true") return;

    const text = textEl.innerText?.trim();

    if (!text) return;

    const words = text.split(/\s+/).length;

    dbg(platformName, "found", words, "words");

    if (words < settings.minWords) return;

    if (words > 800) return;

    const hash = simpleHash(text);

    const now = Date.now();

    const existing = scannedHashes.get(hash);

    // CHANGED: temporary dedupe instead of permanent
    if (existing && now - existing < 15000) {
        return;
    }

    scannedHashes.set(hash, now);

    // cleanup old hashes
    if (scannedHashes.size > 2000) {

        for (const [k, t] of scannedHashes.entries()) {

            if (now - t > 60000) {
                scannedHashes.delete(k);
            }
        }
    }

    dbg("sending to API:", text.slice(0, 60));

    const result = await checkText(text);

    if (!result) {

        dbg("API failed");

        return;
    }

    const score = result.ai_prob;

    dbg("score:", score);

    log(text, words, score);

    if (
        settings.mode === "block" &&
        score > settings.threshold
    ) {

        blockElement(containerEl);

    } else {

        highlightElement(textEl, score);
    }
}

// ----------------------
// SCAN PAGE
// ----------------------
function scanPage() {

    PLATFORMS.forEach(({ container, textNode, name }) => {

        const matches =
            document.querySelectorAll(container);

        dbg(name, "matched", matches.length);

        matches.forEach(containerEl => {

            if (!isVisible(containerEl)) return;

            scanContainer(
                containerEl,
                textNode,
                name
            );
        });
    });
}

// ----------------------
// INITIAL SCAN
// ----------------------
setTimeout(() => {

    loadSettings(() => {

        dbg("settings loaded");

        scanPage();
    });

}, 2000);

// ----------------------
// OBSERVER
// ----------------------
let running = false;

const observer = new MutationObserver(() => {

    if (running) return;

    running = true;

    setTimeout(() => {

        scanPage();

        running = false;

    }, 1500);
});

observer.observe(document.body, {
    childList: true,
    subtree: true
});

// ----------------------
// MESSAGES
// ----------------------
chrome.runtime.onMessage.addListener((msg) => {

    if (msg.action === "exportLogs") {

        const blob = new Blob(
            [JSON.stringify(logs, null, 2)],
            {
                type: "application/json"
            }
        );

        const url = URL.createObjectURL(blob);

        chrome.runtime.sendMessage({
            action: "downloadLogs",
            url
        });
    }

    if (msg.action === "settingsChanged") {

        loadSettings();
    }
});