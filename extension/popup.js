document.addEventListener("DOMContentLoaded", () => {

    const minWordsEl = document.getElementById("minWords");
    const showHumanEl = document.getElementById("showHuman");
    const modeEl      = document.getElementById("mode");
    const biasEl      = document.getElementById("bias");
    const biasDisplay = document.getElementById("biasDisplay");
    const biasHint    = document.getElementById("biasHint");
    const toast       = document.getElementById("toast");

    
    // BIAS → THRESHOLD CONVERSION
    // Slider 0–100 maps to flag threshold 0.95 (lenient)  0.35 (strict).
    // Higher sensitivity = lower probability threshold  more flagging.
    
    function thresholdFromBias(bias) {
        // bias 0   → threshold 0.95 (almost nothing flagged)
        // bias 50  → threshold 0.80 (default)
        // bias 100 → threshold 0.35 (very aggressive)
        return parseFloat((0.95 - (bias / 100) * 0.60).toFixed(2));
    }

    function updateBiasUI(bias) {
        biasDisplay.textContent = bias;
        const threshold = thresholdFromBias(bias);
        const pct = Math.round(threshold * 100);
        biasHint.textContent = "Flags text above " + pct + "% AI probability";
    }

    
    // LOAD SETTINGS
    
    chrome.storage.sync.get(["minWords", "showHuman", "mode", "bias"], (data) => {
        minWordsEl.value  = data.minWords  ?? 30;
        showHumanEl.checked = data.showHuman ?? false;
        modeEl.value      = data.mode      ?? "highlight";
        biasEl.value      = data.bias      ?? 50;
        updateBiasUI(biasEl.value);
    });

    
    // LIVE SLIDER FEEDBACK
    
    biasEl.addEventListener("input", () => {
        updateBiasUI(parseInt(biasEl.value));
    });

    
    // SAVE SETTINGS
    
    document.getElementById("save").onclick = () => {
    const bias = parseInt(biasEl.value);
    const newSettings = {
        minWords:  parseInt(minWordsEl.value),
        showHuman: showHumanEl.checked,
        mode:      modeEl.value,
        bias,
        threshold: thresholdFromBias(bias)
    };

    chrome.storage.sync.set(newSettings, () => {
        chrome.tabs.query({ active: true, currentWindow: true }, ([tab]) => {
            if (tab?.id) {
                // Instead of sendMessage, inject a script directly
                chrome.scripting.executeScript({
                    target: { tabId: tab.id },
                    func: (s) => {
                        // Directly update the settings object in content script scope
                        settings.minWords  = s.minWords;
                        settings.showHuman = s.showHuman;
                        settings.mode      = s.mode;
                        settings.threshold = s.threshold;

                        document.querySelectorAll("[data-ai-score]").forEach(el => {
                            el.dataset.aiDone = "false";
                            const score = parseFloat(el.dataset.aiScore);
                            const mid = s.threshold - 0.15;
                            const badge = el.querySelector(".ai-detector-badge");
                            if (badge) badge.remove();
                            el.style.removeProperty("background");
                            el.style.removeProperty("outline");
                            if (!s.showHuman && score <= s.threshold) return;
                            highlightElement(el, null, score);
                        });
                    },
                    args: [newSettings]
                });
            }
        });

        toast.classList.add("show");
        setTimeout(() => toast.classList.remove("show"), 1800);
    });
};

    
    // DOWNLOAD LOGS
    document.getElementById("download").onclick = async () => {

        const [tab] = await chrome.tabs.query({
            active: true,
            currentWindow: true
        });

        if (tab?.id) {
            chrome.tabs.sendMessage(tab.id, { action: "exportLogs" });
        }
    };
});