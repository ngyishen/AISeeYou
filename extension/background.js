chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
    // Handle Log Downloads
    if (msg.action === "downloadLogs") {
        chrome.downloads.download({
            url: msg.url,
            filename: "ai_detector_logs.json",
            saveAs: true
        });
    }

    // Handle AI API Calls (Bypasses Site CSP)
    if (msg.action === "checkText") {
        fetch("http://127.0.0.1:8000/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ text: msg.text })
        })
        .then(res => {
            if (!res.ok) throw new Error("API responded with " + res.status);
            return res.json();
        })
        .then(data => sendResponse({ success: true, data }))
        .catch(err => sendResponse({ success: false, error: err.message }));

        return true; // Keeps the messaging channel open for the async fetch
    }
});