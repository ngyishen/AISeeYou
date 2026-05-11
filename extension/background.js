chrome.runtime.onMessage.addListener((msg) => {

    if (msg.action === "downloadLogs") {

        chrome.downloads.download({
            url: msg.url,
            filename: "ai_detector_logs.json",
            saveAs: true
        });
    }
});