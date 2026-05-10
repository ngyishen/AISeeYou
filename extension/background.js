chrome.contextMenus.removeAll(() => {

    chrome.contextMenus.create({
        id: "detectAI",
        title: "Detect AI",
        contexts: ["selection"]
    });

});

chrome.contextMenus.onClicked.addListener(async (info, tab) => {

    if (info.menuItemId !== "detectAI") return;

    const selectedText = info.selectionText;

    try {

        const response = await fetch("http://localhost:5000/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                text: selectedText
            })
        });

        const data = await response.json();

        await chrome.scripting.executeScript({
            target: { tabId: tab.id },
            files: ["content.js"]
        });

        chrome.tabs.sendMessage(tab.id, {
            action: "showResult",
            result: data
        });

    } catch (err) {
        console.error(err);
    }

});