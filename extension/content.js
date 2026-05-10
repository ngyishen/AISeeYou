chrome.runtime.onMessage.addListener((message) => {

    if (message.action !== "showResult") return;

    const oldBox = document.getElementById("ai-detector-popup");

    if (oldBox) oldBox.remove();

    const popup = document.createElement("div");

    popup.id = "ai-detector-popup";

    popup.innerHTML = `
        <strong>${message.result.final_label}</strong><br>
        Human: ${message.result.human_prob.toFixed(3)}<br>
        AI: ${message.result.ai_prob.toFixed(3)}
    `;

    popup.style.top = `${window.scrollY + 100}px`;
    popup.style.right = `20px`;

    document.body.appendChild(popup);

    setTimeout(() => {
        popup.remove();
    }, 5000);
});