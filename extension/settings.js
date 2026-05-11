document.getElementById("btn").onclick = async () => {

    const text = document.getElementById("text").value;

    const res = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ text })
    });

    const data = await res.json();

    document.getElementById("result").innerText =
        `AI: ${data.ai_prob.toFixed(3)} | Human: ${data.human_prob.toFixed(3)} | ${data.final_label}`;
};