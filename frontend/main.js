const sendBtn = document.getElementById("sendBtn");
const queryEl = document.getElementById("query");
const categoryEl = document.getElementById("category");
const topkEl = document.getElementById("topk");
const answerEl = document.getElementById("answer");
const citationsEl = document.getElementById("citations");
const liveGuidesEl = document.getElementById("liveGuides");

function renderCitations(citations) {
  if (!citations || citations.length === 0) {
    citationsEl.textContent = "No citations returned.";
    return;
  }
  citationsEl.innerHTML = "";
  const ul = document.createElement("ul");
  citations.forEach((c) => {
    const li = document.createElement("li");
    const source = c.source_url ? ` - ${c.source_url}` : "";
    li.textContent = `${c.guide_title || "Unknown guide"} (step ${c.step_number || "?"})${source}`;
    ul.appendChild(li);
  });
  citationsEl.appendChild(ul);
}

function renderLiveGuides(guides) {
  if (!guides || guides.length === 0) {
    liveGuidesEl.textContent = "No live iFixit guides found (or API unavailable).";
    return;
  }
  liveGuidesEl.innerHTML = "";
  const ul = document.createElement("ul");
  guides.forEach((g) => {
    const li = document.createElement("li");
    li.textContent = `${g.guide_title || "Untitled"}${g.source_url ? ` - ${g.source_url}` : ""}`;
    ul.appendChild(li);
  });
  liveGuidesEl.appendChild(ul);
}

async function sendChat() {
  sendBtn.disabled = true;
  answerEl.textContent = "Thinking...";

  const body = {
    query: queryEl.value.trim(),
    appliance_category: categoryEl.value.trim() || null,
    top_k: Number(topkEl.value || 3),
  };

  try {
    const response = await fetch("/api/v1/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const raw = await response.text();
    let data = {};
    try {
      data = raw ? JSON.parse(raw) : {};
    } catch (parseErr) {
      answerEl.textContent = `Backend returned non-JSON response (${response.status}):\n${raw.slice(0, 500)}`;
      renderCitations([]);
      renderLiveGuides([]);
      return;
    }

    if (!response.ok) {
      answerEl.textContent = `Request failed: ${response.status}\n${JSON.stringify(data, null, 2)}`;
      renderCitations([]);
      renderLiveGuides([]);
      return;
    }

    if (data.guardrail_blocked) {
      answerEl.textContent = `Guardrail blocked:\n${data.message || data.reason || "Safety escalation triggered."}`;
    } else {
      answerEl.textContent = data.answer || "No answer generated.";
    }
    renderCitations(data.citations || []);
    renderLiveGuides(data.live_ifixit_guides || []);
  } catch (err) {
    answerEl.textContent = `Error calling backend: ${err.message}`;
    renderCitations([]);
    renderLiveGuides([]);
  } finally {
    sendBtn.disabled = false;
  }
}

sendBtn.addEventListener("click", sendChat);
