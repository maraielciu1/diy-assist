const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:8000";

export async function sendChatTurn(payload) {
  const response = await fetch(`${API_BASE}/api/v1/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });

  const text = await response.text();
  let data = {};
  try {
    data = text ? JSON.parse(text) : {};
  } catch (error) {
    throw new Error(`Backend returned non-JSON (${response.status}): ${text.slice(0, 300)}`);
  }

  if (!response.ok) {
    throw new Error(data.detail || data.error || `Chat request failed (${response.status})`);
  }

  return data;
}

export async function runRagDebug(payload, strategy = "naive") {
  const response = await fetch(`${API_BASE}/api/v1/rag/${strategy}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });

  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail || data.error || `RAG request failed (${response.status})`);
  }
  return data;
}
