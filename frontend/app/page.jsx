"use client";

import { useEffect, useMemo, useState } from "react";
import { AlertCircle, MessageSquarePlus, PanelLeft, ShieldCheck, Trash2, X } from "lucide-react";

import ChatComposer from "../components/ChatComposer";
import Conversation from "../components/Conversation";
import { sendChatTurn } from "../lib/api";

const STORAGE_KEY = "diy-assist-chats-v1";
const STARTER_QUERIES = [
  "My washer is not draining and makes a humming noise.",
  "My washer is leaking.",
  "I smell gas near my dryer."
];

function makeId() {
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function createChat() {
  const now = new Date().toISOString();
  return {
    id: makeId(),
    title: "New chat",
    createdAt: now,
    updatedAt: now,
    sessionId: null,
    messages: []
  };
}

function titleFromQuery(query) {
  const clean = query.replace(/\s+/g, " ").trim();
  if (!clean) return "New chat";
  return clean.length > 48 ? `${clean.slice(0, 45)}...` : clean;
}

export default function Home() {
  const [chats, setChats] = useState([]);
  const [activeChatId, setActiveChatId] = useState(null);
  const [draft, setDraft] = useState("");
  const [applianceCategory, setApplianceCategory] = useState("Appliance");
  const [brand, setBrand] = useState("");
  const [model, setModel] = useState("");
  const [strategy, setStrategy] = useState("naive");
  const [slmModelName, setSlmModelName] = useState("qwen2.5-3b-instruct-mlx");
  const [isSending, setIsSending] = useState(false);
  const [error, setError] = useState(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [pendingDeleteChatId, setPendingDeleteChatId] = useState(null);

  useEffect(() => {
    try {
      const saved = JSON.parse(window.localStorage.getItem(STORAGE_KEY) || "[]");
      if (Array.isArray(saved) && saved.length) {
        setChats(saved);
        setActiveChatId(saved[0].id);
        return;
      }
    } catch {
      window.localStorage.removeItem(STORAGE_KEY);
    }
    const firstChat = createChat();
    setChats([firstChat]);
    setActiveChatId(firstChat.id);
  }, []);

  useEffect(() => {
    if (chats.length) {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(chats));
    }
  }, [chats]);

  const activeChat = useMemo(
    () => chats.find((chat) => chat.id === activeChatId) || chats[0] || null,
    [activeChatId, chats]
  );

  const messages = activeChat?.messages || [];

  function updateChat(chatId, updater) {
    setChats((current) =>
      current
        .map((chat) => (chat.id === chatId ? updater(chat) : chat))
        .sort((a, b) => new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime())
    );
  }

  function startNewChat() {
    const chat = createChat();
    setChats((current) => [chat, ...current]);
    setActiveChatId(chat.id);
    setDraft("");
    setError(null);
    setSidebarOpen(false);
    setPendingDeleteChatId(null);
  }

  function deleteChat(chatId) {
    const remainingChats = chats.filter((chat) => chat.id !== chatId);
    if (remainingChats.length) {
      setChats(remainingChats);
      if (activeChatId === chatId) {
        setActiveChatId(remainingChats[0].id);
      }
    } else {
      const replacement = createChat();
      setChats([replacement]);
      setActiveChatId(replacement.id);
    }
    setPendingDeleteChatId(null);
    setError(null);
  }

  async function handleSubmit(event) {
    event.preventDefault();
    const query = draft.trim();
    if (!query || isSending || !activeChat) return;

    setError(null);
    setIsSending(true);
    setDraft("");

    const chatId = activeChat.id;
    const userMessage = { id: makeId(), role: "user", content: query };
    const nextTitle = activeChat.messages.length ? activeChat.title : titleFromQuery(query);
    updateChat(chatId, (chat) => ({
      ...chat,
      title: nextTitle,
      updatedAt: new Date().toISOString(),
      messages: [...chat.messages, userMessage]
    }));

    try {
      const payload = {
        query,
        appliance_category: applianceCategory || null,
        brand: brand.trim() || null,
        model: model.trim() || null,
        retrieval_strategy: strategy,
        slm_model_name: slmModelName,
        agent_mode: "react",
        session_id: activeChat.sessionId,
        top_k: 5
      };
      const response = await sendChatTurn(payload);
      updateChat(chatId, (chat) => ({
        ...chat,
        sessionId: response.session_id || chat.sessionId,
        updatedAt: new Date().toISOString(),
        messages: [
          ...chat.messages,
          {
            id: makeId(),
            role: "assistant",
            data: response
          }
        ]
      }));
    } catch (requestError) {
      setError(requestError.message);
      updateChat(chatId, (chat) => ({
        ...chat,
        updatedAt: new Date().toISOString(),
        messages: [
          ...chat.messages,
          {
            id: makeId(),
            role: "assistant",
            data: {
              answer: requestError.message,
              structured: {
                answer_summary: "Could not reach backend.",
                steps: [],
                parts_list: [],
            tools_required: [],
                retrieved_guide_snippets: [],
                tool_trace: []
              },
              citations: [],
              live_ifixit_guides: []
            }
          }
        ]
      }));
    } finally {
      setIsSending(false);
    }
  }

  return (
    <main className={`appShell ${sidebarOpen ? "sidebarOpen" : ""}`}>
      <aside className="sidebar" aria-label="Chat history">
        <div className="sidebarTop">
          <div className="brandMark">
            <ShieldCheck size={24} />
          </div>
          <button className="newChatButton" type="button" onClick={startNewChat}>
            <MessageSquarePlus size={18} />
            New chat
          </button>
          <button className="sidebarCloseButton" type="button" onClick={() => setSidebarOpen(false)} aria-label="Close sidebar">
            <X size={18} />
          </button>
        </div>

        <nav className="chatHistory">
          {chats.map((chat) => (
            <div className={chat.id === activeChat?.id ? "chatHistoryRow active" : "chatHistoryRow"} key={chat.id}>
              <button
                className="chatHistoryItem"
                type="button"
                onClick={() => {
                  setActiveChatId(chat.id);
                  setSidebarOpen(false);
                  setPendingDeleteChatId(null);
                }}
              >
                <span>{chat.title}</span>
                <small>{chat.messages.length ? `${Math.ceil(chat.messages.length / 2)} turns` : "empty"}</small>
              </button>
              {pendingDeleteChatId === chat.id ? (
                <div className="deleteConfirm" aria-label={`Confirm deleting ${chat.title}`}>
                  <button type="button" onClick={() => deleteChat(chat.id)}>Delete</button>
                  <button type="button" onClick={() => setPendingDeleteChatId(null)}>Cancel</button>
                </div>
              ) : (
                <button
                  className="deleteChatButton"
                  type="button"
                  onClick={() => setPendingDeleteChatId(chat.id)}
                  aria-label={`Delete ${chat.title}`}
                >
                  <Trash2 size={15} />
                </button>
              )}
            </div>
          ))}
        </nav>

        <section className="quickPanel">
          <h2>Try</h2>
          {STARTER_QUERIES.map((query) => (
            <button key={query} type="button" onClick={() => setDraft(query)}>
              {query}
            </button>
          ))}
        </section>
      </aside>

      <section className="chatPane">
        <header className="chatHeader">
          <button className="sidebarToggle" type="button" onClick={() => setSidebarOpen((value) => !value)}>
            <PanelLeft size={18} />
          </button>
          <div>
            <h1>DIY-Assist</h1>
            <p>Safe appliance troubleshooting</p>
          </div>
        </header>

        {error ? (
          <div className="errorBanner" role="alert">
            <AlertCircle size={18} />
            <span>{error}</span>
          </div>
        ) : null}

        <Conversation messages={messages} isSending={isSending} />

        <div className="composerDock">
          <ChatComposer
            draft={draft}
            setDraft={setDraft}
            applianceCategory={applianceCategory}
            setApplianceCategory={setApplianceCategory}
            brand={brand}
            setBrand={setBrand}
            model={model}
            setModel={setModel}
            strategy={strategy}
            setStrategy={setStrategy}
            slmModelName={slmModelName}
            setSlmModelName={setSlmModelName}
            isSending={isSending}
            onSubmit={handleSubmit}
          />
        </div>
      </section>
    </main>
  );
}
