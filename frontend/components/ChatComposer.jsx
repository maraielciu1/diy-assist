"use client";

import { useLayoutEffect, useRef } from "react";
import { Send, Settings2 } from "lucide-react";

const APPLIANCES = ["Appliance", "Washer", "Dryer", "Dishwasher", "Refrigerator", "Oven"];
const STRATEGIES = ["naive", "reranked", "hyde", "hyde_reranked"];
const SLM_MODELS = [
  { label: "Qwen 2.5 3B", value: "qwen2.5-3b-instruct-mlx" },
  { label: "Qwen 2.5 7B", value: "qwen2.5-7b-instruct-mlx" }
];

export default function ChatComposer({
  draft,
  setDraft,
  applianceCategory,
  setApplianceCategory,
  brand,
  setBrand,
  model,
  setModel,
  strategy,
  setStrategy,
  slmModelName,
  setSlmModelName,
  isSending,
  onSubmit
}) {
  const textareaRef = useRef(null);

  useLayoutEffect(() => {
    const textarea = textareaRef.current;
    if (!textarea) return;

    textarea.style.height = "46px";
    textarea.style.height = `${Math.min(textarea.scrollHeight, 140)}px`;
  }, [draft]);

  function handleTextareaKeyDown(event) {
    if (event.key !== "Enter" || event.shiftKey || event.nativeEvent.isComposing) return;

    event.preventDefault();
    event.currentTarget.form?.requestSubmit();
  }

  return (
    <form className="composer" onSubmit={onSubmit}>
      <div className="composerTop">
        <details className="composerSettings">
          <summary>
            <Settings2 size={15}/>
            Repair context
          </summary>
        <div className="composerControls" aria-label="Chat controls">
          <label>
            <span>Appliance</span>
            <select value={applianceCategory} onChange={(event) => setApplianceCategory(event.target.value)}>
              {APPLIANCES.map((item) => (
                <option key={item} value={item}>
                  {item}
                </option>
              ))}
            </select>
          </label>
          <label>
            <span>Brand</span>
            <input value={brand} onChange={(event) => setBrand(event.target.value)} placeholder="optional" />
          </label>
          <label>
            <span>Model</span>
            <input value={model} onChange={(event) => setModel(event.target.value)} placeholder="optional" />
          </label>
          <label>
            <span>Strategy</span>
            <select value={strategy} onChange={(event) => setStrategy(event.target.value)}>
              {STRATEGIES.map((item) => (
                <option key={item} value={item}>
                  {item}
                </option>
              ))}
            </select>
          </label>
        </div>
        </details>
        <label className="modelPicker">
          <select value={slmModelName} onChange={(event) => setSlmModelName(event.target.value)}>
            {SLM_MODELS.map((item) => (
              <option key={item.value} value={item.value}>
                {item.label}
              </option>
            ))}
          </select>
        </label>
      </div>
      <div className="inputRow">
        <textarea
          ref={textareaRef}
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={handleTextareaKeyDown}
          placeholder="Describe the symptom, noise, leak location, error code, and when it happens."
          rows={1}
        />
        <button type="submit" disabled={isSending || !draft.trim()} aria-label="Send message">
          <Send size={18} />
          <span>{isSending ? "Sending" : "Send"}</span>
        </button>
      </div>
    </form>
  );
}
