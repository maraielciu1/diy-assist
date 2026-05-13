"use client";

import { AlertTriangle, Bot, CheckCircle2, ExternalLink, History, UserRound, Wrench } from "lucide-react";

function TextBlock({ text }) {
  if (!text) return null;
  return <p className="messageText">{text}</p>;
}

function SourceList({ citations = [] }) {
  if (!citations.length) return null;
  return (
    <section className="responseSection">
      <h3>Sources</h3>
      <ul className="sourceList">
        {citations.slice(0, 5).map((source, index) => {
          const url = source.source_url || source.metadata?.source_url;
          return (
            <li key={`${source.guide_title || "source"}-${index}`}>
              <span>
                {source.guide_title || "Repair guide"}
                {source.step_number ? `, step ${source.step_number}` : ""}
              </span>
              {url ? (
                <a href={url} target="_blank" rel="noreferrer" aria-label="Open source">
                  <ExternalLink size={14} />
                </a>
              ) : null}
            </li>
          );
        })}
      </ul>
    </section>
  );
}

function LiveGuides({ guides = [] }) {
  if (!guides.length) return null;
  return (
    <section className="responseSection">
      <h3>Live iFixit candidates</h3>
      <ul className="sourceList">
        {guides.map((guide, index) => (
          <li key={`${guide.guide_title || "guide"}-${index}`}>
            <span>{guide.guide_title || "Untitled guide"}</span>
            {guide.source_url ? (
              <a href={guide.source_url} target="_blank" rel="noreferrer" aria-label="Open live guide">
                <ExternalLink size={14} />
              </a>
            ) : null}
          </li>
        ))}
      </ul>
    </section>
  );
}

function StructuredResponse({ data }) {
  const structured = data.structured || {};
  const steps = structured.steps || [];
  const parts = structured.parts_list || [];
  const tools = structured.tools_required || [];
  const snippets = structured.retrieved_guide_snippets || [];
  const trace = structured.tool_trace || [];
  const rawFallback = data.answer || data.error || data.details || data.message;
  const fallbackText =
    rawFallback && data.safety_warning && rawFallback === data.safety_warning ? null : rawFallback;

  return (
    <div className="structuredResponse">
      {data.safety_warning ? (
        <section className="safetyBanner" role="alert">
          <AlertTriangle size={20} />
          <div>
            <strong>Safety escalation</strong>
            <p>{data.safety_warning}</p>
          </div>
        </section>
      ) : null}

      {structured.clarifying_question ? (
        <section className="clarifyBox">
          <History size={18} />
          <div>
            <strong>Clarifying question</strong>
            <p>{structured.clarifying_question}</p>
          </div>
        </section>
      ) : null}

      {structured.answer_summary ? (
        <section className="responseSection">
          <h3>Summary</h3>
          <TextBlock text={structured.answer_summary} />
        </section>
      ) : fallbackText ? (
        <TextBlock text={fallbackText} />
      ) : null}

      {structured.likely_issue ? (
        <section className="responseSection issueLine">
          <CheckCircle2 size={18} />
          <div>
            <h3>Likely issue</h3>
            <p>{structured.likely_issue}</p>
          </div>
        </section>
      ) : null}

      {steps.length ? (
        <section className="responseSection">
          <h3>Repair guidance</h3>
          <ol className="stepList">
            {steps.map((step, index) => (
              <li key={`${index}-${step}`}>{step}</li>
            ))}
          </ol>
        </section>
      ) : null}

      {parts.length ? (
        <section className="responseSection">
          <h3>Parts to inspect</h3>
          <div className="partsList">
            {parts.map((part) => (
              <span key={part}>
                <Wrench size={14} /> {part}
              </span>
            ))}
          </div>
        </section>
      ) : null}

      {tools.length ? (
        <section className="responseSection">
          <h3>Tools required</h3>
          <div className="partsList">
            {tools.map((tool) => (
              <span key={tool}>
                <Wrench size={14} /> {tool}
              </span>
            ))}
          </div>
        </section>
      ) : null}

      {snippets.length ? (
        <section className="responseSection">
          <h3>Retrieved guide snippets</h3>
          <div className="snippetList">
            {snippets.map((snippet, index) => (
              <article key={`${snippet.guide_title || "snippet"}-${index}`}>
                <strong>{snippet.guide_title || "Guide snippet"}</strong>
                <p>{snippet.text}</p>
              </article>
            ))}
          </div>
        </section>
      ) : null}

      <SourceList citations={data.citations || []} />
      <LiveGuides guides={data.live_ifixit_guides || []} />

      {trace.length ? (
        <details className="responseSection">
          <summary>Tool trace</summary>
          <div className="snippetList">
            {trace.map((item, index) => (
              <article key={`${item.tool || "tool"}-${index}`}>
                <strong>{item.tool || "tool"}</strong>
                <p>{JSON.stringify(item)}</p>
              </article>
            ))}
          </div>
        </details>
      ) : null}
    </div>
  );
}

function ThinkingIndicator() {
  return (
    <article className="message assistant thinkingMessage" aria-label="Assistant is thinking">
      <div className="avatar" aria-hidden="true">
        <Bot size={18} />
      </div>
      <div className="messageBody thinkingBody">
        <span />
        <span />
        <span />
      </div>
    </article>
  );
}

export default function Conversation({ messages, isSending }) {
  if (!messages.length) {
    return (
      <div className="emptyState">
        <Bot size={30} />
        <h2>What needs fixing?</h2>
        <p>Describe symptom, sound, smell, leak location, error code, and appliance type.</p>
      </div>
    );
  }

  return (
    <div className="conversation" aria-live="polite">
      {messages.map((message) => (
        <article className={`message ${message.role}`} key={message.id}>
          {message.role === "assistant" ? <div className="avatar" aria-hidden="true">
            {message.role === "user" ? <UserRound size={18} /> : <Bot size={18} />}
          </div> : null}
          <div className="messageBody">
            {message.role === "user" ? <TextBlock text={message.content} /> : <StructuredResponse data={message.data} />}
          </div>
          {message.role === "user" ? <div className="avatar" aria-hidden="true">
            <UserRound size={18} />
          </div> : null}
        </article>
      ))}
      {isSending ? <ThinkingIndicator /> : null}
    </div>
  );
}
