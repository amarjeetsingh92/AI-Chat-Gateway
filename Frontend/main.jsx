import { useState, useRef, useEffect, useCallback } from "react";

// ─── Constants ────────────────────────────────────────────────────────────────

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

const MODEL_COLORS = {
  openai: "#10b981",
  anthropic: "#f59e0b",
};

const MODEL_META = {
  "gpt-4o": { provider: "openai", label: "GPT-4o", short: "4o" },
  "gpt-4o-mini": { provider: "openai", label: "GPT-4o Mini", short: "Mini" },
  "claude-opus-4-6": { provider: "anthropic", label: "Claude Opus 4.6", short: "Opus" },
  "claude-sonnet-4-6": { provider: "anthropic", label: "Claude Sonnet 4.6", short: "Sonnet" },
  "claude-haiku-4-5-20251001": { provider: "anthropic", label: "Claude Haiku 4.5", short: "Haiku" },
};

// ─── Hooks ────────────────────────────────────────────────────────────────────

function useSSEChat() {
  const [messages, setMessages] = useState([]);
  const [streaming, setStreaming] = useState(false);
  const [usage, setUsage] = useState(null);
  const [error, setError] = useState(null);
  const abortRef = useRef(null);

  const sendMessage = useCallback(async ({ model, messages: history, systemPrompt, temperature, maxTokens }) => {
    setStreaming(true);
    setError(null);

    const assistantId = Date.now();
    setMessages(prev => [...prev, { id: assistantId, role: "assistant", content: "", model, loading: true }]);

    try {
      const res = await fetch(`${API_BASE}/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model,
          messages: history,
          system_prompt: systemPrompt,
          temperature,
          max_tokens: maxTokens,
          stream: true,
        }),
        signal: abortRef.current?.signal,
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Request failed");
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n\n");
        buffer = lines.pop();

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const data = JSON.parse(line.slice(6));

          if (data.type === "delta") {
            setMessages(prev =>
              prev.map(m =>
                m.id === assistantId
                  ? { ...m, content: m.content + data.content, loading: false }
                  : m
              )
            );
          } else if (data.type === "done") {
            setUsage({ prompt: data.prompt_tokens, completion: data.completion_tokens });
            setMessages(prev =>
              prev.map(m => (m.id === assistantId ? { ...m, loading: false } : m))
            );
          } else if (data.type === "error") {
            throw new Error(data.message);
          }
        }
      }
    } catch (err) {
      if (err.name !== "AbortError") {
        setError(err.message);
        setMessages(prev => prev.filter(m => m.id !== assistantId));
      }
    } finally {
      setStreaming(false);
    }
  }, []);

  const stop = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = new AbortController();
    setStreaming(false);
  }, []);

  const clear = useCallback(() => {
    setMessages([]);
    setUsage(null);
    setError(null);
  }, []);

  return { messages, setMessages, streaming, usage, error, sendMessage, stop, clear };
}

// ─── Components ───────────────────────────────────────────────────────────────

function ModelBadge({ model }) {
  const meta = MODEL_META[model] || { provider: "openai", label: model, short: "?" };
  const color = MODEL_COLORS[meta.provider];
  return (
    <span style={{
      fontSize: "10px", fontFamily: "'JetBrains Mono', monospace",
      padding: "2px 7px", borderRadius: "3px", letterSpacing: "0.05em",
      background: `${color}22`, color, border: `1px solid ${color}44`,
    }}>
      {meta.short}
    </span>
  );
}

function Cursor() {
  const [vis, setVis] = useState(true);
  useEffect(() => {
    const t = setInterval(() => setVis(v => !v), 530);
    return () => clearInterval(t);
  }, []);
  return <span style={{ opacity: vis ? 1 : 0, color: "#64ffda" }}>▋</span>;
}

function MessageBubble({ msg }) {
  const isUser = msg.role === "user";
  return (
    <div style={{
      display: "flex", flexDirection: "column",
      alignItems: isUser ? "flex-end" : "flex-start",
      marginBottom: "20px", animation: "fadeSlide 0.25s ease",
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "6px" }}>
        {!isUser && <ModelBadge model={msg.model} />}
        <span style={{ fontSize: "11px", color: "#4a5568", fontFamily: "'JetBrains Mono', monospace" }}>
          {isUser ? "you" : MODEL_META[msg.model]?.provider || "ai"}
        </span>
      </div>
      <div style={{
        maxWidth: "75%", padding: "14px 18px",
        background: isUser ? "#1a2744" : "#0f1923",
        border: `1px solid ${isUser ? "#2d4a8a" : "#1e2d3d"}`,
        borderRadius: isUser ? "16px 16px 4px 16px" : "16px 16px 16px 4px",
        color: isUser ? "#93c5fd" : "#e2e8f0",
        fontSize: "14px", lineHeight: "1.7",
        fontFamily: "'JetBrains Mono', monospace",
        whiteSpace: "pre-wrap", wordBreak: "break-word",
      }}>
        {msg.content}
        {msg.loading && <Cursor />}
        {!msg.content && msg.loading && (
          <span style={{ color: "#4a5568" }}>thinking...</span>
        )}
      </div>
    </div>
  );
}

function UsageBar({ usage }) {
  if (!usage) return null;
  const total = usage.prompt + usage.completion;
  const promptPct = (usage.prompt / total) * 100;
  return (
    <div style={{
      padding: "8px 16px", background: "#080e18",
      borderTop: "1px solid #1e2d3d", display: "flex",
      alignItems: "center", gap: "12px", fontSize: "11px",
      fontFamily: "'JetBrains Mono', monospace", color: "#4a5568",
    }}>
      <span>tokens</span>
      <div style={{ flex: 1, height: "4px", background: "#1e2d3d", borderRadius: "2px", overflow: "hidden" }}>
        <div style={{
          height: "100%", width: `${promptPct}%`,
          background: "linear-gradient(90deg, #3b82f6, #10b981)",
          transition: "width 0.5s ease",
        }} />
      </div>
      <span style={{ color: "#3b82f6" }}>↑{usage.prompt}</span>
      <span style={{ color: "#10b981" }}>↓{usage.completion}</span>
      <span style={{ color: "#6b7280" }}>= {total}</span>
    </div>
  );
}

// ─── Main App ─────────────────────────────────────────────────────────────────

export default function LLMDashboard() {
  const [model, setModel] = useState("claude-sonnet-4-6");
  const [systemPrompt, setSystemPrompt] = useState("You are a helpful AI assistant.");
  const [temperature, setTemperature] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(1024);
  const [input, setInput] = useState("");
  const [showSettings, setShowSettings] = useState(false);
  const [showUsagePanel, setShowUsagePanel] = useState(false);
  const [globalUsage, setGlobalUsage] = useState(null);

  const { messages, setMessages, streaming, usage, error, sendMessage, stop, clear } = useSSEChat();
  const bottomRef = useRef(null);
  const textareaRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const fetchGlobalUsage = async () => {
    try {
      const res = await fetch(`${API_BASE}/usage`);
      const data = await res.json();
      setGlobalUsage(data);
    } catch { /* offline */ }
  };

  const handleSend = () => {
    if (!input.trim() || streaming) return;
    const userMsg = { id: Date.now(), role: "user", content: input.trim() };
    const history = [...messages, userMsg].filter(m => !m.loading).map(m => ({
      role: m.role, content: m.content,
    }));
    setMessages(prev => [...prev, userMsg]);
    setInput("");
    sendMessage({ model, messages: history, systemPrompt, temperature, maxTokens });
  };

  const handleKey = (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSend(); }
  };

  const currentProvider = MODEL_META[model]?.provider;
  const providerColor = MODEL_COLORS[currentProvider] || "#64ffda";

  return (
    <div style={{
      minHeight: "100vh", background: "#060d16",
      display: "flex", flexDirection: "column",
      fontFamily: "'JetBrains Mono', monospace",
    }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600&family=Syne:wght@600;800&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { background: #060d16; }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: #080e18; }
        ::-webkit-scrollbar-thumb { background: #1e2d3d; border-radius: 2px; }
        @keyframes fadeSlide {
          from { opacity: 0; transform: translateY(8px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes pulse { 0%,100% { opacity:.4 } 50% { opacity:1 } }
        .model-btn { transition: all 0.15s; cursor: pointer; }
        .model-btn:hover { transform: translateY(-1px); }
        .send-btn:hover:not(:disabled) { transform: scale(1.05); }
        .send-btn:disabled { opacity: 0.4; cursor: not-allowed; }
        textarea { resize: none; outline: none; }
        textarea::placeholder { color: #2d4a5a; }
        .settings-panel { animation: fadeSlide 0.2s ease; }
        .nav-btn { background: none; border: none; cursor: pointer; padding: 6px 10px;
          font-family: inherit; font-size: 11px; color: #4a6a80; transition: color 0.15s; }
        .nav-btn:hover { color: #94a3b8; }
      `}</style>

      {/* Header */}
      <div style={{
        borderBottom: "1px solid #1e2d3d", padding: "12px 24px",
        display: "flex", alignItems: "center", justifyContent: "space-between",
        background: "#080e18",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: "16px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
            <div style={{
              width: "8px", height: "8px", borderRadius: "50%",
              background: providerColor,
              boxShadow: `0 0 8px ${providerColor}`,
              animation: streaming ? "pulse 1s infinite" : "none",
            }} />
            <span style={{ fontFamily: "'Syne', sans-serif", fontWeight: 800, fontSize: "16px", color: "#e2e8f0", letterSpacing: "-0.02em" }}>
              LLM Gateway
            </span>
          </div>
          <span style={{ fontSize: "10px", color: "#2d4a5a" }}>v1.0</span>
        </div>

        <div style={{ display: "flex", gap: "4px", alignItems: "center" }}>
          <button className="nav-btn" onClick={() => { setShowUsagePanel(!showUsagePanel); fetchGlobalUsage(); }}>
            analytics
          </button>
          <button className="nav-btn" onClick={() => setShowSettings(!showSettings)}>
            settings
          </button>
          <button className="nav-btn" onClick={clear} style={{ color: "#7f1d1d" }}>
            clear
          </button>
        </div>
      </div>

      {/* Model Selector */}
      <div style={{
        display: "flex", gap: "8px", padding: "12px 24px",
        background: "#080e18", borderBottom: "1px solid #0f1a24",
        overflowX: "auto",
      }}>
        {Object.entries(MODEL_META).map(([id, meta]) => {
          const color = MODEL_COLORS[meta.provider];
          const active = model === id;
          return (
            <button
              key={id}
              className="model-btn"
              onClick={() => setModel(id)}
              style={{
                padding: "6px 14px", borderRadius: "6px", fontSize: "12px",
                background: active ? `${color}18` : "transparent",
                border: `1px solid ${active ? color : "#1e2d3d"}`,
                color: active ? color : "#4a6a80",
                whiteSpace: "nowrap",
              }}
            >
              {meta.label}
            </button>
          );
        })}
      </div>

      {/* Settings Panel */}
      {showSettings && (
        <div className="settings-panel" style={{
          background: "#080e18", borderBottom: "1px solid #1e2d3d",
          padding: "16px 24px", display: "grid",
          gridTemplateColumns: "1fr 160px 160px", gap: "16px",
        }}>
          <div>
            <label style={{ fontSize: "10px", color: "#4a6a80", display: "block", marginBottom: "6px" }}>
              SYSTEM PROMPT
            </label>
            <textarea
              rows={2}
              value={systemPrompt}
              onChange={e => setSystemPrompt(e.target.value)}
              style={{
                width: "100%", background: "#0a1520", border: "1px solid #1e2d3d",
                borderRadius: "6px", padding: "8px 12px", color: "#94a3b8",
                fontSize: "12px", fontFamily: "inherit",
              }}
            />
          </div>
          <div>
            <label style={{ fontSize: "10px", color: "#4a6a80", display: "block", marginBottom: "6px" }}>
              TEMPERATURE: {temperature}
            </label>
            <input type="range" min="0" max="2" step="0.1"
              value={temperature} onChange={e => setTemperature(parseFloat(e.target.value))}
              style={{ width: "100%", accentColor: providerColor }}
            />
          </div>
          <div>
            <label style={{ fontSize: "10px", color: "#4a6a80", display: "block", marginBottom: "6px" }}>
              MAX TOKENS: {maxTokens}
            </label>
            <input type="range" min="128" max="4096" step="128"
              value={maxTokens} onChange={e => setMaxTokens(parseInt(e.target.value))}
              style={{ width: "100%", accentColor: providerColor }}
            />
          </div>
        </div>
      )}

      {/* Usage Panel */}
      {showUsagePanel && globalUsage && (
        <div className="settings-panel" style={{
          background: "#080e18", borderBottom: "1px solid #1e2d3d",
          padding: "16px 24px",
        }}>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(180px, 1fr))", gap: "12px" }}>
            <div style={{ padding: "12px", background: "#0a1520", borderRadius: "8px", border: "1px solid #1e2d3d" }}>
              <div style={{ fontSize: "10px", color: "#4a6a80", marginBottom: "4px" }}>TOTAL REQUESTS</div>
              <div style={{ fontSize: "24px", color: "#e2e8f0", fontWeight: 600 }}>{globalUsage.total_requests}</div>
            </div>
            {Object.entries(globalUsage.by_provider || {}).map(([prov, stats]) => (
              <div key={prov} style={{ padding: "12px", background: "#0a1520", borderRadius: "8px", border: `1px solid ${MODEL_COLORS[prov]}33` }}>
                <div style={{ fontSize: "10px", color: MODEL_COLORS[prov], marginBottom: "4px" }}>{prov.toUpperCase()}</div>
                <div style={{ fontSize: "18px", color: "#e2e8f0", fontWeight: 600 }}>{stats.requests} <span style={{ fontSize: "11px", color: "#4a6a80" }}>reqs</span></div>
                <div style={{ fontSize: "12px", color: "#4a6a80" }}>{stats.total_tokens.toLocaleString()} tokens</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Messages */}
      <div style={{ flex: 1, overflowY: "auto", padding: "24px" }}>
        {messages.length === 0 && (
          <div style={{ textAlign: "center", marginTop: "80px", color: "#1e3a4a" }}>
            <div style={{ fontSize: "48px", marginBottom: "16px", opacity: 0.4 }}>⬡</div>
            <div style={{ fontFamily: "'Syne', sans-serif", fontSize: "20px", color: "#2d4a5a", marginBottom: "8px" }}>
              LLM Gateway
            </div>
            <div style={{ fontSize: "12px" }}>Route prompts across OpenAI & Anthropic models</div>
          </div>
        )}

        {messages.map(msg => <MessageBubble key={msg.id} msg={msg} />)}

        {error && (
          <div style={{
            padding: "12px 16px", background: "#1f0a0a", border: "1px solid #7f1d1d",
            borderRadius: "8px", color: "#fca5a5", fontSize: "13px",
            marginBottom: "16px", animation: "fadeSlide 0.2s ease",
          }}>
            ⚠ {error}
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* Usage bar */}
      <UsageBar usage={usage} />

      {/* Input */}
      <div style={{
        padding: "16px 24px", background: "#080e18",
        borderTop: "1px solid #1e2d3d",
        display: "flex", gap: "12px", alignItems: "flex-end",
      }}>
        <textarea
          ref={textareaRef}
          rows={3}
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={handleKey}
          placeholder="Send a message… (Enter to send, Shift+Enter for newline)"
          style={{
            flex: 1, background: "#0a1520", border: `1px solid ${input ? "#2d4a6a" : "#1e2d3d"}`,
            borderRadius: "10px", padding: "12px 16px",
            color: "#e2e8f0", fontSize: "14px", fontFamily: "inherit",
            lineHeight: "1.6", transition: "border-color 0.2s",
          }}
        />
        <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
          <button
            className="send-btn"
            onClick={streaming ? stop : handleSend}
            disabled={!streaming && !input.trim()}
            style={{
              width: "48px", height: "48px", borderRadius: "10px",
              border: "none", cursor: "pointer",
              background: streaming ? "#7f1d1d" : `linear-gradient(135deg, ${providerColor}, ${providerColor}aa)`,
              color: "#fff", fontSize: "18px",
              display: "flex", alignItems: "center", justifyContent: "center",
              transition: "all 0.15s",
            }}
          >
            {streaming ? "⏹" : "↑"}
          </button>
        </div>
      </div>
    </div>
  );
}
