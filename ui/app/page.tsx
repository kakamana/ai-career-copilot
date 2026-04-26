"use client";

import { useState } from "react";

const API = process.env.NEXT_PUBLIC_API ?? "http://localhost:8000";

type Source = { job_id: string; score: number; snippet: string };
type ChatResponse = {
  answer: string;
  sources: Source[];
  suggested_next_actions: string[];
  profile_used: string;
  disclaimer: string;
};

const DEMO_PROMPT = "I'm a Software Engineer keen to move into a data role. What internal opportunities fit me?";

export default function Home() {
  const [query, setQuery] = useState<string>(DEMO_PROMPT);
  const [empId, setEmpId] = useState<string>("E-00042");
  const [resp, setResp] = useState<ChatResponse | null>(null);
  const [loading, setLoading] = useState(false);

  async function send() {
    setLoading(true);
    try {
      const res = await fetch(`${API}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, employee_id: empId || null, k: 5 }),
      });
      setResp(await res.json());
    } finally {
      setLoading(false);
    }
  }

  function loadDemo() {
    setQuery(DEMO_PROMPT);
    setEmpId("E-00042");
  }

  return (
    <main className="min-h-screen p-8 max-w-5xl mx-auto">
      <h1 className="text-3xl font-bold">AI Career Copilot</h1>
      <p className="opacity-70 mb-6">
        RAG over internal job postings + your profile. Templated answer in the demo,
        LangGraph + LLM in production. Always cites sources.
      </p>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mb-3">
        <div className="md:col-span-2">
          <span className="text-xs uppercase opacity-60">Question</span>
          <textarea
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className="w-full rounded-xl border p-2 text-sm h-24"
          />
        </div>
        <div>
          <span className="text-xs uppercase opacity-60">Employee ID (optional)</span>
          <input
            value={empId}
            onChange={(e) => setEmpId(e.target.value)}
            placeholder="E-00042"
            className="w-full rounded-xl border p-2 text-sm"
          />
          <div className="mt-2 flex gap-2">
            <button onClick={loadDemo} className="text-xs underline opacity-70">
              Use demo prompt
            </button>
          </div>
        </div>
      </div>

      <button
        onClick={send}
        disabled={loading}
        className="rounded-xl px-4 py-2 bg-black text-white disabled:opacity-50"
      >
        {loading ? "Thinking..." : "Ask the copilot"}
      </button>

      {resp && (
        <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="md:col-span-2 rounded-2xl border p-4 whitespace-pre-line">
            <div className="text-xs uppercase opacity-60 mb-2">Answer</div>
            {resp.answer}

            <div className="mt-6">
              <div className="text-xs uppercase opacity-60">Suggested next actions</div>
              <ul className="mt-1 list-disc list-inside text-sm">
                {resp.suggested_next_actions.map((a, i) => <li key={i}>{a}</li>)}
              </ul>
            </div>

            <p className="mt-4 text-xs italic opacity-60">{resp.disclaimer}</p>
          </div>

          <div className="rounded-2xl border p-4">
            <div className="text-xs uppercase opacity-60 mb-2">Sources</div>
            <div className="space-y-3">
              {resp.sources.map((s) => (
                <div key={s.job_id} className="border-l-2 pl-2">
                  <div className="text-sm font-mono">{s.job_id}</div>
                  <div className="text-xs opacity-60">score {s.score.toFixed(3)}</div>
                  <div className="text-xs mt-1">{s.snippet}</div>
                </div>
              ))}
            </div>
            <div className="mt-4 text-xs opacity-60">
              <span className="uppercase">Profile used</span>
              <div className="mt-1">{resp.profile_used}</div>
            </div>
          </div>
        </div>
      )}
    </main>
  );
}
