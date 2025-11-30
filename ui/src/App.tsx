import { useEffect, useRef, useState, type ReactNode } from "react";
import "./app.css";
// import "./App.css"; if needed depending on your machine

const API = "http://127.0.0.1:8000";

type Json = unknown;

type Health = Json;
type ModelMeta = Json;

type ModalProps = {
  title: string;
  open: boolean;
  onClose: () => void;
  children?: ReactNode;
};

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [model, setModel] = useState<ModelMeta | null>(null);
  const [url, setUrl] = useState(
    "https://www.foxnews.com/live-news/anti-trump-no-kings-protests-october-18-2025"
  );
  const [result, setResult] = useState<Record<string, unknown> | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState("");

  const [showHealth, setShowHealth] = useState(false);
  const [showModel, setShowModel] = useState(false);

  const fetchHealth = async () => {
    setErr("");
    try {
      const res = await fetch(`${API}/healthz`);
      const data = await res.json();
      setHealth(data);
      setShowHealth(true);
    } catch (e) {
      setErr(String(e));
    }
  };

  const fetchModel = async () => {
    setErr("");
    try {
      const res = await fetch(`${API}/model`);
      const data = await res.json();
      setModel(data);
      setShowModel(true);
    } catch (e) {
      setErr(String(e));
    }
  };

  const classifyUrl = async () => {
    setErr("");
    setLoading(true);
    setResult(null);
    try {
      const res = await fetch(`${API}/predict_url`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setResult(await res.json());
    } catch (e) {
      setErr(String(e));
    } finally {
      setLoading(false);
    }
  };
  
  const classifyUrlShap = async () => {
  setErr("");
  setLoading(true);
  setResult(null);
  try {
    const res = await fetch(`${API}/predict_url_shap`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    setResult(await res.json());
  } catch (e) {
    setErr(String(e));
  } finally {
    setLoading(false);
  }
};

  return (
    <div className="page">
      <header className="header">
        <div className="title">
          <div className="logo">NB</div>
          <div>
            <h1>News Bias Demo</h1>
          </div>
        </div>

        <div className="actions">
          <button className="btn btn-outline" onClick={fetchHealth}>Health</button>
          <button className="btn btn-outline" onClick={fetchModel}>Model</button>
        </div>
      </header>

      <main className="container">
        <section className="card">
          <div className="card-head">
            <h2>Classify an Article</h2>
          </div>

          <div className="field">
            <label htmlFor="url" className="label">Article URL</label>
            <input
              id="url"
              className="input"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="https://example.com/article"
            />
          </div>

          <div className="row">
            <button className="btn btn-primary" onClick={classifyUrl} disabled={loading}>
              {loading ? <span className="spinner" aria-hidden /> : null}
              {loading ? "Classifying..." : "Classify"}
            </button>
            <button className="btn btn-primary" onClick={classifyUrlShap} disabled={loading}>
              {loading ? <span className="spinner" aria-hidden /> : null}
              {loading ? "Classifying w/ SHAP..." : "Classify w/ SHAP"}
            </button>
          </div>

          {err && (
            <div role="alert" className="alert">
              <strong>Error:</strong> {err}
            </div>
          )}

          {result && (() => {
            const arr = Array.isArray(result) ? result : [result];
            const first = arr[0] as any;
            const summary = first?.summary || first?.summary?.text; // if your field is nested
            return summary ? (
              <>
                <h3 className="result-title">Summary</h3>
                <div className="summary">{summary}</div>
                <div className="divider" />
              </>
            ) : null;
          })()}


          {result && (
            <>
              <div className="divider" />
              <div className="result">
                <h3 className="result-title">Result</h3>
                <pre className="code">{JSON.stringify(result, null, 2)}</pre>
              </div>
            </>
          )}
        </section>
      </main>

      <footer className="footer">
        <span className="muted">CAI 6605 | News Bias Demo By: Daniel Leniz, Louis-Marie Mondesir, and Robert Malloy</span>
        <a href="https://github.com/DanielLeniz/newsbias" className="link">GitHub</a>
      </footer>

      <Modal title="Health" open={showHealth} onClose={() => setShowHealth(false)}>
        {health ? (
          <pre className="code">{JSON.stringify(health, null, 2)}</pre>
        ) : (
          <p className="muted">No health response yet.</p>
        )}
      </Modal>

      <Modal title="Model" open={showModel} onClose={() => setShowModel(false)}>
        {model ? (
          <pre className="code">{JSON.stringify(model, null, 2)}</pre>
        ) : (
          <p className="muted">No model metadata yet.</p>
        )}
      </Modal>
    </div>
  );
}

function Modal({ title, open, onClose, children }: ModalProps) {
  const dialogRef = useRef<HTMLDivElement | null>(null);
  const closeBtnRef = useRef<HTMLButtonElement | null>(null);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose?.();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  useEffect(() => {
    if (open && closeBtnRef.current) {
      closeBtnRef.current.focus();
    }
  }, [open]);

  if (!open) return null;

  return (
    <div
      className="modal-overlay"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose?.();
      }}
    >
      <div
        className="modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="modal-title"
        ref={dialogRef}
      >
        <div className="modal-head">
          <h2 id="modal-title">{title}</h2>
          <button
            ref={closeBtnRef}
            className="icon-btn"
            aria-label="Close"
            onClick={onClose}
          >
            ×
          </button>
        </div>
        <div className="modal-body">{children}</div>
      </div>
    </div>
  );
}
