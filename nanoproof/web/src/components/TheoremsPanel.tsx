import { useState, useEffect, useMemo } from 'react';

interface DatasetEntry {
  name: string;
  theorem_count: number;
}

type Outcome = 'proven' | 'unproven' | 'error';

interface TheoremAttempt {
  step: number;
  outcome: Outcome;
  error: string | null;
  num_simulations: number;
  num_iterations: number;
  num_transitions: number;
  weight_after: number;
}

interface TheoremHistory {
  dataset: string;
  id: string;
  history: TheoremAttempt[];
  current_weight: number;
}

export function TheoremsPanel() {
  const [datasets, setDatasets] = useState<DatasetEntry[]>([]);
  const [dataset, setDataset] = useState<string>('');
  const [theoremId, setTheoremId] = useState<string>('');
  const [submittedQuery, setSubmittedQuery] = useState<{ dataset: string; id: string } | null>(null);
  const [history, setHistory] = useState<TheoremHistory | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const res = await fetch('/api/theorems/datasets');
        if (!res.ok) return;
        const data = await res.json();
        if (!alive) return;
        const ds: DatasetEntry[] = data.datasets || [];
        setDatasets(ds);
        if (ds.length > 0) setDataset((prev) => (prev || ds[0].name));
      } catch {
        // ignore
      }
    })();
    return () => {
      alive = false;
    };
  }, []);

  useEffect(() => {
    if (!submittedQuery) {
      setHistory(null);
      return;
    }
    let alive = true;
    setLoading(true);
    setError(null);
    (async () => {
      try {
        const res = await fetch(
          `/api/theorems/${encodeURIComponent(submittedQuery.dataset)}/${encodeURIComponent(submittedQuery.id)}`,
        );
        if (!alive) return;
        if (!res.ok) {
          setHistory(null);
          setError(`Server returned ${res.status}`);
          return;
        }
        const data = await res.json();
        if (!alive) return;
        setHistory(data);
        if ((data.history || []).length === 0) {
          setError('No attempts recorded for this theorem yet');
        }
      } catch (e) {
        if (alive) {
          setHistory(null);
          setError('Network error');
        }
      } finally {
        if (alive) setLoading(false);
      }
    })();
    return () => {
      alive = false;
    };
  }, [submittedQuery?.dataset, submittedQuery?.id]);

  const summary = useMemo(() => {
    if (!history) return null;
    const proven = history.history.filter((a) => a.outcome === 'proven').length;
    const unproven = history.history.filter((a) => a.outcome === 'unproven').length;
    const errors = history.history.filter((a) => a.outcome === 'error').length;
    return { proven, unproven, errors, total: history.history.length };
  }, [history]);

  const onSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const id = theoremId.trim();
    if (!dataset || !id) return;
    setSubmittedQuery({ dataset, id });
  };

  return (
    <div className="data-panel">
      <div className="data-main" style={{ flex: 1 }}>
        <div className="data-section">
          <div className="data-section-title">Theorem lookup</div>
          <form onSubmit={onSubmit} style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            <select
              value={dataset}
              onChange={(e) => setDataset(e.target.value)}
              style={{ minWidth: 160 }}
            >
              {datasets.length === 0 ? (
                <option value="">(no datasets)</option>
              ) : (
                datasets.map((d) => (
                  <option key={d.name} value={d.name}>
                    {d.name} ({d.theorem_count})
                  </option>
                ))
              )}
            </select>
            <input
              type="text"
              placeholder="theorem id (e.g. lean_workbook_42)"
              value={theoremId}
              onChange={(e) => setTheoremId(e.target.value)}
              style={{ flex: 1, minWidth: 240 }}
            />
            <button type="submit" disabled={!dataset || !theoremId.trim()}>
              Show history
            </button>
          </form>
        </div>

        {submittedQuery && (
          <div className="data-section">
            <div className="data-section-title">
              {submittedQuery.dataset}/{submittedQuery.id}
              {summary && (
                <span className="data-section-count">
                  {summary.proven}p / {summary.unproven}u / {summary.errors}e of {summary.total}
                  {history && (
                    <>
                      {' · weight '}
                      {history.current_weight.toExponential(2)}
                    </>
                  )}
                </span>
              )}
            </div>
            {loading ? (
              <div className="replay-loading">Loading…</div>
            ) : error && (!history || history.history.length === 0) ? (
              <div className="replay-empty">{error}</div>
            ) : history && history.history.length > 0 ? (
              <div className="replay-list">
                {history.history.map((a, i) => (
                  <div key={i} className="proof-item">
                    <span className="proof-name">
                      <span className={`outcome-badge outcome-${a.outcome}`}>{a.outcome}</span>
                      {' '}step {a.step.toString().padStart(5, '0')}
                    </span>
                    <span className="proof-meta">
                      {a.num_simulations} sims · {a.num_iterations} iters
                      {a.outcome === 'proven' && a.num_transitions > 0 && (
                        <>{' · '}{a.num_transitions} trans</>
                      )}
                      {' · w='}{a.weight_after.toExponential(2)}
                      {a.error && <>{' · '}{a.error.slice(0, 80)}</>}
                    </span>
                  </div>
                ))}
              </div>
            ) : null}
          </div>
        )}
      </div>
    </div>
  );
}
