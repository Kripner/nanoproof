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
  proof_size: number | null;
  weight_after: number;
  proof: string | null;
}

interface TheoremHistory {
  dataset: string;
  id: string;
  theorem: string | null;
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
  const [selectedAttemptIdx, setSelectedAttemptIdx] = useState<number | null>(null);

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
      setSelectedAttemptIdx(null);
      return;
    }
    let alive = true;
    setLoading(true);
    setError(null);
    setSelectedAttemptIdx(null);
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
        const data: TheoremHistory = await res.json();
        if (!alive) return;
        setHistory(data);
        const provenIdx = (data.history || []).findIndex((a) => a.outcome === 'proven');
        if (provenIdx >= 0) setSelectedAttemptIdx(provenIdx);
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

  const selectedAttempt =
    selectedAttemptIdx !== null && history ? history.history[selectedAttemptIdx] : null;

  return (
    <div className="theorems-panel">
      <div className="theorems-sidebar">
        <div className="data-section">
          <div className="data-section-title">Theorem lookup</div>
          <form onSubmit={onSubmit} className="theorems-lookup-form">
            <select
              value={dataset}
              onChange={(e) => setDataset(e.target.value)}
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
            />
            <button type="submit" disabled={!dataset || !theoremId.trim()}>
              Look up
            </button>
          </form>
        </div>

        {submittedQuery && (
          <div className="data-section theorems-attempts-section">
            <div className="data-section-title">
              <span>Attempts</span>
              {summary && (
                <span className="data-section-count">
                  {summary.proven}p / {summary.unproven}u / {summary.errors}e
                  {history && (
                    <>
                      {' · w '}
                      {history.current_weight.toExponential(1)}
                    </>
                  )}
                </span>
              )}
            </div>
            {loading ? (
              <div className="replay-loading">Loading...</div>
            ) : error && (!history || history.history.length === 0) ? (
              <div className="replay-empty">{error}</div>
            ) : history && history.history.length > 0 ? (
              <div className="replay-list">
                {history.history.map((a, i) => (
                  <div
                    key={i}
                    className={`proof-item clickable ${i === selectedAttemptIdx ? 'active' : ''}`}
                    onClick={() => setSelectedAttemptIdx(i)}
                  >
                    <span className="proof-name">
                      <span className={`outcome-badge outcome-${a.outcome}`}>
                        {a.outcome[0].toUpperCase()}
                      </span>
                      {' '}step {a.step.toString().padStart(5, '0')}
                    </span>
                    <span className="proof-meta">
                      {a.num_simulations}s/{a.num_iterations}i
                      {a.outcome === 'proven' && a.proof_size != null && (
                        <>{' · '}{a.proof_size}t</>
                      )}
                    </span>
                  </div>
                ))}
              </div>
            ) : (
              <div className="replay-empty">No attempts recorded yet</div>
            )}
          </div>
        )}
      </div>

      <div className="theorems-main">
        {!submittedQuery ? (
          <div className="data-empty">Pick a dataset and theorem id to inspect.</div>
        ) : loading ? (
          <div className="replay-loading">Loading...</div>
        ) : (
          <>
            <div className="data-section theorems-source-section">
              <div className="data-section-title">
                <span>{submittedQuery.dataset}/{submittedQuery.id}</span>
                {history?.theorem && (
                  <span className="data-section-count">
                    {history.theorem.split('\n').length} lines
                  </span>
                )}
              </div>
              {history?.theorem ? (
                <pre className="modal-code state theorem-source">{history.theorem}</pre>
              ) : (
                <div className="replay-empty">
                  Theorem not found in current matchmaker.
                </div>
              )}
            </div>

            <div className="data-section theorems-detail-section">
              {selectedAttempt ? (
                <AttemptDetail attempt={selectedAttempt} />
              ) : history && history.history.length === 0 ? (
                <div className="replay-empty">No attempts recorded for this theorem yet.</div>
              ) : (
                <div className="replay-empty">Select an attempt on the left.</div>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
}

function AttemptDetail({ attempt }: { attempt: TheoremAttempt }) {
  return (
    <>
      <div className="data-section-title">
        <span>
          <span className={`outcome-badge outcome-${attempt.outcome}`}>{attempt.outcome}</span>
          {' '}step {attempt.step.toString().padStart(5, '0')}
        </span>
        <span className="data-section-count">
          {attempt.num_simulations} sims · {attempt.num_iterations} iters
          {attempt.outcome === 'proven' && attempt.proof_size != null && (
            <>{' · '}{attempt.proof_size} tactics</>
          )}
          {' · w='}{attempt.weight_after.toExponential(2)}
        </span>
      </div>
      {attempt.outcome === 'proven' && attempt.proof ? (
        <pre className="modal-code tactic theorem-proof">{attempt.proof}</pre>
      ) : attempt.outcome === 'proven' ? (
        <div className="replay-empty">Proof not available (older run without tree data).</div>
      ) : attempt.error ? (
        <pre className="modal-code state theorem-error">{attempt.error}</pre>
      ) : (
        <div className="replay-empty">
          {attempt.outcome === 'unproven'
            ? 'Search exhausted simulation budget without finding a proof.'
            : 'No proof at this attempt.'}
        </div>
      )}
    </>
  );
}
