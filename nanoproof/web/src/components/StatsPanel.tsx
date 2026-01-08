import { CollectionStats, TrainingStats, EvalProgress, EvalResult } from '../types';

interface StatsPanelProps {
  collection: CollectionStats;
  training: TrainingStats;
  phase: string;
  replayBufferSize: number;
  evalProgress: EvalProgress;
  evalHistory: EvalResult[];
}

function formatMs(ms: number): string {
  if (ms < 1) return `${(ms * 1000).toFixed(0)}μs`;
  if (ms < 1000) return `${ms.toFixed(1)}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
}

export function StatsPanel({ collection, training, phase, replayBufferSize, evalProgress, evalHistory }: StatsPanelProps) {
  const expansionsPerSecond = collection.elapsed > 0 
    ? collection.expansions / collection.elapsed 
    : 0;

  return (
    <div className="card">
      <div className="card-title">
        {phase === 'collecting' ? 'Collection' : phase === 'training' ? 'Training' : phase === 'evaluating' ? 'Evaluation' : 'Stats'}
      </div>

      {phase === 'collecting' && (
        <>
          <div className="stats-grid">
            <div className="stat">
              <div className="stat-value">{collection.proofs_attempted}</div>
              <div className="stat-label">Proof Attempts</div>
            </div>
            <div className="stat">
              <div className="stat-value">{collection.proofs_successful}</div>
              <div className="stat-label">Proofs Found</div>
            </div>
            <div className="stat">
              <div className="stat-value">{(collection.success_rate * 100).toFixed(1)}%</div>
              <div className="stat-label">Success Rate</div>
            </div>
          </div>

          <div className="stats-details">
            <div className="stats-detail-row">
              <span>Replay buffer:</span>
              <span style={{ color: 'var(--accent-cyan)', fontWeight: 600 }}>{replayBufferSize.toLocaleString()}</span>
            </div>
            <div className="stats-detail-row">
              <span>Expansions:</span>
              <span>{collection.expansions.toLocaleString()}</span>
            </div>
            <div className="stats-detail-row">
              <span>Expansions/sec:</span>
              <span style={{ color: 'var(--accent-green)' }}>{expansionsPerSecond.toFixed(1)}</span>
            </div>
            {collection.wait_time_median > 0 && (
              <div className="stats-detail-row">
                <span>Batch wait (med):</span>
                <span>{formatMs(collection.wait_time_median * 1000)}</span>
              </div>
            )}
          </div>
        </>
      )}

      {phase === 'training' && (
        <>
          <div className="stats-grid">
            <div className="stat">
              <div className="stat-value">{training.loss.toFixed(4)}</div>
              <div className="stat-label">Loss</div>
            </div>
            <div className="stat">
              <div className="stat-value">{(training.num_tokens / 1000).toFixed(0)}k</div>
              <div className="stat-label">Tokens</div>
            </div>
            <div className="stat">
              <div className="stat-value">{training.step}</div>
              <div className="stat-label">Step</div>
            </div>
          </div>
          <div className="stats-details">
            <div className="stats-detail-row">
              <span>Replay buffer:</span>
              <span style={{ color: 'var(--accent-cyan)', fontWeight: 600 }}>{replayBufferSize.toLocaleString()}</span>
            </div>
          </div>
        </>
      )}

      {phase === 'evaluating' && (
        <>
          {evalProgress.active ? (
            <>
              <div className="stats-grid">
                <div className="stat">
                  <div className="stat-value">{evalProgress.current}</div>
                  <div className="stat-label">Proofs Attempted</div>
                </div>
                <div className="stat">
                  <div className="stat-value">{evalProgress.solved}</div>
                  <div className="stat-label">Proofs Found</div>
                </div>
                <div className="stat">
                  <div className="stat-value">
                    {evalProgress.current > 0 
                      ? ((evalProgress.solved / evalProgress.current) * 100).toFixed(1) 
                      : 0}%
                  </div>
                  <div className="stat-label">Success Rate</div>
                </div>
                <div className="stat">
                  <div className="stat-value" style={{ color: evalProgress.errors > 0 ? 'var(--accent-red)' : undefined }}>
                    {evalProgress.errors}
                  </div>
                  <div className="stat-label">Errors</div>
                </div>
              </div>

              <div className="progress-bar" style={{ marginTop: 12 }}>
                <div 
                  className="progress-bar-fill eval" 
                  style={{ width: `${evalProgress.progress_percent}%` }} 
                />
              </div>
              <div style={{ fontSize: 'var(--font-sm)', color: 'var(--text-secondary)', marginTop: 4, textAlign: 'center' }}>
                {evalProgress.dataset}: {evalProgress.current} / {evalProgress.total} ({evalProgress.progress_percent.toFixed(1)}%)
                {evalProgress.errors > 0 && <span style={{ color: 'var(--accent-red)' }}> · {evalProgress.errors} errors</span>}
              </div>
            </>
          ) : (
            <div style={{ textAlign: 'center', padding: 20, color: 'var(--accent-yellow)' }}>
              Preparing evaluation...
            </div>
          )}
        </>
      )}

      {phase === 'idle' && (
        <>
          <div style={{ textAlign: 'center', padding: 20, color: 'var(--text-muted)' }}>
            Idle
          </div>
          <div className="stats-details">
            <div className="stats-detail-row">
              <span>Replay buffer:</span>
              <span style={{ color: 'var(--accent-cyan)', fontWeight: 600 }}>{replayBufferSize.toLocaleString()}</span>
            </div>
          </div>
        </>
      )}

      {/* Eval History */}
      {evalHistory.length > 0 && (
        <div style={{ marginTop: 16, paddingTop: 12, borderTop: '1px solid var(--border)' }}>
          <div style={{ fontSize: 'var(--font-sm)', color: 'var(--text-muted)', marginBottom: 8 }}>Eval History</div>
          <div className="eval-history">
            {(() => {
              const datasets: Record<string, typeof evalHistory> = {};
              for (const result of evalHistory) {
                if (!datasets[result.dataset]) datasets[result.dataset] = [];
                datasets[result.dataset].push(result);
              }
              return Object.entries(datasets).map(([dataset, results]) => {
                const recent = results.slice(-5);
                const last = recent[recent.length - 1];
                const prev = recent.length > 1 ? recent[recent.length - 2] : null;
                
                let trend = 'stable';
                if (prev) {
                  if (last.success_rate > prev.success_rate + 0.01) trend = 'improving';
                  else if (last.success_rate < prev.success_rate - 0.01) trend = 'declining';
                }

                return (
                  <div key={dataset} style={{ marginBottom: 8 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 2 }}>
                      <span className="eval-dataset">{dataset}</span>
                      <span className={`eval-rate ${trend}`}>
                        {(last.success_rate * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div style={{ fontSize: 'var(--font-xs)', color: 'var(--text-muted)' }}>
                      {recent.map((r, i) => (
                        <span key={i}>
                          {i > 0 && ' → '}
                          {(r.success_rate * 100).toFixed(1)}%
                        </span>
                      ))}
                    </div>
                  </div>
                );
              });
            })()}
          </div>
        </div>
      )}
    </div>
  );
}
