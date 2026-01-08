import { CollectionStats, TrainingStats, EvalProgress } from '../types';

interface StatsPanelProps {
  collection: CollectionStats;
  training: TrainingStats;
  phase: string;
  replayBufferSize: number;
  evalProgress: EvalProgress;
}

function formatMs(ms: number): string {
  if (ms < 1) return `${(ms * 1000).toFixed(0)}μs`;
  if (ms < 1000) return `${ms.toFixed(1)}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
}

export function StatsPanel({ collection, training, phase, replayBufferSize, evalProgress }: StatsPanelProps) {
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
    </div>
  );
}
