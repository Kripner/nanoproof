import { CollectionStats, TrainingStats, EvalProgress } from '../types';

interface StatsPanelProps {
  collection: CollectionStats;
  training: TrainingStats;
  phase: string;
  replayBufferSize: number;
  evalProgress: EvalProgress;
}

function formatDuration(seconds: number): string {
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${Math.floor(seconds % 60)}s`;
  return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
}

function formatMs(ms: number): string {
  if (ms < 1) return `${(ms * 1000).toFixed(0)}μs`;
  if (ms < 1000) return `${ms.toFixed(1)}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
}

export function StatsPanel({ collection, training, phase, replayBufferSize, evalProgress }: StatsPanelProps) {
  const progress = collection.target_samples > 0 
    ? Math.min(100, (collection.samples_collected / collection.target_samples) * 100)
    : 0;

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
              <div className="stat-label">Attempts</div>
            </div>
            <div className="stat">
              <div className="stat-value">{collection.proofs_successful}</div>
              <div className="stat-label">Solved</div>
            </div>
            <div className="stat">
              <div className="stat-value">{(collection.success_rate * 100).toFixed(1)}%</div>
              <div className="stat-label">Success</div>
            </div>
            <div className="stat">
              <div className="stat-value">{formatDuration(collection.elapsed)}</div>
              <div className="stat-label">Elapsed</div>
            </div>
          </div>

          <div className="progress-bar">
            <div className="progress-bar-fill" style={{ width: `${progress}%` }} />
          </div>
          <div style={{ fontSize: 11, color: 'var(--text-secondary)', marginTop: 4, textAlign: 'center' }}>
            {collection.samples_collected} / {collection.target_samples} transitions ({progress.toFixed(1)}%)
          </div>

          <div style={{ marginTop: 16, fontSize: 12, color: 'var(--text-secondary)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
              <span>Replay buffer:</span>
              <span style={{ color: 'var(--accent-cyan)', fontWeight: 600 }}>{replayBufferSize.toLocaleString()}</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
              <span>Expansions:</span>
              <span>{collection.expansions.toLocaleString()}</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
              <span>Expansions/sec:</span>
              <span style={{ color: 'var(--accent-green)' }}>{expansionsPerSecond.toFixed(1)}</span>
            </div>
            {collection.wait_time_median > 0 && (
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
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
          <div style={{ marginTop: 16, fontSize: 12, color: 'var(--text-secondary)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
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
                  <div className="stat-value">{evalProgress.current}/{evalProgress.total}</div>
                  <div className="stat-label">Progress</div>
                </div>
                <div className="stat">
                  <div className="stat-value">{evalProgress.solved}</div>
                  <div className="stat-label">Solved</div>
                </div>
                <div className="stat">
                  <div className="stat-value">{evalProgress.errors}</div>
                  <div className="stat-label">Errors</div>
                </div>
                <div className="stat">
                  <div className="stat-value">
                    {evalProgress.total > 0 
                      ? ((evalProgress.solved / Math.max(evalProgress.current, 1)) * 100).toFixed(1) 
                      : 0}%
                  </div>
                  <div className="stat-label">Success</div>
                </div>
              </div>

              <div className="progress-bar" style={{ marginTop: 12 }}>
                <div 
                  className="progress-bar-fill eval" 
                  style={{ width: `${evalProgress.progress_percent}%` }} 
                />
              </div>
              <div style={{ fontSize: 11, color: 'var(--text-secondary)', marginTop: 4, textAlign: 'center' }}>
                Evaluating {evalProgress.dataset} ({evalProgress.progress_percent.toFixed(1)}%)
              </div>
            </>
          ) : (
            <div style={{ textAlign: 'center', padding: 20, color: 'var(--accent-yellow)' }}>
              Preparing evaluation...
            </div>
          )}
          <div style={{ marginTop: 16, fontSize: 12, color: 'var(--text-secondary)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span>Replay buffer:</span>
              <span style={{ color: 'var(--accent-cyan)', fontWeight: 600 }}>{replayBufferSize.toLocaleString()}</span>
            </div>
          </div>
        </>
      )}

      {phase === 'idle' && (
        <>
          <div style={{ textAlign: 'center', padding: 20, color: 'var(--text-muted)' }}>
            Idle
          </div>
          <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span>Replay buffer:</span>
              <span style={{ color: 'var(--accent-cyan)', fontWeight: 600 }}>{replayBufferSize.toLocaleString()}</span>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
