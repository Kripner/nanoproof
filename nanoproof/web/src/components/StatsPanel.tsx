import { CollectionStats, TrainingStats } from '../types';

interface StatsPanelProps {
  collection: CollectionStats;
  training: TrainingStats;
  phase: string;
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

export function StatsPanel({ collection, training, phase }: StatsPanelProps) {
  const progress = collection.target_samples > 0 
    ? Math.min(100, (collection.samples_collected / collection.target_samples) * 100)
    : 0;

  return (
    <div className="card">
      <div className="card-title">
        {phase === 'collecting' ? 'Collection' : phase === 'training' ? 'Training' : 'Stats'}
      </div>

      {phase === 'collecting' && (
        <>
          <div className="stats-grid">
            <div className="stat">
              <div className="stat-value">{collection.samples_collected}</div>
              <div className="stat-label">Samples</div>
            </div>
            <div className="stat">
              <div className="stat-value">{collection.proofs_successful}</div>
              <div className="stat-label">Proofs</div>
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
            {collection.samples_collected} / {collection.target_samples} ({progress.toFixed(1)}%)
          </div>

          <div style={{ marginTop: 16, fontSize: 12, color: 'var(--text-secondary)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
              <span>Proofs attempted:</span>
              <span>{collection.proofs_attempted}</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
              <span>Expansions:</span>
              <span>{collection.expansions.toLocaleString()}</span>
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
      )}

      {phase === 'evaluating' && (
        <div style={{ textAlign: 'center', padding: 20, color: 'var(--accent-yellow)' }}>
          Evaluating model...
        </div>
      )}

      {phase === 'idle' && (
        <div style={{ textAlign: 'center', padding: 20, color: 'var(--text-muted)' }}>
          Idle
        </div>
      )}
    </div>
  );
}

