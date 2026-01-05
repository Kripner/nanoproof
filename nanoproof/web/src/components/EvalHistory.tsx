import { EvalResult } from '../types';

interface EvalHistoryProps {
  history: EvalResult[];
}

export function EvalHistory({ history }: EvalHistoryProps) {
  if (history.length === 0) {
    return (
      <div className="card">
        <div className="card-title">Evaluations</div>
        <div style={{ color: 'var(--text-muted)', fontSize: 12, textAlign: 'center', padding: 20 }}>
          No evaluations yet
        </div>
      </div>
    );
  }

  // Group by dataset and get trends
  const datasets: Record<string, EvalResult[]> = {};
  for (const result of history) {
    if (!datasets[result.dataset]) datasets[result.dataset] = [];
    datasets[result.dataset].push(result);
  }

  return (
    <div className="card">
      <div className="card-title">Evaluations</div>
      <div className="eval-history">
        {Object.entries(datasets).map(([dataset, results]) => {
          const recent = results.slice(-5);
          const last = recent[recent.length - 1];
          const prev = recent.length > 1 ? recent[recent.length - 2] : null;
          
          let trend = 'stable';
          if (prev) {
            if (last.success_rate > prev.success_rate + 0.01) trend = 'improving';
            else if (last.success_rate < prev.success_rate - 0.01) trend = 'declining';
          }

          return (
            <div key={dataset} style={{ marginBottom: 12 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                <span className="eval-dataset">{dataset}</span>
                <span className={`eval-rate ${trend}`}>
                  {(last.success_rate * 100).toFixed(1)}%
                </span>
              </div>
              <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                {recent.map((r, i) => (
                  <span key={i}>
                    {i > 0 && ' → '}
                    {(r.success_rate * 100).toFixed(1)}%
                  </span>
                ))}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

