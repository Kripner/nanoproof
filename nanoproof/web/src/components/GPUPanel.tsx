import { GPU } from '../types';

interface GPUPanelProps {
  gpus: GPU[];
}

export function GPUPanel({ gpus }: GPUPanelProps) {
  if (gpus.length === 0) {
    return (
      <div className="card">
        <div className="card-title">GPUs</div>
        <div style={{ color: 'var(--text-muted)', fontSize: 12, textAlign: 'center', padding: 20 }}>
          No GPU data available
        </div>
      </div>
    );
  }

  return (
    <div className="card gpu-card">
      <div className="card-title">GPUs</div>
      <div className="gpu-grid">
        {gpus.map((gpu) => {
          const memPercent = gpu.memory_total > 0 ? (gpu.memory_used / gpu.memory_total) * 100 : 0;
          const memClass = memPercent > 90 ? 'high' : memPercent > 70 ? 'medium' : 'low';
          const utilClass = gpu.utilization > 90 ? 'high' : gpu.utilization > 70 ? 'medium' : 'low';
          
          return (
            <div key={gpu.id} className="gpu-item-compact">
              <div className="gpu-header-compact">
                <span className="gpu-name-compact">GPU {gpu.id}</span>
                <span className="gpu-subname">{gpu.name}</span>
              </div>
              <div className="gpu-metric-inline">
                <span className="gpu-metric-label-inline">Util</span>
                <div className="gpu-bar-inline">
                  <div 
                    className={`gpu-bar-fill ${utilClass}`}
                    style={{ width: `${gpu.utilization}%` }}
                  />
                </div>
                <span className={`gpu-metric-value-inline ${utilClass}`}>{gpu.utilization.toFixed(0)}%</span>
              </div>
              <div className="gpu-metric-inline">
                <span className="gpu-metric-label-inline">Mem</span>
                <div className="gpu-bar-inline">
                  <div 
                    className={`gpu-bar-fill ${memClass}`}
                    style={{ width: `${memPercent}%` }}
                  />
                </div>
                <span className={`gpu-metric-value-inline ${memClass}`}>
                  {(gpu.memory_used / 1024).toFixed(0)}/{(gpu.memory_total / 1024).toFixed(0)}G
                </span>
              </div>
              <div className="gpu-stats-compact">
                <span>Q:{gpu.inference_queue_size}</span>
                <span>W:{gpu.avg_wait_time_ms.toFixed(0)}ms</span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
